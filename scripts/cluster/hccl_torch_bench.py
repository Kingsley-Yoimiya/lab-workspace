#!/usr/bin/env python3
"""HCCL 单 op / 融合算子 microbench（per-iter 打点版）。

用法（torchrun 起）:
  torchrun --nproc_per_node=16 --nnodes=N --node_rank=R \
    --master_addr=... --master_port=... hccl_torch_bench.py \
    --ops all_reduce,barrier,mm_all_reduce,moe_dispatch \
    --sizes 1M,16M,64M,256M \
    --iters 50 \
    --out /afs/.../scale_128.jsonl \
    [--msprof 2 --msprof-dir /afs/.../msprof_128]

本机 dry-run（无 NPU，走 gloo）:
  FAKE_NPU=1 torchrun --nproc_per_node=4 hccl_torch_bench.py --ops barrier \
    --sizes 1K --iters 3 --out /tmp/smoke.jsonl

设计要点：
- per-iter 输出到 {out}.rank{R}.periter.jsonl（下游合并）
- 每 iter 内前 barrier 对齐 → 三段计时（pre / dev / post_fence） → 后 barrier
- msprof 仅对最后 N 个 iter 开启，rank0 only（避免几十 GB）
- 融合算子 shape 走独立 SHAPES 字典；nbytes 只用于 primitive
- 融合算子签名如与本机 CANN 不符，会在 op 初始化阶段抛错并跳过该 op，不影响其他 op
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


FAKE_NPU = os.environ.get("FAKE_NPU") == "1"


def _bytes_list(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip().upper()
        if part.endswith("K"):
            out.append(int(float(part[:-1]) * 1024))
        elif part.endswith("M"):
            out.append(int(float(part[:-1]) * 1024**2))
        elif part.endswith("G"):
            out.append(int(float(part[:-1]) * 1024**3))
        elif part:
            out.append(int(part))
    return out


# ---- 融合算子 shape 集 ----
# 说明（2026-07-29 上机实测）:
#   - mm_all_reduce: 910_93 socVersion 不支持 aclnnMatmulAllReduce → 默认 SKIP。
#     只有更新版 socVersion 才 build 了 kernel。留在表里，脚本会写 err 记录。
#   - moe_combine: 独立调用需要 dispatch 后的真实 layout（A/global_bs）才能 tiling，
#     单 op 场景无法安全构造 → 默认不列入 OPS。要测请手动拼上 dispatch 输出。
#   - moe_dispatch @ 128/256 tokens 需 HCCL_BUFFSIZE=2048（默认 200MB 不够）。
FUSED_SHAPES: dict[str, list[tuple]] = {
    "mm_all_reduce":     [(2048, 4096, 4096), (2048, 8192, 8192), (4096, 8192, 8192)],
    "all_gather_mm":     [(2048, 4096, 4096), (2048, 8192, 8192), (4096, 8192, 8192)],
    "mm_reduce_scatter": [(2048, 4096, 4096), (2048, 8192, 8192), (4096, 8192, 8192)],
    "moe_dispatch":      [(64, 7168, 8),      (128, 7168, 8),     (256, 7168, 8)],
    "moe_combine":       [(64, 7168, 8),      (128, 7168, 8),     (256, 7168, 8)],
    "moe_dispatch_combine": [(64, 7168, 8),   (128, 7168, 8),     (256, 7168, 8)],
}

# MC² TP subgroup 大小（Atlas A2 硬约束：HCCS all-mesh 只到 8）
MC2_TP_SIZE = 8

# 需要 TP subgroup 的融合算子
MC2_OPS = {"mm_all_reduce", "all_gather_mm", "mm_reduce_scatter"}

PRIMITIVE_OPS = {"all_reduce", "all_gather", "reduce_scatter", "all_to_all_single", "broadcast", "barrier"}
FUSED_OPS = set(FUSED_SHAPES.keys())


# ---- Routing 分布生成器（EP dispatch 用） ----
def _make_expert_ids(num_tokens: int, topk: int, num_experts: int,
                      mode: str = "uniform", alpha: float = 2.0,
                      hot_frac: float = 0.3, device=None):
    """按不同分布生成 (num_tokens, topk) 的 expert_ids (int32)。

    mode:
      - uniform:  完全均匀
      - skewed:   前 num_experts//4 个 expert 概率提升 3x
      - hot_expert: 单一 expert 收 hot_frac 比例的 token
      - zipf:     Zipfian P(k) ∝ 1/k^alpha，expert 顺序随机打乱
    """
    import torch
    if mode == "uniform":
        return torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32)
    if mode == "skewed":
        weights = torch.ones(num_experts, dtype=torch.float32)
        weights[: max(1, num_experts // 4)] = 3.0
        weights = weights / weights.sum()
        flat = torch.multinomial(weights, num_tokens * topk, replacement=True)
        return flat.reshape(num_tokens, topk).to(device=device, dtype=torch.int32)
    if mode == "hot_expert":
        ids = torch.randint(1, num_experts, (num_tokens, topk), dtype=torch.int32)
        hot_mask = torch.rand(num_tokens, topk) < hot_frac
        ids = torch.where(hot_mask, torch.zeros_like(ids), ids)
        return ids.to(device=device)
    if mode == "zipf":
        ranks = torch.arange(1, num_experts + 1, dtype=torch.float32)
        weights = 1.0 / (ranks ** alpha)
        weights = weights / weights.sum()
        perm = torch.randperm(num_experts)
        flat = torch.multinomial(weights, num_tokens * topk, replacement=True)
        return perm[flat].reshape(num_tokens, topk).to(device=device, dtype=torch.int32)
    raise ValueError(f"unknown routing mode: {mode}")


def _maybe_inject_skew(rank: int, args) -> None:
    """在 collective 调用前给指定 rank 加 sleep，模拟"这个 rank 计算慢"。"""
    if args.skew_us <= 0:
        return
    skew_ranks = {int(x) for x in args.skew_ranks.split(",") if x.strip()}
    if rank in skew_ranks:
        time.sleep(args.skew_us / 1e6)


def _get_hcom_name(pg, local_rank_in_group: int | None = None,
                    force_init: bool = True) -> str:
    """拿 HCCL 通信域名字（融合算子 API 要求）。fake mode 返回空串。

    重要：HCCL comm handle 是 lazy-init 的。仅拿名字不做 collective 会导致
    dispatch 时报 `HcomGetCommHandleByGroup group=0 not found`。
    force_init=True 时先做一次 dist.barrier(group=pg) 强制 comm 创建。
    """
    if FAKE_NPU:
        return ""
    import torch, torch.distributed as dist
    if local_rank_in_group is None:
        local_rank_in_group = dist.get_rank(group=pg)
    if force_init:
        try:
            dist.barrier(group=pg)  # 强制 lazy comm handle 创建
        except Exception as e:
            print(f"[WARN] force_init barrier(pg) failed: {e}", file=sys.stderr)
    try:
        backend = pg._get_backend(torch.device("npu"))
        return backend.get_hccl_comm_name(local_rank_in_group)
    except Exception:
        return str(pg.group_name if hasattr(pg, "group_name") else "")


def _build_ep_subgroup(world: int, global_rank: int, ep_size: int):
    """建 EP subgroup 供 moe_dispatch/combine 用。返回 (group, ep_rank, hcom, ep_size_actual)。

    HCCL MoE dispatch 的 epWorldSize 上限是 288。千卡上必须切分。
    切法：连续 ep_size 个 rank 一组，共 world//ep_size 组。所有 rank 都要
    调用 dist.new_group 完全一致，不然会 hang。

    若 ep_size >= world，直接返回 WORLD（不切）。
    """
    import torch, torch.distributed as dist
    if FAKE_NPU or ep_size >= world:
        pg = dist.group.WORLD
        return pg, global_rank, _get_hcom_name(pg, global_rank), world

    # 必须整除
    if world % ep_size != 0:
        # 就近向下取整到 world 的因子
        for cand in range(min(ep_size, world), 1, -1):
            if world % cand == 0:
                ep_size = cand
                break

    n_groups = world // ep_size
    my_group = None
    my_rank_in_group = -1
    for g in range(n_groups):
        ranks = list(range(g * ep_size, (g + 1) * ep_size))
        pg = dist.new_group(ranks=ranks)
        if global_rank in ranks:
            my_group = pg
            my_rank_in_group = global_rank - g * ep_size
    hcom = _get_hcom_name(my_group, my_rank_in_group, force_init=True)
    return my_group, my_rank_in_group, hcom, ep_size


def _build_tp_subgroup(world: int, global_rank: int, tp_size: int):
    """建 TP subgroup 供 MC² 三兄弟用。返回 (group, tp_rank_in_group, hcom)。
    若 world 不是 tp_size 倍数，或 world < tp_size，走一个覆盖全 world 的组。"""
    import torch, torch.distributed as dist
    if FAKE_NPU:
        return dist.group.WORLD, global_rank, ""
    if world < tp_size or world % tp_size != 0:
        return dist.group.WORLD, global_rank, _get_hcom_name(dist.group.WORLD, global_rank)
    n_groups = world // tp_size
    my_group = None
    my_rank_in_group = -1
    for g in range(n_groups):
        ranks = list(range(g * tp_size, (g + 1) * tp_size))
        pg = dist.new_group(ranks=ranks)
        if global_rank in ranks:
            my_group = pg
            my_rank_in_group = global_rank - g * tp_size
    hcom = _get_hcom_name(my_group, my_rank_in_group, force_init=True)
    return my_group, my_rank_in_group, hcom


def _make_primitive(op: str, nbytes: int, world: int, elem_size: int, device, dtype):
    """构造 primitive op 的 payload 与调用闭包。返回 (call_fn, meta_dict)。"""
    import torch
    import torch.distributed as dist

    n_elem = max(world, (nbytes // elem_size // world) * world)
    if op == "all_reduce":
        t = torch.randn(n_elem, device=device, dtype=dtype)
        return (lambda: dist.all_reduce(t)), {"nbytes": n_elem * elem_size}
    if op == "broadcast":
        t = torch.randn(n_elem, device=device, dtype=dtype)
        return (lambda: dist.broadcast(t, src=0)), {"nbytes": n_elem * elem_size}
    if op == "all_gather":
        chunk = n_elem // world
        t = torch.randn(chunk, device=device, dtype=dtype)
        out = [torch.empty(chunk, device=device, dtype=dtype) for _ in range(world)]
        return (lambda: dist.all_gather(out, t)), {"nbytes": n_elem * elem_size}
    if op == "reduce_scatter":
        t = torch.randn(n_elem, device=device, dtype=dtype)
        out = torch.empty(n_elem // world, device=device, dtype=dtype)
        return (lambda: dist.reduce_scatter(out, list(t.chunk(world)))), {"nbytes": n_elem * elem_size}
    if op == "all_to_all_single":
        t = torch.randn(n_elem, device=device, dtype=dtype)
        out = torch.empty_like(t)
        return (lambda: dist.all_to_all_single(out, t)), {"nbytes": n_elem * elem_size}
    if op == "barrier":
        return (lambda: dist.barrier()), {"nbytes": 0}
    raise ValueError(f"unknown primitive op {op}")


def _make_fused(op: str, shape: tuple, world: int, device, dtype,
                hcom: str, hcom_tp: str, tp_size: int,
                hcom_ep: str = "", ep_size: int = 0, ep_rank: int = 0,
                routing_mode: str = "uniform",
                routing_alpha: float = 2.0,
                routing_hot_frac: float = 0.3,
                num_experts_override: int = 0) -> tuple[Callable, dict]:
    """构造融合算子调用闭包。签名不匹配当前 CANN 时抛 RuntimeError。
    MC² 三兄弟走 hcom_tp（TP subgroup），MoE 两个走 hcom_ep（EP subgroup，≤288）。
    """
    import torch
    if FAKE_NPU:
        import torch.distributed as dist
        M = shape[0]
        t = torch.randn(M, device=device, dtype=dtype)
        return (lambda: dist.all_reduce(t)), {"shape": shape, "nbytes": M * t.element_size(), "fake": True}

    import torch_npu  # noqa: F401

    # EP op 用 EP subgroup；未传时退回 WORLD hcom（仅 EP <=288 时可用）
    ep_hcom = hcom_ep if hcom_ep else hcom
    ep_ws = ep_size if ep_size > 0 else world
    ep_rid = ep_rank

    if op == "mm_all_reduce":
        M, K, N = shape
        x = torch.randn(M, K, device=device, dtype=dtype)
        w = torch.randn(K, N, device=device, dtype=dtype)
        return (lambda: torch_npu.npu_mm_all_reduce_base(x, w, hcom_tp)), {
            "shape": shape, "M": M, "K": K, "N": N, "tp_size": tp_size,
        }
    if op == "all_gather_mm":
        M, K, N = shape
        x = torch.randn(M, K, device=device, dtype=dtype)
        w = torch.randn(K, N, device=device, dtype=dtype)
        return (lambda: torch_npu.npu_all_gather_base_mm(x, w, hcom_tp, tp_size)), {
            "shape": shape, "M": M, "K": K, "N": N, "tp_size": tp_size,
        }
    if op == "mm_reduce_scatter":
        M, K, N = shape
        x = torch.randn(M, K, device=device, dtype=dtype)
        w = torch.randn(K, N, device=device, dtype=dtype)
        return (lambda: torch_npu.npu_mm_reduce_scatter_base(x, w, hcom_tp, tp_size)), {
            "shape": shape, "M": M, "K": K, "N": N, "tp_size": tp_size,
        }
    if op == "moe_dispatch":
        num_tokens, hidden, topk = shape
        x = torch.randn(num_tokens, hidden, device=device, dtype=dtype)
        num_experts = num_experts_override if num_experts_override > 0 else max(64, min(256, ep_ws * 4))
        eids = _make_expert_ids(num_tokens, topk, num_experts,
                                mode=routing_mode, alpha=routing_alpha,
                                hot_frac=routing_hot_frac, device=device)
        def _call():
            return torch_npu.npu_moe_distribute_dispatch(
                x=x, expert_ids=eids,
                group_ep=ep_hcom, ep_world_size=ep_ws, ep_rank_id=ep_rid,
                moe_expert_num=num_experts,
            )
        return _call, {
            "shape": shape, "num_tokens": num_tokens, "hidden": hidden, "topk": topk,
            "num_experts": num_experts, "routing": routing_mode,
            "ep_world_size": ep_ws,
            "returns_tuple": True,
        }
    if op == "moe_dispatch_combine":
        num_tokens, hidden, topk = shape
        num_experts = num_experts_override if num_experts_override > 0 else max(64, min(256, ep_ws * 4))
        x = torch.randn(num_tokens, hidden, device=device, dtype=dtype)
        eids = _make_expert_ids(num_tokens, topk, num_experts,
                                mode=routing_mode, alpha=routing_alpha,
                                hot_frac=routing_hot_frac, device=device)
        expert_scales = torch.softmax(
            torch.randn(num_tokens, topk, device=device, dtype=torch.float32), dim=-1)
        tp_send_counts = torch.zeros((1,), device=device, dtype=torch.int32)
        def _call():
            d_out = torch_npu.npu_moe_distribute_dispatch(
                x=x, expert_ids=eids,
                group_ep=ep_hcom, ep_world_size=ep_ws, ep_rank_id=ep_rid,
                moe_expert_num=num_experts,
            )
            expand_x = d_out[0]
            expand_idx = d_out[2]
            ep_send_counts = d_out[4]
            return torch_npu.npu_moe_distribute_combine(
                expand_x, eids, expand_idx, ep_send_counts, expert_scales,
                ep_hcom, ep_ws, ep_rid, num_experts,
                tp_send_counts=tp_send_counts,
                tp_world_size=1, tp_rank_id=0,
            )
        return _call, {
            "shape": shape, "num_tokens": num_tokens, "hidden": hidden, "topk": topk,
            "num_experts": num_experts, "routing": routing_mode,
            "ep_world_size": ep_ws,
        }
    if op == "moe_combine":
        # 真实签名（torch_npu 2.7）：
        # npu_moe_distribute_combine(expand_x, expert_ids, expand_idx, ep_send_counts,
        #                            expert_scales, group_ep, ep_world_size, ep_rank_id,
        #                            moe_expert_num, *, ...)
        num_tokens, hidden, topk = shape
        num_experts = num_experts_override if num_experts_override > 0 else max(64, min(256, world * 4))
        expert_ids = torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32)
        expand_x = torch.randn(num_tokens * topk, hidden, device=device, dtype=dtype)
        # expand_idx: 每个 token-topk 展开后的原 token 索引，shape (num_tokens * topk,)
        expand_idx = torch.arange(num_tokens, device=device, dtype=torch.int32).repeat_interleave(topk)
        # ep_send_counts: 每 rank 发向本 rank 的 token 数（int32），shape (ep_world_size,)
        #   随机构造一份，总和 = num_tokens * topk（假设平均分）
        avg = (num_tokens * topk) // world
        rem = (num_tokens * topk) - avg * world
        ep_send_counts = torch.full((world,), avg, device=device, dtype=torch.int32)
        if rem > 0:
            ep_send_counts[:rem] += 1
        # expert_scales: 每 token 对每个 expert 的 gating 权重（必须 fp32）
        expert_scales = torch.softmax(torch.randn(num_tokens, topk, device=device, dtype=torch.float32), dim=-1)
        # tp_send_counts: tiling 校验硬要 dim0 == tp_world_size；这里不做 TP 切，
        # 用 tp_world_size=1 + tp_send_counts=(1,) 骗过校验
        tp_send_counts = torch.zeros((1,), device=device, dtype=torch.int32)
        return (lambda: torch_npu.npu_moe_distribute_combine(
            expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales,
            hcom, world, torch.distributed.get_rank(), num_experts,
            tp_send_counts=tp_send_counts,
            tp_world_size=1, tp_rank_id=0,
        )), {"shape": shape, "num_tokens": num_tokens, "hidden": hidden, "topk": topk, "num_experts": num_experts}
    raise ValueError(f"unknown fused op {op}")


def _compute_bw(op: str, nbytes: int, avg_s: float, world: int) -> tuple[float, float]:
    """returns (alg_bw_GBps, bus_bw_GBps)"""
    if nbytes <= 0 or avg_s <= 0:
        return 0.0, 0.0
    alg_bw = nbytes / avg_s / 1e9
    if op == "all_reduce":
        bus = alg_bw * (2.0 * (world - 1) / world)
    elif op in ("all_gather", "reduce_scatter"):
        bus = alg_bw * ((world - 1) / world)
    else:
        bus = alg_bw
    return alg_bw, bus


def _run_one(op: str, shape_or_bytes, args, world: int, rank: int, host: str,
             device, dtype, elem_size: int, hcom: str, hcom_tp: str, tp_size: int,
             hcom_ep: str, ep_size: int, ep_rank: int,
             out_periter, out_summary):
    """跑一个 (op, shape/nbytes) 组合，写 per-iter + summary。"""
    import torch
    import torch.distributed as dist

    # build closure
    try:
        if op in PRIMITIVE_OPS:
            call_fn, meta = _make_primitive(op, shape_or_bytes, world, elem_size, device, dtype)
        else:
            call_fn, meta = _make_fused(
                op, shape_or_bytes, world, device, dtype, hcom, hcom_tp, tp_size,
                hcom_ep=hcom_ep, ep_size=ep_size, ep_rank=ep_rank,
                routing_mode=args.routing,
                routing_alpha=args.routing_alpha,
                routing_hot_frac=args.routing_hot_frac,
                num_experts_override=args.num_experts,
            )
    except Exception as e:
        # 融合算子签名不匹配 / 参数错 → 记录后跳过
        err = {"record": "hccl_bench_err", "op": op, "shape": shape_or_bytes,
               "world": world, "rank": rank, "err": repr(e), "tb": traceback.format_exc()}
        out_periter.write(json.dumps(err) + "\n")
        if rank == 0:
            print(f"[SKIP] op={op} shape={shape_or_bytes} err={e}", file=sys.stderr)
        return

    # warmup
    for _ in range(args.warmup):
        try:
            call_fn()
        except Exception as e:
            err = {"record": "hccl_bench_err", "op": op, "shape": shape_or_bytes,
                   "world": world, "rank": rank, "phase": "warmup", "err": repr(e)}
            out_periter.write(json.dumps(err) + "\n")
            if rank == 0:
                print(f"[SKIP warmup] op={op} shape={shape_or_bytes} err={e}", file=sys.stderr)
            return
        _sync()
    dist.barrier()

    # msprof only on last N iters, rank0 only
    msprof_from = args.iters - args.msprof if args.msprof > 0 else args.iters + 1
    prof_ctx = None
    if not FAKE_NPU and args.msprof > 0 and rank == 0:
        try:
            import torch_npu
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False,
            )
            prof_ctx = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    f"{args.msprof_dir}/{op}"
                ),
                experimental_config=experimental_config,
            )
        except Exception as e:
            print(f"[WARN] msprof init failed: {e}", file=sys.stderr)
            prof_ctx = None

    per_iter = []
    for i in range(args.iters):
        dist.barrier()                                     # 前对齐

        # skew injection: 指定 rank 上加 sleep，模拟本 rank 计算慢
        _maybe_inject_skew(rank, args)

        t_pre = time.perf_counter_ns()

        if not FAKE_NPU:
            import torch
            ev_launch = torch.npu.Event(enable_timing=True)
            ev_arrival = torch.npu.Event(enable_timing=True)
            ev_launch.record()  # kernel launch 时刻

        # 只对 last N iter 开 profile
        if prof_ctx is not None and i == msprof_from:
            prof_ctx.__enter__()

        try:
            result = call_fn()
        except Exception as e:
            err = {"record": "hccl_bench_err", "op": op, "shape": shape_or_bytes,
                   "world": world, "rank": rank, "iter": i, "err": repr(e)}
            out_periter.write(json.dumps(err) + "\n")
            if rank == 0:
                print(f"[STOP] op={op} iter={i} err={e}", file=sys.stderr)
            break

        # arrival event —— 本 rank collective 完成的设备侧时刻（≈网卡收齐后 kernel 返回）
        if not FAKE_NPU:
            ev_arrival.record()

        if prof_ctx is not None and i == args.iters - 1:
            _sync()
            prof_ctx.__exit__(None, None, None)

        coll_dev_us = None
        arrival_dev_us = None
        if not FAKE_NPU:
            # launch → arrival: 本 rank 从发起到 collective kernel 完成的时间
            # 这就是"本 rank 收齐时刻"的最好近似（不需要外部 barrier）
            arrival_dev_us = ev_launch.elapsed_time(ev_arrival) * 1e3  # ms → us

        _sync()
        t_post = time.perf_counter_ns()
        dist.barrier()                                     # 后 fence
        t_fence = time.perf_counter_ns()

        # coll_dev_us 与 arrival_dev_us 现在语义相同（都是 launch → 完成）
        coll_dev_us = arrival_dev_us

        # ★ 核心新指标（在所有额外 all_gather / cpu() 之前算，避免污染）
        if_local_gate_saved_ns = None
        if arrival_dev_us is not None:
            if_local_gate_saved_ns = (t_fence - t_pre) - int(arrival_dev_us * 1000)

        # 抓 token count 分布（moe_dispatch 返回 tuple[3] = 本 rank per local-expert counts）
        # 注意: 这段 all_gather + cpu().tolist() 每 iter 都做，1024 卡上开销大;
        # 用 args.metadata_every 控制采样频率（默认 5 = 每 5 iter 采一次）
        token_count_local = None
        token_count_max = None
        imbalance_ratio = None
        do_metadata = (i % max(1, args.metadata_every) == 0)
        if do_metadata and not FAKE_NPU and meta.get("returns_tuple") \
                and isinstance(result, (tuple, list)) and len(result) >= 4:
            try:
                counts = result[3].cpu().tolist()
                token_count_local = int(sum(counts))
                # 1024 卡上 all_gather 1024 个小 tensor 也不便宜；只在需要时做
                gathered = [torch.zeros_like(result[3]) for _ in range(world)]
                dist.all_gather(gathered, result[3])
                per_rank_totals = [int(g.sum().item()) for g in gathered]
                token_count_max = max(per_rank_totals) if per_rank_totals else None
                token_count_min = min(per_rank_totals) if per_rank_totals else 1
                if token_count_min > 0:
                    imbalance_ratio = token_count_max / token_count_min
            except Exception:
                pass

        rec = {
            "record": "hccl_bench_iter",
            "op": op, "world": world, "rank": rank, "host": host, "iter": i,
            "coll_host_ns": t_post - t_pre,
            "coll_dev_us": coll_dev_us,
            "arrival_dev_us": arrival_dev_us,
            "post_fence_ns": t_fence - t_post,
            "pre_barrier_ns": t_pre,
            "if_local_gate_saved_ns": if_local_gate_saved_ns,
            "token_count_local": token_count_local,
            "token_count_max": token_count_max,
            "imbalance_ratio": imbalance_ratio,
            "skew_us": args.skew_us,
            "profiled": prof_ctx is not None and i >= msprof_from,
            "dtype": args.dtype,
            **meta,
        }
        # 被 msprof profile 的 iter 会被显著拖慢；只保留到 periter jsonl，不进 summary 统计
        out_periter.write(json.dumps(rec) + "\n")
        if not rec["profiled"]:
            per_iter.append(rec)

    if not per_iter:
        return

    # summary（rank 局部，主机端后续 aggregate 时再算全局 p50/p90/p99）
    host_ns_arr = sorted(r["coll_host_ns"] for r in per_iter)
    n = len(host_ns_arr)
    def pct(p: float) -> int:
        idx = min(n - 1, int(round(p * (n - 1))))
        return host_ns_arr[idx]
    avg_s = sum(host_ns_arr) / n / 1e9
    nbytes = per_iter[-1].get("nbytes", 0) if per_iter else 0
    alg_bw, bus_bw = _compute_bw(op, nbytes, avg_s, world)

    # 新增：if_local_gate_saved_ns 的中位数与均值（EP 主张的量化）
    saved_arr = [r["if_local_gate_saved_ns"] for r in per_iter if r.get("if_local_gate_saved_ns") is not None]
    saved_p50 = sorted(saved_arr)[len(saved_arr)//2] if saved_arr else None
    saved_mean = int(sum(saved_arr) / len(saved_arr)) if saved_arr else None
    # imbalance ratio 中位数
    imb_arr = [r["imbalance_ratio"] for r in per_iter if r.get("imbalance_ratio") is not None]
    imb_p50 = sorted(imb_arr)[len(imb_arr)//2] if imb_arr else None
    # arrival_dev_us 的 p50/p90
    arr_us_arr = sorted([r["arrival_dev_us"] for r in per_iter if r.get("arrival_dev_us") is not None])
    arr_p50 = arr_us_arr[len(arr_us_arr)//2] if arr_us_arr else None
    arr_p90 = arr_us_arr[int(len(arr_us_arr)*0.9)] if len(arr_us_arr) > 1 else None

    summary = {
        "record": "hccl_bench_summary",
        "op": op, "world": world, "rank": rank, "host": host,
        "iters": n, "avg_s": avg_s,
        "p50_ns": pct(0.5), "p90_ns": pct(0.9), "p99_ns": pct(0.99),
        "alg_bw_GBps": alg_bw, "bus_bw_GBps": bus_bw,
        "dtype": args.dtype,
        "shape": per_iter[-1].get("shape"),
        "nbytes": nbytes,
        # EP 研究字段
        "routing": args.routing,
        "skew_us": args.skew_us,
        "saved_rtt_p50_ns": saved_p50,
        "saved_rtt_mean_ns": saved_mean,
        "imbalance_ratio_p50": imb_p50,
        "arrival_dev_p50_us": arr_p50,
        "arrival_dev_p90_us": arr_p90,
    }
    out_summary.write(json.dumps(summary) + "\n")
    if rank == 0:
        shape_str = f"shape={per_iter[-1].get('shape')}" if per_iter[-1].get("shape") else f"nb={nbytes}"
        print(f"op={op} world={world} {shape_str} "
              f"host_p50={pct(0.5)/1e3:.1f}us p90={pct(0.9)/1e3:.1f}us p99={pct(0.99)/1e3:.1f}us "
              f"bus={bus_bw:.2f}GB/s")


def _sync() -> None:
    if FAKE_NPU:
        return
    import torch
    torch.npu.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="all_reduce,all_gather,reduce_scatter,broadcast")
    ap.add_argument("--sizes", default="1M,16M,64M,256M",
                    help="primitive nbytes 列表；融合算子走 FUSED_SHAPES")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", default="fp16",
                    choices=["fp32", "fp16", "bf16"],
                    help="融合算子建议 fp16/bf16；primitive 建议 fp32 对齐旧基线")
    ap.add_argument("--out", required=True,
                    help="e.g. /afs/.../scale_128.jsonl；实际会写 .rank{R}.periter.jsonl 和 .rank{R}.summary.jsonl")
    ap.add_argument("--msprof", type=int, default=0,
                    help="仅对 last N iter 开 msprof，rank0 only；0=关")
    ap.add_argument("--msprof-dir", default="/tmp/msprof",
                    help="msprof 输出目录（推荐指到 AFS）")
    # ---- EP 不均衡研究 ----
    ap.add_argument("--routing", default="uniform",
                    choices=["uniform", "skewed", "hot_expert", "zipf"],
                    help="MoE dispatch expert 路由分布")
    ap.add_argument("--routing-alpha", type=float, default=2.0,
                    help="zipf 分布的 alpha")
    ap.add_argument("--routing-hot-frac", type=float, default=0.3,
                    help="hot_expert 模式下热专家占比")
    ap.add_argument("--skew-us", type=int, default=0,
                    help="给指定 rank 加 sleep（μs），模拟计算慢")
    ap.add_argument("--skew-ranks", default="0",
                    help="被注入 skew 的 rank id，逗号分隔")
    ap.add_argument("--metadata-every", type=int, default=5,
                    help="每 N iter 采一次 imbalance/token_count（all_gather + cpu 昂贵，"
                         "千卡时建议 5-10；1 卡 smoke 可设 1）")
    ap.add_argument("--num-experts", type=int, default=0,
                    help="覆盖 num_experts（0=自动 max(64, min(256, ep_ws*4))）")
    ap.add_argument("--ep-size", type=int, default=256,
                    help="EP subgroup 大小；HCCL MoE dispatch 上限 288，默认 256。"
                         "若 world<=ep_size 则不切分。")
    args = ap.parse_args()

    import torch
    import torch.distributed as dist

    if FAKE_NPU:
        dist.init_process_group(backend="gloo")
        device = torch.device("cpu")
    else:
        import torch_npu  # noqa: F401
        dist.init_process_group(backend="hccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")

    try:
        _run(args, device)
    finally:
        if dist.is_initialized():
            try:
                _sync()
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _run(args, device) -> None:
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    world = dist.get_world_size()
    host = socket.gethostname()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    elem_size = torch.tensor([], dtype=dtype).element_size()
    ops = [x.strip() for x in args.ops.split(",") if x.strip()]
    sizes = _bytes_list(args.sizes)

    hcom = _get_hcom_name(dist.group.WORLD) if not FAKE_NPU else ""

    # MC² TP subgroup 按需建
    tp_size_actual = MC2_TP_SIZE if (world >= MC2_TP_SIZE and world % MC2_TP_SIZE == 0) else world
    need_tp = any(op in MC2_OPS for op in ops)
    if need_tp:
        tp_group, tp_rank_in_group, hcom_tp = _build_tp_subgroup(world, rank, MC2_TP_SIZE)
    else:
        hcom_tp = ""

    # EP subgroup 按需建（HCCL MoE dispatch epWorldSize 上限 288，千卡必须切）
    # 注意 07-30 教训：在 world=1024 内切 4×256 EP subgroup 并行做 all-to-all
    # 会挤爆网卡 CQE / HBM。真正的解法是用 nnodes<=18 直接起 256 卡 torchrun，
    # 而不是 1024 world 内切。EP subgroup 保留，但仅用于 world<=288 场景。
    need_ep = any(op in {"moe_dispatch", "moe_combine", "moe_dispatch_combine"} for op in ops)
    if need_ep and world > 288:
        if rank == 0:
            print(f"[FATAL_HINT] world={world} > 288 且 ops 含 fused MoE。"
                  f"HCCL MoE dispatch EP 域最大 288。**推荐分开跑**：",
                  file=sys.stderr)
            print(f"  - 千卡 primitive run: --ops all_reduce,barrier,... (不含 fused)",
                  file=sys.stderr)
            print(f"  - 256 卡 fused run: nnodes=16 独立 torchrun --ops moe_dispatch,...",
                  file=sys.stderr)
            print(f"[WARN] 继续尝试 ep_size={args.ep_size}, 大概率会 OOM 或 RoCE CQE 错。",
                  file=sys.stderr)
    if need_ep:
        ep_group, ep_rank, hcom_ep, ep_size_actual = _build_ep_subgroup(
            world, rank, args.ep_size)
        if rank == 0:
            print(f"[EP] ep_size={ep_size_actual} n_groups={world // ep_size_actual} "
                  f"hcom_ep={hcom_ep}", file=sys.stderr)
    else:
        hcom_ep = ""
        ep_size_actual = world
        ep_rank = rank

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    periter_path = out_path.parent / f"{out_path.stem}.rank{rank}.periter.jsonl"
    summary_path = out_path.parent / f"{out_path.stem}.rank{rank}.summary.jsonl"

    if rank == 0:
        meta = {
            "record": "run_meta", "world": world, "ops": ops, "sizes_bytes": sizes,
            "iters": args.iters, "warmup": args.warmup, "dtype": args.dtype,
            "msprof": args.msprof, "fake_npu": FAKE_NPU,
            "cann_version": os.environ.get("ASCEND_TOOLKIT_HOME", "?"),
        }
        (out_path.parent / f"{out_path.stem}.meta.json").write_text(json.dumps(meta, indent=2))

    with periter_path.open("a") as fp, summary_path.open("a") as fs:
        for op in ops:
            if op in PRIMITIVE_OPS:
                for nb in sizes:
                    _run_one(op, nb, args, world, rank, host, device, dtype, elem_size,
                             hcom, hcom_tp, tp_size_actual,
                             hcom_ep, ep_size_actual, ep_rank, fp, fs)
            elif op in FUSED_OPS:
                for shape in FUSED_SHAPES[op]:
                    _run_one(op, shape, args, world, rank, host, device, dtype, elem_size,
                             hcom, hcom_tp, tp_size_actual,
                             hcom_ep, ep_size_actual, ep_rank, fp, fs)
            else:
                if rank == 0:
                    print(f"[WARN] unknown op '{op}', skipping", file=sys.stderr)

    print(f"rank{rank} wrote {periter_path.name} + {summary_path.name}")


if __name__ == "__main__":
    main()
