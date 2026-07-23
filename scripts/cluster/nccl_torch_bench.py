#!/usr/bin/env python3
"""PyTorch NCCL collective microbenchmark（沐曦 / CUDA 对标 hccl_torch_bench.py）。

W0.1 计时契约：
  - warmup 后 barrier；计时前再对齐；
  - 计时区仅含目标 collective + 必要 device synchronize；
  - 本 rank synchronize 后立刻停表，再做结果汇总 collective；
  - 全局吞吐按各轮最慢 rank 时间（global_max）；保留每轮 local 原始延迟。

每 rank 各自落盘:
  {out}.rank{R}.jsonl

单机: torchrun --nproc_per_node=8 nccl_torch_bench.py --out ...
多机: torchrun --nnodes=N --node_rank=R --master_addr=... --nproc_per_node=8 ...
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

# 同目录纯计算模块（无 torch 依赖）
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from nccl_torch_bench_metrics import (  # noqa: E402
    build_bench_record,
    parse_bytes_list,
    summarize_case_print,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="all_reduce,all_gather,reduce_scatter,broadcast")
    ap.add_argument("--sizes", default="1M,16M,64M,256M")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    try:
        _run_bench(args)
    finally:
        if dist.is_initialized():
            try:
                torch.cuda.synchronize()
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _do_op(op: str, dist, t, out, world: int) -> None:
    if op == "all_reduce":
        dist.all_reduce(t)
    elif op == "broadcast":
        dist.broadcast(t, src=0)
    elif op == "all_gather":
        dist.all_gather(out, t)
    elif op == "reduce_scatter":
        dist.reduce_scatter(out, list(t.chunk(world)))
    else:
        raise ValueError(f"unsupported op: {op}")


def _run_bench(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    torch.cuda.set_device(local_rank)
    host = socket.gethostname()
    device = f"cuda:{local_rank}"
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_list = [x.strip() for x in visible_devices.split(",") if x.strip()]
    physical_gpu = int(visible_list[local_rank]) if visible_list else local_rank
    print(
        f"GPU_MAPPING rank={rank} local_rank={local_rank} "
        f"physical_gpu={physical_gpu} CUDA_VISIBLE_DEVICES={visible_devices or 'unset'}",
        flush=True,
    )

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    ops = [x.strip() for x in args.ops.split(",") if x.strip()]
    sizes = parse_bytes_list(args.sizes)
    elem_size = torch.tensor([], dtype=dtype).element_size()

    results = []
    for nbytes in sizes:
        n_elem = max(1, nbytes // elem_size)
        n_elem = (n_elem // world) * world
        if n_elem < world:
            n_elem = world

        for op in ops:
            out = None
            if op == "all_reduce":
                t = torch.randn(n_elem, device=device, dtype=dtype)
            elif op == "broadcast":
                t = torch.randn(n_elem, device=device, dtype=dtype)
            elif op == "all_gather":
                chunk = n_elem // world
                t = torch.randn(chunk, device=device, dtype=dtype)
                out = [torch.empty(chunk, device=device, dtype=dtype) for _ in range(world)]
            elif op == "reduce_scatter":
                t = torch.randn(n_elem, device=device, dtype=dtype)
                out = torch.empty(n_elem // world, device=device, dtype=dtype)
            else:
                continue

            # warmup：不计时；完成后 barrier 对齐
            for _ in range(args.warmup):
                _do_op(op, dist, t, out, world)
                torch.cuda.synchronize()
            dist.barrier()

            # 计时区：仅 collective + device completion；每轮原始延迟
            iters_s_local: list[float] = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                _do_op(op, dist, t, out, world)
                torch.cuda.synchronize()
                iters_s_local.append(time.perf_counter() - t0)

            # 停表后再汇总：逐轮 global max，不计入目标计时
            local_t = torch.tensor(iters_s_local, dtype=torch.float64, device=device)
            global_t = local_t.clone()
            dist.all_reduce(global_t, op=dist.ReduceOp.MAX)
            iters_s_global_max = [float(x) for x in global_t.cpu().tolist()]

            data_bytes = n_elem * elem_size
            rec = build_bench_record(
                op=op,
                world_size=world,
                rank=rank,
                host=host,
                local_rank=local_rank,
                nbytes=data_bytes,
                dtype=args.dtype,
                iters_s_local=iters_s_local,
                iters_s_global_max=iters_s_global_max,
            )
            rec["cuda_visible_devices"] = visible_devices or None
            rec["physical_gpu"] = physical_gpu
            results.append(rec)
            if rank == 0:
                print(
                    summarize_case_print(
                        op,
                        world,
                        data_bytes,
                        rec["avg_s_local"],
                        rec["avg_s_global_max"],
                        rec["alg_bw_GBps_local"],
                        rec["bus_bw_GBps_local"],
                        rec["alg_bw_GBps_global_max"],
                        rec["bus_bw_GBps_global_max"],
                    )
                )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    rank_path = path.parent / f"{path.stem}.rank{rank}{path.suffix}"
    with rank_path.open("a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"rank{rank} wrote {rank_path} ({len(results)} lines)")


if __name__ == "__main__":
    main()
