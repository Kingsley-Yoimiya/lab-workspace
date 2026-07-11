#!/usr/bin/env python3
"""PyTorch HCCL P2P microbenchmark（ring 邻接 + star→rank0 抽样）。

采样策略 A+B（O(N)，墙钟可控）:
  A. ring: 每 rank → (rank+1) % world
  B. star: 每 rank ↔ rank0（双向）

大 world（>=64）默认仅 ring，避免 star 在 rank0 上堆积过多 P2P 触发 HCCL SIGSEGV。
边严格串行：全局一次只测一对；size 之间额外 barrier；每边后释放 tensor。

每参与 rank 各自写 JSONL（非 rank0-only）: {out}.rank{R}.jsonl

单机: torchrun --nproc_per_node=16 hccl_p2p_bench.py --out ...
多机: 各节点 torchrun --nnodes=N --node_rank=R --master_addr=... --nproc_per_node=16 ...
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path


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
        else:
            out.append(int(part))
    return out


def _edge_list(world: int, strategies: list[str]) -> list[tuple[int, int]]:
    """去重有向边，保持稳定顺序。"""
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(src: int, dst: int) -> None:
        if src == dst:
            return
        key = (src, dst)
        if key not in seen:
            seen.add(key)
            edges.append(key)

    if "ring" in strategies:
        for i in range(world):
            add(i, (i + 1) % world)
    if "star" in strategies:
        for i in range(1, world):
            add(i, 0)
            add(0, i)
    return edges


def _pattern_ok(recv_t, expected: float, sample: int = 64) -> bool:
    """抽查首尾若干元素，避免大 buffer 全量 allclose 过慢。"""
    import torch

    n = recv_t.numel()
    if n == 0:
        return True
    idx = list(range(min(sample, n)))
    if n > sample:
        idx += list(range(max(0, n - sample), n))
    # unique preserve order
    seen: set[int] = set()
    uniq = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    vals = recv_t.view(-1)[uniq]
    return bool(torch.allclose(vals, torch.full_like(vals, expected), rtol=0, atol=0))


def _sync_barrier() -> None:
    """设备同步后再 barrier，降低 HCCL 未完成 op 撞 barrier 的风险。"""
    import torch
    import torch.distributed as dist

    torch.npu.synchronize()
    dist.barrier()


def main() -> None:
    ap = argparse.ArgumentParser(description="HCCL P2P microbench (ring+star)")
    ap.add_argument(
        "--sizes",
        default="64K,16M",
        help="消息大小锚点，逗号分隔（默认 64KiB 延迟 + 16MiB 带宽）",
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument(
        "--strategies",
        default="",
        help="抽样策略，逗号分隔: ring,star；空=自动（world>=64 仅 ring，否则 ring,star）",
    )
    ap.add_argument("--out", required=True, help="JSONL 路径前缀；每 rank 写 {stem}.rank{R}.jsonl")
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    dist.init_process_group(backend="hccl")
    try:
        _run_bench(args)
    finally:
        if dist.is_initialized():
            try:
                torch.npu.synchronize()
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _run_bench(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % 16))
    torch.npu.set_device(local_rank)
    host = socket.gethostname()
    device = f"npu:{local_rank}"

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    elem_size = torch.tensor([], dtype=dtype).element_size()
    sizes = _bytes_list(args.sizes)
    if args.strategies.strip():
        strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    else:
        # 大 world 默认仅 ring：star 在 rank0 上 O(N) 双向边易触发 HCCL 不稳定
        strategies = ["ring"] if world >= 64 else ["ring", "star"]
    edges = _edge_list(world, strategies)

    if rank == 0:
        print(
            f"hccl_p2p world={world} edges={len(edges)} "
            f"strategies={strategies} sizes={sizes} "
            f"warmup={args.warmup} iters={args.iters}"
        )

    results: list[dict] = []

    for nbytes in sizes:
        n_elem = max(1, nbytes // elem_size)
        data_bytes = n_elem * elem_size
        _sync_barrier()  # size 边界：确保上一 size 全部收尾

        for src, dst in edges:
            # 全局严格串行：一次只测一对，禁止并发 P2P
            send_t = None
            recv_t = None
            if rank == src:
                send_t = torch.full((n_elem,), float(src + 1), device=device, dtype=dtype)
            if rank == dst:
                recv_t = torch.empty((n_elem,), device=device, dtype=dtype)

            for _ in range(args.warmup):
                if rank == src:
                    dist.isend(send_t, dst=dst).wait()
                elif rank == dst:
                    dist.irecv(recv_t, src=src).wait()
            _sync_barrier()

            t0 = time.perf_counter()
            for _ in range(args.iters):
                if rank == src:
                    dist.isend(send_t, dst=dst).wait()
                elif rank == dst:
                    dist.irecv(recv_t, src=src).wait()
            _sync_barrier()
            elapsed = time.perf_counter() - t0
            avg_s = elapsed / max(1, args.iters)

            if rank in (src, dst):
                ok = True
                if rank == dst and recv_t is not None:
                    ok = _pattern_ok(recv_t, float(src + 1))

                bw_GBps = data_bytes / avg_s / 1e9 if avg_s > 0 else 0.0
                lat_us = avg_s * 1e6
                rec = {
                    "record": "hccl_p2p",
                    "src": src,
                    "dst": dst,
                    "nbytes": data_bytes,
                    "avg_s": avg_s,
                    "bw_GBps": bw_GBps,
                    "lat_us": lat_us,
                    "ok": ok,
                    "world_size": world,
                    "host": host,
                    "local_rank": local_rank,
                    "rank": rank,
                    "role": "send" if rank == src else "recv",
                }
                results.append(rec)
                if rank == src:
                    print(
                        f"p2p {src}->{dst} size={data_bytes} "
                        f"avg_ms={avg_s*1e3:.3f} bw={bw_GBps:.2f} GB/s "
                        f"lat_us={lat_us:.1f} ok={ok}"
                    )

            # 释放本边 buffer，避免大 world 长时间堆积
            del send_t, recv_t
            if rank in (src, dst):
                torch.npu.synchronize()

        _sync_barrier()  # size 结束
        if rank == 0:
            print(f"hccl_p2p size={data_bytes} done ({len(edges)} edges)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rank_path = out_path.parent / f"{out_path.stem}.rank{rank}{out_path.suffix}"
    with rank_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"rank{rank} wrote {rank_path} ({len(results)} lines)")


if __name__ == "__main__":
    main()
