#!/usr/bin/env python3
"""PyTorch NCCL P2P microbenchmark（沐曦 / CUDA 对标 hccl_p2p_bench.py）。

ring + star（world>=64 默认仅 ring）；边严格串行。
每 rank 写 {out}.rank{R}.jsonl
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
    import torch

    n = recv_t.numel()
    if n == 0:
        return True
    idx = list(range(min(sample, n)))
    if n > sample:
        idx += list(range(max(0, n - sample), n))
    seen: set[int] = set()
    uniq = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    vals = recv_t.view(-1)[uniq]
    return bool(torch.allclose(vals, torch.full_like(vals, expected), rtol=0, atol=0))


def _sync_barrier() -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.synchronize()
    dist.barrier()


def main() -> None:
    ap = argparse.ArgumentParser(description="NCCL P2P microbench (ring+star)")
    ap.add_argument("--sizes", default="64K,16M")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--strategies", default="")
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


def _run_bench(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    torch.cuda.set_device(local_rank)
    host = socket.gethostname()
    device = f"cuda:{local_rank}"

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    elem_size = torch.tensor([], dtype=dtype).element_size()
    sizes = _bytes_list(args.sizes)
    if args.strategies.strip():
        strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    else:
        strategies = ["ring"] if world >= 64 else ["ring", "star"]
    edges = _edge_list(world, strategies)

    if rank == 0:
        print(
            f"nccl_p2p world={world} edges={len(edges)} "
            f"strategies={strategies} sizes={sizes}"
        )

    results: list[dict] = []

    for nbytes in sizes:
        n_elem = max(1, nbytes // elem_size)
        data_bytes = n_elem * elem_size
        _sync_barrier()

        for src, dst in edges:
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
                    "record": "nccl_p2p",
                    "backend": "nccl",
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

            del send_t, recv_t
            if rank in (src, dst):
                torch.cuda.synchronize()

        _sync_barrier()
        if rank == 0:
            print(f"nccl_p2p size={data_bytes} done ({len(edges)} edges)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rank_path = out_path.parent / f"{out_path.stem}.rank{rank}{out_path.suffix}"
    with rank_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"rank{rank} wrote {rank_path} ({len(results)} lines)")


if __name__ == "__main__":
    main()
