#!/usr/bin/env python3
"""PyTorch NCCL collective microbenchmark (沐曦 / CUDA 对标 hccl_torch_bench.py)。

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
    ops = [x.strip() for x in args.ops.split(",") if x.strip()]
    sizes = _bytes_list(args.sizes)
    elem_size = torch.tensor([], dtype=dtype).element_size()

    results = []
    for nbytes in sizes:
        n_elem = max(1, nbytes // elem_size)
        n_elem = (n_elem // world) * world
        if n_elem < world:
            n_elem = world

        for op in ops:
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

            for _ in range(args.warmup):
                if op == "all_reduce":
                    dist.all_reduce(t)
                elif op == "broadcast":
                    dist.broadcast(t, src=0)
                elif op == "all_gather":
                    dist.all_gather(out, t)
                elif op == "reduce_scatter":
                    dist.reduce_scatter(out, list(t.chunk(world)))
                torch.cuda.synchronize()
            dist.barrier()

            t0 = time.perf_counter()
            for _ in range(args.iters):
                if op == "all_reduce":
                    dist.all_reduce(t)
                elif op == "broadcast":
                    dist.broadcast(t, src=0)
                elif op == "all_gather":
                    dist.all_gather(out, t)
                elif op == "reduce_scatter":
                    dist.reduce_scatter(out, list(t.chunk(world)))
                torch.cuda.synchronize()
            dist.barrier()
            elapsed = time.perf_counter() - t0
            avg_s = elapsed / args.iters
            data_bytes = n_elem * elem_size
            if op == "all_reduce":
                alg_bw = data_bytes / avg_s / 1e9
                bus_bw = alg_bw * (2.0 * (world - 1) / world)
            elif op in ("all_gather", "reduce_scatter"):
                alg_bw = data_bytes / avg_s / 1e9
                bus_bw = alg_bw * ((world - 1) / world)
            else:
                alg_bw = data_bytes / avg_s / 1e9
                bus_bw = alg_bw

            rec = {
                "record": "nccl_bench",
                "backend": "nccl",
                "op": op,
                "world_size": world,
                "rank": rank,
                "host": host,
                "local_rank": local_rank,
                "nbytes": data_bytes,
                "avg_s": avg_s,
                "alg_bw_GBps": alg_bw,
                "bus_bw_GBps": bus_bw,
                "dtype": args.dtype,
            }
            results.append(rec)
            if rank == 0:
                print(
                    f"op={op} world={world} size={data_bytes} "
                    f"avg_ms={avg_s*1e3:.3f} alg={alg_bw:.2f} bus={bus_bw:.2f} GB/s"
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
