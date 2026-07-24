#!/usr/bin/env python3
"""PyTorch NCCL P2P microbenchmark（沐曦 / CUDA 对标 hccl_p2p_bench.py）。

两种模式（--mode）:
  - p2p（默认）: ring + star（world>=64 默认仅 ring）；边严格串行。
  - incast: N→1 并发多对一。所有非 root rank 同时 isend→root，root 预投
    N 个 irecv 并全部 wait；测 root 侧聚合带宽 + 每 sender 公平性/尾延迟。
    用于 ECMP 排查：多对一拥塞下提高 QP 数是否救回带宽。
    注意 root 需 N×size 并发 buffer，故 incast per-sender size 取 16M–32M。
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
    ap = argparse.ArgumentParser(description="NCCL P2P microbench (p2p ring+star / incast N→1)")
    ap.add_argument("--sizes", default="64K,16M")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--strategies", default="")
    ap.add_argument("--mode", default="p2p", choices=["p2p", "incast"],
                    help="p2p=ring/star 串行边; incast=N→1 并发多对一")
    ap.add_argument("--incast-root", type=int, default=0,
                    help="incast 模式下的接收端全局 rank")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    try:
        if args.mode == "incast":
            _run_incast(args)
        else:
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


def _run_incast(args: argparse.Namespace) -> None:
    """N→1 并发多对一。所有非 root rank 同时 isend→root；root 预投 N 个
    irecv 全部 wait。测 root 侧聚合带宽 + 每 sender 完成时间（公平性/尾延迟）。
    ECMP 排查：多对一拥塞把多条流哈希到同一上行时最易暴露；提高 QP 数若
    救回聚合带宽即为 ECMP 极化的强端侧证据。
    """
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    world = dist.get_world_size()
    root = args.incast_root
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    torch.cuda.set_device(local_rank)
    host = socket.gethostname()
    device = f"cuda:{local_rank}"

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    elem_size = torch.tensor([], dtype=dtype).element_size()
    sizes = _bytes_list(args.sizes)
    senders = [r for r in range(world) if r != root]
    n_senders = len(senders)

    if rank == root:
        print(
            f"nccl_incast world={world} root={root} senders={n_senders} "
            f"sizes={sizes} (root buffers={n_senders}×size)",
            flush=True,
        )

    results: list[dict] = []

    for nbytes in sizes:
        n_elem = max(1, nbytes // elem_size)
        data_bytes = n_elem * elem_size

        # root 为每个 sender 预留独立接收 buffer；sender 只需 1 个发送 buffer。
        recv_bufs: dict[int, "torch.Tensor"] = {}
        send_t = None
        if rank == root:
            for s in senders:
                recv_bufs[s] = torch.empty((n_elem,), device=device, dtype=dtype)
        else:
            send_t = torch.full((n_elem,), float(rank + 1), device=device, dtype=dtype)

        def _one_round() -> None:
            # 并发多对一：root 先投所有 irecv，各 sender 同时 isend，全部 wait。
            if rank == root:
                reqs = [dist.irecv(recv_bufs[s], src=s) for s in senders]
                for rq in reqs:
                    rq.wait()
            else:
                dist.isend(send_t, dst=root).wait()

        for _ in range(args.warmup):
            _one_round()
        _sync_barrier()

        # 计时区：每轮 root 侧墙钟即"收齐 N 个 sender"的聚合耗时。
        # sender 侧墙钟即本 sender 自己的完成时间（用于公平性）。
        iters_s: list[float] = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _one_round()
            torch.cuda.synchronize()
            iters_s.append(time.perf_counter() - t0)
        _sync_barrier()

        avg_s = sum(iters_s) / max(1, len(iters_s))
        p50 = sorted(iters_s)[len(iters_s) // 2] if iters_s else 0.0
        p99 = sorted(iters_s)[max(0, int(len(iters_s) * 0.99) - 1)] if iters_s else 0.0

        if rank == root:
            # 聚合带宽：N 个 sender 各 data_bytes，在 avg_s 内全部收齐。
            agg_bytes = data_bytes * n_senders
            agg_bw = agg_bytes / avg_s / 1e9 if avg_s > 0 else 0.0
            per_sender_bw = data_bytes / avg_s / 1e9 if avg_s > 0 else 0.0
            print(
                f"incast N={n_senders}→{root} size={data_bytes} "
                f"avg_ms={avg_s*1e3:.3f} agg_bw={agg_bw:.2f} GB/s "
                f"per_sender_bw={per_sender_bw:.3f} GB/s p50/p99_ms={p50*1e3:.2f}/{p99*1e3:.2f}",
                flush=True,
            )
            rec = {
                "record": "nccl_incast",
                "backend": "nccl",
                "role": "recv_root",
                "root": root,
                "n_senders": n_senders,
                "nbytes": data_bytes,
                "avg_s": avg_s,
                "agg_bw_GBps": agg_bw,
                "per_sender_bw_GBps": per_sender_bw,
                "p50_s": p50,
                "p99_s": p99,
                "iters_s": iters_s,
                "world_size": world,
                "host": host,
                "local_rank": local_rank,
                "rank": rank,
            }
        else:
            sender_bw = data_bytes / avg_s / 1e9 if avg_s > 0 else 0.0
            rec = {
                "record": "nccl_incast",
                "backend": "nccl",
                "role": "send",
                "root": root,
                "n_senders": n_senders,
                "nbytes": data_bytes,
                "avg_s": avg_s,
                "sender_bw_GBps": sender_bw,
                "p50_s": p50,
                "p99_s": p99,
                "world_size": world,
                "host": host,
                "local_rank": local_rank,
                "rank": rank,
            }
        results.append(rec)

        del send_t, recv_bufs
        torch.cuda.synchronize()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rank_path = out_path.parent / f"{out_path.stem}.rank{rank}{out_path.suffix}"
    with rank_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"rank{rank} wrote {rank_path} ({len(results)} lines)")


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
