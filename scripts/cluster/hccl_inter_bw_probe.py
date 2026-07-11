#!/usr/bin/env python3
"""机内 vs 机间 HCCL P2P 带宽探针（严格串行单对，流水线/可选双向）。

无 hccn 时用实测反推机间有效带宽：
  - intra: 同节点两卡（HCCS）
  - inter: 跨节点、同 local_rank 对齐的一对（机间 RoCE）

所有 rank 始终参与 barrier，避免死锁；非本边端点空等。
默认流水线单向（多 in-flight 再统一 wait），可选 --bidir。

用法（8 节点 × 16 卡）:
  export HCCL_BUFFSIZE=2048
  torchrun --nnodes=8 --node_rank=$R --nproc_per_node=16 \\
    --master_addr=$MASTER --master_port=$PORT \\
    hccl_inter_bw_probe.py --out /path/probe.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
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


def _build_edges(
    world: int,
    nproc: int,
    local_samples: list[int],
    modes: list[str],
) -> list[tuple[int, int, str]]:
    nnodes = world // nproc
    edges: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    def add(src: int, dst: int, kind: str) -> None:
        if src == dst or (src, dst) in seen:
            return
        seen.add((src, dst))
        edges.append((src, dst, kind))

    if "intra" in modes:
        pairs = [(0, 1), (0, 8), (7, 8), (1, 9)]
        for node in range(nnodes):
            base = node * nproc
            for a, b in pairs:
                if a < nproc and b < nproc:
                    add(base + a, base + b, "intra")
                    add(base + b, base + a, "intra")

    if "inter" in modes:
        node_pairs: list[tuple[int, int]] = []
        for i in range(nnodes):
            node_pairs.append((i, (i + 1) % nnodes))
        if nnodes >= 4:
            node_pairs.extend([(0, 2), (0, 4)])
        if nnodes >= 8:
            node_pairs.append((1, 5))
        uniq: list[tuple[int, int]] = []
        seen_np: set[tuple[int, int]] = set()
        for a, b in node_pairs:
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            if key not in seen_np:
                seen_np.add(key)
                uniq.append((a, b))
        for na, nb in uniq:
            for lr in local_samples:
                if lr >= nproc:
                    continue
                sa = na * nproc + lr
                sb = nb * nproc + lr
                add(sa, sb, "inter")
                add(sb, sa, "inter")

    return edges


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _bench_uni(
    role: str,
    peer: int,
    send_bufs: list,
    recv_bufs: list,
    warmup: int,
    iters: int,
    inflight: int,
) -> float:
    """流水线单向：每轮挂起 inflight 个请求再统一 wait，减少硬同步低估。"""
    import torch
    import torch.distributed as dist

    nbuf = len(send_bufs)

    def one_round() -> None:
        reqs = []
        for i in range(inflight):
            buf_i = i % nbuf
            if role == "send":
                reqs.append(dist.isend(send_bufs[buf_i], peer))
            else:
                reqs.append(dist.irecv(recv_bufs[buf_i], peer))
        for r in reqs:
            r.wait()
        torch.npu.synchronize()

    for _ in range(warmup):
        one_round()

    samples: list[float] = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        one_round()
        t1 = time.perf_counter()
        # 一轮传了 inflight 个消息；返回「单消息」等效时间
        samples.append((t1 - t0) / inflight)
    return _median(samples)


def _bench_bidir(
    peer: int,
    send_bufs: list,
    recv_bufs: list,
    warmup: int,
    iters: int,
    inflight: int,
) -> float:
    """双向同时：每轮 a↔b 各发 inflight 次，统计单方向等效时间。"""
    import torch
    import torch.distributed as dist

    nbuf = len(send_bufs)

    def one_round() -> None:
        reqs = []
        for i in range(inflight):
            buf_i = i % nbuf
            reqs.append(dist.isend(send_bufs[buf_i], peer))
            reqs.append(dist.irecv(recv_bufs[buf_i], peer))
        for r in reqs:
            r.wait()
        torch.npu.synchronize()

    for _ in range(warmup):
        one_round()

    samples: list[float] = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        one_round()
        t1 = time.perf_counter()
        samples.append((t1 - t0) / inflight)
    return _median(samples)


def _bench_pingpong(
    is_a: bool,
    peer: int,
    send_t,
    recv_t,
    warmup: int,
    iters: int,
) -> float:
    """A→B→A 往返；返回单程等效时间（RTT/2）。"""
    import torch
    import torch.distributed as dist

    def one_rtt() -> None:
        if is_a:
            dist.isend(send_t, peer).wait()
            dist.irecv(recv_t, peer).wait()
        else:
            dist.irecv(recv_t, peer).wait()
            dist.isend(send_t, peer).wait()
        torch.npu.synchronize()

    for _ in range(warmup):
        one_rtt()

    samples: list[float] = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        one_rtt()
        t1 = time.perf_counter()
        samples.append((t1 - t0) / 2.0)  # 单程
    return _median(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1M,16M,64M,256M")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--inflight", type=int, default=4, help="每轮 in-flight 消息数")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--modes", default="intra,inter")
    ap.add_argument("--local-samples", default="0,5,10,15")
    ap.add_argument("--bidir", action="store_true", help="双向同时收发（默认单向流水线）")
    ap.add_argument("--pingpong", action="store_true", help="往返 RTT/2 交叉验证（覆盖 bidir）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")

    host = socket.gethostname()
    gathered: list = [None] * world
    dist.all_gather_object(gathered, host)
    hosts = [str(h) for h in gathered]
    nproc = int(os.environ.get("LOCAL_WORLD_SIZE", "16"))

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    local_samples = [int(x) for x in args.local_samples.split(",") if x.strip()]
    edges = _build_edges(world, nproc, local_samples, modes)
    # bidir/pingpong：每条无向边只测一次
    if args.bidir or args.pingpong:
        canon: list[tuple[int, int, str]] = []
        seen_u: set[tuple[int, int]] = set()
        for s, d, k in edges:
            key = (min(s, d), max(s, d))
            if key in seen_u:
                continue
            seen_u.add(key)
            canon.append((key[0], key[1], k))
        edges = canon

    sizes = _bytes_list(args.sizes)
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    elem_size = torch.tensor([], dtype=dt).element_size()
    if args.pingpong:
        direction = "pingpong"
    elif args.bidir:
        direction = "bidir"
    else:
        direction = "uni_pipe"

    if rank == 0:
        print(
            f"INTER_BW_PROBE world={world} nproc={nproc} nnodes={world // nproc} "
            f"edges={len(edges)} sizes={sizes} modes={modes} dir={direction} "
            f"inflight={args.inflight} iters={args.iters} "
            f"HCCL_BUFFSIZE={os.environ.get('HCCL_BUFFSIZE', '')}",
            flush=True,
        )
        for i, (s, d, k) in enumerate(edges[:12]):
            print(
                f"  [{i}] {k} {s}<->{d}  "
                f"{hosts[s].split('.')[0]}:L{s % nproc} <-> "
                f"{hosts[d].split('.')[0]}:L{d % nproc}",
                flush=True,
            )

    out_path = Path(f"{args.out}.rank{rank}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", encoding="utf-8")

    for nbytes in sizes:
        n_elem = max(1, nbytes // elem_size)
        for src, dst, kind in edges:
            dist.barrier()
            if rank not in (src, dst):
                dist.barrier()
                continue

            # 多 buffer 避免 in-flight 复用同一块
            nbuf = max(2, min(args.inflight, 8))
            send_bufs = [
                torch.empty(n_elem, device=f"npu:{local_rank}", dtype=dt)
                for _ in range(nbuf)
            ]
            recv_bufs = [
                torch.empty(n_elem, device=f"npu:{local_rank}", dtype=dt)
                for _ in range(nbuf)
            ]
            for i, t in enumerate(send_bufs):
                t.fill_(float((src + 17 + i) % 89 + 1))

            peer = dst if rank == src else src
            if args.pingpong:
                avg_s = _bench_pingpong(
                    rank == src, peer, send_bufs[0], recv_bufs[0], args.warmup, args.iters
                )
                role = "pp_a" if rank == src else "pp_b"
            elif args.bidir:
                avg_s = _bench_bidir(
                    peer, send_bufs, recv_bufs, args.warmup, args.iters, args.inflight
                )
                role = "both"
            else:
                role = "send" if rank == src else "recv"
                avg_s = _bench_uni(
                    role, peer, send_bufs, recv_bufs, args.warmup, args.iters, args.inflight
                )

            bw = (nbytes / avg_s) / 1e9 if avg_s > 0 else 0.0
            rec = {
                "record": "hccl_inter_bw",
                "kind": kind,
                "src": src,
                "dst": dst,
                "src_host": hosts[src],
                "dst_host": hosts[dst],
                "src_local_rank": src % nproc,
                "dst_local_rank": dst % nproc,
                "nbytes": nbytes,
                "avg_s": avg_s,
                "bw_GBps": bw,
                "lat_us": avg_s * 1e6,
                "role": role,
                "rank": rank,
                "local_rank": local_rank,
                "host": host,
                "world_size": world,
                "nproc_per_node": nproc,
                "direction": direction,
                "inflight": args.inflight,
                "hccl_buffsize": os.environ.get("HCCL_BUFFSIZE"),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if rank == src:
                print(
                    f"OK {kind} {direction} {src}<->{dst} size={nbytes} "
                    f"bw={bw:.2f} GB/s lat={avg_s*1e6:.1f}us",
                    flush=True,
                )

            del send_bufs, recv_bufs
            torch.npu.empty_cache()
            dist.barrier()

    fh.close()
    dist.barrier()
    if rank == 0:
        print(f"INTER_BW_PROBE_DONE edges={len(edges)} sizes={len(sizes)}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
