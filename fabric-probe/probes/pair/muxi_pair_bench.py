#!/usr/bin/env python3
"""两pod单向MCCL P2P probe。

通信路径复用 scripts/cluster/nccl_p2p_bench.py 已验证的
torch.distributed.isend/irecv + wait 与收端pattern校验；计时停止后才汇总。
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from datetime import timedelta
from pathlib import Path


def _pattern_ok(recv_t, expected: float, sample: int = 64) -> bool:
    import torch

    n = recv_t.numel()
    idx = list(range(min(sample, n)))
    if n > sample:
        idx += list(range(max(0, n - sample), n))
    uniq = list(dict.fromkeys(idx))
    values = recv_t.view(-1)[uniq]
    return bool(torch.allclose(values, torch.full_like(values, expected), rtol=0, atol=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--edge-id", type=int, required=True)
    parser.add_argument("--src-index", type=int, required=True)
    parser.add_argument("--dst-index", type=int, required=True)
    parser.add_argument("--src-pod", required=True)
    parser.add_argument("--dst-pod", required=True)
    parser.add_argument("--hca", default="xscale_0")
    parser.add_argument("--init-timeout-s", type=int, default=60)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    print(
        "PAIR_STAGE before_init "
        f"node_rank={os.environ.get('GROUP_RANK')} rank={os.environ.get('RANK')} "
        f"master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')} "
        f"hca={os.environ.get('MCCL_IB_HCA')}",
        flush=True,
    )
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.init_timeout_s))
    print(f"PAIR_STAGE after_init rank={dist.get_rank()}", flush=True)
    try:
        rank = dist.get_rank()
        if dist.get_world_size() != 2:
            raise RuntimeError("pair probe requires world_size=2")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        n_elem = args.nbytes // torch.tensor([], dtype=torch.float32).element_size()
        send_t = (
            torch.full((n_elem,), float(args.src_index + 1), device="cuda", dtype=torch.float32)
            if rank == 0
            else None
        )
        recv_t = (
            torch.empty((n_elem,), device="cuda", dtype=torch.float32) if rank == 1 else None
        )

        for _ in range(args.warmup):
            if rank == 0:
                dist.isend(send_t, dst=1).wait()
            else:
                dist.irecv(recv_t, src=0).wait()
            torch.cuda.synchronize()
        dist.barrier()
        print(f"PAIR_STAGE warmup_done rank={rank}", flush=True)

        local_times: list[float] = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            if rank == 0:
                dist.isend(send_t, dst=1).wait()
            else:
                dist.irecv(recv_t, src=0).wait()
            torch.cuda.synchronize()
            local_times.append(time.perf_counter() - t0)

        local = torch.tensor(local_times, dtype=torch.float64, device="cuda")
        global_max = local.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        global_times = [float(x) for x in global_max.cpu().tolist()]
        print(f"PAIR_STAGE timed_done rank={rank}", flush=True)
        ok = True if rank == 0 else _pattern_ok(recv_t, float(args.src_index + 1))
        ok_t = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cuda")
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        all_ok = bool(ok_t.item())
        hosts: list[str | None] = [None, None]
        dist.all_gather_object(hosts, socket.gethostname())

        if rank == 0:
            avg_s = statistics.mean(global_times)
            record = {
                "schema_version": "muxi.pair_result.v1",
                "timing_version": "p2p.w0.1",
                "backend": "nccl",
                "primitive": "torch.distributed.isend/irecv",
                "round": args.round,
                "slot": args.slot,
                "edge_id": args.edge_id,
                "src_index": args.src_index,
                "dst_index": args.dst_index,
                "src_pod": args.src_pod,
                "dst_pod": args.dst_pod,
                "src_host": hosts[0],
                "dst_host": hosts[1],
                "src_gpu": 0,
                "dst_gpu": 0,
                "hca": args.hca,
                "nbytes": args.nbytes,
                "warmup": args.warmup,
                "iters": args.iters,
                "iters_s_global_max": global_times,
                "avg_s_global_max": avg_s,
                "bw_GBps": args.nbytes / avg_s / 1e9,
                "lat_us": avg_s * 1e6,
                "pattern_ok": all_ok,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(record) + "\n")
            print(json.dumps(record))
        dist.barrier()
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
