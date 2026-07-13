#!/usr/bin/env python3
"""Ascend 独立模式 / 机内同步：虚拟同步采集（Block A indep 轨）。

独立模式：无 HCCL，每卡本地 forward+backward，落墙钟 step JSONL。
real_sync：可选机内 HCCL AllReduce（单节点校准）。
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path


def _write(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["independent", "real_sync"], required=True)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_npu  # noqa: F401

    local = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    node = int(os.environ.get("NODE_RANK", 0))
    nproc = int(os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("NPUS_PER_NODE") or 16)
    global_rank = int(os.environ.get("GLOBAL_RANK", node * nproc + local))

    torch.npu.set_device(local)
    device = torch.device(f"npu:{local}")
    hostname = socket.gethostname()

    use_dist = args.mode == "real_sync"
    if use_dist:
        import torch.distributed as dist

        dist.init_process_group(backend="hccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        global_rank = rank
    else:
        rank = global_rank
        world = 1

    # 纯 MLP：Ascend 上避免手写 attention 维踩坑；indep 轨只需制造卡间算力抖动
    class Block(nn.Module):
        def __init__(self, h: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(h)
            self.fc1 = nn.Linear(h, 4 * h, bias=False)
            self.fc2 = nn.Linear(4 * h, h, bias=False)
            self.ln2 = nn.LayerNorm(h)
            self.fc3 = nn.Linear(h, 4 * h, bias=False)
            self.fc4 = nn.Linear(4 * h, h, bias=False)

        def forward(self, x):
            y = self.fc2(F.gelu(self.fc1(self.ln1(x))))
            x = x + y
            y = self.fc4(F.gelu(self.fc3(self.ln2(x))))
            return x + y

    model = nn.Sequential(*[Block(args.hidden) for _ in range(args.layers)]).to(
        device=device, dtype=torch.float16
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    B, S, H = args.batch, args.seq, args.hidden

    def one_step() -> None:
        x = torch.randn(B, S, H, device=device, dtype=torch.float16)
        opt.zero_grad(set_to_none=True)
        y = model(x)
        loss = y.float().pow(2).mean()
        loss.backward()
        if use_dist:
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad.mul_(1.0 / world)
        opt.step()
        torch.npu.synchronize()

    out = Path(args.out_dir) / f"step_times_rank{global_rank:03d}.jsonl"
    for _ in range(args.warmup):
        one_step()
    if use_dist:
        dist.barrier()

    t_start = time.time()
    for i in range(1, args.iters + 1):
        t0 = time.time()
        one_step()
        t1 = time.time()
        _write(
            out,
            {
                "iter": i,
                "t0": t0,
                "t1": t1,
                "ms": round((t1 - t0) * 1000.0, 3),
                "rank": global_rank,
                "global_rank": global_rank,
                "node": node,
                "local": local,
                "mode": args.mode,
                "hostname": hostname,
                "t_start": t_start,
            },
        )

    Path(args.out_dir).joinpath(f"done_rank{global_rank:03d}.txt").write_text(
        f"OK {args.mode} rank={global_rank}\n", encoding="utf-8"
    )
    print(f"VSYNC_DONE mode={args.mode} rank={global_rank} iters={args.iters}", flush=True)
    if use_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
