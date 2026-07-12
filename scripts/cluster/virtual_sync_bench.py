#!/usr/bin/env python3
"""MUXI Phase0 虚拟同步基准：本地计算负载 + 墙钟时间戳。

模式:
  --mode independent  纯本地 kernel，不做任何集合通信（实验0虚拟侧 / 实验1）
  --mode real_sync    机内 NCCL AllReduce 真实同步（仅实验0校准，单节点8卡）

每步落盘 JSONL（每卡一份）:
  {iter, t0, t1, ms, rank, global_rank, node, local, mode, hostname}
其中 t0/t1 为 time.time() 墙钟秒。
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
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    local = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    node = int(os.environ.get("NODE_RANK", 0))
    nproc = int(os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("GPUS_PER_NODE") or 8)
    # 独立进程时可能只设 CUDA_VISIBLE_DEVICES=k 且 LOCAL_RANK=0
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis and "," not in vis and local == 0:
        try:
            phys = int(vis.strip())
        except ValueError:
            phys = local
    else:
        phys = local
    global_rank = int(os.environ.get("GLOBAL_RANK", node * nproc + phys))

    torch.cuda.set_device(0 if (vis and "," not in vis) else local)
    device = torch.device("cuda", torch.cuda.current_device())
    hostname = socket.gethostname()

    use_dist = args.mode == "real_sync"
    if use_dist:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        global_rank = rank
    else:
        rank = global_rank
        world = 1

    class Block(nn.Module):
        def __init__(self, h: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(h)
            self.qkv = nn.Linear(h, 3 * h, bias=False)
            self.proj = nn.Linear(h, h, bias=False)
            self.ln2 = nn.LayerNorm(h)
            self.fc1 = nn.Linear(h, 4 * h, bias=False)
            self.fc2 = nn.Linear(4 * h, h, bias=False)

        def forward(self, x):
            h = self.ln1(x)
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            # 简化 attention：按 head 折叠为 matmul，避免 O(S^2) 爆显存
            # 用 chunked 近似：q@k^T 对最后 256 维投影
            q2 = q[..., :256]
            k2 = k[..., :256]
            att = torch.matmul(q2, k2.transpose(-1, -2)) * (256**-0.5)
            att = torch.softmax(att, dim=-1)
            h = torch.matmul(att, v)
            x = x + self.proj(h)
            h = self.ln2(x)
            return x + self.fc2(F.gelu(self.fc1(h)))

    model = nn.Sequential(*[Block(args.hidden) for _ in range(args.layers)]).to(
        device=device, dtype=torch.bfloat16
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    B, S, H = args.batch, args.seq, args.hidden

    def one_step() -> None:
        x = torch.randn(B, S, H, device=device, dtype=torch.bfloat16)
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
        torch.cuda.synchronize()

    out = Path(args.out_dir) / f"step_times_rank{global_rank:03d}.jsonl"
    meta = Path(args.out_dir) / f"meta_rank{global_rank:03d}.json"
    meta.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "global_rank": global_rank,
                "node": node,
                "local": phys,
                "hostname": hostname,
                "hidden": args.hidden,
                "seq": args.seq,
                "layers": args.layers,
                "iters": args.iters,
                "warmup": args.warmup,
                "tag": args.tag,
                "world": world if use_dist else 1,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for _ in range(args.warmup):
        one_step()
    if use_dist:
        dist.barrier()

    # 对齐起点：独立模式靠墙钟约定；真实同步靠 barrier
    t_start = time.time()
    if use_dist:
        # 广播统一起点，便于对比
        t_tensor = torch.tensor([t_start], device=device)
        dist.broadcast(t_tensor, src=0)
        t_start = float(t_tensor.item())

    # 可选 PP-stage 延迟注入（案例二）
    try:
        from delay_inject import maybe_sleep_for_iter
    except Exception:
        maybe_sleep_for_iter = None  # type: ignore

    for i in range(1, args.iters + 1):
        t0 = time.time()
        delayed = False
        if maybe_sleep_for_iter is not None:
            try:
                delayed = bool(maybe_sleep_for_iter(i))
            except Exception:
                delayed = False
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
                "local": phys,
                "mode": args.mode,
                "hostname": hostname,
                "t_start": t_start,
                "delayed": delayed,
            },
        )

    done = Path(args.out_dir) / f"done_rank{global_rank:03d}.txt"
    done.write_text(f"OK {args.mode} rank={global_rank} iters={args.iters}\n", encoding="utf-8")
    print(
        f"VSYNC_DONE mode={args.mode} rank={global_rank} iters={args.iters} out={out}",
        flush=True,
    )
    if use_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
