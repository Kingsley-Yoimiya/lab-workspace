#!/usr/bin/env python3
"""Ascend 训练 MFU 微基准（dense / moe），用 torchrun+HCCL 在 16/32/64/128 上跑。

★ 这是 DP + grad AllReduce，不是 Tensor Parallel。真 TP 见同目录 tp_block_bench_npu.py。

Dense: 小型 GPT 前向+反向（近似 6ND FLOPs）
MoE:  专家路由 + 多专家 FFN（额外 all_to_all 风格通信用 all_gather 近似）

输出 JSONL: mfu, tflops, tokens_per_sec, step_ms, world_size, mode
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense", "moe"], default="dense")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--peak-tflops", type=float, default=320.0, help="单卡峰值 TFLOPS 估计(bf16)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_npu  # noqa: F401

    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.npu.set_device(local)
    device = f"npu:{local}"

    class Block(nn.Module):
        def __init__(self, h: int, ffn: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(h)
            self.qkv = nn.Linear(h, 3 * h, bias=False)
            self.proj = nn.Linear(h, h, bias=False)
            self.ln2 = nn.LayerNorm(h)
            self.fc1 = nn.Linear(h, ffn, bias=False)
            self.fc2 = nn.Linear(ffn, h, bias=False)

        def forward(self, x):
            h = self.ln1(x)
            qkv = self.qkv(h)
            q, k, v = qkv.chunk(3, dim=-1)
            # scaled dot-product attention (no flash, portable)
            att = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
            att = torch.softmax(att, dim=-1)
            h = torch.matmul(att, v)
            x = x + self.proj(h)
            h = self.ln2(x)
            x = x + self.fc2(F.gelu(self.fc1(h)))
            return x

    class MoEBlock(nn.Module):
        def __init__(self, h: int, ffn: int, n_exp: int, topk: int):
            super().__init__()
            self.ln = nn.LayerNorm(h)
            self.gate = nn.Linear(h, n_exp, bias=False)
            self.experts = nn.ModuleList(
                [nn.Sequential(nn.Linear(h, ffn, bias=False), nn.GELU(), nn.Linear(ffn, h, bias=False)) for _ in range(n_exp)]
            )
            self.topk = topk

        def forward(self, x):
            # x: [B,S,H] — 向量化 soft/topk 路由，避免动态 mask（128 卡上会极慢/挂死）
            h = self.ln(x)
            logits = self.gate(h)
            vals, idx = torch.topk(logits, self.topk, dim=-1)
            weights = torch.softmax(vals, dim=-1)  # [B,S,K]
            # 全专家前向再按 topk 加权（可编译、形状稳定）
            stacked = torch.stack([expert(h) for expert in self.experts], dim=-2)  # [B,S,E,H]
            gather_idx = idx.unsqueeze(-1).expand(*idx.shape, h.shape[-1])  # [B,S,K,H]
            picked = torch.gather(stacked, dim=-2, index=gather_idx)  # [B,S,K,H]
            out = (picked * weights.unsqueeze(-1)).sum(dim=-2)
            # 跨 rank 同步，模拟 EP 流量
            dist.all_reduce(out)
            return x + out

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(32000, args.hidden)
            if args.mode == "dense":
                self.blocks = nn.ModuleList([Block(args.hidden, args.ffn) for _ in range(args.layers)])
            else:
                self.blocks = nn.ModuleList(
                    [MoEBlock(args.hidden, args.ffn, args.experts, args.topk) for _ in range(args.layers)]
                )
            self.ln = nn.LayerNorm(args.hidden)
            self.head = nn.Linear(args.hidden, 32000, bias=False)

        def forward(self, idx):
            x = self.emb(idx)
            for b in self.blocks:
                x = b(x)
            return self.head(self.ln(x))

    model = TinyGPT().to(device=device, dtype=torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # param count (local)
    n_params = sum(p.numel() for p in model.parameters())
    # FLOPs approx per token: dense ~ 6 * N_params; moe ~ 6 * (shared + topk*expert)
    if args.mode == "dense":
        flops_per_token = 6.0 * n_params
    else:
        # rough: gate + topk experts
        flops_per_token = 6.0 * n_params * (args.topk / max(args.experts, 1)) * 2

    B, S = args.batch, args.seq
    tokens_per_step = B * S * world  # global tokens if data parallel

    def step():
        idx = torch.randint(0, 32000, (B, S), device=device)
        opt.zero_grad(set_to_none=True)
        logits = model(idx)
        loss = F.cross_entropy(logits.float().reshape(-1, 32000), idx.reshape(-1))
        loss.backward()
        # DP allreduce grads
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad.mul_(1.0 / world)
        opt.step()
        torch.npu.synchronize()
        return float(loss.detach().cpu())

    for _ in range(args.warmup):
        step()
    dist.barrier()

    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(args.iters):
        last_loss = step()
    dist.barrier()
    elapsed = time.perf_counter() - t0
    step_ms = elapsed / args.iters * 1e3
    toks_s = tokens_per_step * args.iters / elapsed
    achieved_tflops = flops_per_token * toks_s / 1e12
    peak = args.peak_tflops * world
    mfu = achieved_tflops / peak if peak > 0 else 0.0

    rec = {
        "record": "train_mfu",
        "mode": args.mode,
        "world_size": world,
        "rank": rank,
        "n_params": n_params,
        "tokens_per_sec": toks_s,
        "step_ms": step_ms,
        "achieved_tflops": achieved_tflops,
        "peak_tflops": peak,
        "mfu": mfu,
        "loss": last_loss,
        "seq": S,
        "hidden": args.hidden,
        "layers": args.layers,
    }
    if rank == 0:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"mode={args.mode} world={world} mfu={mfu*100:.2f}% "
            f"tflops={achieved_tflops:.1f}/{peak:.1f} toks/s={toks_s:.0f} step_ms={step_ms:.1f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
