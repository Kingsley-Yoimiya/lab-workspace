#!/usr/bin/env python3
"""train_bench_probe.py — 仪器化 GPT-2 训练微基准(within-step 分解 + 流式本地落盘)。

相对 train_bench_clean.py 的三处关键改造:
  1. within-step 三段埋点: 用 cuda.Event 分别计 compute_ms / comm_ms,
     并在 backward 后插一个显式 barrier 取 wait_ms(= barrier耗时,
     healthy rank 早到 → wait 大; victim 晚到 → wait≈0)。
     这两路信号穿透 AllReduce 的 step-level 掩蔽,支撑"定位到 straggler rank"。
  2. 流式写 pod 本地盘: warmup 后开句柄,每步 append(写在计时区之外),每
     flush_every 步 fsync;out-dir 默认由编排器指向 /workspace/probe-bundle/out。
  3. --mode host_bound: 用真实 DataLoader(num_workers)喂数,让 host 层注入
     (GC/stress/leak)有关键路径可咬;gpu_bound 则用 device 端随机 idx(纯算)。

所有 config(C0..C4)共用本脚本; 注入由外部 sidecar 进程实现,检测器由环境变量
(PROBING / LD_PRELOAD)叠加。本脚本不感知注入器/检测器的存在。

输出: <out_dir>/rank_<global_rank>.jsonl (每步一行 JSON)
      <out_dir>/warmup_done            (rank0 warmup 完成 marker,供编排器查)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--model", choices=["gpt2", "tiny"], default="gpt2",
                    help="gpt2=约124M GPT-2；tiny 仅用于紧急管线回退")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--ffn", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True,
                    help="pod 本地路径；rank 各写各的 jsonl。")
    # ── 本轮新增 ──
    ap.add_argument("--mode", choices=["gpu_bound", "host_bound"], default="gpu_bound",
                    help="host_bound 用真实 DataLoader,让 host 注入(GC/stress/leak)咬到 data 路径")
    ap.add_argument("--flush-every", type=int, default=5, help="每 N 步 flush+fsync")
    ap.add_argument("--decompose", type=int, default=1,
                    help="1=开 within-step compute/comm/wait 分解(默认); 0=只测 step_ms")
    ap.add_argument("--dl-workers", type=int, default=2, help="host_bound 模式 DataLoader worker 数")
    # 元信息(写进每条记录,便于离线汇聚)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--group", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--round", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", "0")))
    torch.cuda.set_device(local)
    device = f"cuda:{local}"

    # GPT-2 124M: vocab=50257, hidden=768, 12 层, MLP=3072，词表/输出层权重共享。
    # tiny 只保留为资源不足时验证编排、注入和回拉链路的退路。
    if args.model == "gpt2":
        hidden, layers, ffn, vocab = 768, 12, 3072, 50257
    else:
        hidden, layers, ffn, vocab = 256, 4, 1024, 32000
    hidden = args.hidden or hidden
    layers = args.layers or layers
    ffn = args.ffn or ffn

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
            att = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
            att = torch.softmax(att, dim=-1)
            h = torch.matmul(att, v)
            x = x + self.proj(h)
            h = self.ln2(x)
            x = x + self.fc2(F.gelu(self.fc1(h)))
            return x

    class GPT2Bench(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, hidden)
            self.pos = nn.Embedding(args.seq, hidden)
            self.blocks = nn.ModuleList(
                [Block(hidden, ffn) for _ in range(layers)]
            )
            self.ln = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, vocab, bias=False)
            self.head.weight = self.emb.weight  # GPT-2 tying，约124M 参数而非重复输出词表

        def forward(self, idx):
            positions = torch.arange(idx.shape[1], device=idx.device)
            x = self.emb(idx) + self.pos(positions)[None, :, :]
            for b in self.blocks:
                x = b(x)
            return self.head(self.ln(x))

    model = GPT2Bench().to(device=device, dtype=torch.bfloat16)
    model = DDP(model, device_ids=[local], output_device=local)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    B, S = args.batch, args.seq

    # ── 数据供给：两种模式都走 DataLoader(num_workers=2 默认)；host_bound 额外加入 CPU 工作。 ──
    import numpy as np
    from torch.utils.data import DataLoader, Dataset

    class TokenDataset(Dataset):
        def __len__(self):
            return args.iters * 100

        def __getitem__(self, i):
            rng = np.random.default_rng(i)
            buf = rng.integers(0, vocab, size=(S,), dtype=np.int64)
            if args.mode == "host_bound":
                # Loud host 路径可见性：768×768 matmul（仍保持 num_workers=2 / prefetch=2）
                _ = (rng.standard_normal((768, 768)) @ rng.standard_normal((768, 768))).sum()
            return torch.from_numpy(buf)

    dl = DataLoader(
        TokenDataset(), batch_size=B, num_workers=args.dl_workers,
        pin_memory=True, prefetch_factor=2, persistent_workers=True,
    )
    data_iter = iter(dl)

    decompose = bool(args.decompose)
    # 复用同一批 event(避免每步分配)
    if decompose:
        e_c0, e_c1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e_m0, e_m1 = torch.cuda.Event(True), torch.cuda.Event(True)

    def get_batch():
        nonlocal data_iter
        try:
            b = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            b = next(data_iter)
        return b.to(device, non_blocking=True)

    def step_instrumented(force_gc: bool = False):
        """返回 (data_ms, compute_ms, comm_ms, wait_ms, step_ms, loss)。"""
        t0 = time.perf_counter()
        if force_gc:
            import gc as _gc_mod
            _gc_mod.collect()
        idx = get_batch()                       # host_bound: 喂数时间落在此处
        t_data = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        if decompose:
            e_c0.record()
        logits = model(idx)
        loss = F.cross_entropy(logits.float().reshape(-1, vocab), idx.reshape(-1))
        loss.backward()
        if decompose:
            e_c1.record()
            torch.cuda.synchronize()            # compute 段真正算完
            t_arrive = time.perf_counter()      # 本 rank 到达 barrier 的绝对时刻
            dist.barrier()                      # 显式对齐: 慢的还没到, 快的在此等
            t_bar = time.perf_counter()
            e_m0.record()

        if decompose:
            e_m1.record()
        opt.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if decompose:
            compute_ms = e_c0.elapsed_time(e_c1)      # GPU event 计时
            comm_ms = e_m0.elapsed_time(e_m1)
            wait_ms = (t_bar - t_arrive) * 1e3        # barrier 等待(空间信号)
            data_ms = (t_data - t0) * 1e3
        else:
            compute_ms = comm_ms = wait_ms = 0.0
            data_ms = (t_data - t0) * 1e3
        step_ms = (t1 - t0) * 1e3
        return data_ms, compute_ms, comm_ms, wait_ms, step_ms, float(loss.detach().cpu())

    # ── Warmup(同样的训练逻辑, 建立 CUDA context / autotune) ──
    for _ in range(args.warmup):
        step_instrumented()
    dist.barrier()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (out_dir / "warmup_done").write_text(str(time.time()))

    # ── Measure: 流式写 AFS(每步 append, 写在计时区之外, 每 flush_every 步 fsync) ──
    out_file = out_dir / f"rank_{rank:04d}.jsonl"
    meta_common = {
        "rank": rank, "local_rank": local, "node_rank": node_rank, "world_size": world,
        "run_id": args.run_id, "group": args.group, "config": args.config, "round": args.round,
        "mode": args.mode,
    }
    # P3-SW-A Loud：进程内泄漏+GC（外挂 GC 咬不到训练进程）。仅 victim local_rank。
    inline = os.environ.get("INLINE_INJECT", "").strip()
    inline_victim = int(os.environ.get("INLINE_VICTIM_LOCAL_RANK", str(max(0, int(os.environ.get("LOCAL_WORLD_SIZE", "8")) - 1))))
    inline_start = int(os.environ.get("INLINE_INJECT_START", "100"))
    inline_stop = int(os.environ.get("INLINE_INJECT_STOP", "300"))
    do_inline_8a = (
        inline == "8a"
        and local == inline_victim
        and node_rank == 0
    )
    leak_buf: list = []
    ckpt_dir = Path(os.environ.get("CKPT_DIR", "/workspace/probe-bundle/ckpt"))

    f = out_file.open("a", buffering=1)  # line-buffered
    try:
        for i in range(args.iters):
            # 内联 8a：measure 窗内每步泄漏 ~4MB；stop 步在计时区内 gc.collect() 制造尖刺
            if do_inline_8a and i >= inline_start:
                leak_buf.append(bytearray(1024 * 4 * 1024))  # 4MiB
            force_gc = bool(do_inline_8a and (i + 1) == inline_stop)
            data_ms, compute_ms, comm_ms, wait_ms, step_ms, loss = step_instrumented(force_gc=force_gc)
            # 写在计时区之外: 绝不让 AFS I/O 污染 step_ms
            rec = {
                "step": i,
                "data_ms": round(data_ms, 3),
                "compute_ms": round(compute_ms, 3),
                "comm_ms": round(comm_ms, 3),
                "wait_ms": round(wait_ms, 3),
                "step_ms": round(step_ms, 3),
                "loss": round(loss, 6),
                "ts": round(time.time(), 3),
                **meta_common,
            }
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % args.flush_every == 0:
                f.flush()
                os.fsync(f.fileno())
            if rank == 0 and (i + 1) in {100, 300}:
                # 编排器用这两个测量步 marker 实现“总 step 150--350”窗口：
                # warmup=50 时，measure 100/300 分别对应全局 150/350。
                (out_dir / f"step_{i + 1}.marker").write_text(str(time.time()))
            # SOP：每 100 measure 步 checkpoint（9B IO 路径；写在计时外）
            if rank == 0 and (i + 1) % 100 == 0:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"step": i + 1, "model": model.module.state_dict()},
                    ckpt_dir / f"step_{i + 1}.pt",
                )
    finally:
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
        f.close()

    dist.barrier()
    if rank == 0:
        print(f"DONE world={world} out={out_file}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
