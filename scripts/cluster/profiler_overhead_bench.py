#!/usr/bin/env python3
"""MUXI 观测开销基准：8 卡独立或机内 AllReduce，可选 torch.profiler。

Arm 由环境/参数控制：
  --profiler none|torch
  PROBING=0/1 由外部注入

输出每卡 JSONL step_times + summary.json（rank0）。
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from pathlib import Path


def _write(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["independent", "real_sync"], default="real_sync")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--profiler", choices=["none", "torch"], default="none")
    ap.add_argument("--prof-wait", type=int, default=5)
    ap.add_argument("--prof-warmup", type=int, default=5)
    ap.add_argument("--prof-active", type=int, default=10)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    # Hang-on Probing: import activates server when PROBING is set
    if os.environ.get("PROBING", "").strip() not in ("", "0", "false", "off"):
        try:
            import probing  # noqa: F401
        except Exception as e:
            print(f"PROBING_IMPORT_FAIL {e}", flush=True)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    local = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    node = int(os.environ.get("NODE_RANK", 0))
    nproc = int(os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("GPUS_PER_NODE") or 8)
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
    # Record env early (torchrun should inherit PROBING)
    if int(os.environ.get("LOCAL_RANK", local) or 0) == 0:
        print(
            f"ENV_PROBE PROBING={os.environ.get('PROBING')!r} "
            f"TORCH_PROF={os.environ.get('PROBING_TORCH_PROFILING')!r} "
            f"profiler_arg={args.profiler}",
            flush=True,
        )


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
            qkv = self.qkv(h)
            q, k, v = qkv.chunk(3, dim=-1)
            b, s, d = q.shape
            nh = 32
            hd = d // nh
            q = q.view(b, s, nh, hd).transpose(1, 2)
            k = k.view(b, s, nh, hd).transpose(1, 2)
            v = v.view(b, s, nh, hd).transpose(1, 2)
            att = F.scaled_dot_product_attention(q, k, v)
            att = att.transpose(1, 2).contiguous().view(b, s, d)
            x = x + self.proj(att)
            h2 = self.ln2(x)
            x = x + self.fc2(F.gelu(self.fc1(h2)))
            return x

    model = nn.Sequential(*[Block(args.hidden) for _ in range(args.layers)]).to(
        device=device, dtype=torch.bfloat16
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=torch.bfloat16)

    def one_step():
        opt.zero_grad(set_to_none=True)
        y = model(x)
        loss = y.float().pow(2).mean()
        loss.backward()
        if use_dist:
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad.div_(world)
        opt.step()
        return float(loss.detach().item())

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / f"step_times_rank{global_rank:03d}.jsonl"
    if jsonl.exists():
        jsonl.unlink()

    for _ in range(args.warmup):
        one_step()
    if use_dist:
        dist.barrier()

    prof = None
    prof_path = None
    if args.profiler == "torch" and rank == 0:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        prof_path = str(out / "torch_profiler_rank0.json")
        prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=args.prof_wait,
                warmup=args.prof_warmup,
                active=args.prof_active,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out / "tb"))
            if False
            else None,
            record_shapes=False,
            with_stack=False,
        )
        # export chrome trace manually at end
        prof.__enter__()

    t_wall0 = time.time()
    losses = []
    for i in range(1, args.iters + 1):
        t0 = time.time()
        loss = one_step()
        if prof is not None:
            prof.step()
        t1 = time.time()
        losses.append(loss)
        _write(
            jsonl,
            {
                "iter": i,
                "t0": t0,
                "t1": t1,
                "ms": round((t1 - t0) * 1000.0, 3),
                "rank": rank,
                "global_rank": global_rank,
                "node": node,
                "mode": args.mode,
                "hostname": hostname,
                "profiler": args.profiler,
                "probing": os.environ.get("PROBING", "0"),
                "tag": args.tag,
                "loss": loss,
            },
        )
    t_wall1 = time.time()

    if prof is not None:
        prof.__exit__(None, None, None)
        try:
            prof.export_chrome_trace(prof_path)
        except Exception as e:
            (out / "torch_profiler_export_err.txt").write_text(str(e), encoding="utf-8")

    if use_dist:
        dist.barrier()

    # per-rank done marker
    (out / f"done_rank{global_rank:03d}.txt").write_text("ok\n", encoding="utf-8")

    if rank == 0:
        rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
        steady = [r["ms"] for r in rows if r["iter"] > args.warmup]
        if not steady:
            steady = [r["ms"] for r in rows]
        summary = {
            "tag": args.tag,
            "mode": args.mode,
            "profiler": args.profiler,
            "probing": os.environ.get("PROBING", "0"),
            "hostname": hostname,
            "world": world,
            "iters": args.iters,
            "warmup": args.warmup,
            "hidden": args.hidden,
            "seq": args.seq,
            "layers": args.layers,
            "batch": args.batch,
            "wall_s": round(t_wall1 - t_wall0, 3),
            "step_ms_p50": statistics.median(steady),
            "step_ms_p95": sorted(steady)[max(0, int(0.95 * (len(steady) - 1)))],
            "step_ms_mean": statistics.mean(steady),
            "n_steady": len(steady),
            "chrome_trace": prof_path,
        }
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("SUMMARY", json.dumps(summary, ensure_ascii=False))

    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
