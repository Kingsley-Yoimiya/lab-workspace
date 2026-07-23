#!/usr/bin/env python3
"""单卡 Ascend NPU：算子 / Transformer-Block 微基准（Phase2）。

workloads:
  gemm_ffn_up     — [B·S, H] @ [H, 4H]  fwd(+可选 bwd)
  gemm_ffn_down   — [B·S, 4H] @ [4H, H]
  ln              — LayerNorm on [B,S,H]
  block_fwd_bwd   — LayerNorm + Linear FFN（同 virtual_sync MLP block 风格）+ AdamW step
  block_small_ops — 同 block，seq=128（launch 密）

计时：torch.npu.Event；stdout 打印 p50/p95；--out 写 jsonl。

可选注入：--inject-kind / --inject-duty 子进程启动同目录 npu_component_inject.py，结束时 kill。

用法:
  python op_block_bench_npu.py --workload block_fwd_bwd --device 0 --iters 50 --out ops.jsonl
  python op_block_bench_npu.py --workload ln --device 3 --inject-kind cube --inject-duty 0.3 \\
      --factor cube --dose mid --out ops.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional


WORKLOADS = (
    "gemm_ffn_up",
    "gemm_ffn_down",
    "ln",
    "block_fwd_bwd",
    "block_small_ops",
)


def _pctile(xs: list[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def _start_inject(args: argparse.Namespace) -> Optional[subprocess.Popen]:
    if not args.inject_kind:
        return None
    inject_py = Path(__file__).resolve().parent / "npu_component_inject.py"
    if not inject_py.is_file():
        raise FileNotFoundError(f"inject script missing: {inject_py}")
    cmd = [
        sys.executable,
        str(inject_py),
        "--kind",
        args.inject_kind,
        "--device",
        str(args.device),
        "--seconds",
        str(max(args.inject_seconds, 3600)),
        "--duty",
        str(args.inject_duty),
    ]
    if args.inject_period_ms is not None:
        cmd += ["--period-ms", str(args.inject_period_ms)]
    print(f"INJECT_SPAWN {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    # 等注入真正开始（INJECT_START）或最多 15s
    deadline = time.time() + 15.0
    started = False
    buf = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            out = buf + (proc.stdout.read() if proc.stdout else "")
            raise RuntimeError(f"inject exited early rc={proc.returncode}: {out[:800]}")
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            buf += line
            print(f"INJECT_LOG {line.rstrip()}", flush=True)
            if "INJECT_START" in line:
                started = True
                break
        else:
            time.sleep(0.1)
    if not started:
        print("INJECT_WARN no INJECT_START yet; continuing", flush=True)
    time.sleep(2.0)  # 再给一点时间占满
    return proc


def _stop_inject(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        if hasattr(os, "killpg") and proc.pid:
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg") and proc.pid:
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
    print("INJECT_KILLED", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", choices=WORKLOADS, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=1024, help="block_small_ops 会强制用 128")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--ffn", type=int, default=0, help="0 → 4*hidden")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--factor", default="none", help="记录用：注入因素标签")
    ap.add_argument("--dose", default="0", help="记录用：剂量标签")
    ap.add_argument("--inject-kind", default="", choices=["", "cpu", "cube", "vector", "hbm_mte", "placebo"])
    ap.add_argument("--inject-duty", type=float, default=0.3)
    ap.add_argument("--inject-period-ms", type=float, default=None)
    ap.add_argument("--inject-seconds", type=float, default=0.0, help="注入进程最长秒数；0→自动拉长")
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()
    if not args.inject_kind:
        args.inject_kind = ""

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    B = args.batch
    S = 128 if args.workload == "block_small_ops" else args.seq
    H = args.hidden
    FFN = args.ffn if args.ffn > 0 else 4 * H
    inject_on = bool(args.inject_kind)

    class MlpBlock(nn.Module):
        """LayerNorm + Linear FFN，风格对齐 virtual_sync_bench_npu。"""

        def __init__(self, h: int, ffn: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(h)
            self.fc1 = nn.Linear(h, ffn, bias=False)
            self.fc2 = nn.Linear(ffn, h, bias=False)
            self.ln2 = nn.LayerNorm(h)

        def forward(self, x):
            y = self.fc2(F.gelu(self.fc1(self.ln1(x))))
            x = x + y
            return self.ln2(x)

    # build workload callable
    opt = None
    if args.workload in ("block_fwd_bwd", "block_small_ops"):
        model = MlpBlock(H, FFN).to(device=device, dtype=dtype)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def step_fn() -> None:
            x = torch.randn(B, S, H, device=device, dtype=dtype)
            opt.zero_grad(set_to_none=True)
            y = model(x)
            loss = y.float().pow(2).mean()
            loss.backward()
            opt.step()

    elif args.workload == "ln":
        ln = nn.LayerNorm(H).to(device=device, dtype=dtype)

        def step_fn() -> None:
            x = torch.randn(B, S, H, device=device, dtype=dtype, requires_grad=True)
            y = ln(x)
            y.float().pow(2).mean().backward()

    elif args.workload == "gemm_ffn_up":
        w = torch.randn(H, FFN, device=device, dtype=dtype, requires_grad=True)

        def step_fn() -> None:
            x = torch.randn(B * S, H, device=device, dtype=dtype, requires_grad=True)
            y = x @ w
            y.float().pow(2).mean().backward()

    elif args.workload == "gemm_ffn_down":
        w = torch.randn(FFN, H, device=device, dtype=dtype, requires_grad=True)

        def step_fn() -> None:
            x = torch.randn(B * S, FFN, device=device, dtype=dtype, requires_grad=True)
            y = x @ w
            y.float().pow(2).mean().backward()

    else:
        raise ValueError(args.workload)

    def timed_ms(fn: Callable[[], None]) -> float:
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        return float(start.elapsed_time(end))

    inject_proc = None
    try:
        inject_proc = _start_inject(args) if inject_on else None

        for _ in range(args.warmup):
            timed_ms(step_fn)

        ms_list: list[float] = []
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as f:
            for i in range(1, args.iters + 1):
                ms = timed_ms(step_fn)
                ms_list.append(ms)
                rec = {
                    "record": "op_iter",
                    "run_id": args.run_id,
                    "workload": args.workload,
                    "factor": args.factor,
                    "dose": args.dose,
                    "inject_on": inject_on,
                    "inject_kind": args.inject_kind or None,
                    "inject_duty": args.inject_duty if inject_on else None,
                    "device": args.device,
                    "iter": i,
                    "ms": round(ms, 4),
                    "batch": B,
                    "seq": S,
                    "hidden": H,
                    "ffn": FFN,
                    "dtype": args.dtype,
                    "ts": time.time(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        p50 = _pctile(ms_list, 50)
        p95 = _pctile(ms_list, 95)
        summary = {
            "record": "op_summary",
            "run_id": args.run_id,
            "workload": args.workload,
            "factor": args.factor,
            "dose": args.dose,
            "inject_on": inject_on,
            "inject_kind": args.inject_kind or None,
            "device": args.device,
            "iters": args.iters,
            "ms_p50": p50,
            "ms_p95": p95,
            "ms_mean": statistics.mean(ms_list) if ms_list else None,
            "ms_std": statistics.stdev(ms_list) if len(ms_list) > 1 else 0.0,
            "batch": B,
            "seq": S,
            "hidden": H,
            "ffn": FFN,
            "dtype": args.dtype,
            "ts": time.time(),
        }
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        print(
            f"OP_DONE workload={args.workload} device={args.device} "
            f"factor={args.factor} dose={args.dose} inject={inject_on} "
            f"p50={p50:.3f}ms p95={p95:.3f}ms → {args.out}",
            flush=True,
        )
    finally:
        _stop_inject(inject_proc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
