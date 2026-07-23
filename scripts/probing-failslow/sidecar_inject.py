#!/usr/bin/env python3
"""sidecar_inject.py — MetaX C550 外部 GPU 干扰 sidecar。

完全独立于训练进程：有自己的 CUDA context，通过 duty cycle 在指定 device 上施压。
通过 CUDA_VISIBLE_DEVICES 控制目标设备。

用法:
  CUDA_VISIBLE_DEVICES=7 python3 sidecar_inject.py --kind cube --duty 0.8 --warmup-seconds 5 --seconds 120
  CUDA_VISIBLE_DEVICES=7 python3 sidecar_inject.py --kind hbm --duty 0.8 --warmup-seconds 5 --seconds 120

kind:
  cube - 持续 GEMM（抢算力）
  hbm  - 持续 D2D copy（抢显存带宽）
"""
from __future__ import annotations

import argparse
import time
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["cube", "hbm"], default="cube")
    ap.add_argument("--duty", type=float, default=0.3, help="busy fraction per period (0~1)")
    ap.add_argument("--period-ms", type=float, default=200, help="duty cycle period in ms")
    ap.add_argument("--seconds", type=float, default=300, help="total duration")
    ap.add_argument("--size", type=int, default=4096, help="matrix size for cube / MB for hbm")
    ap.add_argument("--warmup-seconds", type=float, default=5.0,
                    help="稳态预热时长；预热完成前不计入故障窗口")
    args = ap.parse_args()

    import torch
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES already restricts to target device
    device = "cuda:0"

    # Warmup
    if args.kind == "cube":
        N = args.size
        A = torch.randn(N, N, device=device, dtype=torch.float16)
        B = torch.randn(N, N, device=device, dtype=torch.float16)
        # Warmup
        for _ in range(5):
            torch.mm(A, B)
        torch.cuda.synchronize()

        def burst():
            torch.mm(A, B)

    elif args.kind == "hbm":
        nbytes = args.size * 1024 * 1024  # MB → bytes
        nelems = nbytes // 2  # fp16
        src = torch.randn(nelems, device=device, dtype=torch.float16)
        dst = torch.empty_like(src)
        for _ in range(3):
            dst.copy_(src)
        torch.cuda.synchronize()

        def burst():
            dst.copy_(src)

    if not 0.0 <= args.duty <= 1.0:
        ap.error("--duty must be within [0, 1]")
    if args.warmup_seconds < 0:
        ap.error("--warmup-seconds must be non-negative")

    if args.warmup_seconds:
        print(
            f"SIDECAR_WARMUP kind={args.kind} seconds={args.warmup_seconds}",
            flush=True,
        )
        warm_end = time.time() + args.warmup_seconds
        while time.time() < warm_end:
            burst()
        torch.cuda.synchronize()

    print(f"SIDECAR_START kind={args.kind} duty={args.duty} period={args.period_ms}ms", flush=True)

    period_s = args.period_ms / 1000.0
    busy_s = period_s * args.duty
    t_end = time.time() + args.seconds
    ops = 0

    while time.time() < t_end:
        # Busy phase
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < busy_s:
            burst()
            ops += 1
        torch.cuda.synchronize()

        # Idle phase
        idle_s = period_s - (time.perf_counter() - t0)
        if idle_s > 0:
            time.sleep(idle_s)

    torch.cuda.synchronize()
    print(f"SIDECAR_STOP ops={ops}", flush=True)


if __name__ == "__main__":
    main()
