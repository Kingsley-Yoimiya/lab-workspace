#!/usr/bin/env python3
"""外部抢占：在指定 GPU 上持续跑大矩阵乘，抢占算力。

用法:
  CUDA_VISIBLE_DEVICES=0 python3 gpu_busy_preempt.py --seconds 120
"""
from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--size", type=int, default=8192)
    args = ap.parse_args()

    import torch

    device = torch.device("cuda", 0)
    a = torch.randn(args.size, args.size, device=device, dtype=torch.bfloat16)
    b = torch.randn(args.size, args.size, device=device, dtype=torch.bfloat16)
    t_end = time.time() + args.seconds
    n = 0
    while time.time() < t_end:
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        n += 1
        if n % 50 == 0:
            print(f"preempt_iters={n} elapsed={args.seconds - (t_end - time.time()):.1f}s", flush=True)
    print(f"PREEMPT_DONE iters={n}", flush=True)


if __name__ == "__main__":
    main()
