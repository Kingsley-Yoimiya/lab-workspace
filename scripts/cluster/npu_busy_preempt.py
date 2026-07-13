#!/usr/bin/env python3
"""NPU 算力抢占：在指定 device 上持续 matmul（实验四 lite）。"""
from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    a = torch.randn(args.size, args.size, device=f"npu:{args.device}", dtype=torch.float16)
    b = torch.randn(args.size, args.size, device=f"npu:{args.device}", dtype=torch.float16)
    t_end = time.time() + args.seconds
    n = 0
    print(f"PREEMPT_START device={args.device} seconds={args.seconds}", flush=True)
    while time.time() < t_end:
        c = torch.matmul(a, b)
        torch.npu.synchronize()
        n += 1
        if n % 20 == 0:
            print(f"preempt_iters={n} device={args.device}", flush=True)
    print(f"PREEMPT_DONE device={args.device} iters={n}", flush=True)


if __name__ == "__main__":
    main()
