#!/usr/bin/env python3
"""NPU 部件干扰 sidecar（Phase1 剂量校准）。

在指定 device 上以第二进程风格持续施压，duty 控制每个 period 内的
busy/idle 比例；设备工作在 burst 边界 synchronize，避免把 sleep 算进占用。

用法示例:
  python3 npu_component_inject.py --kind cube --device 0 --duty 0.3 \\
      --period-ms 200 --seconds 120 --size 4096
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Callable


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="NPU component inject sidecar (Phase1)")
    ap.add_argument(
        "--kind",
        required=True,
        choices=("cpu", "cube", "vector", "hbm_mte", "placebo"),
        help="干扰类型；placebo 起进程但不提交设备工作",
    )
    ap.add_argument("--device", type=int, default=0, help="可见 NPU 逻辑号（容器内常为 0）")
    ap.add_argument("--seconds", type=float, default=120.0, help="总时长（秒）")
    ap.add_argument("--duty", type=float, default=1.0, help="占空比 0–1（period 内 busy 比例）")
    ap.add_argument("--period-ms", type=float, default=200.0, help="duty 周期（毫秒）")
    ap.add_argument("--size", type=int, default=4096, help="cube GEMM 边长")
    ap.add_argument("--elems", type=int, default=1 << 26, help="vector 元素数")
    ap.add_argument("--mb", type=int, default=512, help="hbm_mte D2D copy 缓冲大小（MiB）")
    ap.add_argument(
        "--dtype",
        default="fp16",
        choices=("fp16", "bf16", "fp32"),
        help="cube/vector 数据类型",
    )
    ap.add_argument(
        "--vector-op",
        default="fma",
        choices=("fma", "exp"),
        help="vector 循环算子：a*b+c 或 exp",
    )
    ap.add_argument("--cpu-threads", type=int, default=4, help="cpu 忙线程数")
    ap.add_argument(
        "--cpu-affinity",
        type=str,
        default="",
        help="可选 CPU 亲和性，逗号分隔核号，如 0,1,2,3",
    )
    return ap.parse_args()


def _clamp_duty(duty: float) -> float:
    if duty < 0.0:
        return 0.0
    if duty > 1.0:
        return 1.0
    return float(duty)


def _torch_dtype(torch, name: str):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def _duty_loop(
    seconds: float,
    duty: float,
    period_ms: float,
    burst_fn: Callable[[], None],
    sync_fn: Callable[[], None] | None = None,
    idle_fn: Callable[[float], None] | None = None,
) -> int:
    """在 period 边界按 duty 切 busy/idle；设备类在 burst 结束时 sync。"""
    duty = _clamp_duty(duty)
    period_s = max(period_ms, 1.0) / 1000.0
    t_end = time.time() + seconds
    bursts = 0
    sleep_fn = idle_fn or time.sleep

    if duty <= 0.0:
        # 不起设备工作：仅占进程到时退出（placebo / duty=0）
        while time.time() < t_end:
            remaining = t_end - time.time()
            if remaining <= 0:
                break
            sleep_fn(min(period_s, remaining))
            bursts += 1
        return bursts

    while time.time() < t_end:
        period_start = time.time()
        busy_s = period_s * duty
        # busy 窗：反复提交直到用尽（或总时长到）
        while time.time() < t_end and (time.time() - period_start) < busy_s:
            burst_fn()
        if sync_fn is not None:
            sync_fn()
        bursts += 1
        # idle 窗：在 period 边界对齐
        elapsed = time.time() - period_start
        idle_s = period_s - elapsed
        if idle_s > 0 and time.time() < t_end:
            sleep_fn(min(idle_s, t_end - time.time()))
    return bursts


def _run_placebo(args: argparse.Namespace) -> int:
    return _duty_loop(
        seconds=args.seconds,
        duty=0.0,
        period_ms=args.period_ms,
        burst_fn=lambda: None,
    )


def _run_cpu(args: argparse.Namespace) -> int:
    duty = _clamp_duty(args.duty)
    stop = threading.Event()
    n_threads = max(1, int(args.cpu_threads))
    affinity: set[int] | None = None
    if args.cpu_affinity.strip():
        affinity = {int(x) for x in args.cpu_affinity.split(",") if x.strip() != ""}

    def worker() -> None:
        if affinity is not None:
            try:
                os.sched_setaffinity(0, affinity)
            except (AttributeError, OSError) as exc:
                print(f"INJECT_WARN cpu_affinity_failed err={exc}", flush=True)
        x = 1.0000001
        # 忙等：非 sleep；duty 由主线程门控
        while not stop.is_set():
            x = x * 1.0000001 + 1.0000001
            if x > 1e20:
                x = 1.0000001

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
    for t in threads:
        t.start()

    period_s = max(args.period_ms, 1.0) / 1000.0
    t_end = time.time() + args.seconds
    bursts = 0
    # 用「启停忙线程」近似 duty：busy 窗放行，idle 窗 pause（线程仍在但用 flag）
    # 更简单且符合「非 sleep 压力」：busy 窗什么都不做（线程狂转），idle 窗 stop+restart 成本高。
    # 采用门控：idle 时设 pause event，worker 自旋检查 pause（仍占 CPU 微弱）。
    # 为让 idle 真正降压，idle 窗 join 停线程再重建。
    try:
        if duty <= 0.0:
            stop.set()
            while time.time() < t_end:
                time.sleep(min(period_s, max(0.0, t_end - time.time())))
                bursts += 1
            return bursts

        if duty >= 1.0:
            # 满载：线程持续忙转，主线程只计时
            while time.time() < t_end:
                time.sleep(min(period_s, max(0.0, t_end - time.time())))
                bursts += 1
            return bursts

        while time.time() < t_end:
            period_start = time.time()
            busy_s = period_s * duty
            # busy：线程已在转
            while time.time() < t_end and (time.time() - period_start) < busy_s:
                time.sleep(0.001)
            # idle：停掉忙线程（主线程 sleep 不充当 CPU 压力）
            stop.set()
            for t in threads:
                t.join(timeout=1.0)
            bursts += 1
            elapsed = time.time() - period_start
            idle_s = period_s - elapsed
            if idle_s > 0 and time.time() < t_end:
                time.sleep(min(idle_s, t_end - time.time()))
            if time.time() >= t_end:
                break
            stop = threading.Event()
            threads = [
                threading.Thread(target=worker, daemon=True) for _ in range(n_threads)
            ]
            for t in threads:
                t.start()
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=1.0)
    return bursts


def _run_cube(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    dt = _torch_dtype(torch, args.dtype)
    dev = f"npu:{args.device}"
    a = torch.randn(args.size, args.size, device=dev, dtype=dt)
    b = torch.randn(args.size, args.size, device=dev, dtype=dt)

    def burst() -> None:
        torch.matmul(a, b)

    def sync() -> None:
        torch.npu.synchronize()

    return _duty_loop(args.seconds, args.duty, args.period_ms, burst, sync)


def _run_vector(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    dt = _torch_dtype(torch, args.dtype)
    dev = f"npu:{args.device}"
    a = torch.randn(args.elems, device=dev, dtype=dt)
    b = torch.randn(args.elems, device=dev, dtype=dt)
    c = torch.randn(args.elems, device=dev, dtype=dt)

    if args.vector_op == "exp":
        def burst() -> None:
            torch.exp(a)
    else:
        def burst() -> None:
            a.mul(b).add_(c)

    def sync() -> None:
        torch.npu.synchronize()

    return _duty_loop(args.seconds, args.duty, args.period_ms, burst, sync)


def _run_hbm_mte(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    dev = f"npu:{args.device}"
    elems = max(1, int(args.mb) * 1024 * 1024 // 4)
    src = torch.randn(elems, device=dev, dtype=torch.float32)
    dst = torch.empty(elems, device=dev, dtype=torch.float32)

    def burst() -> None:
        dst.copy_(src)

    def sync() -> None:
        torch.npu.synchronize()

    return _duty_loop(args.seconds, args.duty, args.period_ms, burst, sync)


def main() -> None:
    args = _parse_args()
    duty = _clamp_duty(args.duty)
    kind = args.kind
    if kind != "placebo" and duty <= 0.0:
        # duty=0 等价 placebo：不起设备工作
        kind_eff = "placebo"
    else:
        kind_eff = kind

    print(
        "INJECT_START"
        f" kind={kind}"
        f" kind_eff={kind_eff}"
        f" device={args.device}"
        f" seconds={args.seconds}"
        f" duty={duty}"
        f" period_ms={args.period_ms}"
        f" size={args.size}"
        f" elems={args.elems}"
        f" mb={args.mb}"
        f" dtype={args.dtype}"
        f" vector_op={args.vector_op}"
        f" cpu_threads={args.cpu_threads}"
        f" cpu_affinity={args.cpu_affinity or '-'}"
        ,
        flush=True,
    )
    t0 = time.time()
    if kind_eff == "placebo":
        bursts = _run_placebo(args)
    elif kind_eff == "cpu":
        bursts = _run_cpu(args)
    elif kind_eff == "cube":
        bursts = _run_cube(args)
    elif kind_eff == "vector":
        bursts = _run_vector(args)
    elif kind_eff == "hbm_mte":
        bursts = _run_hbm_mte(args)
    else:
        raise SystemExit(f"unknown kind: {kind_eff}")

    elapsed = time.time() - t0
    print(
        "INJECT_DONE"
        f" kind={kind}"
        f" kind_eff={kind_eff}"
        f" device={args.device}"
        f" bursts={bursts}"
        f" elapsed_s={elapsed:.3f}"
        ,
        flush=True,
    )


if __name__ == "__main__":
    main()
