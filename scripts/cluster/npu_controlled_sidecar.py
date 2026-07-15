#!/usr/bin/env python3
"""预热常驻、可门控的 NPU 部件干扰 sidecar。

控制协议为 stdin/stdout JSONL。进程完成 NPU 初始化和 warmup 后输出 READY；
父进程随后发送 START/STOP，不再用创建/杀进程表示干扰边界。

Examples:
  python3 npu_controlled_sidecar.py --kind cube --device 0
  printf '%s\n' '{"cmd":"start","duty":0.2}' '{"cmd":"stop"}' '{"cmd":"quit"}' |
    python3 npu_controlled_sidecar.py --kind hbm_mte --device 0
"""
from __future__ import annotations

import argparse
import json
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable


def _emit(record: str, **fields) -> None:
    print(json.dumps({"record": record, **fields}, ensure_ascii=False), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kind",
        required=True,
        choices=("cube", "vector", "hbm_mte", "hbm_vector", "small_ops"),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--period-ms", type=float, default=100.0)
    parser.add_argument("--size", type=int, default=4096, help="Cube GEMM 方阵边长")
    parser.add_argument("--elems", type=int, default=1 << 26, help="Vector 元素数")
    parser.add_argument("--mb", type=int, default=512, help="MTE copy 缓冲区 MiB")
    parser.add_argument("--small-op-elems", type=int, default=1 << 18)
    parser.add_argument("--small-op-count", type=int, default=16)
    parser.add_argument("--warmup-iters", type=int, default=5)
    return parser.parse_args()


@dataclass(frozen=True)
class Workload:
    op: Callable[[], None]
    sync: Callable[[], None]
    work_per_op: float
    work_unit: str


def _build_workload(args: argparse.Namespace) -> Workload:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    dev = f"npu:{args.device}"

    if args.kind == "cube":
        a = torch.randn(args.size, args.size, device=dev, dtype=torch.float16)
        b = torch.randn(args.size, args.size, device=dev, dtype=torch.float16)
        out = torch.empty(args.size, args.size, device=dev, dtype=torch.float16)

        def op() -> None:
            torch.mm(a, b, out=out)

        work_per_op = 2.0 * args.size**3
        work_unit = "flop"
    elif args.kind == "vector":
        a = torch.randn(args.elems, device=dev, dtype=torch.float16)
        b = torch.randn(args.elems, device=dev, dtype=torch.float16)
        c = torch.randn(args.elems, device=dev, dtype=torch.float16)
        out = torch.empty(args.elems, device=dev, dtype=torch.float16)

        def op() -> None:
            torch.mul(a, b, out=out)
            out.add_(c)

        work_per_op = 2.0 * args.elems
        work_unit = "flop"
    elif args.kind == "hbm_mte":
        elems = max(1, args.mb * 1024 * 1024 // 4)
        src = torch.randn(elems, device=dev, dtype=torch.float32)
        dst = torch.empty(elems, device=dev, dtype=torch.float32)

        def op() -> None:
            dst.copy_(src)

        work_per_op = float(2 * elems * 4)
        work_unit = "byte"
    elif args.kind == "hbm_vector":
        elems = max(1, args.mb * 1024 * 1024 // 4)
        src = torch.randn(elems, device=dev, dtype=torch.float32)
        dst = torch.empty(elems, device=dev, dtype=torch.float32)

        def op() -> None:
            torch.mul(src, 1.0001, out=dst)

        work_per_op = float(2 * elems * 4)
        work_unit = "byte"
    else:
        a = torch.randn(args.small_op_elems, device=dev, dtype=torch.float16)
        b = torch.randn(args.small_op_elems, device=dev, dtype=torch.float16)
        out = torch.empty(args.small_op_elems, device=dev, dtype=torch.float16)

        def op() -> None:
            for _ in range(args.small_op_count):
                torch.add(a, b, out=out)
                torch.mul(out, b, out=a)

        work_per_op = float(2 * args.small_op_count)
        work_unit = "kernel"

    def sync() -> None:
        torch.npu.synchronize()

    return Workload(op=op, sync=sync, work_per_op=work_per_op, work_unit=work_unit)


class Controller:
    def __init__(self, workload: Workload, period_ms: float):
        self.workload = workload
        self.period_s = max(period_ms, 1.0) / 1000.0
        self.active = threading.Event()
        self.quit = threading.Event()
        self.lock = threading.Lock()
        self.duty = 0.0
        self.pattern = "periodic"
        self.burst_mean_s = 0.02
        self.seed = 0
        self.run_id = 0
        self.done: queue.Queue[dict] = queue.Queue()
        self.worker = threading.Thread(target=self._worker, daemon=True)

    def start_worker(self) -> None:
        self.worker.start()

    def start(
        self,
        duty: float,
        pattern: str = "periodic",
        burst_mean_ms: float = 20.0,
        seed: int = 0,
    ) -> int:
        duty = float(duty)
        if not 0.0 <= duty <= 1.0:
            raise ValueError("duty must be in [0, 1]")
        if pattern not in ("periodic", "poisson"):
            raise ValueError("pattern must be periodic or poisson")
        if burst_mean_ms <= 0:
            raise ValueError("burst_mean_ms must be > 0")
        if self.active.is_set():
            raise RuntimeError("sidecar already active")
        with self.lock:
            self.duty = duty
            self.pattern = pattern
            self.burst_mean_s = burst_mean_ms / 1000.0
            self.seed = int(seed)
            self.run_id += 1
            run_id = self.run_id
        self.active.set()
        return run_id

    def stop(self, timeout_s: float = 30.0) -> dict:
        if not self.active.is_set():
            raise RuntimeError("sidecar is not active")
        self.active.clear()
        try:
            return self.done.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise RuntimeError("timeout waiting for sidecar to become idle") from exc

    def close(self) -> None:
        if self.active.is_set():
            self.active.clear()
            try:
                self.done.get(timeout=30.0)
            except queue.Empty:
                pass
        self.quit.set()
        self.active.set()
        self.worker.join(timeout=30.0)

    def _worker(self) -> None:
        while not self.quit.is_set():
            self.active.wait(timeout=0.1)
            if self.quit.is_set():
                return
            if not self.active.is_set():
                continue
            with self.lock:
                duty = self.duty
                pattern = self.pattern
                burst_mean_s = self.burst_mean_s
                seed = self.seed
                run_id = self.run_id
            started = time.perf_counter()
            busy_wall_s = 0.0
            ops = 0
            periods = 0
            overshoot_s = 0.0

            rng = random.Random(seed)
            while self.active.is_set() and not self.quit.is_set():
                if pattern == "periodic":
                    cycle_start = time.perf_counter()
                    budget_s = self.period_s * duty
                else:
                    if duty < 1.0:
                        off_mean_s = burst_mean_s * (1.0 - duty) / max(duty, 1e-9)
                        off_s = rng.expovariate(1.0 / off_mean_s)
                        sleep_end = time.perf_counter() + off_s
                        while self.active.is_set() and time.perf_counter() < sleep_end:
                            remaining = max(0.0, sleep_end - time.perf_counter())
                            time.sleep(min(0.01, remaining))
                    if not self.active.is_set() or self.quit.is_set():
                        break
                    cycle_start = time.perf_counter()
                    budget_s = rng.expovariate(1.0 / burst_mean_s)

                cycle_busy_s = 0.0
                while (
                    self.active.is_set()
                    and not self.quit.is_set()
                    and cycle_busy_s < budget_s
                ):
                    op_start = time.perf_counter()
                    self.workload.op()
                    self.workload.sync()
                    elapsed = time.perf_counter() - op_start
                    cycle_busy_s += elapsed
                    busy_wall_s += elapsed
                    ops += 1
                periods += 1
                overshoot_s += max(0.0, cycle_busy_s - budget_s)
                if pattern == "periodic":
                    sleep_s = self.period_s - (time.perf_counter() - cycle_start)
                    if sleep_s > 0 and self.active.is_set():
                        time.sleep(sleep_s)

            self.workload.sync()
            ended = time.perf_counter()
            elapsed_s = ended - started
            self.done.put(
                {
                    "run_id": run_id,
                    "target_duty": duty,
                    "pattern": pattern,
                    "burst_mean_ms": burst_mean_s * 1000.0,
                    "seed": seed,
                    "elapsed_s": elapsed_s,
                    "periods": periods,
                    "ops": ops,
                    "busy_wall_s": busy_wall_s,
                    "busy_wall_ratio": busy_wall_s / elapsed_s if elapsed_s > 0 else 0.0,
                    "overshoot_s": overshoot_s,
                    "work": ops * self.workload.work_per_op,
                    "work_unit": self.workload.work_unit,
                    "work_per_s": (
                        ops * self.workload.work_per_op / elapsed_s
                        if elapsed_s > 0
                        else 0.0
                    ),
                }
            )


def main() -> int:
    args = _parse_args()
    if args.period_ms <= 0:
        raise SystemExit("error: --period-ms must be > 0")
    if args.warmup_iters < 0:
        raise SystemExit("error: --warmup-iters must be >= 0")

    initialized_at = time.perf_counter()
    workload = _build_workload(args)
    for _ in range(args.warmup_iters):
        workload.op()
        workload.sync()
    controller = Controller(workload, args.period_ms)
    controller.start_worker()
    _emit(
        "ready",
        kind=args.kind,
        device=args.device,
        period_ms=args.period_ms,
        warmup_iters=args.warmup_iters,
        init_s=time.perf_counter() - initialized_at,
        work_per_op=workload.work_per_op,
        work_unit=workload.work_unit,
        size=args.size,
        elems=args.elems,
        mb=args.mb,
        small_op_elems=args.small_op_elems,
        small_op_count=args.small_op_count,
    )

    try:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                command = json.loads(raw)
                cmd = command.get("cmd", "").lower()
                if cmd == "start":
                    duty = float(command["duty"])
                    pattern = command.get("pattern", "periodic")
                    burst_mean_ms = float(command.get("burst_mean_ms", 20.0))
                    seed = int(command.get("seed", 0))
                    run_id = controller.start(duty, pattern, burst_mean_ms, seed)
                    _emit(
                        "active_start",
                        run_id=run_id,
                        target_duty=duty,
                        pattern=pattern,
                        burst_mean_ms=burst_mean_ms,
                        seed=seed,
                    )
                elif cmd == "stop":
                    stats = controller.stop()
                    _emit("active_stop", **stats)
                elif cmd == "quit":
                    controller.close()
                    _emit("bye")
                    return 0
                elif cmd == "ping":
                    _emit("pong", active=controller.active.is_set())
                else:
                    raise ValueError("cmd must be start, stop, quit, or ping")
            except Exception as exc:  # 协议错误必须返回结构化信息，不能悄悄挂死父进程
                _emit("error", error=type(exc).__name__, message=str(exc))
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
