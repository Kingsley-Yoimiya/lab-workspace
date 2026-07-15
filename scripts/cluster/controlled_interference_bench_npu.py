#!/usr/bin/env python3
"""扫描预热常驻 sidecar 剂量，测量 victim 的吞吐与 step 分位数。

Examples:
  python3 controlled_interference_bench_npu.py \
    --device 0 --workload gemm --inject-kind cube \
    --doses 0,0.1,0.2,0.3,0.4,0.5 --window-s 8 --repeats 3 \
    --out results/controlled-cube.jsonl

  python3 controlled_interference_bench_npu.py \
    --device 0 --workload block --inject-kind hbm_mte \
    --doses 0,0.2,0.4 --window-s 2 --repeats 1 \
    --out results/smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import selectors
import statistics as st
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--workload",
        choices=("gemm", "attention", "norm", "elementwise", "block", "transformer"),
        required=True,
    )
    parser.add_argument(
        "--inject-kind",
        choices=("cube", "vector", "hbm_mte", "hbm_vector", "small_ops"),
        required=True,
    )
    parser.add_argument("--profile", choices=("custom", "small", "large"), default="custom")
    parser.add_argument("--pattern", choices=("periodic", "poisson"), default="periodic")
    parser.add_argument("--burst-mean-ms", type=float, default=20.0)
    parser.add_argument("--doses", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--window-s", type=float, default=8.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--period-ms", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--cooldown-s", type=float, default=0.5)
    parser.add_argument("--gemm-n", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--inject-size", type=int, default=4096)
    parser.add_argument("--inject-elems", type=int, default=1 << 26)
    parser.add_argument("--inject-mb", type=int, default=512)
    parser.add_argument("--inject-small-op-elems", type=int, default=1 << 18)
    parser.add_argument("--inject-small-op-count", type=int, default=16)
    parser.add_argument("--sidecar-warmup-iters", type=int, default=5)
    parser.add_argument("--victim-warmup-iters", type=int, default=10)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    index = (len(xs) - 1) * p / 100.0
    lo = int(index)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (index - lo)


def _apply_profile(args: argparse.Namespace) -> None:
    if args.profile == "small":
        args.gemm_n = 2048
        args.hidden = 2048
        args.seq = 512
        args.heads = 16
    elif args.profile == "large":
        args.gemm_n = 4096
        args.hidden = 4096
        args.seq = 1024
        args.heads = 32


def _build_victim(args: argparse.Namespace):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    dev = torch.device(f"npu:{args.device}")
    if args.workload == "gemm":
        a = torch.randn(args.gemm_n, args.gemm_n, device=dev, dtype=torch.float16)
        b = torch.randn(args.gemm_n, args.gemm_n, device=dev, dtype=torch.float16)
        out = torch.empty(args.gemm_n, args.gemm_n, device=dev, dtype=torch.float16)

        def step() -> None:
            torch.mm(a, b, out=out)
            torch.npu.synchronize()

        return step, "gemm_fp16"

    if args.hidden % args.heads != 0:
        raise ValueError("--hidden must be divisible by --heads")
    head_dim = args.hidden // args.heads

    if args.workload == "attention":
        shape = (1, args.heads, args.seq, head_dim)
        q = torch.randn(shape, device=dev, dtype=torch.float16, requires_grad=True)
        k = torch.randn(shape, device=dev, dtype=torch.float16, requires_grad=True)
        v = torch.randn(shape, device=dev, dtype=torch.float16, requires_grad=True)

        def step() -> None:
            q.grad = None
            k.grad = None
            v.grad = None
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            probs = F.softmax(scores.float(), dim=-1).to(torch.float16)
            output = torch.matmul(probs, v)
            output.float().square().mean().backward()
            torch.npu.synchronize()

        return step, "attention_qk_softmax_pv_fwd_bwd"

    if args.workload == "norm":
        model = nn.LayerNorm(args.hidden).to(device=dev, dtype=torch.float16)
        x = torch.randn(
            1,
            args.seq,
            args.hidden,
            device=dev,
            dtype=torch.float16,
            requires_grad=True,
        )

        def step() -> None:
            x.grad = None
            model.zero_grad(set_to_none=True)
            output = model(x)
            output.float().square().mean().backward()
            torch.npu.synchronize()

        return step, "layernorm_fwd_bwd"

    if args.workload == "elementwise":
        x = torch.randn(
            1,
            args.seq,
            args.hidden,
            device=dev,
            dtype=torch.float16,
            requires_grad=True,
        )
        gate = torch.randn_like(x)
        residual = torch.randn_like(x)

        def step() -> None:
            x.grad = None
            output = F.silu(x) * gate + residual
            output.float().square().mean().backward()
            torch.npu.synchronize()

        return step, "silu_gate_residual_fwd_bwd"

    class MlpBlock(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(hidden)
            self.fc1 = nn.Linear(hidden, 4 * hidden, bias=False)
            self.fc2 = nn.Linear(4 * hidden, hidden, bias=False)
            self.ln2 = nn.LayerNorm(hidden)
            self.fc3 = nn.Linear(hidden, 4 * hidden, bias=False)
            self.fc4 = nn.Linear(4 * hidden, hidden, bias=False)

        def forward(self, x):
            y = self.fc2(F.gelu(self.fc1(self.ln1(x))))
            x = x + y
            y = self.fc4(F.gelu(self.fc3(self.ln2(x))))
            return x + y

    if args.workload == "block":
        model = MlpBlock(args.hidden).to(device=dev, dtype=torch.float16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def step() -> None:
            x = torch.randn(1, args.seq, args.hidden, device=dev, dtype=torch.float16)
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            output.float().square().mean().backward()
            optimizer.step()
            torch.npu.synchronize()

        return step, "mlp_block_fwd_bwd"

    class TransformerLayer(nn.Module):
        def __init__(self, hidden: int, heads: int):
            super().__init__()
            self.hidden = hidden
            self.heads = heads
            self.head_dim = hidden // heads
            self.ln1 = nn.LayerNorm(hidden)
            self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
            self.proj = nn.Linear(hidden, hidden, bias=False)
            self.ln2 = nn.LayerNorm(hidden)
            self.fc1 = nn.Linear(hidden, 4 * hidden, bias=False)
            self.fc2 = nn.Linear(4 * hidden, hidden, bias=False)

        def forward(self, x):
            batch, seq, _ = x.shape
            qkv = self.qkv(self.ln1(x)).reshape(
                batch, seq, 3, self.heads, self.head_dim
            )
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            probs = F.softmax(scores.float(), dim=-1).to(x.dtype)
            context = torch.matmul(probs, v)
            context = context.transpose(1, 2).reshape(batch, seq, self.hidden)
            x = x + self.proj(context)
            return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))

    model = TransformerLayer(args.hidden, args.heads).to(
        device=dev, dtype=torch.float16
    )
    x = torch.randn(
        1,
        args.seq,
        args.hidden,
        device=dev,
        dtype=torch.float16,
        requires_grad=True,
    )

    def step() -> None:
        x.grad = None
        model.zero_grad(set_to_none=True)
        output = model(x)
        output.float().square().mean().backward()
        torch.npu.synchronize()

    return step, "transformer_attention_mlp_fwd_bwd"


def _run_window(step_fn, window_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    values: list[float] = []
    while True:
        t0 = time.perf_counter()
        step_fn()
        t1 = time.perf_counter()
        values.append((t1 - t0) * 1000.0)
        if t1 - started >= window_s:
            break
    elapsed = time.perf_counter() - started
    return {
        "iters": len(values),
        "wall_s": elapsed,
        "iters_per_s": len(values) / elapsed,
        "iter_ms_mean": st.mean(values),
        "iter_ms_p50": _percentile(values, 50),
        "iter_ms_p90": _percentile(values, 90),
        "iter_ms_p95": _percentile(values, 95),
        "iter_ms_p99": _percentile(values, 99),
        "iter_ms_max": max(values),
    }


class Sidecar:
    def __init__(self, args: argparse.Namespace):
        script = Path(__file__).resolve().parent / "npu_controlled_sidecar.py"
        command = [
            sys.executable,
            str(script),
            "--kind",
            args.inject_kind,
            "--device",
            str(args.device),
            "--period-ms",
            str(args.period_ms),
            "--size",
            str(args.inject_size),
            "--elems",
            str(args.inject_elems),
            "--mb",
            str(args.inject_mb),
            "--small-op-elems",
            str(args.inject_small_op_elems),
            "--small-op-count",
            str(args.inject_small_op_count),
            "--warmup-iters",
            str(args.sidecar_warmup_iters),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        if self.process.stdout is None:
            raise RuntimeError("sidecar stdout unavailable")
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.ready = self.read("ready", timeout_s=120.0)

    def send(self, cmd: str, **fields) -> None:
        if self.process.stdin is None:
            raise RuntimeError("sidecar stdin unavailable")
        self.process.stdin.write(json.dumps({"cmd": cmd, **fields}) + "\n")
        self.process.stdin.flush()

    def read(self, expected: str, timeout_s: float = 30.0) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"sidecar exited rc={self.process.returncode}")
            events = self.selector.select(timeout=min(0.5, deadline - time.monotonic()))
            if not events:
                continue
            line = self.process.stdout.readline().strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"SIDECAR_LOG {line}", flush=True)
                continue
            print(f"SIDECAR {json.dumps(record, ensure_ascii=False)}", flush=True)
            if record.get("record") == "error":
                raise RuntimeError(f"sidecar error: {record}")
            if record.get("record") == expected:
                return record
        raise TimeoutError(f"timeout waiting for sidecar record={expected}")

    def start(self, duty: float, pattern: str, burst_mean_ms: float, seed: int) -> dict:
        self.send(
            "start",
            duty=duty,
            pattern=pattern,
            burst_mean_ms=burst_mean_ms,
            seed=seed,
        )
        return self.read("active_start")

    def stop(self) -> dict:
        self.send("stop")
        return self.read("active_stop")

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.send("quit")
            self.read("bye", timeout_s=30.0)
        except Exception:
            self.process.terminate()
        try:
            self.process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10.0)


def _emit(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def _summarize(
    path: Path,
    doses: list[float],
    workload: str,
    inject_kind: str,
    profile: str,
    pattern: str,
) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    windows = [row for row in rows if row.get("record") == "window"]
    baseline = st.median([row["iters_per_s"] for row in windows if row["target_duty"] == 0])
    points = []
    for dose in sorted(doses):
        selected = [row for row in windows if math.isclose(row["target_duty"], dose)]
        throughput = st.median([row["iters_per_s"] for row in selected])
        active = [row for row in selected if row.get("sidecar")]
        point = {
            "target_duty": dose,
            "n": len(selected),
            "victim_iters_per_s_median": throughput,
            "victim_iters_per_s_cv_pct": (
                st.stdev(row["iters_per_s"] for row in selected)
                / st.mean(row["iters_per_s"] for row in selected)
                * 100.0
                if len(selected) > 1
                else 0.0
            ),
            "victim_throughput_drop_pct": (1.0 - throughput / baseline) * 100.0,
            "victim_iter_ms_p50_median": st.median(
                row["iter_ms_p50"] for row in selected
            ),
            "victim_iter_ms_p90_median": st.median(
                row["iter_ms_p90"] for row in selected
            ),
            "victim_iter_ms_p95_median": st.median(
                row["iter_ms_p95"] for row in selected
            ),
            "victim_iter_ms_p99_median": st.median(
                row["iter_ms_p99"] for row in selected
            ),
        }
        if active:
            point.update(
                {
                    "sidecar_busy_wall_ratio_median": st.median(
                        row["sidecar"]["busy_wall_ratio"] for row in active
                    ),
                    "sidecar_ops_per_s_median": st.median(
                        row["sidecar"]["ops"] / row["sidecar"]["elapsed_s"]
                        for row in active
                    ),
                    "sidecar_work_per_s_median": st.median(
                        row["sidecar"]["work_per_s"] for row in active
                    ),
                    "sidecar_work_unit": active[0]["sidecar"]["work_unit"],
                }
            )
        points.append(point)
    drops = [point["victim_throughput_drop_pct"] for point in points]
    monotonic_pairs = sum(
        1 for left, right in zip(drops, drops[1:]) if right + 1.0 >= left
    )
    return {
        "record": "summary",
        "workload": workload,
        "inject_kind": inject_kind,
        "profile": profile,
        "pattern": pattern,
        "baseline_iters_per_s": baseline,
        "points": points,
        "monotonic_pairs_with_1pp_tolerance": monotonic_pairs,
        "monotonic_pairs_total": max(0, len(points) - 1),
    }


def main() -> int:
    args = _parse_args()
    _apply_profile(args)
    doses = [float(value.strip()) for value in args.doses.split(",") if value.strip()]
    if not doses or any(dose < 0 or dose > 1 for dose in doses):
        raise SystemExit("error: --doses must contain comma-separated values in [0, 1]")
    if args.window_s <= 0 or args.repeats <= 0:
        raise SystemExit("error: --window-s and --repeats must be > 0")
    if args.burst_mean_ms <= 0:
        raise SystemExit("error: --burst-mean-ms must be > 0")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"error: output already exists: {output}")

    step_fn, workload_name = _build_victim(args)
    for _ in range(args.victim_warmup_iters):
        step_fn()

    sidecar = Sidecar(args)
    _emit(
        output,
        {
            "record": "run_meta",
            "workload": workload_name,
            "inject_kind": args.inject_kind,
            "profile": args.profile,
            "pattern": args.pattern,
            "burst_mean_ms": args.burst_mean_ms,
            "device": args.device,
            "doses": doses,
            "window_s": args.window_s,
            "repeats": args.repeats,
            "period_ms": args.period_ms,
            "seed": args.seed,
            "gemm_n": args.gemm_n,
            "hidden": args.hidden,
            "seq": args.seq,
            "heads": args.heads,
            "inject_size": args.inject_size,
            "inject_elems": args.inject_elems,
            "inject_mb": args.inject_mb,
            "sidecar_ready": sidecar.ready,
        },
    )
    try:
        for repeat in range(args.repeats):
            order = list(doses)
            random.Random(args.seed + repeat).shuffle(order)
            for order_index, dose in enumerate(order):
                sidecar_stats = None
                window_seed = args.seed + repeat * 1000 + order_index
                if dose > 0:
                    sidecar.start(
                        dose,
                        args.pattern,
                        args.burst_mean_ms,
                        window_seed,
                    )
                stats = _run_window(step_fn, args.window_s)
                if dose > 0:
                    sidecar_stats = sidecar.stop()
                _emit(
                    output,
                    {
                        "record": "window",
                        "repeat": repeat,
                        "order_index": order_index,
                        "target_duty": dose,
                        "workload": workload_name,
                        "inject_kind": args.inject_kind,
                        "profile": args.profile,
                        "pattern": args.pattern,
                        "burst_mean_ms": args.burst_mean_ms,
                        "window_seed": window_seed,
                        **stats,
                        "sidecar": sidecar_stats,
                    },
                )
                time.sleep(args.cooldown_s)
    finally:
        sidecar.close()

    summary = _summarize(
        output,
        doses,
        workload_name,
        args.inject_kind,
        args.profile,
        args.pattern,
    )
    _emit(output, summary)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"summary_path: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
