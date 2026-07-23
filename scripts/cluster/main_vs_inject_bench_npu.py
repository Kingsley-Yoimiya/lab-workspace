#!/usr/bin/env python3
"""长窗口：不同干扰对「主进程」吞吐/逐步时间的影响（补实验）。

相对 Phase2 短窗 Event 测时的改进：
  - victim 连续跑数秒，主指标用 wall-clock iters/s 与逐步 iter_ms 分位数
  - ABBA：off → on → on → off，量化稳态吞吐下降
  - timeline：同一长窗内 quiet → burst → quiet，看突发开启/关闭对逐步时间的冲击

用法（容器内 device 常为 0）:
  python3 main_vs_inject_bench_npu.py --device 0 --protocol abba --out abba.jsonl
  python3 main_vs_inject_bench_npu.py --device 0 --protocol timeline \\
      --factors cube,hbm_mte --window-s 24 --out timeline.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics as st
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


def _pct(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    if not ys:
        return float("nan")
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def _start_inject(
    kind: str, device: int, duty: float, period_ms: float, size: int, mb: int
) -> Optional[subprocess.Popen]:
    if kind in ("", "none", "off", "placebo"):
        return None
    inject_py = Path(__file__).resolve().parent / "npu_component_inject.py"
    if not inject_py.is_file():
        raise FileNotFoundError(inject_py)
    cmd = [
        sys.executable,
        str(inject_py),
        "--kind",
        kind,
        "--device",
        str(device),
        "--seconds",
        "3600",
        "--duty",
        str(duty),
        "--period-ms",
        str(period_ms),
        "--size",
        str(size),
        "--mb",
        str(mb),
    ]
    print(f"INJECT_SPAWN {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    deadline = time.time() + 20.0
    started = False
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"inject died: {out[:500]}")
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            print(f"INJECT_LOG {line.rstrip()}", flush=True)
            if "INJECT_START" in line:
                started = True
                break
        else:
            time.sleep(0.05)
    if not started:
        print("INJECT_WARN no INJECT_START; continue", flush=True)
    time.sleep(1.0)
    return proc


def _stop_inject(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
    print("INJECT_KILLED", flush=True)


def _build_victim(workload: str, device: int, gemm_n: int, hidden: int, seq: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    dev = torch.device(f"npu:{device}")

    if workload == "gemm":
        a = torch.randn(gemm_n, gemm_n, device=dev, dtype=torch.float16)
        b = torch.randn(gemm_n, gemm_n, device=dev, dtype=torch.float16)

        def step():
            _ = torch.matmul(a, b)
            torch.npu.synchronize()

        flops_per = 2.0 * gemm_n * gemm_n * gemm_n
        return step, flops_per, "gemm_fp16"

    if workload == "block":

        class Block(nn.Module):
            def __init__(self, h: int):
                super().__init__()
                self.ln1 = nn.LayerNorm(h)
                self.fc1 = nn.Linear(h, 4 * h, bias=False)
                self.fc2 = nn.Linear(4 * h, h, bias=False)
                self.ln2 = nn.LayerNorm(h)
                self.fc3 = nn.Linear(h, 4 * h, bias=False)
                self.fc4 = nn.Linear(4 * h, h, bias=False)

            def forward(self, x):
                y = self.fc2(F.gelu(self.fc1(self.ln1(x))))
                x = x + y
                y = self.fc4(F.gelu(self.fc3(self.ln2(x))))
                return x + y

        model = Block(hidden).to(device=dev, dtype=torch.float16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        B, S, H = 1, seq, hidden

        def step():
            x = torch.randn(B, S, H, device=dev, dtype=torch.float16)
            opt.zero_grad(set_to_none=True)
            y = model(x)
            loss = y.float().pow(2).mean()
            loss.backward()
            opt.step()
            torch.npu.synchronize()

        return step, 0.0, "mlp_block_fwd_bwd"

    raise ValueError(f"unknown workload {workload}")


def _summarize_iters(iter_ms: list[float], wall_s: float) -> dict[str, Any]:
    n = len(iter_ms)
    return {
        "iters": n,
        "wall_s": wall_s,
        "iters_per_s": n / wall_s if wall_s > 0 else 0.0,
        "iter_ms_p50": _pct(iter_ms, 50),
        "iter_ms_p90": _pct(iter_ms, 90),
        "iter_ms_p95": _pct(iter_ms, 95),
        "iter_ms_p99": _pct(iter_ms, 99),
        "iter_ms_max": max(iter_ms) if iter_ms else float("nan"),
        "iter_ms_mean": st.mean(iter_ms) if iter_ms else float("nan"),
    }


def _run_window(
    step_fn,
    window_s: float,
    warmup_iters: int,
    *,
    inject_state: str = "off",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for _ in range(warmup_iters):
        step_fn()
    t0 = time.perf_counter()
    iter_ms: list[float] = []
    timeline: list[dict[str, Any]] = []
    n = 0
    while True:
        i0 = time.perf_counter()
        step_fn()
        i1 = time.perf_counter()
        ms = (i1 - i0) * 1000.0
        iter_ms.append(ms)
        timeline.append(
            {
                "iter": n,
                "t_rel_s": i0 - t0,
                "iter_ms": ms,
                "inject_state": inject_state,
            }
        )
        n += 1
        if (i1 - t0) >= window_s:
            break
    wall_s = time.perf_counter() - t0
    return _summarize_iters(iter_ms, wall_s), timeline


def _run_timeline_burst(
    step_fn,
    segment_s: float,
    warmup_iters: int,
    factor: str,
    device: int,
    duty: float,
    period_ms: float,
    inject_size: int,
    inject_mb: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """quiet → continuous burst → quiet；记录逐步时间与事件。"""
    for _ in range(warmup_iters):
        step_fn()

    events: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    iter_ms_by_phase: dict[str, list[float]] = {
        "pre": [],
        "burst": [],
        "post": [],
    }

    t0 = time.perf_counter()
    n = 0
    proc = None
    phase = "pre"
    phase_deadline = t0 + segment_s
    burst_start = t0 + segment_s
    burst_end = t0 + 2 * segment_s
    end_t = t0 + 3 * segment_s

    events.append({"event": "victim_start", "t_rel_s": 0.0})

    try:
        while True:
            now = time.perf_counter()
            if phase == "pre" and now >= burst_start:
                proc = _start_inject(
                    factor, device, duty, period_ms, inject_size, inject_mb
                )
                phase = "burst"
                phase_deadline = burst_end
                events.append(
                    {
                        "event": "inject_start",
                        "factor": factor,
                        "t_rel_s": time.perf_counter() - t0,
                    }
                )
            elif phase == "burst" and now >= burst_end:
                _stop_inject(proc)
                proc = None
                phase = "post"
                phase_deadline = end_t
                events.append(
                    {
                        "event": "inject_stop",
                        "factor": factor,
                        "t_rel_s": time.perf_counter() - t0,
                    }
                )
            elif phase == "post" and now >= end_t:
                break

            i0 = time.perf_counter()
            step_fn()
            i1 = time.perf_counter()
            ms = (i1 - i0) * 1000.0
            iter_ms_by_phase[phase].append(ms)
            timeline.append(
                {
                    "iter": n,
                    "t_rel_s": i0 - t0,
                    "iter_ms": ms,
                    "inject_state": phase,
                    "factor": factor,
                }
            )
            n += 1
            if i1 >= end_t and phase == "post":
                break
            # 防止 inject 启动阻塞后越过 deadline 却无 step
            _ = phase_deadline
    finally:
        _stop_inject(proc)

    wall_s = time.perf_counter() - t0
    events.append({"event": "victim_end", "t_rel_s": wall_s})

    phase_stats = {}
    for ph, xs in iter_ms_by_phase.items():
        phase_stats[ph] = _summarize_iters(xs, sum(xs) / 1000.0 if xs else 0.0)

    pre_p50 = phase_stats["pre"].get("iter_ms_p50") or float("nan")
    burst_p50 = phase_stats["burst"].get("iter_ms_p50") or float("nan")
    post_p50 = phase_stats["post"].get("iter_ms_p50") or float("nan")
    summary = {
        "iters_total": n,
        "wall_s": wall_s,
        "segment_s": segment_s,
        "pre": phase_stats["pre"],
        "burst": phase_stats["burst"],
        "post": phase_stats["post"],
        "burst_vs_pre_slowdown_pct": (
            (burst_p50 / pre_p50 - 1.0) * 100.0 if pre_p50 and pre_p50 > 0 else float("nan")
        ),
        "post_vs_pre_slowdown_pct": (
            (post_p50 / pre_p50 - 1.0) * 100.0 if pre_p50 and pre_p50 > 0 else float("nan")
        ),
    }
    return summary, timeline, events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--workload", choices=("gemm", "block"), default="gemm")
    ap.add_argument("--protocol", choices=("abba", "timeline"), default="abba")
    ap.add_argument("--factors", default="placebo,cube,vector,hbm_mte")
    ap.add_argument(
        "--window-s",
        type=float,
        default=8.0,
        help="abba: 每窗时长；timeline: 每段（pre/burst/post）时长",
    )
    ap.add_argument("--warmup-iters", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3, help="每因素重复轮数")
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--period-ms", type=float, default=200.0)
    ap.add_argument("--gemm-n", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--inject-size", type=int, default=4096)
    ap.add_argument("--inject-mb", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cooldown-s", type=float, default=2.0)
    ap.add_argument(
        "--emit-timeline",
        action="store_true",
        help="把逐步时间写入同目录 <stem>.timeline.jsonl",
    )
    args = ap.parse_args()

    import torch_npu  # noqa: F401

    step_fn, flops_per, wl_name = _build_victim(
        args.workload, args.device, args.gemm_n, args.hidden, args.seq
    )
    factors = [x.strip() for x in args.factors.split(",") if x.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tl_path = out_path.with_suffix(".timeline.jsonl")
    if args.emit_timeline or args.protocol == "timeline":
        if tl_path.exists():
            tl_path.unlink()

    def emit(rec: dict) -> None:
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    def emit_tl(rec: dict) -> None:
        with tl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    emit(
        {
            "record": "run_meta",
            "phase": "main_vs_inject",
            "protocol": args.protocol,
            "workload": wl_name,
            "device": args.device,
            "window_s": args.window_s,
            "repeats": args.repeats,
            "duty": args.duty,
            "factors": factors,
        }
    )

    if args.protocol == "abba":
        abba = [("off", False), ("on", True), ("on", True), ("off", False)]
        for factor in factors:
            for rep in range(args.repeats):
                for phase_i, (phase_name, inject_on) in enumerate(abba):
                    if factor == "placebo":
                        inject_on = False
                    proc = None
                    try:
                        if inject_on:
                            proc = _start_inject(
                                kind=factor,
                                device=args.device,
                                duty=args.duty,
                                period_ms=args.period_ms,
                                size=args.inject_size,
                                mb=args.inject_mb,
                            )
                        else:
                            time.sleep(0.3)
                        state = "on" if inject_on else "off"
                        stats, timeline = _run_window(
                            step_fn,
                            args.window_s,
                            args.warmup_iters,
                            inject_state=state,
                        )
                        rec = {
                            "record": "window",
                            "factor": factor,
                            "inject_on": bool(inject_on),
                            "inject_kind": factor if inject_on else "none",
                            "abba_phase": phase_name,
                            "abba_idx": phase_i,
                            "repeat": rep,
                            "workload": wl_name,
                            "device": args.device,
                            "duty": args.duty if inject_on else 0.0,
                            **stats,
                        }
                        if flops_per > 0 and stats["wall_s"] > 0:
                            rec["tflops"] = (
                                (stats["iters"] * flops_per) / stats["wall_s"] / 1e12
                            )
                        emit(rec)
                        if args.emit_timeline:
                            for row in timeline:
                                emit_tl(
                                    {
                                        "record": "timeline_iter",
                                        "factor": factor,
                                        "repeat": rep,
                                        "abba_idx": phase_i,
                                        **row,
                                    }
                                )
                    finally:
                        _stop_inject(proc)
                        time.sleep(args.cooldown_s)

        rows = []
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("record") == "window":
                rows.append(r)

        print("\n=== SUMMARY (iters_per_s) ===", flush=True)
        for factor in factors:
            off = [r["iters_per_s"] for r in rows if r["factor"] == factor and not r["inject_on"]]
            on = [r["iters_per_s"] for r in rows if r["factor"] == factor and r["inject_on"]]
            if not off:
                continue
            off_med = st.median(off)
            on_med = st.median(on) if on else off_med
            drop = (1.0 - on_med / off_med) * 100.0 if off_med > 0 else 0.0
            off_p50 = st.median(
                [r["iter_ms_p50"] for r in rows if r["factor"] == factor and not r["inject_on"]]
            )
            on_p50 = (
                st.median(
                    [r["iter_ms_p50"] for r in rows if r["factor"] == factor and r["inject_on"]]
                )
                if on
                else off_p50
            )
            summ = {
                "record": "factor_summary",
                "factor": factor,
                "off_iters_per_s_med": off_med,
                "on_iters_per_s_med": on_med,
                "main_slowdown_pct": drop,
                "off_iter_ms_p50_med": off_p50,
                "on_iter_ms_p50_med": on_p50,
                "iter_ms_p50_increase_pct": (
                    (on_p50 / off_p50 - 1.0) * 100.0 if off_p50 > 0 else 0.0
                ),
                "n_off": len(off),
                "n_on": len(on),
            }
            emit(summ)
            print(
                f"  {factor:10s} off={off_med:.3f}/s on={on_med:.3f}/s "
                f"thru_drop={drop:.1f}%  iter_p50 {off_p50:.2f}->{on_p50:.2f}ms "
                f"({summ['iter_ms_p50_increase_pct']:.1f}%)",
                flush=True,
            )
        return

    # timeline protocol
    for factor in factors:
        if factor == "placebo":
            continue
        for rep in range(args.repeats):
            summary, timeline, events = _run_timeline_burst(
                step_fn,
                args.window_s,
                args.warmup_iters,
                factor,
                args.device,
                args.duty,
                args.period_ms,
                args.inject_size,
                args.inject_mb,
            )
            emit(
                {
                    "record": "timeline_summary",
                    "factor": factor,
                    "repeat": rep,
                    "workload": wl_name,
                    **summary,
                }
            )
            for ev in events:
                emit({"record": "timeline_event", "factor": factor, "repeat": rep, **ev})
            for row in timeline:
                emit_tl(
                    {
                        "record": "timeline_iter",
                        "factor": factor,
                        "repeat": rep,
                        **row,
                    }
                )
            time.sleep(args.cooldown_s)

    print("\n=== TIMELINE SUMMARY ===", flush=True)
    for line in out_path.read_text().splitlines():
        r = json.loads(line)
        if r.get("record") != "timeline_summary":
            continue
        print(
            f"  {r['factor']:10s} rep={r['repeat']} "
            f"pre_p50={r['pre']['iter_ms_p50']:.2f}ms "
            f"burst_p50={r['burst']['iter_ms_p50']:.2f}ms "
            f"post_p50={r['post']['iter_ms_p50']:.2f}ms "
            f"burst_slowdown={r['burst_vs_pre_slowdown_pct']:.1f}% "
            f"post_residual={r['post_vs_pre_slowdown_pct']:.1f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
