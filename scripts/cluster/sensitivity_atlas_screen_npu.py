#!/usr/bin/env python3
"""单进程执行完整敏感度图谱，避免每个 case 重复导入 torch_npu。

Examples:
  python3 sensitivity_atlas_screen_npu.py \
    --device 0 --out-dir results/atlas-screen
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from argparse import Namespace
from pathlib import Path

import controlled_interference_bench_npu as bench


INJECTORS = ("cube", "vector", "hbm_mte", "hbm_vector", "small_ops")
WORKLOADS = ("gemm", "attention", "norm", "elementwise", "block", "transformer")
PROFILES = ("small", "large")
PATTERNS = ("periodic", "poisson")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--doses", default="0,0.1,0.3,0.5")
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--victim-warmup-iters", type=int, default=3)
    parser.add_argument("--sidecar-warmup-iters", type=int, default=3)
    parser.add_argument("--cooldown-s", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _case_args(config: argparse.Namespace, injector: str, workload: str, profile: str) -> Namespace:
    args = Namespace(
        device=config.device,
        workload=workload,
        inject_kind=injector,
        profile=profile,
        pattern="periodic",
        burst_mean_ms=20.0,
        period_ms=100.0,
        seed=config.seed,
        cooldown_s=config.cooldown_s,
        gemm_n=4096,
        hidden=4096,
        seq=1024,
        heads=32,
        inject_size=4096,
        inject_elems=1 << 26,
        inject_mb=512,
        inject_small_op_elems=1 << 18,
        inject_small_op_count=16,
        sidecar_warmup_iters=config.sidecar_warmup_iters,
        victim_warmup_iters=config.victim_warmup_iters,
    )
    bench._apply_profile(args)
    return args


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    config = _parse_args()
    doses = [float(value) for value in config.doses.split(",")]
    if not doses or any(dose < 0 or dose > 1 for dose in doses):
        raise SystemExit("error: --doses must contain comma-separated values in [0, 1]")
    if config.out_dir.exists() and not config.resume and any(config.out_dir.iterdir()):
        raise SystemExit(
            f"error: non-empty output directory already exists: {config.out_dir}; "
            "use a new path or --resume"
        )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = config.out_dir / "matrix_manifest.jsonl"
    total = len(INJECTORS) * len(WORKLOADS) * len(PROFILES) * len(PATTERNS)
    case_index = 0
    started = time.monotonic()

    import torch

    for injector in INJECTORS:
        sidecar_args = _case_args(config, injector, "gemm", "small")
        sidecar = bench.Sidecar(sidecar_args)
        try:
            for workload in WORKLOADS:
                for profile in PROFILES:
                    args = _case_args(config, injector, workload, profile)
                    step_fn, workload_name = bench._build_victim(args)
                    for _ in range(config.victim_warmup_iters):
                        step_fn()

                    for pattern in PATTERNS:
                        stem = f"{injector}__{workload}__{profile}__{pattern}"
                        output = config.out_dir / f"{stem}.jsonl"
                        summary_path = output.with_suffix(".summary.json")
                        base_record = {
                            "case_index": case_index,
                            "inject_kind": injector,
                            "workload": workload,
                            "profile": profile,
                            "pattern": pattern,
                            "output_path": output.name,
                            "summary_path": summary_path.name,
                        }
                        case_index += 1
                        print(
                            f"CASE {case_index}/{total} {injector} {workload} "
                            f"{profile} {pattern}",
                            flush=True,
                        )
                        if summary_path.exists() and config.resume:
                            _append(manifest, {**base_record, "status": "existing"})
                            continue
                        case_started = time.monotonic()
                        bench._emit(
                            output,
                            {
                                "record": "run_meta",
                                "workload": workload_name,
                                "inject_kind": injector,
                                "profile": profile,
                                "pattern": pattern,
                                "burst_mean_ms": args.burst_mean_ms,
                                "device": config.device,
                                "doses": doses,
                                "window_s": config.window_s,
                                "repeats": 1,
                                "period_ms": args.period_ms,
                                "seed": config.seed,
                                "gemm_n": args.gemm_n,
                                "hidden": args.hidden,
                                "seq": args.seq,
                                "heads": args.heads,
                                "sidecar_ready": sidecar.ready,
                            },
                        )
                        order = list(doses)
                        random.Random(config.seed + case_index).shuffle(order)
                        for order_index, dose in enumerate(order):
                            sidecar_stats = None
                            window_seed = config.seed + case_index * 100 + order_index
                            if dose > 0:
                                sidecar.start(
                                    dose,
                                    pattern,
                                    args.burst_mean_ms,
                                    window_seed,
                                )
                            stats = bench._run_window(step_fn, config.window_s)
                            if dose > 0:
                                sidecar_stats = sidecar.stop()
                            bench._emit(
                                output,
                                {
                                    "record": "window",
                                    "repeat": 0,
                                    "order_index": order_index,
                                    "target_duty": dose,
                                    "workload": workload_name,
                                    "inject_kind": injector,
                                    "profile": profile,
                                    "pattern": pattern,
                                    "burst_mean_ms": args.burst_mean_ms,
                                    "window_seed": window_seed,
                                    **stats,
                                    "sidecar": sidecar_stats,
                                },
                            )
                            time.sleep(config.cooldown_s)
                        summary = bench._summarize(
                            output,
                            doses,
                            workload_name,
                            injector,
                            profile,
                            pattern,
                        )
                        bench._emit(output, summary)
                        summary_path.write_text(
                            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8",
                        )
                        _append(
                            manifest,
                            {
                                **base_record,
                                "status": "ok",
                                "returncode": 0,
                                "elapsed_s": time.monotonic() - case_started,
                            },
                        )

                    del step_fn
                    gc.collect()
                    torch.npu.empty_cache()
        finally:
            sidecar.close()

    done = {
        "record": "matrix_done",
        "mode": "screen",
        "cases": case_index,
        "failures": 0,
        "elapsed_s": time.monotonic() - started,
    }
    (config.out_dir / "MATRIX_DONE.json").write_text(
        json.dumps(done, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(done, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
