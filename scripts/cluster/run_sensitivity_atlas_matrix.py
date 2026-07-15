#!/usr/bin/env python3
"""编排资源干扰 × victim × profile × 时间模式敏感度矩阵。

Examples:
  python3 run_sensitivity_atlas_matrix.py --mode smoke --out-dir results/atlas-smoke
  python3 run_sensitivity_atlas_matrix.py --mode screen --out-dir results/atlas-screen
  python3 run_sensitivity_atlas_matrix.py --mode screen --out-dir results/atlas --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


INJECTORS = ("cube", "vector", "hbm_mte", "hbm_vector", "small_ops")
WORKLOADS = ("gemm", "attention", "norm", "elementwise", "block", "transformer")
PROFILES = ("small", "large")
PATTERNS = ("periodic", "poisson")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=("smoke", "screen", "targeted"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _cases(mode: str) -> list[tuple[str, str, str, str]]:
    if mode == "smoke":
        cases = [(injector, "gemm", "small", "periodic") for injector in INJECTORS]
        cases.extend(
            ("cube", workload, "small", "periodic")
            for workload in WORKLOADS
            if workload != "gemm"
        )
        cases.extend(
            [
                ("cube", "attention", "large", "poisson"),
                ("hbm_mte", "transformer", "large", "poisson"),
            ]
        )
        return cases
    if mode == "targeted":
        return [
            ("hbm_vector", "norm", "small", "periodic"),
            ("hbm_vector", "norm", "small", "poisson"),
            ("hbm_mte", "norm", "small", "periodic"),
            ("hbm_mte", "norm", "small", "poisson"),
            ("hbm_mte", "elementwise", "small", "periodic"),
            ("hbm_mte", "elementwise", "small", "poisson"),
            ("cube", "gemm", "small", "periodic"),
            ("cube", "gemm", "large", "periodic"),
            ("hbm_mte", "block", "small", "periodic"),
            ("hbm_mte", "block", "small", "poisson"),
            ("cube", "elementwise", "small", "periodic"),
            ("vector", "attention", "small", "periodic"),
        ]
    return [
        (injector, workload, profile, pattern)
        for injector in INJECTORS
        for workload in WORKLOADS
        for profile in PROFILES
        for pattern in PATTERNS
    ]


def _append_manifest(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = _parse_args()
    if args.out_dir.exists() and not args.resume and any(args.out_dir.iterdir()):
        raise SystemExit(
            f"error: non-empty output directory already exists: {args.out_dir}; "
            "use a new path or --resume"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "matrix_manifest.jsonl"
    bench = Path(__file__).resolve().parent / "controlled_interference_bench_npu.py"
    cases = _cases(args.mode)
    doses = "0,0.5" if args.mode == "smoke" else "0,0.1,0.3,0.5"
    window_s = "1" if args.mode == "smoke" else "2"
    sidecar_warmup = "2" if args.mode == "smoke" else "3"
    victim_warmup = "2" if args.mode == "smoke" else "3"
    repeats = "5" if args.mode == "targeted" else "1"
    failures = 0
    started = time.monotonic()

    for index, (injector, workload, profile, pattern) in enumerate(cases):
        stem = f"{injector}__{workload}__{profile}__{pattern}"
        output = args.out_dir / f"{stem}.jsonl"
        summary = output.with_suffix(".summary.json")
        base_record = {
            "case_index": index,
            "inject_kind": injector,
            "workload": workload,
            "profile": profile,
            "pattern": pattern,
            "output_path": output.name,
            "summary_path": summary.name,
        }
        if summary.exists() and args.resume:
            _append_manifest(manifest, {**base_record, "status": "existing"})
            continue

        command = [
            sys.executable,
            str(bench),
            "--device",
            str(args.device),
            "--inject-kind",
            injector,
            "--workload",
            workload,
            "--profile",
            profile,
            "--pattern",
            pattern,
            "--doses",
            doses,
            "--window-s",
            window_s,
            "--repeats",
            repeats,
            "--sidecar-warmup-iters",
            sidecar_warmup,
            "--victim-warmup-iters",
            victim_warmup,
            "--cooldown-s",
            "0.2",
            "--out",
            str(output),
        ]
        print(
            f"CASE {index + 1}/{len(cases)} {injector} {workload} "
            f"{profile} {pattern}",
            flush=True,
        )
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        case_started = time.monotonic()
        completed = subprocess.run(command, check=False)
        record = {
            **base_record,
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "elapsed_s": time.monotonic() - case_started,
        }
        _append_manifest(manifest, record)
        if completed.returncode != 0:
            failures += 1

    summary_record = {
        "record": "matrix_done",
        "mode": args.mode,
        "cases": len(cases),
        "failures": failures,
        "elapsed_s": time.monotonic() - started,
    }
    (args.out_dir / "MATRIX_DONE.json").write_text(
        json.dumps(summary_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_record, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
