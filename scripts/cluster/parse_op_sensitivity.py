#!/usr/bin/env python3
"""解析 op_block_bench_npu JSONL：按 workload×factor×dose 汇总 slowdown，打印热图，选 Top-2 因素。

slowdown = inject_ms_p50 / baseline_ms_p50 - 1
baseline：同 workload、同 device（若有）、factor=none 或 dose=0 / inject_on=false

用法:
  python parse_op_sensitivity.py --jsonl ops/*.jsonl --out summary.json
  python parse_op_sensitivity.py --jsonl a.jsonl --jsonl b.jsonl --out summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _resolve(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        matches = sorted(glob.glob(pat, recursive=True))
        if not matches and Path(pat).is_file():
            matches = [pat]
        if not matches:
            raise FileNotFoundError(f"no files match: {pat}")
        for m in matches:
            key = str(Path(m).resolve())
            if key not in seen:
                seen.add(key)
                out.append(Path(m))
    return out


def _load_summaries(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("record") == "op_summary":
                    rows.append(rec)
                elif rec.get("record") == "op_iter":
                    continue
                elif "ms_p50" in rec and "workload" in rec:
                    # 兼容无 record 字段的摘要行
                    rows.append(rec)
    return rows


def _is_baseline(rec: dict) -> bool:
    if rec.get("inject_on") is False:
        return True
    factor = str(rec.get("factor") or "none").lower()
    dose = str(rec.get("dose") or "0").lower()
    if factor in ("none", "baseline", "0", ""):
        return True
    if dose in ("0", "none", "baseline", "off"):
        return True
    return False


def _key_wd(rec: dict) -> str:
    return str(rec.get("workload") or "?")


def _factor(rec: dict) -> str:
    return str(rec.get("factor") or rec.get("inject_kind") or "none")


def _dose(rec: dict) -> str:
    return str(rec.get("dose") or "0")


def summarize(rows: list[dict]) -> dict:
    # baseline p50 by (workload, device)
    base: dict[tuple, list[float]] = defaultdict(list)
    inj: dict[tuple, list[float]] = defaultdict(list)
    # key for inj: (workload, factor, dose, device)

    for r in rows:
        wl = _key_wd(r)
        dev = r.get("device")
        p50 = r.get("ms_p50")
        if p50 is None:
            continue
        p50 = float(p50)
        if _is_baseline(r):
            base[(wl, dev)].append(p50)
            base[(wl, None)].append(p50)  # device-agnostic fallback
        else:
            inj[(wl, _factor(r), _dose(r), dev)].append(p50)

    cells: list[dict[str, Any]] = []
    # group by workload × factor × dose (median across devices)
    grouped: dict[tuple, list[float]] = defaultdict(list)

    for (wl, factor, dose, dev), vals in inj.items():
        base_vals = base.get((wl, dev)) or base.get((wl, None)) or []
        if not base_vals or not vals:
            continue
        b = statistics.median(base_vals)
        m = statistics.median(vals)
        if b <= 0:
            continue
        slowdown = m / b - 1.0
        cell = {
            "workload": wl,
            "factor": factor,
            "dose": dose,
            "device": dev,
            "baseline_ms_p50": b,
            "inject_ms_p50": m,
            "slowdown": slowdown,
            "n_base": len(base_vals),
            "n_inj": len(vals),
        }
        cells.append(cell)
        grouped[(wl, factor, dose)].append(slowdown)

    matrix: list[dict] = []
    for (wl, factor, dose), sds in sorted(grouped.items()):
        matrix.append(
            {
                "workload": wl,
                "factor": factor,
                "dose": dose,
                "slowdown_mean": statistics.mean(sds),
                "slowdown_median": statistics.median(sds),
                "slowdown_abs_mean": statistics.mean([abs(x) for x in sds]),
                "n": len(sds),
            }
        )

    # Top-2 factors by mean |slowdown| across all workloads/doses
    by_factor: dict[str, list[float]] = defaultdict(list)
    for m in matrix:
        by_factor[m["factor"]].append(abs(m["slowdown_mean"]))
    factor_rank = sorted(
        (
            {
                "factor": f,
                "mean_abs_slowdown": statistics.mean(vs),
                "n_cells": len(vs),
            }
            for f, vs in by_factor.items()
        ),
        key=lambda x: -x["mean_abs_slowdown"],
    )
    top2 = factor_rank[:2]

    return {
        "n_summaries": len(rows),
        "n_cells": len(cells),
        "cells": cells,
        "matrix": matrix,
        "factor_rank": factor_rank,
        "top2_factors": top2,
    }


def _print_heatmap(result: dict) -> None:
    matrix = result["matrix"]
    if not matrix:
        print("(empty matrix — need baseline + inject op_summary rows)")
        return
    workloads = sorted({m["workload"] for m in matrix})
    factors = sorted({m["factor"] for m in matrix})
    doses = sorted({m["dose"] for m in matrix})

    # 每个 factor×dose 一列；行=workload
    cols: list[tuple[str, str]] = []
    for f in factors:
        for d in doses:
            if any(m["factor"] == f and m["dose"] == d for m in matrix):
                cols.append((f, d))

    header = f"{'workload':20s}" + "".join(f" | {f}:{d:>4s}" for f, d in cols)
    print(header)
    print("-" * len(header))
    lookup = {(m["workload"], m["factor"], m["dose"]): m for m in matrix}
    for wl in workloads:
        parts = [f"{wl:20s}"]
        for f, d in cols:
            m = lookup.get((wl, f, d))
            if m is None:
                parts.append(f" | {'n/a':>8s}")
            else:
                parts.append(f" | {m['slowdown_mean']*100:+7.1f}%")
        print("".join(parts))

    print()
    print("Top-2 factors by mean |slowdown|:")
    for i, t in enumerate(result["top2_factors"], 1):
        print(f"  {i}. {t['factor']}: mean|slowdown|={t['mean_abs_slowdown']*100:.2f}% (n={t['n_cells']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", action="append", default=[], help="op bench jsonl 或 glob（可多次）")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.jsonl:
        ap.error("至少提供一个 --jsonl")

    paths = _resolve(args.jsonl)
    rows = _load_summaries(paths)
    result = summarize(rows)
    result["sources"] = [str(p) for p in paths]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_heatmap(result)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
