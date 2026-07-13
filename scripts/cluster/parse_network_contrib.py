#!/usr/bin/env python3
"""Block A：gap_real / gap_indep / network_contrib vs N。"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def gap_from_step_jsonl(scale_dir: Path, drop_first: int = 10) -> dict | None:
    by_iter: dict[int, list[float]] = defaultdict(list)
    n_ranks = 0
    for p in sorted(scale_dir.glob("step_times_rank*.jsonl")):
        n_ranks += 1
        for line in p.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            it = int(rec.get("iter", 0))
            if it <= drop_first:
                continue
            by_iter[it].append(float(rec["ms"]))
    gaps = []
    meds = []
    for it, vals in sorted(by_iter.items()):
        if len(vals) < 2:
            continue
        gaps.append(max(vals) - statistics.median(vals))
        meds.append(statistics.median(vals))
    if not gaps:
        return None
    return {
        "gap_median_ms": statistics.median(gaps),
        "gap_mean_ms": statistics.fmean(gaps),
        "median_step_ms": statistics.median(meds),
        "n_iters": len(gaps),
        "n_ranks": n_ranks,
    }


def gap_from_csv_row(row: dict) -> dict | None:
    if not row.get("gap_median_ms"):
        return None
    return {
        "gap_median_ms": float(row["gap_median_ms"]),
        "median_step_ms": float(row["median_step_ms"]) if row.get("median_step_ms") else None,
        "n_iters": int(float(row["n_iters"])) if row.get("n_iters") else None,
        "n_ranks": int(float(row["n_ranks"])) if row.get("n_ranks") else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indep-root", type=Path, required=True)
    ap.add_argument("--real-csv", type=Path, action="append", default=[])
    ap.add_argument("--real-scale-root", type=Path, action="append", default=[])
    ap.add_argument("--drop-first", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    indep: dict[int, dict] = {}
    for d in sorted(args.indep_root.glob("scale_*")):
        try:
            n = int(d.name.split("_", 1)[1])
        except ValueError:
            continue
        g = gap_from_step_jsonl(d, args.drop_first)
        if g:
            indep[n] = g

    real: dict[int, dict] = {}
    for csvp in args.real_csv:
        if not csvp.exists():
            continue
        for row in csv.DictReader(csvp.open(encoding="utf-8")):
            if not row.get("world_npu") or not row.get("gap_median_ms"):
                continue
            n = int(float(row["world_npu"]))
            g = gap_from_csv_row(row)
            if g:
                real[n] = g
    for root in args.real_scale_root:
        for d in sorted(root.glob("scale_*")):
            try:
                n = int(d.name.split("_", 1)[1])
            except ValueError:
                continue
            g = gap_from_step_jsonl(d, args.drop_first)
            if g:
                real[n] = g

    ns = sorted(set(indep) | set(real))
    rows = []
    for n in ns:
        r = real.get(n) or {}
        i = indep.get(n) or {}
        gr = r.get("gap_median_ms")
        gi = i.get("gap_median_ms")
        net = (gr - gi) if (gr is not None and gi is not None) else None
        rows.append(
            {
                "world_npu": n,
                "gap_real_ms": gr,
                "gap_indep_ms": gi,
                "network_contrib_ms": net,
                "median_step_real_ms": r.get("median_step_ms"),
                "median_step_indep_ms": i.get("median_step_ms"),
                "cross_node": n > 16,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["world_npu"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"CSV → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
