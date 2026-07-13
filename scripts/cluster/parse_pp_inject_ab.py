#!/usr/bin/env python3
"""实验三：baseline vs PP-stage 注入，按 stage 对比 step ms。"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_steps(scale_dir: Path, drop_first: int = 5) -> dict[int, list[dict]]:
    by: dict[int, list[dict]] = defaultdict(list)
    for p in sorted(scale_dir.glob("step_times_rank*.jsonl")):
        for line in p.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if int(rec.get("iter", 0)) <= drop_first:
                continue
            by[int(rec["rank"])].append(rec)
    return by


def stage_of(rank: int, world: int, pp: int) -> int:
    return rank // max(1, world // pp)


def summarize(by: dict[int, list[dict]], world: int, pp: int) -> dict:
    stage_med: dict[int, list[float]] = defaultdict(list)
    all_med = []
    delayed_n = 0
    total_n = 0
    for rank, rows in by.items():
        if not rows:
            continue
        ms = [float(r["ms"]) for r in rows]
        med = statistics.median(ms)
        all_med.append(med)
        st = int(rows[0].get("pp_stage", stage_of(rank, world, pp)))
        stage_med[st].append(med)
        delayed_n += sum(1 for r in rows if r.get("delayed"))
        total_n += len(rows)
    sm = {str(k): statistics.median(v) for k, v in sorted(stage_med.items())}
    return {
        "global_median_ms": statistics.median(all_med) if all_med else None,
        "stage_median_ms": sm,
        "stage_spread_ms": (max(sm.values()) - min(sm.values())) if len(sm) >= 2 else None,
        "delayed_frac": delayed_n / max(1, total_n),
        "n_ranks": len(by),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--inject", type=Path, required=True)
    ap.add_argument("--pp", type=int, default=2)
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--drop-first", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base = summarize(load_steps(args.baseline, args.drop_first), args.world, args.pp)
    inj = summarize(load_steps(args.inject, args.drop_first), args.world, args.pp)
    out = {
        "baseline": base,
        "inject": inj,
        "delta_global_median_ms": None,
        "delta_stage_spread_ms": None,
        "verdict": "",
    }
    if base["global_median_ms"] and inj["global_median_ms"]:
        out["delta_global_median_ms"] = inj["global_median_ms"] - base["global_median_ms"]
    if base.get("stage_spread_ms") is not None and inj.get("stage_spread_ms") is not None:
        out["delta_stage_spread_ms"] = inj["stage_spread_ms"] - base["stage_spread_ms"]
    # 判据：全局中位变化弱，stage spread 明显放大
    dg = out["delta_global_median_ms"] or 0
    ds = out["delta_stage_spread_ms"] or 0
    if ds > max(50.0, abs(dg) * 2):
        out["verdict"] = "PASS: stage slice sees inject; global median weak"
    elif ds > 0:
        out["verdict"] = "WEAK: stage spread rose but modest"
    else:
        out["verdict"] = "FAIL: no clear stage contrast"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
