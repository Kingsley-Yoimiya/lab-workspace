#!/usr/bin/env python3
"""B1：按 PP stage 分组对比 baseline vs inject 的 step ms。"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
    return ys[k]


def load(dir: Path) -> dict[int, dict]:
    by: dict[int, dict] = {}
    for p in sorted(dir.glob("step_times_rank*.jsonl")):
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        r = int(rows[0]["global_rank"])
        steady = [x for x in rows if int(x["iter"]) > 100]
        ms = [float(x["ms"]) for x in steady]
        delayed_ms = [float(x["ms"]) for x in steady if x.get("delayed")]
        by[r] = {
            "ms": ms,
            "delayed_ms": delayed_ms,
            "n_delayed": len(delayed_ms),
            "n": len(ms),
        }
    return by


def stage(rank: int, pp: int = 4, world: int = 8) -> int:
    return rank // max(1, world // pp)


def summarize(by: dict[int, dict], pp: int = 4) -> dict:
    per_stage_med: dict[int, list[float]] = defaultdict(list)
    per_stage_p95: dict[int, list[float]] = defaultdict(list)
    delayed_frac = []
    for r, rec in by.items():
        ms = rec["ms"]
        if not ms:
            continue
        st = stage(r, pp)
        per_stage_med[st].append(statistics.median(ms))
        per_stage_p95[st].append(_pct(ms, 95))
        delayed_frac.append(rec["n_delayed"] / max(1, rec["n"]))
    stage_med = {str(k): statistics.median(v) for k, v in sorted(per_stage_med.items())}
    stage_p95 = {str(k): statistics.median(v) for k, v in sorted(per_stage_p95.items())}
    return {
        "global_median_ms": statistics.median([statistics.median(v["ms"]) for v in by.values() if v["ms"]]),
        "global_p95_ms": statistics.median([_pct(v["ms"], 95) for v in by.values() if v["ms"]]),
        "stage_median_ms": stage_med,
        "stage_p95_ms": stage_p95,
        "stage_spread_ms": (
            max(stage_med.values()) - min(stage_med.values()) if stage_med else None
        ),
        "stage_p95_spread_ms": (
            max(stage_p95.values()) - min(stage_p95.values()) if stage_p95 else None
        ),
        "mean_delayed_frac": statistics.mean(delayed_frac) if delayed_frac else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--inject", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base = summarize(load(args.baseline))
    inj = summarize(load(args.inject))
    # 判据：全局中位变化弱，但注入 stage 的 p95 明显抬升（间歇 sleep 被中位数稀释）
    base_spread = base.get("stage_p95_spread_ms") or base["stage_spread_ms"] or 0.0
    inj_spread = inj.get("stage_p95_spread_ms") or inj["stage_spread_ms"] or 0.0
    global_delta = abs(inj["global_median_ms"] - base["global_median_ms"])
    rec = {
        "baseline": base,
        "inject": inj,
        "global_delta_ms": inj["global_median_ms"] - base["global_median_ms"],
        "stage_spread_baseline": base["stage_spread_ms"],
        "stage_spread_inject": inj["stage_spread_ms"],
        "stage_p95_spread_baseline": base.get("stage_p95_spread_ms"),
        "stage_p95_spread_inject": inj.get("stage_p95_spread_ms"),
        "masking_ok": (
            inj_spread > max(base_spread * 1.5, 20.0)
            and global_delta < max(inj_spread * 0.5, 15.0)
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(rec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
