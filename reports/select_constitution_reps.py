#!/usr/bin/env python3
"""从多次 constitution JSONL 跨 run 选慢/中/快代表卡。

主键指标：
  - card: func_tflops, hbm_gbps, vector_gflops, mte_gbps
  - sustained_steady: 从 gemm_sustained_sample 按 device 取后半段 window 的 tflops median
    （禁止用 card.sustained_tflops 作主键；那是末窗）

跨 run 每卡 median + CV；CV 过高标 unstable；输出 reps.json + stdout 表。

用法:
  python select_constitution_reps.py \\
    --jsonl 'results/npu-dev-1/*/constitution*.jsonl' \\
    --out reps.json

  python select_constitution_reps.py \\
    --jsonl run1.jsonl --jsonl run2.jsonl --cv-thresh 0.03 --out reps.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


PRIMARY = (
    "func_tflops",
    "hbm_gbps",
    "sustained_steady",
    "vector_gflops",
    "mte_gbps",
)

# Cube / HBM 权重更高；其余等权较低
WEIGHTS = {
    "func_tflops": 1.5,
    "hbm_gbps": 1.2,
    "sustained_steady": 1.3,
    "vector_gflops": 0.8,
    "mte_gbps": 0.8,
}


def _median(xs: list[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def _mean(xs: list[float]) -> Optional[float]:
    return statistics.mean(xs) if xs else None


def _stdev(xs: list[float]) -> Optional[float]:
    if len(xs) < 2:
        return 0.0 if xs else None
    return statistics.stdev(xs)


def _cv(xs: list[float]) -> Optional[float]:
    m = _mean(xs)
    s = _stdev(xs)
    if m is None or s is None or abs(m) < 1e-12:
        return None
    return s / abs(m)


def _resolve_paths(patterns: list[str]) -> list[Path]:
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


def _device_key(rec: dict) -> tuple:
    host = rec.get("host") or ""
    device = rec.get("device")
    if device is None:
        device = rec.get("card_id")
    return (str(host), int(device) if device is not None else -1)


def _load_run(path: Path) -> tuple[dict, dict]:
    """Return (cards_by_device, sustained_samples_by_device)."""
    cards: dict[tuple, dict] = {}
    sus: dict[tuple, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = rec.get("record")
            key = _device_key(rec)
            if kind == "card":
                cards[key] = rec
            elif kind == "gemm_sustained_sample":
                sus[key].append(rec)
    return cards, sus


def _sustained_steady(samples: list[dict]) -> Optional[float]:
    if not samples:
        return None
    ordered = sorted(
        samples,
        key=lambda s: (float(s.get("t_s") or 0.0), int(s.get("iter") or 0)),
    )
    half = len(ordered) // 2
    tail = ordered[half:] if half < len(ordered) else ordered
    vals = [float(s["tflops"]) for s in tail if s.get("tflops") is not None]
    return _median(vals)


def select_reps(
    run_paths: list[Path],
    cv_thresh: float = 0.03,
    cv_ratio: float = 2.0,
) -> dict:
    # run -> device -> metric -> value
    per_run: list[dict[tuple, dict[str, Optional[float]]]] = []
    run_ids: list[str] = []

    for path in run_paths:
        cards, sus = _load_run(path)
        run_map: dict[tuple, dict[str, Optional[float]]] = {}
        devices = set(cards) | set(sus)
        for key in devices:
            card = cards.get(key, {})
            row: dict[str, Optional[float]] = {
                "func_tflops": card.get("func_tflops"),
                "hbm_gbps": card.get("hbm_gbps"),
                "vector_gflops": card.get("vector_gflops"),
                "mte_gbps": card.get("mte_gbps"),
                # 显式不用 card.sustained_tflops
                "sustained_steady": _sustained_steady(sus.get(key, [])),
                "card_sustained_tflops_last_window": card.get("sustained_tflops"),
            }
            # coerce numerics
            for m in list(row):
                v = row[m]
                if v is not None:
                    try:
                        row[m] = float(v)
                    except (TypeError, ValueError):
                        row[m] = None
            run_map[key] = row
        per_run.append(run_map)
        run_ids.append(path.name)

    all_devices = sorted({d for rm in per_run for d in rm})
    cards_out: list[dict] = []

    # aggregate across runs
    med_by_metric: dict[str, dict[tuple, float]] = {m: {} for m in PRIMARY}

    for key in all_devices:
        metrics_series: dict[str, list[float]] = {m: [] for m in PRIMARY}
        last_window_series: list[float] = []
        for rm in per_run:
            row = rm.get(key)
            if not row:
                continue
            for m in PRIMARY:
                v = row.get(m)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    metrics_series[m].append(float(v))
            lw = row.get("card_sustained_tflops_last_window")
            if lw is not None:
                last_window_series.append(float(lw))

        entry: dict[str, Any] = {
            "host": key[0],
            "device": key[1],
            "n_runs": max(len(v) for v in metrics_series.values()) if metrics_series else 0,
            "metrics": {},
            "unstable": False,
            "unstable_reasons": [],
        }
        for m in PRIMARY:
            xs = metrics_series[m]
            med = _median(xs)
            cv = _cv(xs)
            entry["metrics"][m] = {
                "median": med,
                "mean": _mean(xs),
                "std": _stdev(xs),
                "cv": cv,
                "n": len(xs),
                "values": xs,
            }
            if med is not None:
                med_by_metric[m][key] = med
        entry["card_sustained_tflops_last_window_median"] = _median(last_window_series)
        cards_out.append(entry)

    # cluster medians & relative
    cluster_med = {
        m: _median(list(med_by_metric[m].values())) for m in PRIMARY
    }

    # score: higher throughput → higher score (fast)
    score: dict[tuple, float] = {}
    for entry in cards_out:
        key = (entry["host"], entry["device"])
        zs = []
        wsum = 0.0
        for m in PRIMARY:
            med = entry["metrics"][m]["median"]
            cmed = cluster_med.get(m)
            if med is None or cmed is None or abs(cmed) < 1e-12:
                continue
            rel = (med - cmed) / cmed
            # store rel for table
            entry["metrics"][m]["rel"] = rel
            w = WEIGHTS.get(m, 1.0)
            zs.append(rel * w)
            wsum += w
        score[key] = (sum(zs) / wsum) if zs and wsum > 0 else 0.0
        entry["score"] = score[key]

    # unstable: func_tflops CV high vs peer median CV, or above absolute thresh
    func_cvs = [e["metrics"]["func_tflops"]["cv"] for e in cards_out
                if e["metrics"]["func_tflops"]["cv"] is not None]
    peer_cv_med = _median(func_cvs) or 0.0
    for entry in cards_out:
        cv = entry["metrics"]["func_tflops"]["cv"]
        reasons = []
        if cv is not None and cv > cv_thresh:
            reasons.append(f"func_tflops_cv={cv:.4f}>{cv_thresh}")
        if cv is not None and peer_cv_med > 0 and cv > cv_ratio * peer_cv_med:
            reasons.append(f"func_tflops_cv={cv:.4f}>{cv_ratio}x_peer_med({peer_cv_med:.4f})")
        # also flag if any primary CV is extreme
        for m in PRIMARY:
            mcv = entry["metrics"][m]["cv"]
            if mcv is not None and mcv > max(cv_thresh * 2, 0.08):
                reasons.append(f"{m}_cv={mcv:.4f}")
        entry["unstable"] = bool(reasons)
        entry["unstable_reasons"] = reasons

    stable = [e for e in cards_out if not e["unstable"]]
    pool = stable if len(stable) >= 3 else cards_out
    by_score = sorted(pool, key=lambda e: e["score"])
    slow = by_score[0] if by_score else None
    fast = by_score[-1] if by_score else None
    if by_score:
        target = statistics.median([e["score"] for e in by_score])
        mid = min(by_score, key=lambda e: abs(e["score"] - target))
    else:
        mid = None

    def _rep(e: Optional[dict]) -> Optional[dict]:
        if e is None:
            return None
        return {
            "host": e["host"],
            "device": e["device"],
            "score": e["score"],
            "unstable": e["unstable"],
            "metrics": {m: e["metrics"][m]["median"] for m in PRIMARY},
        }

    result = {
        "n_runs": len(run_paths),
        "run_files": [str(p) for p in run_paths],
        "run_ids": run_ids,
        "primary_metrics": list(PRIMARY),
        "note": (
            "sustained_steady = median(tflops) over second-half gemm_sustained_sample windows; "
            "card.sustained_tflops (last window) is NOT used as primary key"
        ),
        "cluster_median": cluster_med,
        "cv_thresh": cv_thresh,
        "cv_ratio": cv_ratio,
        "reps": {
            "slow": _rep(slow),
            "mid": _rep(mid),
            "fast": _rep(fast),
        },
        "cards": cards_out,
        "unstable_devices": [
            {"host": e["host"], "device": e["device"], "reasons": e["unstable_reasons"]}
            for e in cards_out
            if e["unstable"]
        ],
    }
    return result


def _print_table(result: dict) -> None:
    headers = [
        "host", "dev", "score", "flag",
        "func", "hbm", "sus_med", "vec", "mte",
        "cv_func",
    ]
    print(" | ".join(f"{h:>10}" for h in headers))
    print("-+-".join("-" * 10 for _ in headers))
    for e in sorted(result["cards"], key=lambda x: x["score"]):
        flag = "unstable" if e["unstable"] else ""
        mets = e["metrics"]

        def fmt(m: str, key: str = "median") -> str:
            v = mets[m].get(key)
            return f"{v:.2f}" if isinstance(v, (int, float)) else "  n/a"

        row = [
            (e["host"] or "")[-10:],
            str(e["device"]),
            f"{e['score']:+.3f}",
            flag,
            fmt("func_tflops"),
            fmt("hbm_gbps"),
            fmt("sustained_steady"),
            fmt("vector_gflops"),
            fmt("mte_gbps"),
            (
                f"{mets['func_tflops']['cv']:.4f}"
                if mets["func_tflops"].get("cv") is not None
                else "  n/a"
            ),
        ]
        print(" | ".join(f"{c:>10}" for c in row))

    print()
    reps = result["reps"]
    for kind in ("slow", "mid", "fast"):
        r = reps.get(kind)
        if r:
            print(
                f"REP {kind:5s}: device={r['device']} host={r['host']} "
                f"score={r['score']:+.4f} metrics={r['metrics']}"
            )
        else:
            print(f"REP {kind:5s}: (none)")
    if result["unstable_devices"]:
        print("unstable:", result["unstable_devices"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--jsonl",
        action="append",
        default=[],
        help="constitution JSONL path or glob（可多次）",
    )
    ap.add_argument("--out", type=Path, required=True, help="输出 reps.json")
    ap.add_argument("--cv-thresh", type=float, default=0.03, help="func_tflops CV 绝对门限")
    ap.add_argument("--cv-ratio", type=float, default=2.0, help="相对同伴 median CV 倍数门限")
    args = ap.parse_args()
    if not args.jsonl:
        ap.error("至少提供一个 --jsonl")

    paths = _resolve_paths(args.jsonl)
    result = select_reps(paths, cv_thresh=args.cv_thresh, cv_ratio=args.cv_ratio)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_table(result)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
