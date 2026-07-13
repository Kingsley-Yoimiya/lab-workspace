#!/usr/bin/env python3
"""解析 Dense FailSlow 每卡 step_times JSONL，汇总 gap vs N。

主指标：稳态窗内各 iter 的 (max_rank_ms - median_rank_ms) 再对 iter 取 median；
旁证：吞吐/MFU（可选读 rank0 训练日志）。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# 复用吞吐解析
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
try:
    from parse_train_mfu_log import parse_text, steady as steady_tflop
except Exception:
    parse_text = None  # type: ignore
    steady_tflop = None  # type: ignore


def _pct(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def load_step_times(scale_dir: Path) -> dict[int, list[dict]]:
    """rank -> list of {iter, ms}"""
    by_rank: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(scale_dir.glob("step_times_rank*.jsonl")):
        for line in path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rank = int(rec.get("rank", -1))
            if rank < 0 or "ms" not in rec:
                continue
            by_rank[rank].append({"iter": int(rec.get("iter", 0)), "ms": float(rec["ms"])})
    for rank in by_rank:
        by_rank[rank].sort(key=lambda x: x["iter"])
    return by_rank


def gap_stats(by_rank: dict[int, list[dict]], drop_first: int = 20) -> dict | None:
    if not by_rank:
        return None
    # iter -> list of ms across ranks
    by_iter: dict[int, list[float]] = defaultdict(list)
    for rows in by_rank.values():
        for r in rows:
            by_iter[int(r["iter"])].append(float(r["ms"]))
    iters_sorted = sorted(by_iter)
    if drop_first:
        iters_sorted = [i for i in iters_sorted if i > drop_first]
    gaps: list[float] = []
    medians: list[float] = []
    maxes: list[float] = []
    p99_p50: list[float] = []
    for it in iters_sorted:
        vals = sorted(by_iter[it])
        if len(vals) < 2:
            continue
        med = statistics.median(vals)
        mx = max(vals)
        gaps.append(mx - med)
        medians.append(med)
        maxes.append(mx)
        p50 = _pct(vals, 50) or med
        p99 = _pct(vals, 99) or mx
        if p50 > 0:
            p99_p50.append(p99 / p50)
    if not gaps:
        return None
    gaps_s = sorted(gaps)
    return {
        "n_iters": len(gaps),
        "n_ranks": len(by_rank),
        "gap_median_ms": statistics.median(gaps),
        "gap_mean_ms": statistics.fmean(gaps),
        "gap_p90_ms": _pct(gaps_s, 90),
        "gap_p99_ms": _pct(gaps_s, 99),
        "median_step_ms": statistics.median(medians),
        "max_step_median_ms": statistics.median(maxes),
        "p99_over_p50_median": statistics.median(p99_p50) if p99_p50 else None,
    }


def parse_scale_dir(scale_dir: Path, drop_first: int, peak: float) -> dict:
    world = None
    name = scale_dir.name
    if name.startswith("scale_"):
        try:
            world = int(name.split("_", 1)[1])
        except ValueError:
            world = None
    by_rank = load_step_times(scale_dir)
    gap = gap_stats(by_rank, drop_first=drop_first)
    out: dict = {
        "scale_dir": str(scale_dir),
        "world_npu": world,
        "step_files": len(list(scale_dir.glob("step_times_rank*.jsonl"))),
        "gap": gap,
        "mfu": None,
        "tflops_median": None,
    }
    if parse_text and steady_tflop:
        texts = []
        for f in sorted(scale_dir.glob("rank*.log")) + sorted(scale_dir.glob("train_*.log")):
            texts.append(f.read_text(errors="ignore"))
        rows = parse_text("\n".join(texts))
        st = steady_tflop(rows, drop_first=min(drop_first, 1) if drop_first else 1)
        if st:
            out["tflops_median"] = st["tflops_median"]
            out["mfu"] = (st["tflops_median"] / peak) if peak > 0 else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="dense_failslow/<stamp> 根目录或单个 scale_*")
    ap.add_argument("--drop-first", type=int, default=20)
    ap.add_argument("--peak", type=float, default=292.79)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    root: Path = args.root
    if (root / "scale_16").exists() or any(root.glob("scale_*")):
        scale_dirs = sorted(root.glob("scale_*"), key=lambda p: int(p.name.split("_")[1]) if "_" in p.name else 0)
    else:
        scale_dirs = [root]

    rows = [parse_scale_dir(d, args.drop_first, args.peak) for d in scale_dirs]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print("world\tgap_med_ms\tmedian_step_ms\tp99/p50\ttflops_med\tmfu\tstep_files")
        for r in rows:
            g = r.get("gap") or {}
            print(
                f"{r.get('world_npu')}\t{g.get('gap_median_ms')}\t{g.get('median_step_ms')}\t"
                f"{g.get('p99_over_p50_median')}\t{r.get('tflops_median')}\t{r.get('mfu')}\t{r.get('step_files')}"
            )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "world_npu",
                    "gap_median_ms",
                    "gap_mean_ms",
                    "gap_p90_ms",
                    "median_step_ms",
                    "p99_over_p50_median",
                    "tflops_median",
                    "mfu",
                    "n_iters",
                    "n_ranks",
                    "step_files",
                ],
            )
            w.writeheader()
            for r in rows:
                g = r.get("gap") or {}
                w.writerow(
                    {
                        "world_npu": r.get("world_npu"),
                        "gap_median_ms": g.get("gap_median_ms"),
                        "gap_mean_ms": g.get("gap_mean_ms"),
                        "gap_p90_ms": g.get("gap_p90_ms"),
                        "median_step_ms": g.get("median_step_ms"),
                        "p99_over_p50_median": g.get("p99_over_p50_median"),
                        "tflops_median": r.get("tflops_median"),
                        "mfu": r.get("mfu"),
                        "n_iters": g.get("n_iters"),
                        "n_ranks": g.get("n_ranks"),
                        "step_files": r.get("step_files"),
                    }
                )
        print(f"CSV → {args.csv}", file=sys.stderr)
    return 0 if any((r.get("gap") for r in rows)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
