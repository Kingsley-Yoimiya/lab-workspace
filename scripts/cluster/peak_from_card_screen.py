#!/usr/bin/env python3
"""从 CARD_SCREEN cluster.json 读取 median func_tflops，作为训练 MFU 分母峰值。

默认路径：
  logs/card-screen-128-20260710_224218/results/perf128.cluster.json
  （也接受同目录下的 cluster.json）

用法：
  python3 scripts/cluster/peak_from_card_screen.py
  python3 scripts/cluster/peak_from_card_screen.py --results-dir /path/to/results
  python3 scripts/cluster/peak_from_card_screen.py --json /path/to/xxx.cluster.json --world-size 16
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


# .../project/lab-workspace/scripts/cluster/this.py → parents[4]=random-thing
DEFAULT_RESULTS = (
    Path(__file__).resolve().parents[4]
    / "logs"
    / "card-screen-128-20260710_224218"
    / "results"
)


def find_cluster_json(results_dir: Path) -> Path:
    candidates = [
        results_dir / "perf128.cluster.json",
        results_dir / "cluster.json",
    ]
    candidates.extend(sorted(results_dir.glob("*.cluster.json")))
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"no *.cluster.json under {results_dir}")


def load_func_tflops(path: Path) -> tuple[float, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta: dict = {"source": str(path)}
    if isinstance(data, dict) and "medians" in data and "func_tflops" in data["medians"]:
        peak = float(data["medians"]["func_tflops"])
        meta["from"] = "medians.func_tflops"
        meta["n_cards"] = data.get("n_cards")
        return peak, meta
    # fallback: cards[]
    cards = data.get("cards") if isinstance(data, dict) else data
    vals = []
    for c in cards or []:
        if isinstance(c, dict) and c.get("func_tflops") is not None:
            vals.append(float(c["func_tflops"]))
    if not vals:
        raise ValueError(f"no func_tflops in {path}")
    peak = float(statistics.median(vals))
    meta["from"] = "median(cards.func_tflops)"
    meta["n_cards"] = len(vals)
    return peak, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Peak TFLOPS from card-screen cluster.json")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--json", type=Path, default=None, help="直接指定 cluster.json")
    ap.add_argument("--world-size", type=int, default=None, help="若给出则打印 peak×world")
    ap.add_argument("--field", default="func_tflops", help="备用字段名（仅 cards 回退时）")
    args = ap.parse_args()

    path = args.json if args.json else find_cluster_json(args.results_dir)
    peak, meta = load_func_tflops(path)
    print(f"peak_tflops_per_card={peak:.6f}")
    print(f"source={meta['source']}")
    print(f"from={meta['from']}")
    if meta.get("n_cards") is not None:
        print(f"n_cards={meta['n_cards']}")
    if args.world_size:
        total = peak * args.world_size
        print(f"world_size={args.world_size}")
        print(f"peak_tflops_total={total:.6f}")
        print(f"# MFU = achieved_tflops / {total:.6f}")
    else:
        print("# MFU denom per card; total = peak_tflops_per_card * world_size")
    return 0


if __name__ == "__main__":
    sys.exit(main())
