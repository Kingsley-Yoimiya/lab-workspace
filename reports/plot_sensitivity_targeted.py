#!/usr/bin/env python3
"""绘制重点敏感组合的五次重复剂量曲线。

Example:
  python3 reports/plot_sensitivity_targeted.py \
    --result-dir /path/to/targeted-r5 \
    --out-dir /path/to/figs
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports"))
from plot_style import COLORS, apply_plot_style, save_fig, style_axes  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _quantile(values: list[float], q: float) -> float:
    xs = sorted(values)
    index = (len(xs) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (index - lo)


def main() -> int:
    args = _parse_args()
    manifest = [
        json.loads(line)
        for line in (args.result_dir / "matrix_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    apply_plot_style((17, 13.5))
    fig, axes = plt.subplots(3, 4)
    for index, (axis, record) in enumerate(zip(axes.flat, manifest)):
        raw_path = args.result_dir / record["output_path"]
        rows = [
            json.loads(line)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        windows = [row for row in rows if row.get("record") == "window"]
        baseline = st.median(
            row["iters_per_s"] for row in windows if row["target_duty"] == 0
        )
        xs, ys, low, high = [], [], [], []
        for duty in sorted({row["target_duty"] for row in windows}):
            selected = [row for row in windows if row["target_duty"] == duty]
            drops = [(1.0 - row["iters_per_s"] / baseline) * 100.0 for row in selected]
            actual = [
                0.0 if row["sidecar"] is None else row["sidecar"]["busy_wall_ratio"] * 100.0
                for row in selected
            ]
            median = st.median(drops)
            xs.append(st.median(actual))
            ys.append(median)
            low.append(median - _quantile(drops, 0.1))
            high.append(_quantile(drops, 0.9) - median)
        axis.errorbar(
            xs,
            ys,
            yerr=[low, high],
            color=COLORS[index % len(COLORS)],
            marker="o",
            linewidth=2,
            capsize=3,
        )
        axis.set_title(
            f"{record['inject_kind']} → {record['workload']}\n"
            f"{record['profile']} / {record['pattern']}",
            fontsize=15,
        )
        axis.set_xlabel("实测 sidecar busy (%)", fontsize=13)
        axis.set_ylabel("吞吐下降 (%)", fontsize=13)
        axis.tick_params(labelsize=12)
        style_axes(axis)
    fig.suptitle("重点组合的剂量—响应与五次重复区间", fontsize=22, y=0.975)
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.075,
        top=0.89,
        wspace=0.30,
        hspace=0.62,
    )
    output = save_fig(fig, args.out_dir / "sensitivity_targeted_curves.svg")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
