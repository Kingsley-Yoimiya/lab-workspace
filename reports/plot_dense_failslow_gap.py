#!/usr/bin/env python3
"""Dense FailSlow：gap vs N SVG（plot_style）。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_plot_style, save_fig  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--title", default="Dense FailSlow: max−median step gap vs N")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    xs, ys, meds = [], [], []
    for r in rows:
        if not r.get("world_npu") or not r.get("gap_median_ms"):
            continue
        xs.append(int(float(r["world_npu"])))
        ys.append(float(r["gap_median_ms"]))
        if r.get("median_step_ms"):
            meds.append(float(r["median_step_ms"]))
        else:
            meds.append(float("nan"))

    apply_plot_style(figsize=(8, 4.5))
    fig, ax1 = plt.subplots()
    ax1.plot(xs, ys, marker="o", linewidth=2, label="gap = max−median (ms)")
    ax1.set_xlabel("NPU 数 N")
    ax1.set_ylabel("稳态 gap (ms)")
    ax1.set_title(args.title)

    if any(m == m for m in meds):  # not all nan
        ax2 = ax1.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.plot(xs, meds, marker="s", linestyle="--", color="tab:gray", label="median step (ms)")
        ax2.set_ylabel("median step time (ms)")
        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lab1 + lab2, loc="best")
    else:
        ax1.legend(loc="best")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, args.out)
    print(f"SVG → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
