#!/usr/bin/env python3
"""出图：虚拟同步 gap vs 规模 N（MUXI Phase0）。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT = Path(__file__).resolve()
_REPORTS = _SCRIPT.parents[2] / "reports"
sys.path.insert(0, str(_REPORTS))
from plot_style import apply_plot_style, save_fig, style_axes  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    by_scale: dict[int, list[float]] = defaultdict(list)
    with args.csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_scale[int(row["scale"])].append(float(row["gap_med_ms"]))

    scales = sorted(by_scale)
    meds = [float(np.median(by_scale[s])) for s in scales]
    means = [float(np.mean(by_scale[s])) for s in scales]
    stds = [float(np.std(by_scale[s])) if len(by_scale[s]) > 1 else 0.0 for s in scales]

    apply_plot_style((9, 5))
    fig, ax = plt.subplots()
    ax.errorbar(
        scales,
        meds,
        yerr=stds,
        fmt="o-",
        color="tab:orange",
        capsize=4,
        label="虚拟同步 gap 中位数（多子集）",
    )
    ax.plot(scales, means, "s--", color="tab:green", alpha=0.7, label="均值")
    ax.set_xlabel("子集规模 N（卡数）")
    ax.set_ylabel("差距指标 (ms)\n最慢累计 − 中位独立耗时")
    ax.set_title("MUXI 128 卡：虚拟同步差距随规模变化")
    ax.legend(frameon=False)
    ax.set_xticks(scales)

    style_axes(ax)
    caption = (
        "差距指标：事后对独立时间戳做虚拟同步屏障重构——"
        "第 i 步取各卡累计完成墙钟的最大值差分，再减去该步独立 step 中位数；"
        "负载为本地 attention/MLP 前向+反向（无 AllReduce）。"
        "数据来自 16×8 沐曦卡同时独立跑（默认战役 ITERS=3000，稳态丢前 100）；"
        "子集含按物理节点块抽取与随机抽样。"
    )
    out = save_fig(fig, args.out, also_png=True)
    args.out.with_suffix(".caption.md").write_text(caption + "\n", encoding="utf-8")
    print(f"WROTE {out}")

    if args.summary and args.summary.exists():
        print(args.summary.read_text(encoding="utf-8")[:2000])


if __name__ == "__main__":
    main()
