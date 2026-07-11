#!/usr/bin/env python3
"""从 jumphost ledger 生成 Dense TP×PP 弱扩展 MFU 曲线（默认 plot_style）。"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

REPORTS = Path(__file__).resolve().parent
sys.path.insert(0, str(REPORTS))
from plot_style import COLORS, apply_plot_style, save_fig, style_axes  # noqa: E402

LEDGER = REPORTS / "rounds" / "mfu_tp_pp_scale_bundle" / "jumphost" / "ledger.md"
# 回退
if not LEDGER.is_file():
    LEDGER = REPORTS / "rounds" / "mfu_tp_pp_scale_ledger.md"
OUT_DIR = REPORTS / "rounds" / "mfu_tp_pp_scale_bundle" / "figs"
CSV_PATH = REPORTS / "rounds" / "mfu_tp_pp_scale_bundle" / "mfu_tp_pp_scale_results.csv"
PEAK = 292.79

# 展示顺序（有意义并行；不含 TP1PP1）
TOPO_ORDER = ["4/2/1", "4/4/1", "8/1/1", "8/2/1", "8/4/1"]
TOPO_LABEL = {
    "4/2/1": "TP4 PP2",
    "4/4/1": "TP4 PP4",
    "8/1/1": "TP8 PP1",
    "8/2/1": "TP8 PP2",
    "8/4/1": "TP8 PP4",
}


def parse_ledger(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 10 or parts[0] in ("round", "------:"):
            continue
        if parts[9] not in ("ok", "leftover"):
            continue
        if parts[2] != "dense":
            continue
        # 跳过 leftover 重复
        if parts[9] == "leftover":
            continue
        try:
            scale = int(parts[3])
            tflop = float(parts[7])
            mfu = float(parts[8])
            dp = int(parts[5])
        except ValueError:
            continue
        rows.append(
            {
                "id": parts[1],
                "mode": parts[2],
                "scale": scale,
                "tp_pp_ep": parts[4],
                "dp": dp,
                "gbs": parts[6],
                "tflop": tflop,
                "mfu": mfu,
                "status": parts[9],
            }
        )
    # 同拓扑同 scale 留最后一次
    uniq: dict[tuple, dict] = {}
    for r in rows:
        uniq[(r["tp_pp_ep"], r["scale"])] = r
    return sorted(uniq.values(), key=lambda r: (r["tp_pp_ep"], r["scale"]))


def main() -> None:
    rows = parse_ledger(LEDGER)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["tp_pp_ep", "label", "scale", "dp", "gbs", "tflop", "mfu_pct", "peak_tflops"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "tp_pp_ep": r["tp_pp_ep"],
                    "label": TOPO_LABEL.get(r["tp_pp_ep"], r["tp_pp_ep"]),
                    "scale": r["scale"],
                    "dp": r["dp"],
                    "gbs": r["gbs"],
                    "tflop": r["tflop"],
                    "mfu_pct": r["mfu"],
                    "peak_tflops": PEAK,
                }
            )

    by_topo: dict[str, dict[int, float]] = defaultdict(dict)
    by_topo_tflop: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        by_topo[r["tp_pp_ep"]][r["scale"]] = r["mfu"]
        by_topo_tflop[r["tp_pp_ep"]][r["scale"]] = r["tflop"]

    import matplotlib.pyplot as plt

    # 图1：MFU vs 卡数
    apply_plot_style((9.5, 5.2))
    fig, ax = plt.subplots()
    for i, topo in enumerate(TOPO_ORDER):
        if topo not in by_topo:
            continue
        xs = sorted(by_topo[topo])
        ys = [by_topo[topo][x] for x in xs]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=8,
            color=COLORS[i % len(COLORS)],
            label=TOPO_LABEL.get(topo, topo),
        )
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=11,
                color=COLORS[i % len(COLORS)],
            )
    style_axes(ax)
    ax.set_xlabel("卡数 (NPU)")
    ax.set_ylabel("MFU (%)")
    ax.set_xticks([16, 32, 64, 128])
    ax.set_ylim(0, 50)
    ax.legend(loc="best", ncol=2)
    ax.set_title("华为 Ascend Dense：固定 TP×PP 扩 DP 的 MFU")
    p1 = save_fig(fig, OUT_DIR / "mfu_vs_scale_dense.svg", also_png=True)

    # 图2：相对 16 卡效率（无 16 的拓扑用最小 scale 作基线）
    apply_plot_style((9.5, 5.2))
    fig, ax = plt.subplots()
    for i, topo in enumerate(TOPO_ORDER):
        if topo not in by_topo:
            continue
        pts = by_topo[topo]
        base_s = 16 if 16 in pts else min(pts)
        base = pts[base_s]
        xs = sorted(pts)
        ys = [pts[x] / base * 100.0 for x in xs]
        ax.plot(
            xs,
            ys,
            marker="s",
            linewidth=2.2,
            markersize=7,
            color=COLORS[i % len(COLORS)],
            label=f"{TOPO_LABEL.get(topo, topo)} (基线@{base_s})",
        )
    style_axes(ax)
    ax.axhline(100, color="#808080", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("卡数 (NPU)")
    ax.set_ylabel("相对效率 (%)")
    ax.set_xticks([16, 32, 64, 128])
    ax.set_ylim(60, 110)
    ax.legend(loc="best", fontsize=14)
    ax.set_title("弱扩展相对效率（相对本拓扑基线卡数）")
    p2 = save_fig(fig, OUT_DIR / "mfu_relative_efficiency_dense.svg", also_png=True)

    # 图3：TFLOP/卡
    apply_plot_style((9.5, 5.2))
    fig, ax = plt.subplots()
    for i, topo in enumerate(TOPO_ORDER):
        if topo not in by_topo_tflop:
            continue
        xs = sorted(by_topo_tflop[topo])
        ys = [by_topo_tflop[topo][x] for x in xs]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=8,
            color=COLORS[i % len(COLORS)],
            label=TOPO_LABEL.get(topo, topo),
        )
    style_axes(ax)
    ax.set_xlabel("卡数 (NPU)")
    ax.set_ylabel("稳态 TFLOP/s/GPU")
    ax.set_xticks([16, 32, 64, 128])
    ax.legend(loc="best", ncol=2)
    ax.set_title(f"吞吐（peak={PEAK} TFLOPS/卡）")
    p3 = save_fig(fig, OUT_DIR / "tflop_vs_scale_dense.svg", also_png=True)

    print(f"CSV → {CSV_PATH}")
    print(f"FIG → {p1}")
    print(f"FIG → {p2}")
    print(f"FIG → {p3}")
    print(f"n_points={len(rows)}")


if __name__ == "__main__":
    main()
