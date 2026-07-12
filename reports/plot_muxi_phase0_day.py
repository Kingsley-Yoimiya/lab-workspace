#!/usr/bin/env python3
"""MUXI Phase0 一天战役交付图：A1 校准、A2 gap、B1 掩蔽、C1 连通性。

默认样式见 plot_style.py（大字号、去顶右边框、y 点线网格、SVG）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

_REPORTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPORTS))
from plot_style import COLORS, HATCHES, apply_plot_style, save_fig, style_axes  # noqa: E402


def plot_a1(analysis: Path, out: Path) -> None:
    labels, ratios = [], []
    for name, label in (
        ("A1_master", "master"),
        ("A1_worker3", "worker-3"),
        ("A1_worker10", "worker-10"),
    ):
        p = analysis / name / "gap_summary.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())["exp0"]
        labels.append(label)
        ratios.append(float(rec["gap_ratio_virt_over_real"]))

    apply_plot_style((8, 4.5))
    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    bars = ax.bar(x, ratios, color=COLORS[0], edgecolor="black", linewidth=0.6)
    for b, h in zip(bars, HATCHES):
        b.set_hatch(h)
    ax.axhspan(0.3, 3.0, color="tab:green", alpha=0.12, label="校准通过带 [0.3, 3]")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("virt/real gap 比")
    ax.set_xlabel("物理节点（各 8 卡 Exp0×3000）")
    ax.set_title("A1：虚拟同步相对真实 AllReduce 校准")
    ax.set_ylim(0, max(3.2, max(ratios) * 1.15))
    ax.legend(frameon=False, loc="upper right")
    for i, v in enumerate(ratios):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=16)
    style_axes(ax)
    save_fig(fig, out, also_png=True)
    out.with_suffix(".caption.md").write_text(
        "纵轴：虚拟屏障重构差距中位数 ÷ 机内 NCCL AllReduce 真实同步差距中位数。"
        "负载为本地 attention/MLP 前向+反向（bf16，hidden=4096）；"
        "数据来自 virtual_sync_bench.py Exp0，ITERS=3000 稳态丢前 100。"
        "三条柱均落在 [0.3, 3] 即校准通过。\n",
        encoding="utf-8",
    )


def plot_a2(analysis: Path, out: Path) -> None:
    s = json.loads((analysis / "A2" / "gap_summary.json").read_text())["exp1"]
    scales = sorted(int(k) for k in s)
    meds = [s[str(n)]["gap_med_of_meds_ms"] for n in scales]
    means = [s[str(n)]["gap_mean_of_meds_ms"] for n in scales]
    stds = [s[str(n)]["gap_std_of_meds_ms"] for n in scales]

    apply_plot_style((9, 5))
    fig, ax = plt.subplots()
    ax.errorbar(
        scales,
        meds,
        yerr=stds,
        fmt="o-",
        color=COLORS[0],
        capsize=4,
        label="gap 中位 of 中位（多子集）",
    )
    ax.plot(scales, means, "s--", color=COLORS[1], alpha=0.8, label="均值")
    ax.set_xlabel("子集规模 N（卡数）")
    ax.set_ylabel("差距指标 (ms)\n最慢累计 − 中位独立耗时")
    ax.set_title("A2/A3：虚拟同步差距随规模上升")
    ax.set_xticks(scales)
    ax.legend(frameon=False)
    style_axes(ax)
    save_fig(fig, out, also_png=True)
    out.with_suffix(".caption.md").write_text(
        "差距指标：事后对独立时间戳做虚拟屏障重构——"
        "第 i 步取各卡累计完成墙钟的最大值差分，再减去该步独立 step 中位数；"
        "无 AllReduce。数据来自 16×8 沐曦卡同时独立跑 3000 step（稳态丢前 100）；"
        "子集含物理节点块 + 随机抽样（n_random≈20）。\n",
        encoding="utf-8",
    )


def plot_b1(analysis: Path, out: Path) -> None:
    rows = json.loads((analysis / "B1_scan_summary.json").read_text())
    tags = [r["tag"] for r in rows]
    short = ["80ms/20/3", "200ms/10/5", "400ms/5/3"]
    global_d = [r["global_delta_ms"] for r in rows]
    p95 = [r["inject_stage1_p95"] for r in rows]
    base_p95 = rows[0]["baseline_stage1_p95"]

    apply_plot_style((10, 5))
    fig, ax1 = plt.subplots()
    x = np.arange(len(tags))
    w = 0.36
    b1 = ax1.bar(
        x - w / 2,
        global_d,
        w,
        color=COLORS[0],
        edgecolor="black",
        linewidth=0.6,
        label="全局中位 Δ (ms)",
    )
    b1[0].set_hatch(HATCHES[0])
    for b, h in zip(b1, HATCHES):
        b.set_hatch(h)
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_ylabel("全局中位变化 (ms)")
    ax1.set_xlabel("注入配置 DELAY_MS / EVERY / BURST（仅 stage1）")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short)

    ax2 = ax1.twinx()
    ax2.plot(x, p95, "o-", color=COLORS[2], lw=2, markersize=8, label="stage1 p95 (ms)")
    ax2.axhline(base_p95, color=COLORS[1], ls="--", lw=1.5, label=f"baseline stage1 p95={base_p95:.0f}")
    ax2.set_ylabel("stage1 step p95 (ms)")
    ax2.spines["top"].set_visible(False)
    # twin 右边框保留作第二轴，但颜色压灰
    ax2.spines["right"].set_color("#808080")
    ax2.tick_params(axis="y", colors="#808080")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    ax1.set_title("B1：PP stage 注入 — 全局弱、切片强（掩蔽）")
    style_axes(ax1)
    save_fig(fig, out, also_png=True)
    out.with_suffix(".caption.md").write_text(
        "假 PP 映射 PP_SIZE=4：rank 块→stage；对 stage1 间歇 time.sleep（delay_inject.py）。"
        "左轴：全体卡 step 中位数相对 baseline 的变化（几乎无感）；"
        "右轴：注入 stage 的 step p95（随 DELAY_MS 抬升）。"
        "说明间歇延迟被中位数稀释，但按 stage 分组可见——案例二「切片掩蔽」。\n",
        encoding="utf-8",
    )


def plot_c1(c1_dir: Path, out: Path) -> None:
    mat = json.loads((c1_dir / "pair_matrix.json").read_text())
    flat = mat["flat"]
    n = mat["summary"]["n"]
    # 编码：0=self, 1=L3 reachable (refused), 2=ok/ping, 3=timeout/unresolved
    grid = np.zeros((n, n), dtype=int)
    for r in flat:
        i, j = r["i"], r["j"]
        st = r["status"]
        err = r.get("err") or ""
        if st == "self":
            grid[i, j] = 0
        elif st in ("ok", "ping_ok"):
            grid[i, j] = 2
        elif "refused" in err.lower():
            grid[i, j] = 1
        else:
            grid[i, j] = 3

    cmap = ListedColormap(["#4daf4a", "#377eb8", "#ff7f00", "#e41a1c"])
    apply_plot_style((8.5, 7.5))
    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_xlabel("目标节点 j")
    ax.set_ylabel("源节点 i（master + worker-0…14）")
    ax.set_title("C1：16×16 节点对连通性（管理面 TCP）")
    ax.set_xticks(range(0, n, 2))
    ax.set_yticks(range(0, n, 2))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["self", "L3可达\n(TCP refused)", "TCP/ping OK", "超时/未解析"])
    style_axes(ax)
    ax.grid(False)
    save_fig(fig, out, also_png=True)
    out.with_suffix(".caption.md").write_text(
        "在 master 上对 16 个 Pod IP 做 TCP connect（22/29500/8080/443）与 ping。"
        "Connection refused = 报文到达对端但无监听（L3 路由通）；"
        "机间 RoCE 训练面另记 expected_fail_proxy_connect，不把 eth0 TCP 冒充 MFU。\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--day-root",
        type=Path,
        default=Path("/Users/yinjinrun/random-thing/logs/muxi-day-20260713_002719"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/yinjinrun/random-thing/docs/muxi/phase0-day-20260713/figs"),
    )
    args = ap.parse_args()
    analysis = args.day_root / "results" / "analysis"
    c1 = args.day_root / "results" / "C1"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_a1(analysis, args.out_dir / "A1_calibration.svg")
    plot_a2(analysis, args.out_dir / "A2_gap_vs_scale.svg")
    plot_b1(analysis, args.out_dir / "B1_pp_mask_scan.svg")
    plot_c1(c1, args.out_dir / "C1_pair_matrix.svg")

    # 同步已有热力图
    for name in ("A4_power_heatmap.svg", "A4_clock_heatmap.svg", "A4_power_heatmap.png", "A4_clock_heatmap.png"):
        src = analysis / name
        if src.exists():
            (args.out_dir / name).write_bytes(src.read_bytes())
            cap = analysis / name.replace(".svg", ".caption.md").replace(".png", ".caption.md")
            if cap.exists() and name.endswith(".svg"):
                (args.out_dir / (name.replace(".svg", ".caption.md"))).write_text(
                    cap.read_text(encoding="utf-8"), encoding="utf-8"
                )
    print(f"FIGS -> {args.out_dir}")


if __name__ == "__main__":
    main()
