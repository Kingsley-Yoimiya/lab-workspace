#!/usr/bin/env python3
"""体质筛查 · 增强可视化（补 plot_card_constitution 未覆盖/不够清晰的图）。

读 JSONL record=card，输出雷达/平行坐标、HBM 四模式、相关矩阵、launch boxplot、
CDF、快慢卡对比、host×device 热力图、sustained vs func 散点。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from plot_style import (
    apply_plot_style,
    annot_fontsize_for_grid,
    hatch_bar_kwargs,
    natural_host_key,
    save_fig,
    short_host_label,
    style_axes,
    style_heatmap_axes,
)

# ── 指标定义 ──────────────────────────────────────────────────────────────

METRIC_LABEL: dict[str, str] = {
    "func_tflops": "Cube func TFLOPS",
    "hbm_gbps": "HBM GB/s",
    "sustained_tflops": "Sustained TFLOPS",
    "vector_gflops": "Vector GFLOPS",
    "scalar_elems_per_s": "Scalar elems/s",
    "mte_gbps": "MTE copy GB/s",
    "cube_vector_tflops": "Cube+Vector TFLOPS",
    "sfu_gflops": "SFU GFLOPS",
    "hbm_mode_seq_copy_gbps": "HBM 顺序拷贝 GB/s",
    "hbm_mode_strided_gbps": "HBM 跨步 GB/s",
    "hbm_mode_read_heavy_gbps": "HBM 读密集 GB/s",
    "hbm_mode_write_heavy_gbps": "HBM 写密集 GB/s",
    "launch_sync_p50_us": "Launch sync p50 (μs)",
    "launch_sync_p99_us": "Launch sync p99 (μs)",
    "launch_host_overhead_p50_us": "Host overhead p50 (μs)",
    "launch_host_overhead_p99_us": "Host overhead p99 (μs)",
    "launch_burst_p50_us": "Burst total p50 (μs)",
    "launch_burst_per_kernel_p50_us": "Burst/kernel p50 (μs)",
}

# 雷达/平行坐标用（越高越好；launch 类取倒数归一化）
RADAR_METRICS = [
    "func_tflops",
    "sustained_tflops",
    "vector_gflops",
    "mte_gbps",
    "sfu_gflops",
    "hbm_gbps",
    "cube_vector_tflops",
    "scalar_elems_per_s",
]

HBM_MODES = [
    ("hbm_mode_seq_copy_gbps", "顺序拷贝"),
    ("hbm_mode_strided_gbps", "跨步"),
    ("hbm_mode_read_heavy_gbps", "读密集"),
    ("hbm_mode_write_heavy_gbps", "写密集"),
]

CORR_METRICS = [
    ("func_tflops", "Cube"),
    ("vector_gflops", "Vector"),
    ("sfu_gflops", "SFU"),
    ("mte_gbps", "MTE"),
]

LAUNCH_KEYS = [
    ("launch_sync_p99_us", "Launch sync p99"),
    ("launch_host_overhead_p99_us", "Host overhead p99"),
    ("launch_burst_p50_us", "Burst total p50"),
]

CDF_METRICS = [
    "func_tflops",
    "hbm_gbps",
    "vector_gflops",
    "mte_gbps",
    "sfu_gflops",
    "sustained_tflops",
]

HEATMAP_METRICS = [
    "vector_gflops",
    "mte_gbps",
    "sfu_gflops",
    "scalar_elems_per_s",
]

SMALL_MULTIPLE_METRICS = [
    ("func_tflops", "Cube TFLOPS"),
    ("sustained_tflops", "Sustained TFLOPS"),
    ("hbm_gbps", "HBM GB/s"),
    ("vector_gflops", "Vector GFLOPS"),
    ("mte_gbps", "MTE GB/s"),
    ("sfu_gflops", "SFU GFLOPS"),
]


# ── 工具函数 ──────────────────────────────────────────────────────────────

def short_host(host: str) -> str:
    return short_host_label(host)


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if abs(v) >= 1e6:
            return f"{v:.2e}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.4g}"
    return str(v)


def metric_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    ordered = sorted(values)
    mean = statistics.mean(ordered)
    std = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    cv = (std / mean * 100.0) if mean else 0.0
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "mean": mean,
        "std": std,
        "cv_pct": cv,
        "min": ordered[0],
        "max": ordered[-1],
    }


def values_for(cards: list[dict], key: str) -> list[float]:
    vals: list[float] = []
    for c in cards:
        v = c.get(key)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return vals


def load_cards(path: Path) -> list[dict]:
    cards: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("record") == "card":
                cards.append(obj)
    return cards


def card_label(c: dict) -> str:
    return f"{short_host(c.get('host', '?'))}.{c.get('device')}"


def _try_plt():
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None, None
    apply_plot_style()
    return plt, np


# ── 图 1：雷达 + 平行坐标 ────────────────────────────────────────────────

def plot_radar_and_parallel(cards: list[dict], fig_dir: Path, plt, np) -> list[str]:
    hosts = sorted({c.get("host", "?") for c in cards}, key=natural_host_key)
    if len(hosts) < 2:
        return []

    # 集群中位数作为归一化基准
    cluster_medians: dict[str, float] = {}
    for key in RADAR_METRICS:
        vals = values_for(cards, key)
        if vals:
            cluster_medians[key] = statistics.median(vals)

    if not cluster_medians:
        return []

    host_norm: dict[str, list[float]] = {}
    for h in hosts:
        hc = [c for c in cards if c.get("host") == h]
        row = []
        for key in RADAR_METRICS:
            med = cluster_medians.get(key)
            if not med:
                row.append(0.0)
                continue
            hvals = values_for(hc, key)
            hmed = statistics.median(hvals) if hvals else med
            row.append(hmed / med)  # 1.0 = 集群中位
        host_norm[h] = row

    labels = [METRIC_LABEL.get(k, k).split()[0] for k in RADAR_METRICS]
    n_axes = len(RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(11, 9), subplot_kw=dict(polar=True))
    cmap = plt.get_cmap("tab10")
    for i, h in enumerate(hosts):
        vals = host_norm[h] + host_norm[h][:1]
        ax.plot(angles, vals, linewidth=1.8, color=cmap(i % 10), label=short_host(h))
        ax.fill(angles, vals, alpha=0.08, color=cmap(i % 10))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.85, 1.15)
    ax.axhline(1.0, color="#888", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("各 Host 多指标中位数归一化雷达图\n（1.0 = 集群中位）", fontsize=13, pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.05), fontsize=9, framealpha=0.5)
    fig.subplots_adjust(right=0.78)
    name_radar = "radar_host_median_norm.svg"
    save_fig(fig, fig_dir / name_radar)

    # 平行坐标
    fig2, ax2 = plt.subplots(figsize=(14, 6.5))
    x = range(n_axes)
    for i, h in enumerate(hosts):
        ax2.plot(x, host_norm[h], marker="o", markersize=5, linewidth=1.5,
                 color=cmap(i % 10), label=short_host(h), alpha=0.85)
    ax2.axhline(1.0, color="#888", linestyle="--", linewidth=1, alpha=0.7, label="集群中位")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([METRIC_LABEL.get(k, k) for k in RADAR_METRICS],
                        rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("相对集群中位数比值")
    ax2.set_title("各 Host 多指标中位数平行坐标对比", fontsize=13)
    ax2.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), framealpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.85, 1.15)
    style_axes(ax2)
    fig2.tight_layout()
    name_par = "parallel_host_median_norm.svg"
    save_fig(fig2, fig_dir / name_par)
    return [name_radar, name_par]


# ── 图 2：HBM 四模式 grouped bar ─────────────────────────────────────────

def plot_hbm_modes_grouped(cards: list[dict], fig_dir: Path, plt, np) -> str | None:
    hosts = sorted({c.get("host", "?") for c in cards}, key=natural_host_key)
    mode_labels = [m[1] for m in HBM_MODES]
    n_modes = len(HBM_MODES)

    # 全卡中位
    cluster_med = []
    for key, _ in HBM_MODES:
        vals = values_for(cards, key)
        cluster_med.append(statistics.median(vals) if vals else 0.0)

    # 每 host 中位
    host_med: list[list[float]] = []
    host_labels = ["全集群"] + [short_host(h) for h in hosts]
    host_med.append(cluster_med)
    for h in hosts:
        hc = [c for c in cards if c.get("host") == h]
        row = []
        for key, _ in HBM_MODES:
            vals = values_for(hc, key)
            row.append(statistics.median(vals) if vals else 0.0)
        host_med.append(row)

    n_groups = len(host_labels)
    x = np.arange(n_groups)
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(12, n_groups * 1.2), 6))
    for j in range(n_modes):
        offset = (j - (n_modes - 1) / 2) * width
        bars = [host_med[i][j] for i in range(n_groups)]
        ax.bar(x + offset, bars, **hatch_bar_kwargs(j, width=width), label=mode_labels[j])

    ax.set_xticks(x)
    ax.set_xticklabels(host_labels, rotation=30, ha="right", fontsize=14)
    ax.set_ylabel("带宽 (GB/s)")
    ax.set_title("HBM 四模式带宽 · 全卡中位 & 各 Host 中位", fontsize=13)
    ax.legend(fontsize=14)
    style_axes(ax)
    fig.tight_layout()
    name = "hbm_modes_grouped_bar.svg"
    save_fig(fig, fig_dir / name)
    return name


# ── 图 3：相关矩阵热力图 ──────────────────────────────────────────────────

def plot_corr_heatmap(cards: list[dict], fig_dir: Path, plt, np) -> str | None:
    keys = [k for k, _ in CORR_METRICS]
    labels = [lbl for _, lbl in CORR_METRICS]
    n = len(keys)
    matrix = np.full((n, n), np.nan)
    data_cols: list[list[float]] = []
    for key in keys:
        col = values_for(cards, key)
        if len(col) < 3:
            return None
        data_cols.append(col)

    # 对齐：只取四指标都有值的卡
    aligned: list[list[float]] = [[] for _ in keys]
    for c in cards:
        row = []
        ok = True
        for key in keys:
            v = c.get(key)
            if v is None:
                ok = False
                break
            row.append(float(v))
        if ok:
            for i, v in enumerate(row):
                aligned[i].append(v)

    if len(aligned[0]) < 3:
        return None

    arr = np.array(aligned)
    corr = np.corrcoef(arr)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                fontsize=annot_fontsize_for_grid(n, n),
                color="white" if abs(corr[i, j]) > 0.6 else "black",
            )
    ax.set_title("Cube / Vector / SFU / MTE 相关矩阵", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    style_heatmap_axes(ax, tick_fs=12)
    fig.tight_layout()
    name = "corr_cube_vector_sfu_mte.svg"
    save_fig(fig, fig_dir / name)
    return name


# ── 图 4：launch boxplot ──────────────────────────────────────────────────

def plot_launch_boxplot(cards: list[dict], fig_dir: Path, plt) -> str | None:
    hosts = sorted({c.get("host", "?") for c in cards}, key=natural_host_key)
    n_metrics = len(LAUNCH_KEYS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5.5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    drawn = False
    for ax, (key, title) in zip(axes, LAUNCH_KEYS):
        series, labels = [], []
        for h in hosts:
            vals = values_for([c for c in cards if c.get("host") == h], key)
            if vals:
                series.append(vals)
                labels.append(short_host(h))
        if not series:
            ax.set_visible(False)
            continue
        drawn = True
        # 离群点易把主体箱线挤扁：对数轴 + 短 host 名
        bp = ax.boxplot(series, patch_artist=True, widths=0.55, showfliers=False,
                        boxprops=dict(facecolor="#4C78A8", alpha=0.7),
                        medianprops=dict(color="#E45756", linewidth=2))
        for i, vals in enumerate(series, start=1):
            ax.scatter([i] * len(vals), vals, alpha=0.3, s=12, color="#72B7B2", zorder=3)
        rot = 0 if max(len(x) for x in labels) <= 10 else 30
        ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center", fontsize=10)
        ax.set_ylabel("延迟 (μs)")
        ax.set_title(title, fontsize=11)
        if key in ("launch_host_overhead_p99_us", "launch_burst_p50_us"):
            ax.set_yscale("log")
        style_axes(ax)

    if not drawn:
        plt.close(fig)
        return None
    fig.suptitle("Launch 延迟按 Host 分布", fontsize=13, y=1.02)
    fig.tight_layout()
    name = "box_launch_by_host.svg"
    save_fig(fig, fig_dir / name)
    return name


# ── 图 5：CDF ─────────────────────────────────────────────────────────────

def plot_cdf_panel(cards: list[dict], fig_dir: Path, plt, np) -> str | None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    drawn = 0
    cmap = plt.get_cmap("tab10")
    hosts = sorted({c.get("host", "?") for c in cards}, key=natural_host_key)

    for ax, key in zip(axes_flat, CDF_METRICS):
        vals = values_for(cards, key)
        if len(vals) < 2:
            ax.set_visible(False)
            continue
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, color="#4C78A8", linewidth=2, label="全集群")
        med = statistics.median(vals)
        ax.axvline(med, color="#E45756", linestyle="--", linewidth=1.2,
                   label=f"中位={med:.4g}")
        ax.set_xlabel(METRIC_LABEL.get(key, key))
        ax.set_ylabel("累积概率")
        ax.set_title(METRIC_LABEL.get(key, key), fontsize=10)
        ax.legend(fontsize=7)
        style_axes(ax)
        drawn += 1

    if drawn == 0:
        plt.close(fig)
        return None
    for ax in axes_flat[drawn:]:
        ax.set_visible(False)
    fig.suptitle("核心指标累积分布函数 (CDF)", fontsize=14, y=1.01)
    fig.tight_layout()
    name = "cdf_core_metrics.svg"
    save_fig(fig, fig_dir / name)
    return name


# ── 图 6：最慢/最快 10 卡 small multiples ─────────────────────────────────

def plot_extreme_cards(cards: list[dict], fig_dir: Path, plt, np) -> str | None:
    # 用 sustained_tflops 排序（稳定性主指标）
    ranked = [(c, float(c["sustained_tflops"])) for c in cards
              if c.get("sustained_tflops") is not None]
    if len(ranked) < 10:
        return None
    ranked.sort(key=lambda x: x[1])
    slow10 = [c for c, _ in ranked[:10]]
    fast10 = [c for c, _ in ranked[-10:]]
    cluster_medians = {key: statistics.median(values_for(cards, key))
                       for key, _ in SMALL_MULTIPLE_METRICS
                       if values_for(cards, key)}

    n_metrics = len(SMALL_MULTIPLE_METRICS)
    fig, axes = plt.subplots(2, n_metrics, figsize=(3.2 * n_metrics, 8), sharey="row")
    groups = [("最慢 10 卡", slow10, "#E45756"), ("最快 10 卡", fast10, "#54A24B")]

    for row, (gtitle, group, color) in enumerate(groups):
        for col, (key, mlabel) in enumerate(SMALL_MULTIPLE_METRICS):
            ax = axes[row, col]
            vals = [float(c[key]) for c in group if c.get(key) is not None]
            labels = [card_label(c) for c in group if c.get(key) is not None]
            if not vals:
                ax.set_visible(False)
                continue
            med = cluster_medians.get(key, statistics.median(vals))
            rel = [(v / med - 1) * 100 if med else 0 for v in vals]
            y_pos = range(len(rel))
            hatch = "//" if row == 0 else "\\\\"
            for yi, rval in zip(y_pos, rel):
                ec = color if (rval >= 0 if row == 1 else rval >= 0) else "#E45756"
                if row == 0 and rval < 0:
                    ec = "#E45756"
                ax.barh(yi, rval, color="none", edgecolor=ec, hatch=hatch,
                        linewidth=2.0, height=0.7)
            ax.axvline(0, color="#333", linewidth=0.8)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_xlabel("相对集群中位偏差 (%)")
            if col == 0:
                ax.set_ylabel(gtitle, fontsize=10)
            ax.set_title(mlabel, fontsize=9)
            style_axes(ax)
            ax.invert_yaxis()

    fig.suptitle("最慢 vs 最快 10 卡 · 多指标相对偏差", fontsize=14, y=1.01)
    fig.tight_layout()
    name = "extreme10_small_multiples.svg"
    save_fig(fig, fig_dir / name)
    return name


# ── 图 7：host×device 绝对值热力图 ────────────────────────────────────────

def plot_host_device_heatmap(cards: list[dict], key: str, fig_dir: Path, plt, np) -> str | None:
    hosts = sorted({c.get("host", "?") for c in cards}, key=natural_host_key)
    devices = sorted({int(c["device"]) for c in cards if c.get("device") is not None})
    if not hosts or not devices:
        return None
    matrix = np.full((len(hosts), len(devices)), np.nan)
    for c in cards:
        v = c.get(key)
        if v is None or c.get("host") not in hosts or c.get("device") is None:
            continue
        try:
            hi = hosts.index(c["host"])
            di = devices.index(int(c["device"]))
            matrix[hi, di] = float(v)
        except (ValueError, TypeError):
            continue
    if np.all(np.isnan(matrix)):
        return None

    label = METRIC_LABEL.get(key, key)
    vmin, vmax = np.nanpercentile(matrix, [5, 95])
    if vmin == vmax:
        vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)

    fig, ax = plt.subplots(figsize=(max(13.5, len(devices) * 0.85), max(6.0, len(hosts) * 0.78)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(devices)))
    ax.set_xticklabels([str(d) for d in devices])
    ax.set_yticks(range(len(hosts)))
    ax.set_yticklabels([short_host(h) for h in hosts])
    ax.set_xlabel("Device")
    ax.set_ylabel("Host")
    ax.set_title(f"{label} · Host × Device（偏离中位 ≥0.5% 才标数）", fontsize=14)
    annot_fs = 8.0
    med = float(np.nanmedian(matrix))
    mid = (float(vmin) + float(vmax)) / 2.0
    for i in range(len(hosts)):
        for j in range(len(devices)):
            v = matrix[i, j]
            if math.isnan(v) or not med:
                continue
            if abs(v - med) / abs(med) < 0.005:
                continue
            if abs(v) >= 1e6:
                txt = f"{v/1e6:.1f}M"
            elif abs(v) >= 100:
                txt = f"{v:.0f}"
            else:
                txt = f"{v:.2g}"
            ax.text(
                j, i, txt, ha="center", va="center", fontsize=annot_fs,
                color="white" if v > mid else "black",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    style_heatmap_axes(ax)
    fig.tight_layout()
    name = f"heatmap_host_device_{key}.svg"
    save_fig(fig, fig_dir / name)
    return name


# ── 图 8：sustained vs func 散点（稳定性）──────────────────────────────────

def plot_sustained_vs_func(cards: list[dict], fig_dir: Path, plt) -> str | None:
    pts = []
    for c in cards:
        sus, func = c.get("sustained_tflops"), c.get("func_tflops")
        if sus is None or func is None:
            continue
        pts.append((float(func), float(sus), short_host(c.get("host", "?")),
                    card_label(c)))
    if len(pts) < 2:
        return None

    hosts = sorted({p[2] for p in pts})
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, h in enumerate(hosts):
        sub = [p for p in pts if p[2] == h]
        xs = [p[0] for p in sub]
        ys = [p[1] for p in sub]
        ax.scatter(xs, ys, s=40, alpha=0.75, color=cmap(i % 10), label=h, edgecolors="white", linewidth=0.3)

    # y=x 参考线 & 稳定性带
    all_x = [p[0] for p in pts]
    all_y = [p[1] for p in pts]
    lo, hi = min(min(all_x), min(all_y)) - 5, max(max(all_x), max(all_y)) + 5
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="y = x（理想一致）")
    ratios = [p[1] / p[0] for p in pts if p[0] > 0]
    med_ratio = statistics.median(ratios)
    ax.plot([lo, hi], [lo * med_ratio, hi * med_ratio], color="#F58518", linestyle=":",
            linewidth=1.2, label=f"中位比 sustained/func = {med_ratio:.3f}")

    ax.set_xlabel("Cube func TFLOPS")
    ax.set_ylabel("Sustained TFLOPS")
    ax.set_title("Sustained vs Func 散点 · 稳定性分析", fontsize=13)
    ax.legend(fontsize=7, loc="best")
    style_axes(ax)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    name = "scatter_sustained_vs_func.svg"
    save_fig(fig, fig_dir / name)
    return name


# ── Markdown 报告 ────────────────────────────────────────────────────────

def write_markdown(
    cards: list[dict],
    source: Path,
    avail: list[tuple[str, str, dict]],
    skipped: list[str],
    fig_names: list[str],
    out_path: Path,
    fig_dir_name: str,
) -> None:
    lines = [
        "# Constitution 增强可视化报告",
        "",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 卡数: {len(cards)}",
        f"- 数据源: {source}",
        "",
        "> 补充 plot_card_constitution 未覆盖的多维对比、CDF、相关矩阵、快慢卡分析。",
        "",
    ]
    if skipped:
        lines += ["## 跳过说明", ""]
        for s in skipped:
            lines.append(f"- {s}")
        lines.append("")

    lines += [
        "## 核心指标 median / CV 摘要",
        "",
        "| 指标 | n | median | CV% | min | max |",
        "|------|---|--------|-----|-----|-----|",
    ]
    for key, label, st in avail:
        lines.append(
            f"| {label} | {st['n']} | {fmt(st['median'])} | {fmt(st['cv_pct'])} | "
            f"{fmt(st['min'])} | {fmt(st['max'])} |"
        )

    hosts = sorted({c.get("host", "?") for c in cards}, key=natural_host_key)
    lines += ["", "## 元数据", ""]
    lines.append(f"- hosts ({len(hosts)}): {', '.join(short_host(h) for h in hosts)}")

    if fig_names:
        lines += ["", "## 图表", ""]
        for fn in fig_names:
            title = fn.replace(".svg", "").replace("_", " ")
            lines += [f"### {title}", "", f"![{title}]({fig_dir_name}/{fn})", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── 主流程 ───────────────────────────────────────────────────────────────

CORE_STATS_KEYS = [
    "func_tflops", "hbm_gbps", "sustained_tflops", "vector_gflops",
    "scalar_elems_per_s", "mte_gbps", "cube_vector_tflops", "sfu_gflops",
    "hbm_mode_seq_copy_gbps", "hbm_mode_strided_gbps",
    "hbm_mode_read_heavy_gbps", "hbm_mode_write_heavy_gbps",
    "launch_sync_p99_us", "launch_host_overhead_p99_us", "launch_burst_p50_us",
]


def generate(
    jsonl: Path,
    out_dir: Path,
    stamp: str = "constitution_extra_20260711",
) -> tuple[Path, Path]:
    cards = load_cards(jsonl)
    if not cards:
        raise SystemExit(f"无 record=card 行：{jsonl}")

    report_path = out_dir / f"{stamp}.md"
    fig_dir = out_dir / f"{stamp}_figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    avail = []
    for key in CORE_STATS_KEYS:
        vals = values_for(cards, key)
        st = metric_stats(vals)
        if st.get("n", 0) >= 1:
            avail.append((key, METRIC_LABEL.get(key, key), st))

    skipped: list[str] = []
    fig_names: list[str] = []

    plt, np = _try_plt()
    if plt is None:
        skipped.append("matplotlib/numpy 不可用，跳过全部作图")
    else:
        fns = plot_radar_and_parallel(cards, fig_dir, plt, np)
        fig_names.extend(fns)

        fn = plot_hbm_modes_grouped(cards, fig_dir, plt, np)
        if fn:
            fig_names.append(fn)
        else:
            skipped.append("HBM 四模式 grouped bar：数据不足")

        fn = plot_corr_heatmap(cards, fig_dir, plt, np)
        if fn:
            fig_names.append(fn)
        else:
            skipped.append("相关矩阵：配对点数 < 3")

        fn = plot_launch_boxplot(cards, fig_dir, plt)
        if fn:
            fig_names.append(fn)
        else:
            skipped.append("Launch boxplot：数据不足")

        fn = plot_cdf_panel(cards, fig_dir, plt, np)
        if fn:
            fig_names.append(fn)
        else:
            skipped.append("CDF：核心指标数据不足")

        fn = plot_extreme_cards(cards, fig_dir, plt, np)
        if fn:
            fig_names.append(fn)
        else:
            skipped.append("快慢 10 卡对比：sustained 数据不足")

        for key in HEATMAP_METRICS:
            fn = plot_host_device_heatmap(cards, key, fig_dir, plt, np)
            if fn:
                fig_names.append(fn)
            else:
                skipped.append(f"热力图 `{key}`：数据不足")

        fn = plot_sustained_vs_func(cards, fig_dir, plt)
        if fn:
            fig_names.append(fn)
        else:
            skipped.append("sustained vs func 散点：数据不足")

    write_markdown(cards, jsonl, avail, skipped, fig_names, report_path, fig_dir.name)
    return report_path, fig_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="体质筛查增强可视化")
    ap.add_argument(
        "--jsonl", type=Path,
        default=Path(
            "/Users/yinjinrun/random-thing/logs/card-constitution-128-20260711_134309"
            "-constitution128/results/constitution128.merged.jsonl"
        ),
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parent / "rounds",
    )
    ap.add_argument("--stamp", type=str, default="constitution_extra_20260711")
    args = ap.parse_args()

    report_path, fig_dir = generate(args.jsonl, args.out_dir, stamp=args.stamp)
    svgs = sorted(fig_dir.glob("*.svg"))
    print(f"wrote {report_path}")
    print(f"figs  {fig_dir} ({len(svgs)} svg)")
    for p in svgs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
