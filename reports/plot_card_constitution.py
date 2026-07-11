#!/usr/bin/env python3
"""体质筛查 · 分布优先可视化。

读 JSONL record=card（及可选 round 行），输出统计表 + hist / heatmap /
box_by_host / sorted_bar / 正交散点；缺字段跳过并在 md 注明。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

NUMERIC_METRICS: list[tuple[str, str]] = [
    ("func_tflops", "Cube func TFLOPS"),
    ("hbm_gbps", "HBM GB/s"),
    ("sustained_tflops", "Sustained TFLOPS"),
    ("vector_gflops", "Vector GFLOPS"),
    ("scalar_elems_per_s", "Scalar elems/s"),
    ("mte_gbps", "MTE copy GB/s"),
    ("cube_vector_tflops", "Cube+Vector TFLOPS"),
    ("sfu_gflops", "SFU GFLOPS"),
    ("launch_sync_p50_us", "Launch sync p50 (us)"),
    ("launch_sync_p99_us", "Launch sync p99 (us)"),
    ("launch_host_overhead_p50_us", "Host overhead p50 (us)"),
    ("launch_host_overhead_p99_us", "Host overhead p99 (us)"),
    ("launch_burst_p50_us", "Burst total p50 (us)"),
    ("launch_burst_per_kernel_p50_us", "Burst/kernel p50 (us)"),
    ("health_temp_c", "Health temp (C)"),
    ("health_power_w", "Health power (W)"),
    ("aicore_freq_mhz", "AICore freq (MHz)"),
    ("hbm_temp_c", "HBM temp (C)"),
    ("board_temp_c", "Board temp (C)"),
    ("aicore_util_pct", "AICore util %"),
    ("aicpu_util_pct", "AICPU util %"),
    ("ctrlcpu_util_pct", "CtrlCPU util %"),
    ("mem_bw_util_pct", "MemBW util %"),
    ("power_w", "Power (W)"),
    ("power_limit_w", "Power limit (W)"),
    ("shape_sweep_peak_tflops", "Shape sweep peak TFLOPS"),
]

METRIC_LABEL = dict(NUMERIC_METRICS)

# 正交散点：缺任一轴则跳过
SCATTER_PAIRS: list[tuple[str, str, str]] = [
    ("func_tflops", "vector_gflops", "Cube × Vector"),
    ("hbm_gbps", "mte_gbps", "HBM × MTE"),
    ("power_w", "func_tflops", "Power × Cube"),
    ("health_power_w", "func_tflops", "Health power × Cube"),
    ("power_w", "hbm_gbps", "Power × HBM"),
    ("health_power_w", "hbm_gbps", "Health power × HBM"),
    ("launch_host_overhead_p50_us", "ctrlcpu_util_pct", "Launch overhead × CtrlCPU"),
]

# heatmap / box / sorted_bar 优先画这些；其余有数据也画
CORE_FOR_LAYOUT = [
    "func_tflops",
    "hbm_gbps",
    "sustained_tflops",
    "vector_gflops",
    "mte_gbps",
    "cube_vector_tflops",
    "sfu_gflops",
    "scalar_elems_per_s",
    "power_w",
    "health_power_w",
    "health_temp_c",
    "aicore_freq_mhz",
]


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


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
        "p5": _percentile(ordered, 5),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
    }


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.4g}"
    return str(v)


def short_host(host: str) -> str:
    for prefix in ("huawei-8node-copy-", "huawei-8node-", "ascend-"):
        if host.startswith(prefix):
            return host[len(prefix):]
    return host


def relmed(value: float, median: float) -> float:
    if not median:
        return 0.0
    return (value - median) / median * 100.0


def discover_jsonl(data_dir: Path) -> list[Path]:
    patterns = [str(data_dir / "**" / "*.jsonl"), str(data_dir / "*.jsonl")]
    found: list[Path] = []
    for pat in patterns:
        found.extend(Path(p) for p in glob.glob(pat, recursive=True))
    return sorted(set(found))


def load_records(paths: list[Path]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {
        "card": [],
        "gemm_sustained_sample": [],
        "gemm_shape_sample": [],
    }
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rtype = obj.get("record")
                if rtype in out:
                    out[rtype].append(obj)
    return out


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


def available_metrics(cards: list[dict], min_n: int = 2) -> list[tuple[str, str, dict]]:
    result = []
    for key, label in NUMERIC_METRICS:
        vals = values_for(cards, key)
        st = metric_stats(vals)
        if st.get("n", 0) >= min_n:
            result.append((key, label, st))
    return result


def _try_plt():
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None, None
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt, np


def plot_hist(cards: list[dict], key: str, label: str, st: dict, fig_dir: Path, plt) -> str | None:
    vals = values_for(cards, key)
    if len(vals) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(vals, bins=min(30, max(5, len(vals) // 4)), color="#4C78A8", alpha=0.85, edgecolor="white")
    ax.axvline(st["median"], color="#E45756", linestyle="--",
               label=f"median={st['median']:.4g}")
    ax.set_title(label)
    ax.set_xlabel(key)
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    name = f"hist_{key}.png"
    fig.savefig(fig_dir / name, dpi=130)
    plt.close(fig)
    return name


def plot_heatmap_relmed(cards: list[dict], key: str, label: str, median: float,
                        fig_dir: Path, plt, np) -> str | None:
    hosts = sorted({c.get("host", "?") for c in cards})
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
            matrix[hi, di] = relmed(float(v), median)
        except (ValueError, TypeError):
            continue
    if np.all(np.isnan(matrix)):
        return None

    finite = matrix[~np.isnan(matrix)]
    lim = float(max(5.0, np.percentile(np.abs(finite), 95))) if len(finite) else 5.0

    fig, ax = plt.subplots(figsize=(max(8, len(devices) * 0.55), max(3.5, len(hosts) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(devices)))
    ax.set_xticklabels([str(d) for d in devices], fontsize=8)
    ax.set_yticks(range(len(hosts)))
    ax.set_yticklabels([short_host(h) for h in hosts], fontsize=8)
    ax.set_xlabel("device")
    ax.set_ylabel("host")
    ax.set_title(f"{label}\n相对中位数偏差 (%)")
    for i in range(len(hosts)):
        for j in range(len(devices)):
            v = matrix[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=5.5,
                        color="black" if abs(v) < lim * 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    name = f"heatmap_relmed_{key}.png"
    fig.savefig(fig_dir / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return name


def plot_box_by_host(cards: list[dict], key: str, label: str, fig_dir: Path, plt) -> str | None:
    hosts = sorted({c.get("host", "?") for c in cards})
    series = []
    labels = []
    for h in hosts:
        vals = [float(c[key]) for c in cards
                if c.get("host") == h and c.get(key) is not None]
        if vals:
            series.append(vals)
            labels.append(short_host(h))
    if len(series) < 1:
        return None
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 4.5))
    bp = ax.boxplot(series, patch_artist=True, widths=0.55,
                    boxprops=dict(facecolor="#4C78A8", alpha=0.7),
                    medianprops=dict(color="#E45756", linewidth=2))
    for i, vals in enumerate(series, start=1):
        ax.scatter([i] * len(vals), vals, alpha=0.35, s=14, color="#72B7B2", zorder=3)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(label)
    ax.set_title(f"按 host 分布 · {label}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    name = f"box_by_host_{key}.png"
    fig.savefig(fig_dir / name, dpi=130)
    plt.close(fig)
    return name


def plot_sorted_bar(cards: list[dict], key: str, label: str, median: float,
                    fig_dir: Path, plt) -> str | None:
    rows = [(c, float(c[key])) for c in cards if c.get(key) is not None]
    if len(rows) < 2:
        return None
    rows.sort(key=lambda x: x[1])
    labels = [f"{short_host(c.get('host', '?'))}:d{c.get('device')}" for c, _ in rows]
    vals = [v for _, v in rows]
    fig_w = max(10, len(vals) * 0.08)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    colors = ["#E45756" if v < median else "#4C78A8" for v in vals]
    ax.bar(range(len(vals)), vals, color=colors, width=0.85)
    ax.axhline(median, color="#F58518", linestyle="--", linewidth=1.2,
               label=f"median={median:.4g}")
    ax.set_ylabel(label)
    ax.set_title(f"排序条形 · {label}")
    step = max(1, len(vals) // 16)
    ax.set_xticks(range(0, len(vals), step))
    ax.set_xticklabels([labels[i] for i in range(0, len(vals), step)],
                       rotation=60, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    name = f"sorted_bar_{key}.png"
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)
    return name


def plot_bar_host_mean_std(cards: list[dict], key: str, label: str,
                           fig_dir: Path, plt, np) -> str | None:
    hosts = sorted({c.get("host", "?") for c in cards})
    means, stds, labels = [], [], []
    for h in hosts:
        vals = values_for([c for c in cards if c.get("host") == h], key)
        if not vals:
            continue
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
        labels.append(short_host(h))
    if len(means) < 1:
        return None
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 4.5))
    x = range(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color="#4C78A8", alpha=0.85, ecolor="#333")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(label)
    ax.set_title(f"host 均值 ± σ · {label}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    name = f"bar_host_mean_std_{key}.png"
    fig.savefig(fig_dir / name, dpi=130)
    plt.close(fig)
    return name


def plot_scatter(cards: list[dict], xkey: str, ykey: str, title: str,
                 fig_dir: Path, plt) -> str | None:
    pts = []
    for c in cards:
        xv, yv = c.get(xkey), c.get(ykey)
        if xv is None or yv is None:
            continue
        try:
            pts.append((float(xv), float(yv), short_host(c.get("host", "?"))))
        except (TypeError, ValueError):
            continue
    if len(pts) < 2:
        return None
    hosts = sorted({p[2] for p in pts})
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, h in enumerate(hosts):
        xs = [p[0] for p in pts if p[2] == h]
        ys = [p[1] for p in pts if p[2] == h]
        ax.scatter(xs, ys, s=28, alpha=0.75, color=cmap(i % 10), label=h)
    ax.set_xlabel(METRIC_LABEL.get(xkey, xkey))
    ax.set_ylabel(METRIC_LABEL.get(ykey, ykey))
    ax.set_title(title)
    if len(hosts) <= 12:
        ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    name = f"scatter_{xkey}_vs_{ykey}.png"
    fig.savefig(fig_dir / name, dpi=130)
    plt.close(fig)
    return name


def plot_box_overview(avail: list[tuple[str, str, dict]], cards: list[dict],
                      fig_dir: Path, plt) -> str | None:
    # 最多 6 个核心指标
    keys = [k for k, _, _ in avail if k in CORE_FOR_LAYOUT][:6]
    if len(keys) < 2:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 4.5))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        vals = values_for(cards, key)
        ax.boxplot([vals], widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="#4C78A8", alpha=0.7),
                   medianprops=dict(color="#E45756", linewidth=2))
        ax.scatter([1] * len(vals), vals, alpha=0.25, s=10, color="#72B7B2", zorder=3)
        ax.set_title(METRIC_LABEL.get(key, key), fontsize=9)
        ax.set_xticks([1])
        ax.set_xticklabels([f"n={len(vals)}"])
    fig.suptitle("核心指标总览箱线", fontsize=12, y=1.02)
    fig.tight_layout()
    name = "box_overview.png"
    fig.savefig(fig_dir / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return name


def plot_sustained_timeseries(samples: list[dict], cards: list[dict],
                              fig_dir: Path, plt) -> str | None:
    if not samples or not cards:
        return None
    with_sus = [c for c in cards if c.get("sustained_tflops") is not None]
    if len(with_sus) < 2:
        return None
    ordered = sorted(with_sus, key=lambda c: float(c["sustained_tflops"]))
    picks = [
        ("p05", ordered[max(0, int(len(ordered) * 0.05))]),
        ("p50", ordered[len(ordered) // 2]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (tag, card) in zip(axes, picks):
        host, dev = card["host"], card["device"]
        pts = [s for s in samples if s.get("host") == host and s.get("device") == dev]
        pts.sort(key=lambda s: s.get("t_s", 0))
        if not pts:
            ax.set_title(f"{tag} 无 sample")
            continue
        ax.plot([s.get("t_s", i) for i, s in enumerate(pts)],
                [s.get("tflops") for s in pts], color="#4C78A8", linewidth=1.2)
        ax.axhline(float(card["sustained_tflops"]), color="#E45756", linestyle="--",
                   label=f"汇总 {float(card['sustained_tflops']):.1f}")
        ax.set_title(f"sustained {tag}\n{short_host(host)}:d{dev}")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("TFLOPS")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("sustained 抽样时序（分位卡）", fontsize=12)
    fig.tight_layout()
    name = "timeseries_sustained_p05_p50.png"
    fig.savefig(fig_dir / name, dpi=130)
    plt.close(fig)
    return name


def plot_shape_curves(samples: list[dict], cards: list[dict],
                      fig_dir: Path, plt) -> str | None:
    if not samples:
        return None
    with_sus = [c for c in cards if c.get("sustained_tflops") is not None]
    if not with_sus:
        with_sus = cards
    if not with_sus:
        return None
    ordered = sorted(with_sus, key=lambda c: float(c.get("sustained_tflops") or c.get("func_tflops") or 0))
    picks = [
        ("低", ordered[0], "#E45756"),
        ("中", ordered[len(ordered) // 2], "#4C78A8"),
        ("高", ordered[-1], "#54A24B"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    drawn = 0
    for tag, card, color in picks:
        pts = [s for s in samples
               if s.get("host") == card.get("host") and s.get("device") == card.get("device")
               and s.get("tflops") is not None and s.get("n") is not None]
        if not pts:
            continue
        pts.sort(key=lambda s: s["n"])
        ax.plot([p["n"] for p in pts], [p["tflops"] for p in pts],
                marker="o", markersize=3, linewidth=1.4, color=color,
                label=f"{tag} ({short_host(card.get('host', '?'))}:d{card.get('device')})")
        drawn += 1
    if drawn == 0:
        plt.close(fig)
        return None
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (log2)")
    ax.set_ylabel("TFLOPS")
    ax.set_title("Shape Sweep: TFLOPS vs N")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    name = "shape_tflops_vs_n.png"
    fig.savefig(fig_dir / name, dpi=130)
    plt.close(fig)
    return name


def write_markdown(
    cards: list[dict],
    sources: list[Path],
    avail: list[tuple[str, str, dict]],
    skipped: list[str],
    fig_names: list[str],
    out_path: Path,
    fig_dir_name: str,
) -> None:
    lines = [
        "# Card Constitution 分布报告",
        "",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 卡数: {len(cards)}",
        f"- 数据源: {', '.join(str(p) for p in sources[:8])}"
        + (" …" if len(sources) > 8 else ""),
        "",
        "> 本报告只做分布统计与可视化，不强调 slow / 坏卡判定。",
        "",
    ]
    if skipped:
        lines += ["## 跳过说明", ""]
        for s in skipped:
            lines.append(f"- {s}")
        lines.append("")

    lines += [
        "## 指标分布",
        "",
        "| 指标 | n | median | mean | std | CV% | min | max | p5 | p50 | p95 |",
        "|------|---|--------|------|-----|-----|-----|-----|----|----|-----|",
    ]
    for key, label, st in avail:
        lines.append(
            f"| {label} | {st['n']} | {fmt(st['median'])} | {fmt(st['mean'])} | "
            f"{fmt(st['std'])} | {fmt(st['cv_pct'])} | {fmt(st['min'])} | "
            f"{fmt(st['max'])} | {fmt(st['p5'])} | {fmt(st['p50'])} | {fmt(st['p95'])} |"
        )

    lines += ["", "## 相对中位数偏差", ""]
    lines.append("偏差 = `(值 - 集群中位数) / 集群中位数 × 100%`。")
    lines.append("")
    for key, label, st in avail:
        med = st["median"]
        devs = [relmed(float(c[key]), med) for c in cards if c.get(key) is not None]
        if not devs:
            continue
        abs_mean = sum(abs(d) for d in devs) / len(devs)
        lines.append(
            f"- **{label}** (`{key}`): [{min(devs):+.2f}%, {max(devs):+.2f}%]，"
            f"|偏差|均值 {abs_mean:.2f}%"
        )

    lines += ["", "## 元数据", ""]
    hosts = sorted({c.get("host", "?") for c in cards})
    backends = sorted({str(c.get("backend", "?")) for c in cards})
    lines.append(f"- hosts ({len(hosts)}): {', '.join(short_host(h) for h in hosts[:16])}"
                 + (" …" if len(hosts) > 16 else ""))
    lines.append(f"- backends: {', '.join(backends)}")
    timing_methods = sorted({c.get("launch_timing_method") for c in cards
                             if c.get("launch_timing_method")})
    if timing_methods:
        lines.append(f"- launch_timing_method: {', '.join(str(t) for t in timing_methods)}")

    if fig_names:
        lines += ["", "## 图表", ""]
        for fn in fig_names:
            title = fn.replace(".png", "").replace("_", " ")
            lines += [f"### {title}", "", f"![{title}]({fig_dir_name}/{fn})", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(
    paths: list[Path],
    out_dir: Path,
    stamp: str | None = None,
    include_optional: bool = True,
) -> tuple[Path, Path]:
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("未找到 JSONL：请指定 --data-dir 或 --jsonl")

    records = load_records(paths)
    cards = records["card"]
    if not cards:
        raise SystemExit(f"无 record=card 行：{paths}")

    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"card_constitution_{stamp}.md"
    fig_dir = out_dir / f"card_constitution_{stamp}_figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    avail = available_metrics(cards, min_n=2)
    avail_keys = {k for k, _, _ in avail}
    skipped: list[str] = []
    for key, label in NUMERIC_METRICS:
        if key in avail_keys:
            continue
        n = len(values_for(cards, key))
        if n == 0:
            skipped.append(f"`{key}`（{label}）：字段缺失或全空，跳过")
        else:
            skipped.append(f"`{key}`（{label}）：仅 n={n}，不足 2，跳过分布图")

    fig_names: list[str] = []
    plt, np = _try_plt()
    if plt is None:
        skipped.append("matplotlib/numpy 不可用：跳过全部作图")
    else:
        # 总览
        fn = plot_box_overview(avail, cards, fig_dir, plt)
        if fn:
            fig_names.append(fn)

        layout_keys = [k for k, _, _ in avail if k in CORE_FOR_LAYOUT]
        # 其余有数据的也画 hist；heatmap/box/sorted 优先 core，其余也画
        for key, label, st in avail:
            fn = plot_hist(cards, key, label, st, fig_dir, plt)
            if fn:
                fig_names.append(fn)

        for key, label, st in avail:
            fn = plot_heatmap_relmed(cards, key, label, st["median"], fig_dir, plt, np)
            if fn:
                fig_names.append(fn)
            fn = plot_box_by_host(cards, key, label, fig_dir, plt)
            if fn:
                fig_names.append(fn)
            fn = plot_sorted_bar(cards, key, label, st["median"], fig_dir, plt)
            if fn:
                fig_names.append(fn)
            if include_optional and key in layout_keys:
                fn = plot_bar_host_mean_std(cards, key, label, fig_dir, plt, np)
                if fn:
                    fig_names.append(fn)

        for xkey, ykey, title in SCATTER_PAIRS:
            if xkey not in avail_keys or ykey not in avail_keys:
                skipped.append(
                    f"散点 `{xkey}` × `{ykey}`（{title}）：缺轴字段，跳过"
                )
                continue
            # 避免 power_w 与 health_power_w 重复画同一 y
            fn = plot_scatter(cards, xkey, ykey, title, fig_dir, plt)
            if fn:
                fig_names.append(fn)
            else:
                skipped.append(f"散点 `{xkey}` × `{ykey}`：配对点数 < 2，跳过")

        if include_optional:
            fn = plot_sustained_timeseries(
                records["gemm_sustained_sample"], cards, fig_dir, plt)
            if fn:
                fig_names.append(fn)
            else:
                skipped.append("`gemm_sustained_sample`：无可用时序，跳过 timeseries")
            fn = plot_shape_curves(records["gemm_shape_sample"], cards, fig_dir, plt)
            if fn:
                fig_names.append(fn)
            else:
                skipped.append("`gemm_shape_sample`：无可用曲线，跳过 shape")

    # 去重保持顺序
    seen = set()
    uniq_figs = []
    for fn in fig_names:
        if fn not in seen:
            seen.add(fn)
            uniq_figs.append(fn)

    write_markdown(
        cards, paths, avail, skipped, uniq_figs, report_path, fig_dir.name,
    )
    (fig_dir / "skipped.json").write_text(
        json.dumps({"skipped": skipped, "figures": uniq_figs, "n_cards": len(cards)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path, fig_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="体质筛查分布优先可视化")
    ap.add_argument("--data-dir", type=Path,
                    help="含 JSONL 的 logs/results 目录（递归搜索 *.jsonl）")
    ap.add_argument("--jsonl", type=Path, action="append", default=[],
                    help="单个 JSONL 文件，可重复")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "rounds")
    ap.add_argument("--stamp", type=str, default=None,
                    help="输出时间戳后缀；默认当前时间")
    ap.add_argument("--no-optional", action="store_true",
                    help="跳过可选图（host mean±σ / timeseries / shape）")
    args = ap.parse_args()

    paths: list[Path] = list(args.jsonl)
    if args.data_dir:
        paths.extend(discover_jsonl(args.data_dir))
    paths = sorted(set(paths))

    report_path, fig_dir = generate(
        paths,
        args.out_dir,
        stamp=args.stamp,
        include_optional=not args.no_optional,
    )
    n_figs = len(list(fig_dir.glob("*.png")))
    print(f"wrote {report_path}")
    print(f"figs  {fig_dir} ({n_figs} png)")


if __name__ == "__main__":
    main()
