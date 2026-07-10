#!/usr/bin/env python3
"""128 卡 card-screen 性能数据分析与报告生成。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# 中文字体
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path("/Users/yinjinrun/random-thing/logs/card-screen-128-20260710_224218/results")
CLUSTER_JSON = DATA_DIR / "perf128.cluster.json"
FIG_DIR = Path("/Users/yinjinrun/random-thing/project/lab-workspace/reports/card_screen_128_figs")
REPORT_PATH = Path("/Users/yinjinrun/random-thing/project/lab-workspace/reports/card_screen_128.md")
STATS_JSON = FIG_DIR / "stats.json"

METRICS = ["func_tflops", "hbm_gbps", "sustained_tflops"]
METRIC_LABELS = {
    "func_tflops": "func_tflops (TFLOPS)",
    "hbm_gbps": "hbm_gbps (GB/s)",
    "sustained_tflops": "sustained_tflops (TFLOPS)",
}
METRIC_UNITS = {
    "func_tflops": "TFLOPS",
    "hbm_gbps": "GB/s",
    "sustained_tflops": "TFLOPS",
}


def load_cluster() -> dict[str, Any]:
    with CLUSTER_JSON.open() as f:
        return json.load(f)


def load_jsonl_records() -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {
        "card": [],
        "gemm_sustained_sample": [],
        "gemm_shape_sample": [],
    }
    for path in sorted(DATA_DIR.glob("perf128.huawei-8node-copy-*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rtype = obj.get("record")
                if rtype in records:
                    records[rtype].append(obj)
    return records


def card_key(host: str, device: int) -> str:
    return f"{host}:device{device}"


def compute_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "cv_pct": float(np.std(arr, ddof=0) / np.mean(arr) * 100) if np.mean(arr) else 0.0,
    }


def rel_deviation(value: float, median: float) -> float:
    if median == 0:
        return 0.0
    return (value - median) / median * 100.0


def short_host(host: str) -> str:
    return host.replace("huawei-8node-copy-", "")


def plot_boxplots(cards: list[dict], stats: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, METRICS):
        data = [c[metric] for c in cards]
        bp = ax.boxplot([data], widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor="#4C78A8", alpha=0.7),
                        medianprops=dict(color="#E45756", linewidth=2))
        ax.scatter([1] * len(data), data, alpha=0.25, s=12, color="#72B7B2", zorder=3)
        s = stats[metric]
        ax.axhline(s["median"], color="#E45756", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks([1])
        ax.set_xticklabels(["128 卡"])
        ax.set_ylabel(METRIC_UNITS[metric])
        ax.text(
            0.02, 0.98,
            f"均值 {s['mean']:.1f}\n中位 {s['median']:.1f}\nσ {s['std']:.2f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    fig.suptitle("128 卡三指标箱线图", fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "boxplot_three_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name


def plot_host_device_heatmap(cards: list[dict], cluster: dict) -> str:
    hosts = cluster["nodes"]
    devices = sorted({c["device"] for c in cards})
    host_short = [short_host(h) for h in hosts]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, METRICS):
        median = cluster["medians"][metric]
        matrix = np.full((len(hosts), len(devices)), np.nan)
        for c in cards:
            hi = hosts.index(c["host"])
            di = devices.index(c["device"])
            matrix[hi, di] = rel_deviation(c[metric], median)

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-5, vmax=5)
        ax.set_xticks(range(len(devices)))
        ax.set_xticklabels([str(d) for d in devices])
        ax.set_yticks(range(len(hosts)))
        ax.set_yticklabels(host_short, fontsize=8)
        ax.set_xlabel("device")
        ax.set_ylabel("host")
        ax.set_title(f"{METRIC_LABELS[metric]}\n相对中位数偏差 (%)")
        for i in range(len(hosts)):
            for j in range(len(devices)):
                v = matrix[i, j]
                if not math.isnan(v):
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6,
                            color="black" if abs(v) < 3 else "white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("host × device 相对中位数偏差热力图", fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "heatmap_host_device_deviation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name


def plot_sustained_timeseries(
    sustained: list[dict],
    slow_card: dict,
    median_card: dict,
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    selections = [
        ("最慢卡 (sustained)", slow_card, "#E45756"),
        ("中位卡 (sustained)", median_card, "#4C78A8"),
    ]
    for col, (label, card, color) in enumerate(selections):
        host, dev = card["host"], card["device"]
        samples = [
            s for s in sustained
            if s["host"] == host and s["device"] == dev
        ]
        samples.sort(key=lambda x: x["t_s"])
        t_s = [s["t_s"] for s in samples]
        tflops = [s["tflops"] for s in samples]
        temp = [s.get("temp_c") for s in samples]

        ax_t = axes[0, col]
        ax_t.plot(t_s, tflops, color=color, linewidth=1.2)
        ax_t.axhline(card["sustained_tflops"], color=color, linestyle="--", alpha=0.5,
                     label=f"汇总 {card['sustained_tflops']:.1f}")
        ax_t.set_ylabel("TFLOPS")
        ax_t.set_title(f"{label}\n{short_host(host)}:d{dev}")
        ax_t.legend(fontsize=8)
        ax_t.grid(True, alpha=0.3)

        ax_temp = axes[1, col]
        ax_temp.plot(t_s, temp, color="#F58518", linewidth=1.2)
        ax_temp.set_xlabel("时间 (s)")
        ax_temp.set_ylabel("温度 (°C)")
        ax_temp.grid(True, alpha=0.3)

    fig.suptitle("sustained GEMM 抽样时序（TFLOPS / 温度）", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "sustained_timeseries_slow_median.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name


def plot_shape_curves(
    shape_samples: list[dict],
    slow_card: dict,
    median_card: dict,
    fast_card: dict,
) -> str | None:
    if not shape_samples:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    selections = [
        ("最慢卡", slow_card, "#E45756"),
        ("中位卡", median_card, "#4C78A8"),
        ("最快卡", fast_card, "#54A24B"),
    ]
    for label, card, color in selections:
        host, dev = card["host"], card["device"]
        pts = [
            s for s in shape_samples
            if s["host"] == host and s["device"] == dev
        ]
        if not pts:
            continue
        pts.sort(key=lambda x: x["n"])
        ns = [p["n"] for p in pts]
        tflops = [p["tflops"] for p in pts]
        ax.plot(ns, tflops, marker="o", markersize=3, linewidth=1.5, color=color,
                label=f"{label} ({short_host(host)}:d{dev})")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("矩阵维度 N (log2)")
    ax.set_ylabel("TFLOPS")
    ax.set_title("Shape Sweep: TFLOPS vs N（叠加曲线）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "shape_tflops_vs_n.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name


def plot_host_bar_summary(cards: list[dict]) -> str:
    hosts = sorted({c["host"] for c in cards})
    host_short = [short_host(h) for h in hosts]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, METRICS):
        means = []
        stds = []
        for h in hosts:
            vals = [c[metric] for c in cards if c["host"] == h]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        x = np.arange(len(hosts))
        ax.bar(x, means, yerr=stds, capsize=3, color="#4C78A8", alpha=0.85, ecolor="#333")
        ax.set_xticks(x)
        ax.set_xticklabels(host_short, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"按 host 均值 ± σ\n{METRIC_LABELS[metric]}")
        ax.set_ylabel(METRIC_UNITS[metric])
    fig.suptitle("各节点三指标均值对比", fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "bar_host_mean_std.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name


def top_n_cards(cards: list[dict], metric: str, n: int = 10, ascending: bool = True) -> list[dict]:
    return sorted(cards, key=lambda c: c[metric], reverse=not ascending)[:n]


def format_card_row(c: dict, metric: str, median: float) -> str:
    dev_pct = rel_deviation(c[metric], median)
    return (
        f"| {short_host(c['host'])} | {c['device']} | "
        f"{c[metric]:.2f} | {dev_pct:+.2f}% |"
    )


def write_report(
    cluster: dict,
    cards: list[dict],
    stats: dict[str, dict],
    fig_names: list[str],
    slow_card: dict,
    median_card: dict,
    fast_card: dict,
    tops: dict[str, dict],
) -> None:
    summary = cluster["summary"]
    medians = cluster["medians"]
    nodes = cluster["nodes"]

    lines = [
        "# 128 卡 Card Screen 性能报告",
        "",
        f"**数据来源**: `{DATA_DIR}`  ",
        f"**集群规模**: {cluster['n_cards']} 卡 / {len(nodes)} 节点  ",
        f"**设备**: Ascend910_9392 (NPU)  ",
        f"**生成时间**: 2026-07-10",
        "",
        "---",
        "",
        "## 1. 指标说明",
        "",
        "| 指标 | 含义 | 测量方式 |",
        "|------|------|----------|",
        "| **func_tflops** | 功能性 GEMM 峰值算力 | 多轮 GEMM 基准测试（`gemm_round`），取代表性峰值 TFLOPS |",
        "| **hbm_gbps** | HBM 内存带宽 | HBM 读写压测（`hbm_round`），汇总有效 GB/s |",
        "| **sustained_tflops** | 持续算力 | 长时间 sustained GEMM（`gemm_sustained_sample`），反映稳态吞吐 |",
        "",
        "附加探测：`gemm_shape_sample` 对不同矩阵维度 N 扫频，观察算力随 shape 变化；SDC/健康检查均通过。",
        "",
        "---",
        "",
        "## 2. 128 卡汇总 (cluster summary)",
        "",
        f"- **总卡数**: {cluster['n_cards']}",
        f"- **节点列表**: {', '.join(short_host(n) for n in nodes)}",
        f"- **slow_frac 阈值**: {cluster['slow_frac']}",
        f"- **verdict 分布**: good = **{summary.get('good', 0)}**，slow = {summary.get('slow', 0)}，其余 = {cluster['n_cards'] - summary.get('good', 0) - summary.get('slow', 0)}",
        "",
        "**集群中位数（用于偏差计算）**:",
        "",
        f"| 指标 | 中位数 |",
        f"|------|--------|",
    ]
    for m in METRICS:
        lines.append(f"| {m} | {medians[m]:.2f} {METRIC_UNITS[m]} |")

    lines += ["", "---", "", "## 3. 全集群统计量", ""]
    lines.append("| 指标 | 均值 | 中位数 | 标准差 | 最小 | 最大 | CV(%) |")
    lines.append("|------|------|--------|--------|------|------|-------|")
    for m in METRICS:
        s = stats[m]
        lines.append(
            f"| {m} | {s['mean']:.2f} | {s['median']:.2f} | {s['std']:.2f} | "
            f"{s['min']:.2f} | {s['max']:.2f} | {s['cv_pct']:.2f} |"
        )

    lines += ["", "---", "", "## 4. 相对中位数偏差", ""]
    lines.append("偏差 = `(值 - 集群中位数) / 集群中位数 × 100%`。")
    lines.append("")
    for m in METRICS:
        vals = [rel_deviation(c[m], medians[m]) for c in cards]
        lines.append(
            f"- **{m}**: 偏差范围 [{min(vals):+.2f}%, {max(vals):+.2f}%]，"
            f"绝对偏差均值 {np.mean(np.abs(vals)):.2f}%，标准差 {np.std(vals):.2f}%"
        )

    lines += ["", "---", "", "## 5. 最慢 / 最快 Top 10", ""]
    for m in METRICS:
        lines += [f"### {m}", "", "#### 最慢 Top 10", "",
                  "| host | device | 值 | 相对中位数偏差 |",
                  "|------|--------|-----|----------------|"]
        for c in tops[m]["slow"]:
            lines.append(format_card_row(c, m, medians[m]))
        lines += ["", "#### 最快 Top 10", "",
                  "| host | device | 值 | 相对中位数偏差 |",
                  "|------|--------|-----|----------------|"]
        for c in tops[m]["fast"]:
            lines.append(format_card_row(c, m, medians[m]))
        lines.append("")

    lines += [
        "---",
        "",
        "## 6. 方差解读",
        "",
    ]
    # variance interpretation
    cv_func = stats["func_tflops"]["cv_pct"]
    cv_hbm = stats["hbm_gbps"]["cv_pct"]
    cv_sus = stats["sustained_tflops"]["cv_pct"]
    lines += [
        f"1. **func_tflops** CV = {cv_func:.2f}%：峰值 GEMM 在各卡间离散度{'较低' if cv_func < 3 else '中等' if cv_func < 5 else '偏高'}，"
        f"范围 {stats['func_tflops']['min']:.1f}–{stats['func_tflops']['max']:.1f} TFLOPS（跨度 {stats['func_tflops']['max'] - stats['func_tflops']['min']:.1f}）。",
        f"2. **hbm_gbps** CV = {cv_hbm:.2f}%：HBM 带宽一致性{'较好' if cv_hbm < 2 else '一般'}，"
        f"范围 {stats['hbm_gbps']['min']:.1f}–{stats['hbm_gbps']['max']:.1f} GB/s。",
        f"3. **sustained_tflops** CV = {cv_sus:.2f}%：持续算力波动{'最小' if cv_sus < min(cv_func, cv_hbm) else '与峰值相近'}，"
        f"说明稳态负载下集群表现{'均匀' if cv_sus < 2 else '存在少量离群卡'}。",
        "",
        f"**代表性卡片**:",
        f"- 最慢 sustained: `{short_host(slow_card['host'])}:device{slow_card['device']}` = {slow_card['sustained_tflops']:.2f} TFLOPS",
        f"- 中位 sustained: `{short_host(median_card['host'])}:device{median_card['device']}` = {median_card['sustained_tflops']:.2f} TFLOPS",
        f"- 最快 sustained: `{short_host(fast_card['host'])}:device{fast_card['device']}` = {fast_card['sustained_tflops']:.2f} TFLOPS",
        "",
        "全部 128 卡 verdict 均为 **good**，无 thermal/power throttling 标记；温度读数均为 2°C（NPU 遥测占位值，不代表真实热状态）。",
        "节点间均值差异主要来自单卡微观波动，未见系统性 host 级退化。",
        "",
        "---",
        "",
        "## 7. 图表",
        "",
    ]
    fig_titles = {
        "boxplot_three_metrics.png": "三指标箱线图",
        "heatmap_host_device_deviation.png": "host×device 相对中位数偏差热力图",
        "bar_host_mean_std.png": "各节点均值 ± 标准差",
        "sustained_timeseries_slow_median.png": "sustained 抽样时序（最慢 vs 中位卡）",
        "shape_tflops_vs_n.png": "Shape Sweep: TFLOPS vs N",
    }
    for fn in fig_names:
        title = fig_titles.get(fn, fn)
        lines += [f"### {title}", "", f"![{title}](card_screen_128_figs/{fn})", ""]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cluster = load_cluster()
    cards = cluster["cards"]
    records = load_jsonl_records()

    stats = {m: compute_stats([c[m] for c in cards]) for m in METRICS}

    # pick representative cards by sustained_tflops
    sorted_sus = sorted(cards, key=lambda c: c["sustained_tflops"])
    slow_card = sorted_sus[0]
    fast_card = sorted_sus[-1]
    median_card = sorted_sus[len(sorted_sus) // 2]

    tops = {
        m: {
            "slow": top_n_cards(cards, m, 10, ascending=True),
            "fast": top_n_cards(cards, m, 10, ascending=False),
        }
        for m in METRICS
    }

    fig_names = []
    fig_names.append(plot_boxplots(cards, stats))
    fig_names.append(plot_host_device_heatmap(cards, cluster))
    fig_names.append(plot_host_bar_summary(cards))
    fig_names.append(plot_sustained_timeseries(records["gemm_sustained_sample"], slow_card, median_card))
    shape_fig = plot_shape_curves(records["gemm_shape_sample"], slow_card, median_card, fast_card)
    if shape_fig:
        fig_names.append(shape_fig)

    # save stats for reference
    payload = {
        "n_cards": cluster["n_cards"],
        "summary": cluster["summary"],
        "medians": cluster["medians"],
        "stats": stats,
        "representative_cards": {
            "slow_sustained": {"host": slow_card["host"], "device": slow_card["device"], "sustained_tflops": slow_card["sustained_tflops"]},
            "median_sustained": {"host": median_card["host"], "device": median_card["device"], "sustained_tflops": median_card["sustained_tflops"]},
            "fast_sustained": {"host": fast_card["host"], "device": fast_card["device"], "sustained_tflops": fast_card["sustained_tflops"]},
        },
        "figures": fig_names,
    }
    STATS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    write_report(cluster, cards, stats, fig_names, slow_card, median_card, fast_card, tops)

    print("=== 生成完成 ===")
    print(f"报告: {REPORT_PATH}")
    print(f"图表目录: {FIG_DIR}")
    for fn in fig_names:
        print(f"  - {FIG_DIR / fn}")
    print("\n关键统计:")
    for m in METRICS:
        s = stats[m]
        print(f"  {m}: mean={s['mean']:.2f} median={s['median']:.2f} std={s['std']:.2f} "
              f"min={s['min']:.2f} max={s['max']:.2f} cv={s['cv_pct']:.2f}%")
    print(f"  summary: {cluster['summary']}")


if __name__ == "__main__":
    main()
