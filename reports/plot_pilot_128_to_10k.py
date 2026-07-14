#!/usr/bin/env python3
"""为《128 卡初筛：万卡训练预期》生成主线图。

只画正文需要的五张新图；HCCL、HBM 热图与 MFU 图复用现有产物。
所有输出均为 SVG，并使用 reports/plot_style.py。
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from plot_style import COLORS, apply_plot_style, natural_host_key, save_fig, short_host_label, style_axes


LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
ROUNDS = LAB_ROOT / "reports" / "rounds"
OUT_DIR = ROUNDS / "pilot_128_to_10k_figs"

ASCEND_JSONL = REPO_ROOT / "logs/card-fillgap-20260711_140301/results/constitution128.merged.jsonl"
MUXI_JSONL = (
    REPO_ROOT
    / "logs/muxi-constitution-20260711_232400-muxi-constitution128/results/constitution128.merged.jsonl"
)
ASCEND_PUBLISH = ROUNDS / "ascend_publish_20260713"


STABLE_METRICS = [
    ("func_tflops", "方阵 GEMM"),
    ("sustained_tflops", "持续 GEMM"),
    ("shape_sweep_peak_tflops", "BNMK 峰值"),
    ("hbm_gbps", "HBM"),
    ("mte_gbps", "纯搬运"),
]

LAUNCH_METRICS = [
    ("launch_sync_p50_us", "空 sync"),
    ("launch_host_overhead_p50_us", "Host 发射"),
    ("launch_burst_per_kernel_p50_us", "Burst 每核"),
]


def load_cards(path: Path) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("record") == "card":
                cards.append(obj)
    return cards


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def numeric_values(cards: list[dict[str, Any]], key: str) -> list[float]:
    return [float(c[key]) for c in cards if isinstance(c.get(key), (int, float))]


def plot_zero_event_upper_bound() -> Path:
    """零事件时二项比例的单侧 95% 上界：1-alpha^(1/n)。"""
    apply_plot_style((10, 5.8))
    fig, ax = plt.subplots()
    ns = list(range(32, 10001))
    upper_pct = [(1.0 - 0.05 ** (1.0 / n)) * 100.0 for n in ns]
    ax.plot(ns, upper_pct, color=COLORS[0], linewidth=2.6, label="零事件时 95% 单侧上界")
    marks = [128, 512, 2048, 3000, 10000]
    for idx, n in enumerate(marks):
        y = (1.0 - 0.05 ** (1.0 / n)) * 100.0
        ax.scatter([n], [y], color=COLORS[(idx + 1) % len(COLORS)], s=58, zorder=3)
        ax.annotate(
            f"{n} 卡\n{y:.2f}%",
            (n, y),
            xytext=(7, 8 if idx % 2 == 0 else -28),
            textcoords="offset points",
            fontsize=13,
        )
    ax.axhline(0.1, color=COLORS[2], linestyle="--", linewidth=1.8, label="0.1% 目标上界")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("完成同口径筛查且未检出事件的卡数")
    ax.set_ylabel("真实事件率 95% 单侧上界（%）")
    ax.set_title("128 卡零事件仍不足以约束万卡低概率风险")
    ax.legend(loc="upper right")
    style_axes(ax)
    return save_fig(fig, OUT_DIR / "sample_zero_event_upper_bound.svg")


def plot_stable_metric_distribution(
    ascend_cards: list[dict[str, Any]], muxi_cards: list[dict[str, Any]]
) -> Path:
    """各侧内部归一化，不比较两侧绝对峰值。"""
    apply_plot_style((15, 6.2))
    fig, axes = plt.subplots(1, 2, sharey=True)
    muxi_non_bad = [c for c in muxi_cards if c.get("verdict") != "bad"]
    for ax, cards, title in [
        (axes[0], ascend_cards, "昇腾 Ascend910（128 卡）"),
        (axes[1], muxi_non_bad, "沐曦 C550（127 个非 bad 记录）"),
    ]:
        data: list[list[float]] = []
        labels: list[str] = []
        for key, label in STABLE_METRICS:
            vals = numeric_values(cards, key)
            if not vals:
                continue
            med = statistics.median(vals)
            data.append([(v / med - 1.0) * 100.0 for v in vals])
            labels.append(label)
        bp = ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True, widths=0.62)
        for idx, box in enumerate(bp["boxes"]):
            box.set(facecolor="none", edgecolor=COLORS[idx % len(COLORS)], linewidth=2.0)
        for median in bp["medians"]:
            median.set(color="#333333", linewidth=2.0)
        for whisker in bp["whiskers"]:
            whisker.set(color="#777777", linewidth=1.2)
        for cap in bp["caps"]:
            cap.set(color="#777777", linewidth=1.2)
        for flier in bp["fliers"]:
            flier.set(marker="o", markersize=3.5, alpha=0.55)
        ax.axhline(0.0, color="#555555", linewidth=1.2)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        style_axes(ax)
    axes[0].set_ylabel("相对本侧该指标中位数偏差（%）")
    fig.suptitle("稳定性能主体与异常尾部：先看形态，不做跨厂商绝对值排名", y=1.02)
    return save_fig(fig, OUT_DIR / "stable_metric_relative_distribution.svg")


def plot_failslow_measurement_chain() -> Path:
    apply_plot_style((17, 5.8))
    indep = read_csv(ASCEND_PUBLISH / "gap_indep.csv")
    real = read_csv(ASCEND_PUBLISH / "gap_real_gbsprop.csv")
    network = read_csv(ASCEND_PUBLISH / "network_contrib_final.csv")

    fig, axes = plt.subplots(1, 3)

    x1 = [int(r["world_npu"]) for r in indep]
    y1 = [float(r["gap_median_ms"]) for r in indep]
    axes[0].plot(x1, y1, marker="o", linewidth=2.4, color=COLORS[0])
    axes[0].set_title("A. 设备独立执行尾部")
    axes[0].set_xlabel("卡数")
    axes[0].set_ylabel("max−median gap（ms）")
    style_axes(axes[0])

    x2 = [int(r["world_npu"]) for r in real]
    y2 = [float(r["gap_median_ms"]) for r in real]
    axes[1].plot(x2, y2, marker="o", linewidth=2.4, color=COLORS[1])
    axes[1].set_title("B. 真实 Dense 同步后可见 gap")
    axes[1].set_xlabel("卡数")
    axes[1].set_ylabel("max−median gap（ms）")
    style_axes(axes[1])

    network_valid = [r for r in network if r["network_contrib_ms"]]
    x3 = [int(r["world_npu"]) for r in network_valid]
    y3 = [float(r["network_contrib_ms"]) for r in network_valid]
    axes[2].plot(x3, y3, marker="o", linewidth=2.4, color=COLORS[2])
    axes[2].axvline(32, color="#666666", linestyle="--", linewidth=1.4)
    axes[2].annotate("首次跨节点", (32, y3[x3.index(32)]), xytext=(8, 16), textcoords="offset points", fontsize=13)
    axes[2].set_title("C. 两工作负载 gap 残差")
    axes[2].set_xlabel("卡数")
    axes[2].set_ylabel("gap_real−gap_indep（ms）")
    style_axes(axes[2])

    fig.suptitle("Fail-slow 测量链：独立尾部、同步可见离散与启发式残差", y=1.02)
    return save_fig(fig, OUT_DIR / "failslow_measurement_chain.svg")


def plot_sync_injection_amplification() -> Path:
    apply_plot_style((12.5, 5.8))
    data = json.loads((ASCEND_PUBLISH / "exp3_delayed_iter_analysis.json").read_text(encoding="utf-8"))
    labels = ["正常迭代", "延迟注入迭代"]
    gap = [data["gap_median_normal_ms"], data["gap_median_delayed_ms"]]
    step_s = [data["step_median_normal_ms"] / 1000.0, data["step_median_delayed_ms"] / 1000.0]

    fig, axes = plt.subplots(1, 2)
    for idx, val in enumerate(gap):
        axes[0].bar(
            idx,
            val,
            width=0.55,
            color="none",
            edgecolor=COLORS[idx],
            hatch=["//", "xx"][idx],
            linewidth=2.0,
        )
        axes[0].text(idx, val + 0.025, f"{val:.2f} ms", ha="center", va="bottom", fontsize=15)
    axes[0].set_xticks([0, 1], labels)
    axes[0].set_ylabel("rank max−median gap（ms）")
    axes[0].set_title("表面 rank gap 几乎不变")
    style_axes(axes[0])

    for idx, val in enumerate(step_s):
        axes[1].bar(
            idx,
            val,
            width=0.55,
            color="none",
            edgecolor=COLORS[idx],
            hatch=["//", "xx"][idx],
            linewidth=2.0,
        )
        axes[1].text(idx, val + 0.15, f"{val:.2f} s", ha="center", va="bottom", fontsize=15)
    axes[1].annotate(
        f"+{data['step_lift_ms'] / 1000.0:.2f} s",
        xy=(1, step_s[1]),
        xytext=(-38, 30),
        textcoords="offset points",
        fontsize=16,
        color=COLORS[2],
        arrowprops={"arrowstyle": "->", "color": COLORS[2]},
    )
    axes[1].set_xticks([0, 1], labels)
    axes[1].set_ylabel("全局 step 中位（s）")
    axes[1].set_title("同步把局部慢吸收到全局 step")
    style_axes(axes[1])

    fig.suptitle("16 卡延迟注入：只看 rank gap 会漏掉真实 fail-slow 损失", y=1.02)
    return save_fig(fig, OUT_DIR / "sync_injection_amplification.svg")


def host_relative_profiles(cards: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[float]]]:
    hosts = sorted({str(c["host"]) for c in cards}, key=natural_host_key)
    profiles: dict[str, list[float]] = {}
    for key, _label in LAUNCH_METRICS:
        global_vals = numeric_values(cards, key)
        global_median = statistics.median(global_vals)
        rels: list[float] = []
        for host in hosts:
            vals = [
                float(c[key])
                for c in cards
                if c.get("host") == host and isinstance(c.get(key), (int, float))
            ]
            rels.append(statistics.median(vals) / global_median if vals else math.nan)
        profiles[key] = rels
    return hosts, profiles


def plot_launch_host_relative(
    ascend_cards: list[dict[str, Any]], muxi_cards: list[dict[str, Any]]
) -> Path:
    apply_plot_style((17, 7.0))
    fig, axes = plt.subplots(2, 1, sharey=False)
    for ax, cards, title in [
        (axes[0], ascend_cards, "昇腾 Ascend910：各 host 中位 / 本侧全卡中位"),
        (axes[1], muxi_cards, "沐曦 C550：各 host 中位 / 本侧全卡中位"),
    ]:
        hosts, profiles = host_relative_profiles(cards)
        x = list(range(len(hosts)))
        for idx, (key, label) in enumerate(LAUNCH_METRICS):
            ax.plot(
                x,
                profiles[key],
                marker="o",
                linewidth=2.0,
                color=COLORS[idx],
                label=label,
            )
        ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2)
        ax.set_xticks(x, [short_host_label(h) for h in hosts])
        ax.set_ylabel("相对本侧中位（×）")
        ax.set_title(title)
        ax.legend(ncol=3, loc="upper left")
        style_axes(ax)
    axes[1].set_xlabel("Host")
    fig.suptitle("Launch 是运行时尾部信号：只比较各侧内部 host 偏离，不比较跨栈绝对微秒", y=1.01)
    return save_fig(fig, OUT_DIR / "launch_host_relative_profile.svg")


def main() -> None:
    ascend_cards = load_cards(ASCEND_JSONL)
    muxi_cards = load_cards(MUXI_JSONL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_zero_event_upper_bound(),
        plot_stable_metric_distribution(ascend_cards, muxi_cards),
        plot_failslow_measurement_chain(),
        plot_sync_injection_amplification(),
        plot_launch_host_relative(ascend_cards, muxi_cards),
    ]
    print(f"昇腾 card 记录：{len(ascend_cards)}；沐曦 card 记录：{len(muxi_cards)}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
