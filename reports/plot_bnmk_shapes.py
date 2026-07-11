#!/usr/bin/env python3
"""BNMK GEMM shape 可视化：读 gemm_bnmk_sample JSONL，输出 TFLOPS 分布图与中文报告。"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROUND_STAMP = "20260711"
FIG_DIR = Path(__file__).resolve().parent / "rounds" / f"bnmk_shapes_{ROUND_STAMP}_figs"
MD_PATH = Path(__file__).resolve().parent / "rounds" / f"bnmk_shapes_{ROUND_STAMP}.md"

LOG_ROOT = Path("/Users/yinjinrun/random-thing/logs")
FILLGAP_GLOB = "card-fillgap-*/results"
CONSTITUTION_GLOB = "card-constitution-128-*-constitution128/results"


def short_host(h: str) -> str:
    if not h:
        return "?"
    parts = h.split(".")
    return parts[0] if parts else h


def shape_label(row: dict[str, Any]) -> str:
    if row.get("label"):
        return str(row["label"])
    b = int(row.get("B", 1))
    m = int(row["M"])
    n = int(row["N"])
    k = int(row["K"])
    return f"B{b}_M{m}_N{n}_K{k}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_bnmk_samples(paths: list[Path]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for p in paths:
        for row in load_jsonl(p):
            if row.get("record") == "gemm_bnmk_sample":
                samples.append(row)
            elif row.get("record") == "card" and row.get("shape_sweep_peak_tflops") is not None:
                # 兼容 card 行上的 shape 汇总字段（非 BNMK 明细）
                continue
    return samples


def discover_jsonl_dirs() -> list[Path]:
    """按优先级：fillgap results > constitution merged 目录。"""
    candidates: list[Path] = []

    fillgap_dirs = sorted(LOG_ROOT.glob(FILLGAP_GLOB), reverse=True)
    for d in fillgap_dirs:
        if d.is_dir():
            candidates.append(d)

    const_dirs = sorted(LOG_ROOT.glob(CONSTITUTION_GLOB), reverse=True)
    for d in const_dirs:
        if d.is_dir():
            candidates.append(d)

    return candidates


def resolve_data_dir(data_dir: str | None) -> tuple[Path, list[Path]]:
    if data_dir:
        root = Path(data_dir).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"--data-dir 不存在：{root}")
        jsonls = sorted(root.glob("**/*.jsonl"))
        return root, jsonls

    for d in discover_jsonl_dirs():
        jsonls = sorted(d.glob("**/*.jsonl"))
        if jsonls:
            return d, jsonls
    raise SystemExit("未找到 JSONL 数据目录，请指定 --data-dir")


def setup_mpl():
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.sans-serif": [
            "PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS",
            "SimHei", "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })
    return plt, np


def style_axes(ax) -> None:
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)


def save_fig(fig, name: str, plt) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


def plot_box_by_label(samples: list[dict], plt, np) -> str | None:
    by_label: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        t = s.get("tflops")
        if t is None:
            continue
        by_label[shape_label(s)].append(float(t))
    labels = sorted(by_label.keys(), key=lambda lb: statistics.median(by_label[lb]))
    if not labels:
        return None

    data = [by_label[lb] for lb in labels]
    fig_w = max(12, len(labels) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C78A8")
        patch.set_alpha(0.65)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("TFLOPS")
    ax.set_title("BNMK Shape · TFLOPS 箱线图（按 label）")
    style_axes(ax)
    fig.tight_layout()
    return save_fig(fig, "bnmk_tflops_box_by_label.png", plt)


def plot_bar_median_by_label(samples: list[dict], plt, np) -> str | None:
    by_label: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        t = s.get("tflops")
        if t is None:
            continue
        by_label[shape_label(s)].append(float(t))
    labels = sorted(by_label.keys(), key=lambda lb: statistics.median(by_label[lb]))
    if not labels:
        return None

    medians = [statistics.median(by_label[lb]) for lb in labels]
    fig_w = max(12, len(labels) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    x = np.arange(len(labels))
    bars = ax.bar(x, medians, color="#54A24B", alpha=0.85, edgecolor="white")
    for bar, m in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{m:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("TFLOPS（中位数）")
    ax.set_title("BNMK Shape · 各 label 中位 TFLOPS")
    style_axes(ax)
    fig.tight_layout()
    return save_fig(fig, "bnmk_tflops_bar_median_by_label.png", plt)


def plot_host_heatmap(samples: list[dict], plt, np) -> str | None:
    hosts = sorted({s.get("host", "?") for s in samples})
    labels = sorted({shape_label(s) for s in samples})
    if not hosts or not labels:
        return None

    matrix = np.full((len(hosts), len(labels)), np.nan)
    counts = np.zeros((len(hosts), len(labels)), dtype=int)
    for s in samples:
        t = s.get("tflops")
        if t is None:
            continue
        hi = hosts.index(s.get("host", "?"))
        li = labels.index(shape_label(s))
        if math.isnan(matrix[hi, li]):
            matrix[hi, li] = float(t)
        else:
            matrix[hi, li] = (matrix[hi, li] * counts[hi, li] + float(t)) / (counts[hi, li] + 1)
        counts[hi, li] += 1

    if np.all(np.isnan(matrix)):
        return None

    fig, ax = plt.subplots(figsize=(max(14, len(labels) * 0.45), max(5, len(hosts) * 0.55)))
    vmin = float(np.nanpercentile(matrix, 5))
    vmax = float(np.nanpercentile(matrix, 95))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(hosts)))
    ax.set_yticklabels([short_host(h) for h in hosts], fontsize=8)
    ax.set_xlabel("Shape label")
    ax.set_ylabel("Host")
    ax.set_title("BNMK · Host × Shape TFLOPS 热力图")
    for i in range(len(hosts)):
        for j in range(len(labels)):
            v = matrix[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if v > (vmin + vmax) / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="TFLOPS")
    fig.tight_layout()
    return save_fig(fig, "bnmk_host_shape_heatmap.png", plt)


def build_stats(samples: list[dict]) -> list[dict[str, Any]]:
    by_label: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        t = s.get("tflops")
        if t is not None:
            by_label[shape_label(s)].append(float(t))
    rows = []
    for lb in sorted(by_label.keys()):
        vals = sorted(by_label[lb])
        rows.append({
            "label": lb,
            "n": len(vals),
            "median": statistics.median(vals),
            "mean": statistics.mean(vals),
            "min": vals[0],
            "max": vals[-1],
        })
    return rows


def write_empty_md(data_dir: Path, jsonl_count: int) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = "\n".join([
        f"# BNMK Shape 报告 · {ROUND_STAMP}",
        "",
        f"> 生成时间：{ts}",
        f"> 数据目录：`{data_dir}`",
        "",
        "## 状态",
        "",
        f"已扫描 **{jsonl_count}** 个 JSONL 文件，未发现 `record=gemm_bnmk_sample` 行。",
        "",
        "可能原因：",
        "- fillgap / constitution128 批次尚未启用 `gemm_bnmk_sweep` 探针；",
        "- 数据仍在采集或合流中。",
        "",
        "待有数据后重新运行：",
        "",
        "```bash",
        "python reports/plot_bnmk_shapes.py --data-dir <含 jsonl 的目录>",
        "```",
        "",
    ]) + "\n"
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text(md, encoding="utf-8")


def write_md(samples: list[dict], data_dir: Path, figs: list[str], stats: list[dict]) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    hosts = sorted({s.get("host", "?") for s in samples})
    lines = [
        f"# BNMK Shape 报告 · {ROUND_STAMP}",
        "",
        f"> 生成时间：{ts}",
        f"> 数据目录：`{data_dir}`",
        "",
        "## 摘要",
        "",
        f"- 样本数：**{len(samples)}**",
        f"- Shape 种类：**{len(stats)}**",
        f"- 节点数：**{len(hosts)}**",
        "",
        "## 各 Shape TFLOPS 统计",
        "",
        "| Label | N | 中位数 | 均值 | 最小 | 最大 |",
        "|-------|---|--------|------|------|------|",
    ]
    for st in stats:
        lines.append(
            f"| {st['label']} | {st['n']} | {st['median']:.2f} | {st['mean']:.2f} | "
            f"{st['min']:.2f} | {st['max']:.2f} |"
        )

    lines += ["", "## 图表", ""]
    fig_titles = {
        "bnmk_tflops_box_by_label.png": "TFLOPS 箱线图（按 label）",
        "bnmk_tflops_bar_median_by_label.png": "各 label 中位 TFLOPS 柱状图",
        "bnmk_host_shape_heatmap.png": "Host × Shape 热力图",
    }
    for fn in figs:
        title = fig_titles.get(fn, fn)
        lines += [f"### {title}", "", f"![{title}](bnmk_shapes_{ROUND_STAMP}_figs/{fn})", ""]

    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="BNMK GEMM shape TFLOPS 可视化")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="含 *.jsonl 的数据目录（默认自动探测 fillgap / constitution merged）",
    )
    args = parser.parse_args()

    data_dir, jsonl_paths = resolve_data_dir(args.data_dir)
    print(f"数据目录: {data_dir}")
    print(f"JSONL 文件: {len(jsonl_paths)}")

    samples = load_bnmk_samples(jsonl_paths)
    if not samples:
        print(f"[提示] 未发现 gemm_bnmk_sample（0 样本），已写入空报告 -> {MD_PATH}")
        write_empty_md(data_dir, len(jsonl_paths))
        return

    plt, np = setup_mpl()
    figs: list[str] = []
    for plot_fn in (plot_box_by_label, plot_bar_median_by_label, plot_host_heatmap):
        name = plot_fn(samples, plt, np)
        if name:
            figs.append(name)

    stats = build_stats(samples)
    write_md(samples, data_dir, figs, stats)

    print(f"样本数: {len(samples)} · shapes: {len(stats)}")
    print(f"图表 ({len(figs)}) -> {FIG_DIR}")
    print(f"报告 -> {MD_PATH}")


if __name__ == "__main__":
    main()
