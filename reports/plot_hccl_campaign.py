#!/usr/bin/env python3
"""HCCL 通信战役 · 可视化与 Markdown 报告。

读取 pipeline-comm 产出的 HCCL collective / P2P JSONL，生成中文标注 PNG 与 md 报告。
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ── 固定路径 ──────────────────────────────────────────────────────────────
LOG_ROOT = Path("/Users/yinjinrun/random-thing/logs/pipeline-comm-20260711_134811")
HCCL_DIR = LOG_ROOT / "hccl-results"
P2P_DIR = LOG_ROOT / "p2p-results"
TOPO_RAW_DIR = LOG_ROOT / "hccl-topo" / "raw"

FIG_DIR = Path(__file__).resolve().parent / "rounds" / "hccl_campaign_20260711_figs"
MD_PATH = Path(__file__).resolve().parent / "rounds" / "hccl_campaign_20260711.md"

WORLD_SCALES = [16, 32, 64, 128]
P2P_SCALES = [16, 128]
OPS = ["all_reduce", "all_gather", "reduce_scatter", "broadcast"]
OP_CN = {
    "all_reduce": "All-Reduce",
    "all_gather": "All-Gather",
    "reduce_scatter": "Reduce-Scatter",
    "broadcast": "Broadcast",
}
LARGE_NBYTES = 268435456  # 256 MB
SIZE_LABELS = {
    1048576: "1 MB",
    16777216: "16 MB",
    67108864: "64 MB",
    268435456: "256 MB",
}
WORLD_COLORS = {16: "#4C78A8", 32: "#F58518", 64: "#54A24B", 128: "#E45756"}
P2P_SIZE_CN = {65536: "64 KB", 16777216: "16 MB"}


def fmt_bytes(n: int) -> str:
    return SIZE_LABELS.get(n, f"{n / (1024**2):.0f} MB")


def fmt_bw(v: float) -> str:
    return f"{v:.2f}"


def fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
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


def median(vals: list[float]) -> float:
    return statistics.median(vals) if vals else float("nan")


def mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else float("nan")


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
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    return plt, np


def style_axes(ax) -> None:
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)


def save_fig(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    return name


def load_hccl() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ws in WORLD_SCALES:
        rows.extend(load_jsonl(HCCL_DIR / f"scale_{ws}.jsonl"))
    return rows


def load_p2p() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ws in P2P_SCALES:
        rows.extend(load_jsonl(P2P_DIR / f"scale_{ws}.jsonl"))
    return rows


def hccl_agg(rows: list[dict]) -> dict[tuple, list[float]]:
    """(op, world_size, nbytes) -> [bus_bw_GBps per rank]."""
    bucket: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("record") != "hccl_bench":
            continue
        key = (r["op"], int(r["world_size"]), int(r["nbytes"]))
        bucket[key].append(float(r["bus_bw_GBps"]))
    return bucket


def hccl_rank_series(rows: list[dict], op: str, ws: int, nbytes: int) -> list[float]:
    vals: list[float] = []
    for r in rows:
        if (
            r.get("record") == "hccl_bench"
            and r["op"] == op
            and int(r["world_size"]) == ws
            and int(r["nbytes"]) == nbytes
        ):
            vals.append(float(r["bus_bw_GBps"]))
    return vals


def p2p_edge_key(r: dict) -> tuple[int, int, int, int]:
    """去重边：(world, src, dst, nbytes)。"""
    return (int(r["world_size"]), int(r["src"]), int(r["dst"]), int(r["nbytes"]))


def p2p_edge_kind(src: int, dst: int, world: int) -> str:
    if src == 0 or dst == 0:
        return "星型(经 rank0)"
    if abs(src - dst) == 1:
        return "环相邻"
    if (src == world - 1 and dst == 0) or (src == 0 and dst == world - 1):
        return "环闭合"
    # 跨节点块（每 16 rank 一块）
    if src // 16 != dst // 16:
        return "跨节点块"
    return "其他"


def dedupe_p2p_edges(rows: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in rows:
        if r.get("record") != "hccl_p2p":
            continue
        k = p2p_edge_key(r)
        if k in seen:
            continue
        seen.add(k)
        rr = dict(r)
        rr["edge_kind"] = p2p_edge_kind(int(r["src"]), int(r["dst"]), int(r["world_size"]))
        rr["edge_label"] = f"{r['src']}→{r['dst']}"
        out.append(rr)
    return out


def parse_hccs_matrix(raw_text: str) -> list[list[str]] | None:
    """从 npu-smi topo raw 文本解析 NPU×NPU 矩阵。"""
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    matrix: list[list[str]] = []
    for ln in lines:
        if ln.startswith("NPU") and ("HCCS" in ln or "PIX" in ln or "PHB" in ln or "SYS" in ln):
            parts = ln.split()
            if len(parts) >= 2:
                matrix.append(parts[1:])
        elif "\t" in ln or "  " in ln:
            parts = ln.replace("\t", " ").split()
            if len(parts) >= 2 and any(
                tok in ("HCCS", "PIX", "PHB", "SYS", "X") for tok in parts[1:]
            ):
                matrix.append(parts[1:])
    if len(matrix) < 2:
        return None
    return matrix


def find_topo_raw() -> Path | None:
    if not TOPO_RAW_DIR.is_dir():
        return None
    candidates = sorted(TOPO_RAW_DIR.glob("*.raw.txt"))
    if not candidates:
        return None
    for p in candidates:
        if "master-0" in p.name or "master_0" in p.name:
            return p
    return candidates[0]


def plot_hccl_curves(rows: list[dict], agg: dict, plt, np) -> list[str]:
    figs: list[str] = []
    sizes = sorted({k[2] for k in agg})
    x_labels = [fmt_bytes(s) for s in sizes]
    x_pos = np.arange(len(sizes))

    for op in OPS:
        fig, ax = plt.subplots(figsize=(10, 6))
        for ws in WORLD_SCALES:
            ys = []
            for sz in sizes:
                vals = agg.get((op, ws, sz), [])
                ys.append(median(vals))
            ax.plot(
                x_pos, ys, marker="o", linewidth=2.2, markersize=7,
                color=WORLD_COLORS[ws], label=f"world={ws}",
            )
        ax.set_xticks(x_pos, x_labels)
        ax.set_xlabel("消息大小")
        ax.set_ylabel("Bus 带宽 (GB/s)")
        ax.set_title(f"{OP_CN[op]} · Bus 带宽 vs 消息大小")
        style_axes(ax)
        ax.legend(title="规模", loc="best")
        fig.tight_layout()
        name = f"hccl_bus_bw_vs_size_{op}.png"
        save_fig(fig, name)
        plt.close(fig)
        figs.append(name)
    return figs


def compute_256mb_means(rows: list[dict]) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for op in OPS:
        for ws in WORLD_SCALES:
            vals = hccl_rank_series(rows, op, ws, LARGE_NBYTES)
            out[(op, ws)] = mean(vals)
    return out


def plot_256mb_step_and_retention(means: dict, plt, np) -> tuple[list[str], dict[str, dict[int, float]]]:
    figs: list[str] = []
    retention: dict[str, dict[int, float]] = {op: {} for op in OPS}
    baseline = {op: means[(op, 16)] for op in OPS}
    for op in OPS:
        for ws in WORLD_SCALES:
            b = baseline[op]
            retention[op][ws] = (means[(op, ws)] / b * 100.0) if b else float("nan")

    # 阶梯图
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(WORLD_SCALES))
    for op in OPS:
        ys = [means[(op, ws)] for ws in WORLD_SCALES]
        ax.plot(x, ys, marker="s", linewidth=2.2, markersize=8, label=OP_CN[op])
    ax.set_xticks(x, [str(w) for w in WORLD_SCALES])
    ax.set_xlabel("World 规模")
    ax.set_ylabel("Bus 带宽 (GB/s)")
    ax.set_title("256 MB 大消息 · 四算子 Bus 带宽随 World 规模")
    style_axes(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    name = "hccl_256mb_step_bus_bw.png"
    save_fig(fig, name)
    plt.close(fig)
    figs.append(name)

    # 分算子阶梯图（更清晰）
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ax, op in zip(axes.flat, OPS):
        ys = [means[(op, ws)] for ws in WORLD_SCALES]
        ax.step(x, ys, where="mid", color=WORLD_COLORS[128], linewidth=2.5)
        ax.scatter(x, ys, color=WORLD_COLORS[16], s=60, zorder=3)
        for i, (w, y) in enumerate(zip(WORLD_SCALES, ys)):
            ax.annotate(fmt_bw(y), (i, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9)
        ax.set_title(OP_CN[op])
        ax.set_ylabel("Bus 带宽 (GB/s)")
        style_axes(ax)
    for ax in axes[1]:
        ax.set_xticks(x, [str(w) for w in WORLD_SCALES])
        ax.set_xlabel("World 规模")
    fig.suptitle("256 MB · 各算子 Bus 带宽阶梯图", fontsize=15, y=1.01)
    fig.tight_layout()
    name = "hccl_256mb_step_per_op.png"
    save_fig(fig, name)
    plt.close(fig)
    figs.append(name)

    # 保持率柱状图
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.18
    ops_x = np.arange(len(OPS))
    for i, ws in enumerate(WORLD_SCALES):
        vals = [retention[op][ws] for op in OPS]
        offset = (i - 1.5) * width
        bars = ax.bar(ops_x + offset, vals, width, label=f"world={ws}",
                      color=WORLD_COLORS[ws], alpha=0.9)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        fmt_pct(v), ha="center", va="bottom", fontsize=8, rotation=0)
    ax.axhline(100, color="#666", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xticks(ops_x, [OP_CN[op] for op in OPS])
    ax.set_ylabel("保持率 (相对 world=16, %)")
    ax.set_title("256 MB · Bus 带宽保持率（基准 world=16 = 100%）")
    ax.set_ylim(0, max(110, max(retention[op][ws] for op in OPS for ws in WORLD_SCALES) + 8))
    style_axes(ax)
    ax.legend(title="规模", ncol=4, loc="upper right")
    fig.tight_layout()
    name = "hccl_256mb_retention_bar.png"
    save_fig(fig, name)
    plt.close(fig)
    figs.append(name)

    return figs, retention


def plot_rank_distribution(rows: list[dict], plt, np) -> list[str]:
    figs: list[str] = []

    # 256MB violin：每算子一张，x=world
    for op in OPS:
        fig, ax = plt.subplots(figsize=(10, 6))
        data = []
        positions = []
        labels = []
        for i, ws in enumerate(WORLD_SCALES):
            vals = hccl_rank_series(rows, op, ws, LARGE_NBYTES)
            if vals:
                data.append(vals)
                positions.append(i)
                labels.append(str(ws))
        if data:
            parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor("#4C78A8")
                pc.set_alpha(0.65)
        ax.set_xticks(positions, labels)
        ax.set_xlabel("World 规模")
        ax.set_ylabel("Bus 带宽 (GB/s)")
        ax.set_title(f"{OP_CN[op]} · 256 MB Rank 分布（Violin）")
        style_axes(ax)
        fig.tight_layout()
        name = f"hccl_rank_violin_256mb_{op}.png"
        save_fig(fig, name)
        plt.close(fig)
        figs.append(name)

    # 256MB box：四算子合图
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, op in zip(axes.flat, OPS):
        data = [hccl_rank_series(rows, op, ws, LARGE_NBYTES) for ws in WORLD_SCALES]
        bp = ax.boxplot(data, tick_labels=[str(w) for w in WORLD_SCALES], patch_artist=True)
        for patch, ws in zip(bp["boxes"], WORLD_SCALES):
            patch.set_facecolor(WORLD_COLORS[ws])
            patch.set_alpha(0.55)
        ax.set_xlabel("World 规模")
        ax.set_ylabel("Bus 带宽 (GB/s)")
        ax.set_title(OP_CN[op])
        style_axes(ax)
    fig.suptitle("256 MB · Rank Bus 带宽箱线图", fontsize=15, y=1.01)
    fig.tight_layout()
    name = "hccl_rank_box_256mb_all_ops.png"
    save_fig(fig, name)
    plt.close(fig)
    figs.append(name)

    # 各 world × 算子 histogram
    for ws in WORLD_SCALES:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        for ax, op in zip(axes.flat, OPS):
            vals = hccl_rank_series(rows, op, ws, LARGE_NBYTES)
            if vals:
                ax.hist(vals, bins=min(12, max(4, len(vals))), color=WORLD_COLORS[ws],
                        alpha=0.8, edgecolor="white")
                ax.axvline(median(vals), color="#E45756", linestyle="--",
                           label=f"中位数={fmt_bw(median(vals))}")
                ax.legend(fontsize=9)
            ax.set_xlabel("Bus 带宽 (GB/s)")
            ax.set_ylabel("Rank 计数")
            ax.set_title(OP_CN[op])
            style_axes(ax)
        fig.suptitle(f"world={ws} · 256 MB Rank 直方图", fontsize=15, y=1.01)
        fig.tight_layout()
        name = f"hccl_rank_hist_w{ws}_256mb.png"
        save_fig(fig, name)
        plt.close(fig)
        figs.append(name)

    return figs


def plot_p2p(edges: list[dict], plt, np) -> list[str]:
    figs: list[str] = []
    if not edges:
        return figs

    # 按消息大小 · 边类型 · world 的 violin
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, ws in zip(axes, P2P_SCALES):
        sub = [e for e in edges if int(e["world_size"]) == ws]
        kinds = sorted({e["edge_kind"] for e in sub})
        sizes = sorted({int(e["nbytes"]) for e in sub})
        for sz in sizes:
            data = []
            labels = []
            for k in kinds:
                vals = [float(e["bw_GBps"]) for e in sub
                        if e["edge_kind"] == k and int(e["nbytes"]) == sz]
                if vals:
                    data.append(vals)
                    labels.append(k)
            if data:
                pos = np.arange(len(labels))
                parts = ax.violinplot(data, positions=pos, showmeans=True)
                for pc in parts["bodies"]:
                    pc.set_alpha(0.7)
                ax.set_xticks(pos, labels, rotation=20, ha="right")
                ax.set_title(f"world={ws} · {P2P_SIZE_CN.get(sz, str(sz))}")
        ax.set_ylabel("带宽 (GB/s)")
        style_axes(ax)
    fig.suptitle("P2P · 边类型带宽分布", fontsize=14, y=1.02)
    fig.tight_layout()
    name = "p2p_bw_violin_by_kind_size.png"
    save_fig(fig, name)
    plt.close(fig)
    figs.append(name)

    # 按消息大小汇总 box（world 分面）
    for sz in sorted({int(e["nbytes"]) for e in edges}):
        fig, ax = plt.subplots(figsize=(8, 6))
        data = []
        labels = []
        for ws in P2P_SCALES:
            vals = [float(e["bw_GBps"]) for e in edges
                    if int(e["world_size"]) == ws and int(e["nbytes"]) == sz]
            if vals:
                data.append(vals)
                labels.append(f"world={ws}")
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            for patch, ws in zip(bp["boxes"], P2P_SCALES[: len(data)]):
                patch.set_facecolor(WORLD_COLORS.get(ws, "#888"))
                patch.set_alpha(0.6)
            ax.set_ylabel("带宽 (GB/s)")
            ax.set_title(f"P2P · {P2P_SIZE_CN.get(sz, str(sz))} 带宽分布 (16 vs 128)")
            style_axes(ax)
            fig.tight_layout()
            name = f"p2p_box_compare_w16_w128_{sz}.png"
            save_fig(fig, name)
            plt.close(fig)
            figs.append(name)

    # 慢边 TopK（按 16MB 大消息）
    big = [e for e in edges if int(e["nbytes"]) == 16777216]
    if big:
        big_sorted = sorted(big, key=lambda e: float(e["bw_GBps"]))
        topk = big_sorted[: min(15, len(big_sorted))]
        fig, ax = plt.subplots(figsize=(11, 7))
        labels = [f"w{e['world_size']} {e['edge_label']} ({e['edge_kind']})" for e in topk]
        vals = [float(e["bw_GBps"]) for e in topk]
        y = np.arange(len(topk))
        ax.barh(y, vals, color="#E45756", alpha=0.85)
        ax.set_yticks(y, labels, fontsize=9)
        ax.set_xlabel("带宽 (GB/s)")
        ax.set_title("P2P 慢边 Top-15（16 MB 消息，带宽升序）")
        style_axes(ax)
        ax.invert_yaxis()
        fig.tight_layout()
        name = "p2p_slow_edges_top15_16mb.png"
        save_fig(fig, name)
        plt.close(fig)
        figs.append(name)

        # 快边 TopK 对照
        fast = big_sorted[-min(15, len(big_sorted)):]
        fig, ax = plt.subplots(figsize=(11, 7))
        labels = [f"w{e['world_size']} {e['edge_label']} ({e['edge_kind']})" for e in fast]
        vals = [float(e["bw_GBps"]) for e in fast]
        y = np.arange(len(fast))
        ax.barh(y, vals, color="#54A24B", alpha=0.85)
        ax.set_yticks(y, labels, fontsize=9)
        ax.set_xlabel("带宽 (GB/s)")
        ax.set_title("P2P 快边 Top-15（16 MB 消息，带宽降序）")
        style_axes(ax)
        ax.invert_yaxis()
        fig.tight_layout()
        name = "p2p_fast_edges_top15_16mb.png"
        save_fig(fig, name)
        plt.close(fig)
        figs.append(name)

    # 16 vs 128 同边类型对比柱状（16MB）
    fig, ax = plt.subplots(figsize=(10, 6))
    kinds = sorted({e["edge_kind"] for e in edges if int(e["nbytes"]) == 16777216})
    x = np.arange(len(kinds))
    width = 0.35
    for i, ws in enumerate(P2P_SCALES):
        vals = []
        for k in kinds:
            ev = [float(e["bw_GBps"]) for e in edges
                  if e["edge_kind"] == k and int(e["world_size"]) == ws
                  and int(e["nbytes"]) == 16777216]
            vals.append(mean(ev) if ev else 0)
        ax.bar(x + (i - 0.5) * width, vals, width, label=f"world={ws}",
               color=WORLD_COLORS[ws], alpha=0.85)
    ax.set_xticks(x, kinds, rotation=15, ha="right")
    ax.set_ylabel("平均带宽 (GB/s)")
    ax.set_title("P2P · 16 MB 边类型均值对比 (world 16 vs 128)")
    style_axes(ax)
    ax.legend()
    fig.tight_layout()
    name = "p2p_kind_mean_compare_16mb.png"
    save_fig(fig, name)
    plt.close(fig)
    figs.append(name)

    return figs


def affinity_to_num(val: str) -> float:
    m = {"HCCS": 4.0, "PIX": 3.0, "PHB": 2.0, "SYS": 1.0, "X": 0.0}
    return m.get(val.strip(), 0.5)


def plot_topo_heatmap(raw_path: Path, plt, np) -> str | None:
    text = raw_path.read_text(encoding="utf-8", errors="replace")
    matrix = parse_hccs_matrix(text)
    if not matrix:
        return None
    n = len(matrix)
    data = np.zeros((n, n))
    for i, row in enumerate(matrix):
        for j, cell in enumerate(row[:n]):
            data[i, j] = affinity_to_num(cell)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=4)
    ax.set_xticks(range(n), [f"NPU{j}" for j in range(n)], rotation=45, ha="right")
    ax.set_yticks(range(n), [f"NPU{i}" for i in range(n)])
    for i in range(n):
        for j in range(n):
            if i < len(matrix) and j < len(matrix[i]):
                ax.text(j, i, matrix[i][j], ha="center", va="center", fontsize=7,
                        color="black" if data[i, j] < 2.5 else "white")
    ax.set_title(f"机内 HCCS 拓扑热力图 · {raw_path.name}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("亲和等级 (HCCS>PIX>PHB>SYS)")
    fig.tight_layout()
    name = "topo_hccs_heatmap_master0.png"
    save_fig(fig, name)
    plt.close(fig)
    return name


def build_md(
    all_figs: list[str],
    retention: dict[str, dict[int, float]],
    means_256: dict[tuple[str, int], float],
    hccl_rows: list[dict],
    p2p_edges: list[dict],
    topo_note: str,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# HCCL 通信战役报告 · 20260711",
        "",
        f"> 生成时间：{ts}  ",
        f"> 数据源：`{LOG_ROOT}`",
        "",
        "## 摘要",
        "",
        "本报告汇总 All-Reduce / All-Gather / Reduce-Scatter / Broadcast 四算子在 "
        "world=16/32/64/128 下的 Bus 带宽曲线，256 MB 大消息的扩展性阶梯图与保持率，"
        "Rank 级分布，以及 P2P 边级带宽对比。",
        "",
        "### 256 MB 保持率（相对 world=16）",
        "",
        "| 算子 | w=16 | w=32 | w=64 | w=128 |",
        "|------|------|------|------|-------|",
    ]
    for op in OPS:
        cells = [OP_CN[op]]
        for ws in WORLD_SCALES:
            cells.append(fmt_pct(retention[op][ws]))
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "### 256 MB 平均 Bus 带宽 (GB/s)",
        "",
        "| 算子 | w=16 | w=32 | w=64 | w=128 |",
        "|------|------|------|------|-------|",
    ]
    for op in OPS:
        cells = [OP_CN[op]]
        for ws in WORLD_SCALES:
            cells.append(fmt_bw(means_256[(op, ws)]))
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        f"- HCCL 记录数：{len(hccl_rows)}",
        f"- P2P 去重边数：{len(p2p_edges)}",
        f"- 拓扑：{topo_note}",
        "",
        "## 1. Collective · Bus 带宽 vs 消息大小",
        "",
    ]
    for op in OPS:
        fn = f"hccl_bus_bw_vs_size_{op}.png"
        if fn in all_figs:
            lines += [f"### {OP_CN[op]}", "", f"![{OP_CN[op]}](hccl_campaign_20260711_figs/{fn})", ""]

    lines += [
        "## 2. 256 MB 大消息扩展性",
        "",
        "![阶梯图](hccl_campaign_20260711_figs/hccl_256mb_step_bus_bw.png)",
        "",
        "![分算子阶梯](hccl_campaign_20260711_figs/hccl_256mb_step_per_op.png)",
        "",
        "![保持率](hccl_campaign_20260711_figs/hccl_256mb_retention_bar.png)",
        "",
        "## 3. Rank 分布（256 MB）",
        "",
    ]
    for op in OPS:
        fn = f"hccl_rank_violin_256mb_{op}.png"
        if fn in all_figs:
            lines += [f"### {OP_CN[op]}", "", f"![violin](hccl_campaign_20260711_figs/{fn})", ""]
    lines += [
        "![箱线图汇总](hccl_campaign_20260711_figs/hccl_rank_box_256mb_all_ops.png)",
        "",
    ]
    for ws in WORLD_SCALES:
        fn = f"hccl_rank_hist_w{ws}_256mb.png"
        if fn in all_figs:
            lines += [f"### world={ws}", "", f"![hist w{ws}](hccl_campaign_20260711_figs/{fn})", ""]

    lines += ["## 4. P2P", ""]
    for fn in all_figs:
        if fn.startswith("p2p_"):
            lines += [f"![{fn}](hccl_campaign_20260711_figs/{fn})", ""]

    if any(f.startswith("topo_") for f in all_figs):
        lines += ["## 5. 机内拓扑", "", "![HCCS](hccl_campaign_20260711_figs/topo_hccs_heatmap_master0.png)", ""]

    lines += [
        "## 附录 · 图文件清单",
        "",
    ]
    for fn in sorted(all_figs):
        lines.append(f"- `{fn}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    plt, np = setup_mpl()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    hccl_rows = load_hccl()
    p2p_rows = load_p2p()
    p2p_edges = dedupe_p2p_edges(p2p_rows)
    agg = hccl_agg(hccl_rows)

    all_figs: list[str] = []
    all_figs.extend(plot_hccl_curves(hccl_rows, agg, plt, np))

    means_256 = compute_256mb_means(hccl_rows)
    step_figs, retention = plot_256mb_step_and_retention(means_256, plt, np)
    all_figs.extend(step_figs)

    all_figs.extend(plot_rank_distribution(hccl_rows, plt, np))
    all_figs.extend(plot_p2p(p2p_edges, plt, np))

    topo_note = "未找到 hccl-topo raw 文件，跳过拓扑热力图"
    topo_raw = find_topo_raw()
    if topo_raw:
        topo_fig = plot_topo_heatmap(topo_raw, plt, np)
        if topo_fig:
            all_figs.append(topo_fig)
            topo_note = f"已解析 `{topo_raw.name}`"
        else:
            topo_note = f"`{topo_raw.name}` 解析失败"

    md = build_md(all_figs, retention, means_256, hccl_rows, p2p_edges, topo_note)
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text(md, encoding="utf-8")

    print(f"Wrote {len(all_figs)} figures -> {FIG_DIR}")
    print(f"Report -> {MD_PATH}")
    print("\n256 MB retention (vs world=16):")
    for op in OPS:
        parts = [f"{OP_CN[op]}:"]
        for ws in WORLD_SCALES:
            parts.append(f"w{ws}={fmt_pct(retention[op][ws])}")
        print("  " + "  ".join(parts))


if __name__ == "__main__":
    main()
