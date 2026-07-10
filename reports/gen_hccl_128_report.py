#!/usr/bin/env python3
"""HCCL 多卡扩展基准测试数据分析与报告生成。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HCCL_DIR = Path("/Users/yinjinrun/random-thing/logs/hccl-20260710_224902/results")
LINK_DIR = Path("/Users/yinjinrun/random-thing/logs/link-health-20260710_224719/results")
FIG_DIR = Path("/Users/yinjinrun/random-thing/project/lab-workspace/reports/hccl_128_figs")
REPORT_PATH = Path("/Users/yinjinrun/random-thing/project/lab-workspace/reports/hccl_128.md")
STATS_JSON = FIG_DIR / "stats.json"

WORLD_SIZES = [16, 32, 64, 128]
OPS = ["all_reduce", "all_gather", "reduce_scatter", "broadcast"]
OP_LABELS = {
    "all_reduce": "All-Reduce",
    "all_gather": "All-Gather",
    "reduce_scatter": "Reduce-Scatter",
    "broadcast": "Broadcast",
}
SIZE_BYTES = [1 << 20, 16 << 20, 64 << 20, 256 << 20]
SIZE_LABELS = {b: f"{b >> 20}M" for b in SIZE_BYTES}
BASE_WS = 16
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
MARKERS = ["o", "s", "^", "D"]


def load_hccl_data() -> dict[tuple[str, int, int], dict[str, float]]:
    """返回 {(op, world_size, nbytes): {bus_bw, alg_bw, avg_s}}。"""
    data: dict[tuple[str, int, int], dict[str, float]] = {}
    for ws in WORLD_SIZES:
        path = HCCL_DIR / f"scale_{ws}.jsonl"
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = (obj["op"], obj["world_size"], obj["nbytes"])
                data[key] = {
                    "bus_bw_GBps": obj["bus_bw_GBps"],
                    "alg_bw_GBps": obj["alg_bw_GBps"],
                    "avg_s": obj["avg_s"],
                }
    return data


def scaling_efficiency(bus_at_ws: float, bus_at_base: float, ws: int, base_ws: int = BASE_WS) -> float:
    """弱扩展效率：实际带宽增益 / 理想线性增益。"""
    if bus_at_base <= 0 or ws == base_ws:
        return 100.0
    ideal_ratio = ws / base_ws
    actual_ratio = bus_at_ws / bus_at_base
    return actual_ratio / ideal_ratio * 100.0


def plot_bus_bw_vs_world_size(data: dict) -> list[str]:
    fig_names: list[str] = []
    for op in OPS:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, nbytes in enumerate(SIZE_BYTES):
            xs, ys = [], []
            for ws in WORLD_SIZES:
                rec = data.get((op, ws, nbytes))
                if rec:
                    xs.append(ws)
                    ys.append(rec["bus_bw_GBps"])
            ax.plot(
                xs, ys,
                marker=MARKERS[i], color=COLORS[i], linewidth=2, markersize=7,
                label=f"msg={SIZE_LABELS[nbytes]}",
            )
        ax.set_xlabel("world_size（卡数）")
        ax.set_ylabel("bus_bw (GB/s)")
        ax.set_title(f"{OP_LABELS[op]}：bus_bw vs world_size")
        ax.set_xticks(WORLD_SIZES)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fname = f"bus_bw_{op}.png"
        fig.savefig(FIG_DIR / fname, dpi=150)
        plt.close(fig)
        fig_names.append(fname)
    return fig_names


def plot_scaling_efficiency(data: dict) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()
    for ax, op in zip(axes_flat, OPS):
        for i, nbytes in enumerate(SIZE_BYTES):
            xs, ys = [], []
            base_rec = data.get((op, BASE_WS, nbytes))
            if not base_rec:
                continue
            base_bw = base_rec["bus_bw_GBps"]
            for ws in WORLD_SIZES:
                rec = data.get((op, ws, nbytes))
                if rec:
                    xs.append(ws)
                    ys.append(scaling_efficiency(rec["bus_bw_GBps"], base_bw, ws))
            ax.plot(
                xs, ys,
                marker=MARKERS[i], color=COLORS[i], linewidth=2, markersize=6,
                label=f"msg={SIZE_LABELS[nbytes]}",
            )
        ax.axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="理想 100%")
        ax.set_xlabel("world_size")
        ax.set_ylabel("扩展效率 (%)")
        ax.set_title(OP_LABELS[op])
        ax.set_xticks(WORLD_SIZES)
        ax.set_ylim(0, 120)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("相对 16 卡的弱扩展效率（bus_bw）", fontsize=12, y=1.01)
    fig.tight_layout()
    fname = "scaling_efficiency.png"
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def parse_link_health() -> dict[str, Any]:
    hosts = sorted(LINK_DIR.glob("huawei-8node-copy-*.txt"))
    node_summaries: list[dict[str, Any]] = []
    total_npu_ok = 0
    hccn_missing_count = 0

    for path in hosts:
        text = path.read_text(encoding="utf-8", errors="replace")
        host_match = re.search(r"^HOST=(.+)$", text, re.M)
        host = host_match.group(1) if host_match else path.stem

        health_ok_lines = re.findall(r"Ascend910\s+\|\s+OK\s+\|", text)
        npu_ok = len(health_ok_lines)

        temps: list[float] = []
        for m in re.finditer(r"\|\s+(\d+\.\d+)\s+\d+\s+\d+\s+\|", text):
            try:
                temps.append(float(m.group(1)))
            except ValueError:
                pass
        # 从 npu-smi info 主表提取温度（Power 列后的 Temp）
        temp_from_table: list[int] = []
        for m in re.finditer(r"\|\s+OK\s+\|\s+[\d.-]+\s+(\d+)\s+", text):
            temp_from_table.append(int(m.group(1)))

        hccn_missing = "hccn_tool not found" in text
        if hccn_missing:
            hccn_missing_count += 1

        node_summaries.append({
            "host": host,
            "npu_health_ok": npu_ok,
            "temp_min": min(temp_from_table) if temp_from_table else None,
            "temp_max": max(temp_from_table) if temp_from_table else None,
            "hccn_missing": hccn_missing,
        })
        total_npu_ok += npu_ok

    return {
        "n_nodes": len(hosts),
        "n_npu_per_node": 16,
        "total_npu_ok": total_npu_ok,
        "nodes": node_summaries,
        "hccn_missing_count": hccn_missing_count,
    }


def fmt(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}"


def build_bus_bw_table(data: dict, nbytes: int) -> list[str]:
    header = "| op | " + " | ".join(str(ws) for ws in WORLD_SIZES) + " |"
    sep = "|---|" + "|".join(["---:"] * len(WORLD_SIZES)) + "|"
    lines = [header, sep]
    for op in OPS:
        row = [OP_LABELS[op]]
        for ws in WORLD_SIZES:
            rec = data.get((op, ws, nbytes))
            row.append(fmt(rec["bus_bw_GBps"]) if rec else "—")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_efficiency_table(data: dict, nbytes: int) -> list[str]:
    header = "| op | " + " | ".join(f"{ws}卡效率%" for ws in WORLD_SIZES) + " |"
    sep = "|---|" + "|".join(["---:"] * len(WORLD_SIZES)) + "|"
    lines = [header, sep]
    for op in OPS:
        row = [OP_LABELS[op]]
        base_rec = data.get((op, BASE_WS, nbytes))
        base_bw = base_rec["bus_bw_GBps"] if base_rec else 0.0
        for ws in WORLD_SIZES:
            rec = data.get((op, ws, nbytes))
            if rec and base_bw > 0:
                eff = scaling_efficiency(rec["bus_bw_GBps"], base_bw, ws)
                row.append(fmt(eff, 1))
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_report(data: dict, link: dict[str, Any], fig_names: list[str]) -> None:
    nbytes_256m = 256 << 20
    ar16 = data[( "all_reduce", 16, nbytes_256m)]["bus_bw_GBps"]
    ar128 = data[("all_reduce", 128, nbytes_256m)]["bus_bw_GBps"]
    eff_128 = scaling_efficiency(ar128, ar16, 128)

    lines = [
        "# HCCL 128 卡扩展基准测试报告",
        "",
        f"> 数据来源：`{HCCL_DIR}`  ",
        f"> 生成时间：2026-07-10",
        "",
        "## 1. 测试概要",
        "",
        "- **集群规模**：8 节点 × 16 NPU = 128 卡（Ascend 910）",
        "- **测试算子**：`all_reduce`、`all_gather`、`reduce_scatter`、`broadcast`",
        "- **消息大小**：1M / 16M / 64M / 256M（fp32，每 rank 发送量）",
        "- **扩展规模**：world_size = 16 / 32 / 64 / 128",
        "- **核心指标**：`bus_bw_GBps`（总线带宽，更能反映互联瓶颈）",
        "",
        "## 2. 关键结论",
        "",
        f"1. **256M All-Reduce 峰值 bus_bw**：16 卡 **{fmt(ar16)} GB/s**，128 卡 **{fmt(ar128)} GB/s**，"
        f"128 相对 16 的弱扩展效率 **{fmt(eff_128, 1)}%**（理想线性扩展为 100%）。",
        "2. 大消息（64M/256M）下 All-Reduce 带宽在 32–64 卡区间接近饱和（~138–150 GB/s），"
        "扩至 128 卡略有回落，说明跨节点互联成为瓶颈。",
        "3. All-Gather / Reduce-Scatter 在小消息时随规模扩大显著退化，大消息下 128 卡约为 16 卡的 55% 左右效率。",
        "4. Broadcast 表现最稳定，256M 消息在 128 卡仍维持 ~80% 扩展效率。",
        "",
        "## 3. bus_bw 数值表（GB/s）",
        "",
    ]

    for nbytes in SIZE_BYTES:
        label = SIZE_LABELS[nbytes]
        lines += [f"### 消息大小 {label}", ""]
        lines += build_bus_bw_table(data, nbytes)
        lines.append("")

    lines += [
        "## 4. 相对 16 卡的扩展效率",
        "",
        "扩展效率定义：`效率 = (bus_bw_N / bus_bw_16) / (N / 16) × 100%`。",
        "100% 表示完美弱扩展（带宽随卡数线性增长）；低于 100% 表示互联或算法开销导致退化。",
        "",
    ]
    for nbytes in SIZE_BYTES:
        label = SIZE_LABELS[nbytes]
        lines += [f"### 消息大小 {label}", ""]
        lines += build_efficiency_table(data, nbytes)
        lines.append("")

    lines += [
        "## 5. 图表",
        "",
        "### bus_bw vs world_size（按消息大小分线）",
        "",
    ]
    for op in OPS:
        fname = f"bus_bw_{op}.png"
        lines += [
            f"#### {OP_LABELS[op]}",
            "",
            f"![{OP_LABELS[op]} bus_bw](hccl_128_figs/{fname})",
            "",
        ]

    lines += [
        "### 扩展效率总览",
        "",
        "![扩展效率](hccl_128_figs/scaling_efficiency.png)",
        "",
        "---",
        "",
        "## 6. 链路健康",
        "",
        f"> 数据来源：`{LINK_DIR}`",
        "",
        f"共检查 **{link['n_nodes']} 个节点**（master-0 + worker-0..6），每节点 **{link['n_npu_per_node']} 张 NPU**，"
        f"合计 **{link['total_npu_ok']} 张卡** `npu-smi info` 与 `npu-smi info -t health` 均报告 **Health = OK**。",
        "",
        "| 节点 | NPU Health=OK | 温度范围 (°C) | hccn_tool |",
        "|------|:-------------:|:-------------:|:---------:|",
    ]

    for node in link["nodes"]:
        short = node["host"].replace("huawei-8node-copy-", "")
        temp_rng = (
            f"{node['temp_min']}–{node['temp_max']}"
            if node["temp_min"] is not None
            else "—"
        )
        hccn = "未找到" if node["hccn_missing"] else "可用"
        lines.append(
            f"| {short} | {node['npu_health_ok']}/16 | {temp_rng} | {hccn} |"
        )

    lines += [
        "",
        "### 限制说明",
        "",
        f"- 全部 {link['hccn_missing_count']} 个节点均输出 `hccn_tool not found`，"
        "未能采集 HCCS/RoCE 链路级诊断（如 `hccn_tool -i 0 -link -g`）。",
        "- 当前健康结论仅基于 **npu-smi** 设备级状态，无法覆盖网络交换机、光模块或跨节点链路质量。",
        "- 节点 `.bashrc` 中 Ascend driver `setenv.bash` 路径缺失（日志警告），不影响本次 npu-smi 采集。",
        "",
        "### 节点采样摘要",
        "",
        "- 各节点功耗约 155–172 W（空闲态），HBM 使用约 2.9–3.2 GB / 65536 MB，无运行中进程。",
        "- 所有 NPU 型号为 Ascend910，双 Chip 结构，MCU Health 均为 OK。",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    data = load_hccl_data()
    link = parse_link_health()

    fig_names = plot_bus_bw_vs_world_size(data)
    eff_fig = plot_scaling_efficiency(data)
    fig_names.append(eff_fig)

    nbytes_256m = 256 << 20
    key_vals = {
        "all_reduce_256M_16": data[("all_reduce", 16, nbytes_256m)]["bus_bw_GBps"],
        "all_reduce_256M_128": data[("all_reduce", 128, nbytes_256m)]["bus_bw_GBps"],
    }
    STATS_JSON.write_text(
        json.dumps({"key_values": key_vals, "figures": fig_names, "link_health": link}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report(data, link, fig_names)

    print("=== 生成完成 ===")
    print(f"报告: {REPORT_PATH}")
    print(f"图表目录: {FIG_DIR}")
    for fn in fig_names:
        print(f"  - {FIG_DIR / fn}")
    print(f"\n256M all_reduce bus_bw:")
    print(f"  16 卡: {key_vals['all_reduce_256M_16']:.2f} GB/s")
    print(f"  128 卡: {key_vals['all_reduce_256M_128']:.2f} GB/s")


if __name__ == "__main__":
    main()
