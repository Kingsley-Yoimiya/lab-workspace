#!/usr/bin/env python3
"""npu-dev-1 Phase0–3 交付图：大字号 + 语义图注。

数据：myportal/results/npu-dev-1/
出图：lab-workspace/reports/rounds/npu-dev1-phase03-delivery/
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_style import (
    COLORS,
    FONT_SIZE_LG,
    FONT_SIZE_MD,
    FONT_SIZE_SM,
    apply_plot_style,
    hatch_bar_kwargs,
    save_fig,
    style_axes,
)

RES = Path("/Users/yinjinrun/Codespace/myportal/results/npu-dev-1")
OUT = Path(
    "/Users/yinjinrun/random-thing/project/lab-workspace/reports/rounds/npu-dev1-phase03-delivery"
)
OUT.mkdir(parents=True, exist_ok=True)


def _load_reps():
    return json.loads((RES / "reps_final.json").read_text())


def _load_dose():
    p = RES / "20260715_123840-phase1-dose-d11" / "dose_table.json"
    return json.loads(p.read_text())


def _load_p3():
    p = RES / "20260715_131215-phase3-tracks" / "summary_global.json"
    return json.loads(p.read_text())


def fig_card_profile():
    """16 卡跨 5 run 的主指标相对偏差。"""
    reps = _load_reps()
    cards = sorted(reps["cards"], key=lambda c: c["device"])
    cluster = reps["cluster_median"]
    metrics = [
        ("func_tflops", "Cube 算力\nfunc TFLOPS"),
        ("sustained_steady", "稳态算力\nsustained 中位"),
        ("hbm_gbps", "HBM 带宽\nGB/s"),
        ("mte_gbps", "MTE copy\nGB/s"),
        ("vector_gflops", "Vector FMA\nGFLOPS"),
    ]
    apply_plot_style(figsize=(12, 5.5))
    fig, ax = plt.subplots()
    x = np.arange(16)
    width = 0.16
    for i, (key, lab) in enumerate(metrics):
        rel = []
        for c in cards:
            v = c["metrics"][key]["median"]
            rel.append((v / cluster[key] - 1.0) * 100.0)
        ax.bar(
            x + (i - 2) * width,
            rel,
            label=lab.replace("\n", " "),
            **hatch_bar_kwargs(i, width=width),
        )
    # mark reps
    for name, d in [("慢", 11), ("中", 8), ("快", 2)]:
        ax.axvline(d, color="#333", ls="--", lw=1.2, alpha=0.7)
        ax.text(d, ax.get_ylim()[1] if False else 1.5, name, ha="center", fontsize=FONT_SIZE_SM)

    ax.axhline(0, color="#666", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c["device"]) for c in cards])
    ax.set_xlabel("物理卡号 (device)")
    ax.set_ylabel("相对集群中位数偏差 (%)")
    ax.set_title("16×910B2C 体质主指标（5 次 constitution 跨 run 中位）")
    ax.legend(loc="upper left", fontsize=FONT_SIZE_SM - 2, ncol=2)
    style_axes(ax)
    save_fig(fig, OUT / "card_profile_relmed.svg")
    plt.close(fig)

    # caption md snippet
    return """### 图：卡体质相对偏差

**含义**：各卡相对本机 16 卡中位数的偏差；正=更快/更宽。

**字段**
- `func_tflops`：Stage A 方阵 GEMM（bf16 `a@b`）中位算力，含正确性门控
- `sustained_steady`：满载 GEMM 后半段窗口 TFLOPS 中位（**不是** card 末窗 `sustained_tflops`）
- `hbm_gbps`：大数组 `dst=src*2` 一读一写带宽
- `mte_gbps`：`Tensor.copy_` 纯 DMA/MTE 吞吐
- `vector_gflops`：逐元素 FMA（Vector ALU）

**采集**：`npu-smi` + `torch_npu`；CARD_SCREEN `config.constitution128.yaml`；16 卡并发 ×5 run。

**代表卡**：慢=11，中=8，快=2（虚线）。
"""


def fig_phase1_dose():
    dose = _load_dose()
    order = ["0", "light", "mid", "heavy"]
    factors = ["cube", "vector", "hbm_mte", "cpu"]
    labels = {
        "cube": "Cube（第二进程 GEMM）",
        "vector": "Vector（FMA sidecar）",
        "hbm_mte": "HBM/MTE（D2D copy）",
        "cpu": "CPU busy threads",
    }
    apply_plot_style(figsize=(10, 5))
    fig, ax = plt.subplots()
    x = np.arange(len(order))
    width = 0.2
    for i, f in enumerate(factors):
        ys = []
        for lab in order:
            info = dose.get(f, {}).get(lab, {})
            ys.append(float(info.get("median_drop_pct") or 0.0))
        ax.bar(
            x + (i - 1.5) * width,
            ys,
            label=labels[f],
            **hatch_bar_kwargs(i, width=width),
        )
    ax.axhline(0, color="#666", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["0\n(placebo)", "light\n~duty0.1", "mid\n~duty0.3", "heavy\n~duty0.6"])
    ax.set_ylabel("sentinel 实测性能下降 (%)")
    ax.set_title("Phase1 剂量校准（慢卡 d11）：注入 → sentinel 下降%")
    ax.legend(loc="upper left", fontsize=FONT_SIZE_SM - 2)
    style_axes(ax)
    save_fig(fig, OUT / "phase1_dose_drop.svg")
    plt.close(fig)
    return """### 图：Phase1 剂量–响应

**含义**：在代表慢卡 d11 上开第二进程干扰，CARD_SCREEN sentinel 主指标相对 placebo 的下降%。
下降越大 = 该部件争用越有效。

**sentinel 主指标**
- Cube → `func_tflops`（GEMM）
- Vector → `vector_gflops`（FMA）
- HBM/MTE → `mte_gbps`（`copy_`）
- CPU → `launch_host_overhead_p50_us`（host 侧 launch 开销）

**条件**：镜像 `vllm-ascend:v0.19.1rc1`；`ASCEND_RT_VISIBLE_DEVICES=11`；duty 初值 0/0.1/0.3/0.6。

**读图**：Cube/Vector/HBM 可打到 44–63%；CPU busy 对本机 launch 几乎无效（~0%）。
"""


def fig_phase3_steps():
    p3 = _load_p3()
    tracks = [
        ("indep", "independent\n无 HCCL"),
        ("real_sync", "real_sync\nDP AllReduce"),
        ("tp2", "真 TP=2"),
        ("tp4", "真 TP=4"),
    ]
    apply_plot_style(figsize=(9, 5))
    fig, ax = plt.subplots()
    xs = np.arange(len(tracks))
    p50 = [p3[k]["global_p50"] for k, _ in tracks]
    p95 = [p3[k]["global_p95"] for k, _ in tracks]
    ax.bar(xs - 0.18, p50, label="global step p50", **hatch_bar_kwargs(0, width=0.36))
    ax.bar(xs + 0.18, p95, label="global step p95", **hatch_bar_kwargs(1, width=0.36))
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in tracks])
    ax.set_ylabel("全局 step 时间 (ms)")
    ax.set_title("Phase3：independent / DP-sync / 真 TP（主指标= max_rank）")
    ax.legend(fontsize=FONT_SIZE_SM)
    style_axes(ax)
    save_fig(fig, OUT / "phase3_global_step.svg")
    plt.close(fig)
    return """### 图：Phase3 全局 step

**含义**：`global_step_ms = max_over_ranks(local_step_ms)`。同步训练看全局，不看单卡局部。

**轨道**
- independent：16 卡各跑 Transformer-MLP Block，无 process group
- real_sync：同计算 + **grad AllReduce（DP）**，标签不是 TP
- 真 TP2/TP4：`tp_block_bench_npu.py` ColumnParallel+AG / RowParallel+RS

**条件**：`torch_npu` + HCCL；Block hidden=4096 seq=1024 layers=2；40 iter。

**读图**：indep≈21.4ms → real_sync≈31.3ms（同步约 +46%）；rank 间隙极小（同步会「藏慢卡」）。
"""


def main():
    caps = []
    caps.append(fig_card_profile())
    caps.append(fig_phase1_dose())
    caps.append(fig_phase3_steps())
    md = OUT / "RESULTS.md"
    body = [
        "# npu-dev-1 Phase0–3 实验结果（交付）",
        "",
        "> 实验已在公司 `npu-dev-1`（16×Ascend 910B2C）跑完；原始 jsonl 在 `myportal/results/npu-dev-1/`。",
        "",
        "## 一句话结论",
        "",
        "1. **卡间自然差异**：HBM/MTE/Vector 极齐（CV≪1%）；稳态 Cube 有约 ±3% 量级差；代表卡慢=11 / 中=8 / 快=2。",
        "2. **可注入争用**：第二进程可把 Cube/Vector/HBM sentinel 打掉 ~45–63%；CPU busy 几乎无效。",
        "3. **短窗算子测时几乎看不到争用**（Phase2 Event 计时 slowdown≈0%），与 Phase1 长窗对照——测时窗口决定是否暴露争用。",
        "4. **同步代价**：independent 21.4ms → DP real_sync 31.3ms（+46%）；真 TP2/TP4 轨道已跑通。",
        "",
        "## 图",
        "",
        "![卡体质](card_profile_relmed.svg)",
        "",
        caps[0],
        "",
        "![Phase1剂量](phase1_dose_drop.svg)",
        "",
        caps[1],
        "",
        "![Phase3 step](phase3_global_step.svg)",
        "",
        caps[2],
        "",
        "## 数据路径",
        "",
        "| 内容 | 路径 |",
        "|------|------|",
        "| 选卡 | `results/npu-dev-1/reps_final.json` |",
        "| Phase1 dose | `.../20260715_123840-phase1-dose-d11/` |",
        "| Phase2 | `.../20260715_130900-phase2-ops-d11/` |",
        "| Phase3 | `.../20260715_131215-phase3-tracks/summary_global.json` |",
        "| 计划进度 | `plans/npu-dev1-variation-causality.md` |",
        "",
    ]
    md.write_text("\n".join(body), encoding="utf-8")
    print("wrote", md)
    print("figs", list(OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
