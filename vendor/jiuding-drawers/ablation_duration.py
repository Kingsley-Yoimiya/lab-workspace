"""
时延扰动消融实验（symmetric 模式）。

自变量：对 communication 或 GPU 时延施加 0–100% 的扰动。
因变量：精度相对于基线（e0）的变化量 Δ Precision (%)。

数据处理规则：
  - paper_data      : 4 个模型（GPT-13B/175B, Llama-70B, Mixtral-8x7B），
                      每个 config 取 start_variant 编号更大的那个，再对 4 个模型取均值。
  - configuration_set: 6 个 175B 配置变体，均取 START_5_para，取均值。
  - moe_balance      : 5 个 MOE_BALANCE 变体，均取 START_27，取均值。

输出（./output/）：
  ablation_duration_comm_full.pdf    通信扰动，所有 11 个扰动点
  ablation_duration_comm_sparse.pdf  通信扰动，稀疏 6 个点 (0,20,40,60,80,100)
  ablation_duration_gpu_full.pdf     GPU 扰动，所有 11 个扰动点
  ablation_duration_gpu_sparse.pdf   GPU 扰动，稀疏 6 个点
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% Style（与工作区其他脚本一致）
SECONDARY_COLOR = "#808080"
FONT_SIZE_SM = 18
FONT_SIZE_LG = 22

DATASET_COLORS = {
    "paper_data":        "tab:blue",
    "configuration_set": "tab:orange",
    "moe_balance":       "tab:green",
}
DATASET_MARKERS = {
    "paper_data":        "o",
    "configuration_set": "s",
    "moe_balance":       "^",
}
DATASET_LABELS = {
    "paper_data":        "End-to-End",
    "configuration_set": "Optim. Strategies",
    "moe_balance":       "Expert Imbalance",
}

plt.rcParams.update(
    {
        "figure.figsize":    (8, 4.5),
        "font.size":         FONT_SIZE_SM,
        "axes.titlesize":    FONT_SIZE_LG,
        "axes.labelsize":    FONT_SIZE_LG,
        "font.family":       "Calibri",
        "legend.framealpha": 0.5,
        "legend.fontsize":   FONT_SIZE_SM,
        "axes.spines.right": False,
        "axes.spines.top":   False,
        "axes.edgecolor":    SECONDARY_COLOR,
        "xtick.color":       SECONDARY_COLOR,
        "ytick.color":       SECONDARY_COLOR,
        "xtick.labelsize":   FONT_SIZE_LG,
        "ytick.labelsize":   FONT_SIZE_LG,
        "grid.color":        SECONDARY_COLOR,
        "grid.linestyle":    ":",
        "grid.linewidth":    1,
        "axes.grid":         True,
        "axes.grid.axis":    "y",
    }
)

# %% 常量
EPS_LEVELS    = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
EPS_COLS      = [f"e{e}_precision" for e in EPS_LEVELS]
SPARSE_LEVELS = [0, 20, 40, 60, 80, 100]


# %% 数据加载与处理
def _start_num(sv: str) -> int:
    """从 start_variant 字符串中提取最后一个整数，用于比较大小。"""
    nums = re.findall(r"\d+", str(sv))
    return int(nums[-1]) if nums else 0


def load_results(csv_path: str) -> dict:
    """
    返回 {(dataset, kind): delta_array}，
    delta_array shape=(11,)，单位为 %，相对 e0 基线的变化量。
    """
    df = pd.read_csv(csv_path)
    df = df[df["perturb_mode"] == "symmetric"].copy()
    df["_snum"] = df["start_variant"].apply(_start_num)

    out = {}
    for dataset in ["paper_data", "configuration_set", "moe_balance"]:
        sub = df[df["dataset"] == dataset].copy()

        if dataset == "paper_data":
            # 每个 config（模型）取 start_variant 编号更大的所有行（保留 kind/perturb_mode 维度）
            max_snum = sub.groupby("config")["_snum"].transform("max")
            sub = sub[sub["_snum"] == max_snum]
        elif dataset == "configuration_set":
            sub = sub[sub["start_variant"] == "START_5_para"]
        else:  # moe_balance
            sub = sub[sub["start_variant"] == "START_27"]

        for kind in ["communication", "gpu"]:
            rows = sub[sub["kind"] == kind]
            avg_prec = rows[EPS_COLS].mean().values   # (11,)
            baseline  = avg_prec[0]                   # e0 = original
            delta     = (avg_prec - baseline) / baseline * 100
            out[(dataset, kind)] = delta
            print(
                f"  [{dataset:>18s} | {kind:>13s}]  "
                f"baseline={baseline:.4f}  "
                f"e50Δ={delta[5]:+.2f}%  e100Δ={delta[10]:+.2f}%"
            )

    return out


# %% 绘图
def _plot_kind(
    results: dict,
    kind: str,
    levels: list,
    out_path: str,
) -> None:
    idx = [EPS_LEVELS.index(e) for e in levels]

    fig = plt.figure()
    ax  = fig.add_subplot()
    ax.grid(True, alpha=0.5)

    for dataset in ["paper_data", "configuration_set", "moe_balance"]:
        delta = results[(dataset, kind)]
        y     = delta[idx]
        ax.plot(
            levels,
            y,
            label=DATASET_LABELS[dataset],
            color=DATASET_COLORS[dataset],
            marker=DATASET_MARKERS[dataset],
            linewidth=2,
            markersize=7,
        )

    # 零变化基准线
    ax.axhline(0, color=SECONDARY_COLOR, linestyle="--", linewidth=1.2, alpha=0.7)

    kind_label = "Communication" if kind == "communication" else "GPU Compute"
    ax.set_xlabel("Perturbation Level (%)")
    ax.set_ylabel("Δ Precision (%)")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])

    ax.legend(
        framealpha=0.3,
        prop={"size": FONT_SIZE_SM - 2},
        ncols=1,
        labelspacing=0.2,
        loc="best",
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved: {out_path}")
    plt.close(fig)


# %% 主入口
if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "input", "summary_duration_all_wide.csv")
    out_dir  = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Loading & processing data ===")
    results = load_results(csv_path)
    print()

    for kind in ["communication", "gpu"]:
        _plot_kind(
            results, kind,
            levels=EPS_LEVELS,
            out_path=os.path.join(out_dir, f"ablation_duration_{kind}_full.pdf"),
        )
        _plot_kind(
            results, kind,
            levels=SPARSE_LEVELS,
            out_path=os.path.join(out_dir, f"ablation_duration_{kind}_sparse.pdf"),
        )
