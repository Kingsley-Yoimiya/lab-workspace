"""
依赖关系消融实验 - 绝对精度 + 误差棒 + 原始散点可视化。

6 根柱子：首柱为 baseline（"None"，无移除），后 5 根为各依赖类型移除后的精度。
均值柱 + 标准差误差棒 + 各 case 原始散点（jitter），按精度降序排列。
x 轴标题补充说明语义，不再绘制参考线。

数据来源（合并所有 wide 格式文件，取每个 config 较大的 start_variant）：
  - main_summary_dep_only_wide.csv      (4 种模型架构)
  - configure_summary_dep_only_wide.csv (6 种 GPT-175b 配置变体)
  - moe_balance_summary_dep_only_wide.csv (5 种 Mixtral MoE balance 配置)

Output: output/ablation_dep_type_vs_baseline.pdf
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
COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

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

# %% 常量：依赖类型
DEP_COLS = [
    "communication_to_communication_precision",
    "communication_to_compute_precision",
    "compute_to_communication_precision",
    "compute_to_compute_precision",
    "between_compute_and_communication_precision",
]
DEP_LABELS = [
    "Comm→Comm",
    "Comm→Comp",
    "Comp→Comm",
    "Comp→Comp",
    "Comp↔Comm",
]

# 三个 wide 格式文件，每个文件代表一组实验场景
WIDE_CSV_FILES = [
    "main_summary_dep_only_wide.csv",          # 4 种模型架构
    "configure_summary_dep_only_wide.csv",     # 6 种 GPT-175b 配置变体
    "moe_balance_summary_dep_only_wide.csv",   # 5 种 Mixtral MoE balance 配置
]


# %% 数据加载
def _start_num(sv: str) -> int:
    nums = re.findall(r"\d+", str(sv))
    return int(nums[-1]) if nums else 0


def load_all_cases(input_dir: str) -> pd.DataFrame:
    """
    合并所有 wide 格式 CSV，每个 config 取 start_variant 编号最大的行，
    返回包含所有 case 的 DataFrame（行 = case，列 = dep_cols + original_precision）。
    """
    frames = []
    for fname in WIDE_CSV_FILES:
        path = os.path.join(input_dir, fname)
        df = pd.read_csv(path)
        sv_col  = df.columns[1]
        cfg_col = df.columns[0]
        df["_snum"] = df[sv_col].apply(_start_num)
        # 每个 config 只保留编号最大的 start_variant
        df = df.loc[df.groupby(cfg_col)["_snum"].idxmax()].copy()
        df["_source"] = fname
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


# %% 绘图
if __name__ == "__main__":
    input_dir = os.path.join(os.path.dirname(__file__), "input")
    out_dir   = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    all_cases = load_all_cases(input_dir)
    n_cases   = len(all_cases)

    # 构建 6 组数据：baseline("None") 置首，5 种依赖类型按精度均值降序跟随
    dep_vals   = [all_cases[col].values for col in DEP_COLS]
    dep_means  = np.array([v.mean() for v in dep_vals])
    dep_order  = np.argsort(dep_means)[::-1]           # 依赖类型按精度降序

    base_vals  = all_cases["original_precision"].values
    all_vals   = [base_vals] + [dep_vals[i] for i in dep_order]
    all_labels = ["None"] + [DEP_LABELS[i] for i in dep_order]
    all_means  = np.array([v.mean() for v in all_vals])
    all_stds   = np.array([v.std(ddof=1) for v in all_vals])

    # 颜色：baseline 用浅灰，其余用 COLORS
    bar_colors = ["#b8b8b8"] + [COLORS[k] for k in range(len(DEP_COLS))]

    orig_avg = float(base_vals.mean())
    orig_std = float(base_vals.std(ddof=1))

    print(f"=== Dep ablation ({n_cases} cases, baseline + 5 dep types) ===")
    for lbl, mean, std in zip(all_labels, all_means, all_stds):
        delta = (mean - orig_avg) / orig_avg * 100 if lbl != "None" else 0.0
        print(f"  {lbl:15s}: {mean:.4f} ± {std:.4f}  (Δ {delta:+.1f}%)")

    n_bars = len(all_labels)
    x      = np.arange(n_bars)

    # Y 轴范围：涵盖所有散点，上方留标注空间
    # 转换为百分比
    all_vals   = [v * 100 for v in all_vals]
    all_means  = all_means * 100
    all_stds   = all_stds  * 100
    orig_avg   = orig_avg  * 100
    orig_std   = orig_std  * 100

    all_medians = np.array([np.median(v) for v in all_vals])

    all_raw = np.concatenate(all_vals)
    y_bot   = 48.0                         # 问题4：底部留小边距
    y_top   = all_raw.max() + 6.5          # 问题2：顶部留足标注+高散点空间

    rng = np.random.default_rng(42)        # 固定 seed，jitter 可复现

    fig, ax = plt.subplots()
    ax.grid(True, axis="y", alpha=0.5, zorder=0)

    # ── 柱子（高度 = 中位数，与散点视觉重心对齐）─────────────
    ax.bar(
        x, all_medians - y_bot,
        bottom=y_bot,
        width=0.55,
        color=bar_colors,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    # ── 各 case 原始散点（jitter，空心圆） ────────────────────
    JITTER_W = 0.18
    for i, (vals, color) in enumerate(zip(all_vals, bar_colors)):
        jitter = rng.uniform(-JITTER_W, JITTER_W, size=len(vals))
        ax.scatter(
            x[i] + jitter, vals,
            s=22,
            color=color,
            alpha=0.65,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )

    # ── 顶部均值标注（固定高度，整齐对齐）────────────────────
    label_y = y_top - 0.8
    for i, (median, color) in enumerate(zip(all_medians, bar_colors)):
        text_color = "#444444" if color == "#b8b8b8" else color
        ax.text(
            x[i], label_y,
            f"{median:.1f}%",
            ha="center", va="top",
            fontsize=12, fontweight="medium",
            color=text_color,
            zorder=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=FONT_SIZE_SM, rotation=15, ha="right")
    ax.set_xlabel("Removed Dependency Type", fontsize=FONT_SIZE_SM)
    ax.set_ylabel("Precision (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_ylim(y_bot, y_top)

    plt.tight_layout()

    out_path = os.path.join(out_dir, "ablation_dep_type_vs_baseline.pdf")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    print(f"\nSaved: {out_path}")
    plt.close(fig)
