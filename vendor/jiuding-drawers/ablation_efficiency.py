"""
GPT-3-175b 稠密模型：不同仿真层级随集群规模变化的对比（分组柱状图）。
seq_length=2048, bs=1024, mbs=8

纵轴为线性刻度；通过将柱子的 bottom 设为负的 MINIMAL、高度为 y+MINIMAL，
把“零刻度”视觉上抬高（与 scale-end-to-end-dense.py 一致），便于突出 autoswitch
相对 Flow-level 的加速；柱顶数值留出垂直间距，避免与柱子重合。
"""
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# %% Style（与 scale-end-to-end-dense.py 等脚本一致）
SECONDARY_COLOR = "#808080"
FONT_SIZE_SM = 18
FONT_SIZE_LG = 22
COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:purple",
    "tab:green",
    "tab:red",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]
HATCHS = ["oo", "//", "xx", "\\", "..", "+", "O", "*"]

plt.rcParams.update(
    {
        # 略增高画布，纵向上更舒展（原 4.5in）
        "figure.figsize": (10, 6.2),
        "font.size": FONT_SIZE_SM,
        "axes.titlesize": FONT_SIZE_LG,
        "axes.labelsize": FONT_SIZE_LG,
        "font.family": "Calibri",
        "legend.framealpha": 0.5,
        "legend.fontsize": FONT_SIZE_SM,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.edgecolor": SECONDARY_COLOR,
        "xtick.color": SECONDARY_COLOR,
        "ytick.color": SECONDARY_COLOR,
        "xtick.labelsize": FONT_SIZE_LG,
        "ytick.labelsize": FONT_SIZE_LG,
        "grid.color": SECONDARY_COLOR,
        "grid.linestyle": ":",
        "grid.linewidth": 1,
        "axes.grid": True,
        "axes.grid.axis": "y",
    }
)


def _format_bar_value(y: float) -> str:
    if y >= 100:
        return f"{y:.0f}"
    if y >= 10:
        return f"{y:.2g}"
    return f"{y:.3g}"


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot()

    x_labels = ["256", "512", "1024"]
    # 顺序：collective-level, autoswitch, Flow-level
    ydata = [
        [1.1140587, 0.549024, 0.274340284, "collective-level"],
        [11.369599, 16.13368, 23.55015675, "autoswitch"],
        [91.681965, 109.68816, 118.6512901, "Flow-level"],
    ]

    # 抬高“零线”：柱底在 -MINIMAL，柱高为 y+MINIMAL，柱顶仍在真实 y（线性刻度）
    MINIMAL = 2.0
    n_series = len(ydata)
    bar_width = 0.22
    ax.grid(True, alpha=0.5)

    all_tops = []
    for idx in range(n_series):
        x_positions = [
            i - 0.36 + idx * (0.92 / n_series) for i in range(len(x_labels))
        ]
        bar_values = ydata[idx][: len(x_labels)]
        bar_heights = [v + MINIMAL for v in bar_values]
        ax.bar(
            x_positions,
            bar_heights,
            bottom=-MINIMAL,
            align="center",
            hatch=HATCHS[idx],
            color="none",
            edgecolor=COLORS[idx],
            linewidth=2,
            width=bar_width,
            label=ydata[idx][-1],
        )
        all_tops.extend(bar_values)

    ymax_data = max(all_tops)
    # 柱顶标签统一抬高到真实值之上，避免与柱体重叠
    label_dy = max(1.8, 0.022 * ymax_data)

    for idx in range(n_series):
        x_positions = [
            i - 0.36 + idx * (0.92 / n_series) for i in range(len(x_labels))
        ]
        bar_values = ydata[idx][: len(x_labels)]
        for x, y in zip(x_positions, bar_values):
            ax.text(
                x,
                y + label_dy,
                _format_bar_value(y),
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="medium",
                color=COLORS[idx],
            )

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=20, rotation=0)
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("E2E Runtime (s)")

    # 纵轴略抬高上限，给柱顶数字与图例之间留出空隙
    ax.set_ylim(-MINIMAL, ymax_data * 1.28)
    # 主刻度稀疏一些（约 0, 50, 100, 150），避免过密
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

    def _ytick_fmt(v: float, _pos: int) -> str:
        if v < 0:
            return ""
        return f"{v:g}"

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_ytick_fmt))

    # 图例放在坐标轴上方（loc=lower center + bbox 在轴外），避免与右上角 Flow 标签遮挡
    ax.legend(
        framealpha=0.3,
        prop={"size": 18},
        labelspacing=0.2,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.07),
        borderaxespad=0.0,
    )
    # rect 顶部留白，给轴外图例；savefig 仍用 bbox_inches=tight 防裁切
    plt.tight_layout(rect=(0, 0, 1, 0.84))

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ablation_executor_gpt3_175b_dense.pdf")
    # 必须先 savefig 再 show：若先 show 并关闭窗口，figure 会被清空，savefig 会得到空白 PDF。
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved: {out_path}")
    plt.close(fig)
