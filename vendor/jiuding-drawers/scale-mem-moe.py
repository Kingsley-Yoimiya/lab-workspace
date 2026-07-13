import matplotlib.pyplot as plt
import numpy as np

# %% Style
SECONDARY_COLOR = "#808080"
FONT_SIZE_SM = 18
FONT_SIZE_MD = 20
FONT_SIZE_LG = 22
COLORS = [
    # "tab:blue",
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
HATCHS = ["//", "xx", "\\", "oo", "..", "+", "O", "*"]

plt.rcParams.update(
    {
        "figure.figsize": (10, 4.5),
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
        # enable grid
        "axes.grid": True,
        # disable x axis grid
        "axes.grid.axis": "y",
    }
)

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot()

    x_labels = ["128", "256", "512"]

    # A: tp4_pp4_ep4
    # B: tp4_pp4_ep8

    # company_y = [0.0,0.42857]
    # bics_y = [0.141558,0.17316]
    # ft4_y = [0, 0.1]
    ydata = [
        # [2982, 3236, 4162, 1983, "ground_truth"],
        # [0.214, 0.217, 0.224, 0.5, 0.5, 0.5, "HyperSim"],
        # [0.217, 0.218, 0.22, 1.56, 2.22, 4.32, "SimAI"],
        # [21, 34, 52.8, 22.89, 37, 58.4, "Astra-Sim"],
        [0.214, 0.217, 0.224, "HyperSim"],
        [1.3, 2.0, 4.07, "SimAI"],
        [21, 34, 52.8, "Astra-Sim"],
    ]

    MINIMAL = 3.0
    ax.grid(True, alpha=0.5)
    for idx in range(0, len(ydata)):
        x_positions = [i - 0.36 + idx * (0.75 / len(ydata)) for i in range(0, len(x_labels))]
        bar_values = ydata[idx][0:len(x_labels)]
        bar_values_shift = [x + MINIMAL for x in bar_values]
        hdl = ax.bar(
            x_positions,
            bar_values_shift,
            bottom=-MINIMAL, 
            align="center",
            hatch=HATCHS[idx],
            color="none",
            edgecolor=COLORS[idx],
            linewidth=2,
            width=0.24,
            label=ydata[idx][-1],
        )
        for x, y, z in zip(x_positions, bar_values, bar_values_shift):
            if y > 0:  # 只显示大于0的值
                # 在柱子顶部中央显示数值
                formatted_text = f'{y:.1f}'
                ax.text(x, z, formatted_text, 
                       ha='center', va='bottom',  # 水平居中，垂直底部对齐
                       fontsize=16, fontweight='medium',
                       color=COLORS[idx])  # 使用和柱子相同的颜色


    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=20, rotation=0)
    ax.set_ylabel("Memory Usage (GB)")
    ax.set_ylim(-MINIMAL)
    ax.set_xlabel("Cluster Size")
    # ax.set_ylim(0, 1.1)
    # ax.set_yscale("log")

    # ax.legend(framealpha=0.3, prop={"size": 18}, draggable=True, ncols=2, labelspacing=0.2, loc="upper left")
    ax.legend(framealpha=0.3, prop={"size": 18},  labelspacing=0.2, loc="upper left")
    plt.tight_layout()
    plt.savefig('./output/scale_mem_moe.pdf')
