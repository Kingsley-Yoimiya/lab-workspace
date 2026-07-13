import matplotlib.pyplot as plt
import numpy as np

# %% Style
SECONDARY_COLOR = "#808080"
FONT_SIZE_SM = 18
FONT_SIZE_MD = 20
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
HATCHS = ["--", "//", "xx", "\\", "oo", "..", "+", "O", "*"]

plt.rcParams.update(
    {
        "figure.figsize": (8, 4.5),
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

    x_labels = ["EP\n(Uncongested)", "EP\n(Congested)"]
    # company_y = [0.0,0.42857]
    # bics_y = [0.141558,0.17316]
    # ft4_y = [0, 0.1]
    ydata = [
        [32.39, 27.17, "Groundtruth"],
        [32.38, 27.17, "HyperSim"],
    ]

    ax.grid(True, alpha=0.5)
    for idx in range(0, len(ydata)):
        x_positions = [i - 0.36 + idx * (0.75 / len(ydata)) for i in range(0, len(x_labels))]
        bar_values = ydata[idx][0:len(x_labels)]
        hdl = ax.bar(
            x_positions,
            bar_values,
            align="center",
            hatch=HATCHS[idx],
            color="none",
            edgecolor=COLORS[idx],
            linewidth=2,
            width=0.37,
            label=ydata[idx][-1],
        )
        for x, y in zip(x_positions, bar_values):
            if y > 0:  # 只显示大于0的值
                # 在柱子顶部中央显示数值
                ax.text(x, y, f'{y}', 
                       ha='center', va='bottom',  # 水平居中，垂直底部对齐
                       fontsize=16, fontweight='medium',
                       color=COLORS[idx])  # 使用和柱子相同的颜色


    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=20, rotation=0)
    ax.set_ylabel("Avg Bandwidth (GB/s)")
    # ax.set_ylim(0, 1.1)
    # ax.set_yscale("log")

    max_value = max([max(ydata[i][0:len(x_labels)]) for i in range(len(ydata))])
    ax.set_ylim(0, max_value * 1.45) 

    # 创建legend对象
    legend = ax.legend(framealpha=0.3, prop={"size": 18}, ncol=2, labelspacing=0.2, loc="upper left")
    # 设置为可拖动
    legend.set_draggable(True)
    plt.tight_layout()
    plt.savefig('./output/network_overlap_ep_ep.pdf')
