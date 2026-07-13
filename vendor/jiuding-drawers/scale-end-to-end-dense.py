import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# %% Style
SECONDARY_COLOR = "#808080"
FONT_SIZE_SM = 18
FONT_SIZE_MD = 20
FONT_SIZE_LG = 22
COLORS = [
    # "tab:blue",
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

    x_labels = ["256", "512", "1024"]
    ydata = [
        [110+1.7, 115+0.7, 120+0.38, "HyperSim(Hybrid-level)"],
        [110+89, 115+100, 120+118, "HyperSim(Flow-level)"],
        [123403, 412498, 800000, "SimAI"],
        [500000, 500000, 800000, "Astra-sim"],
    ]
    
    # ✅ 新增：预处理 ydata，将 >=500k 的改为 500k
    THRESHOLD = 500000
    for i in range(len(ydata)):
        for j in range(len(ydata[i]) - 1):  # 最后一个是标签，跳过
            if ydata[i][j] >= THRESHOLD:
                ydata[i][j] = THRESHOLD
    
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 2))
    EngFormatter = ticker.EngFormatter(places=0)

    MINIMAL = 50000
    ax.grid(True, alpha=0.5)
    for idx in range(0, len(ydata)):
        x_positions = [i - 0.36 + idx * (0.92 / len(ydata)) for i in range(0, len(x_labels))]
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
            width=0.22,
            label=ydata[idx][-1],
        )
        for x, y, z in zip(x_positions, bar_values, bar_values_shift):
            if y > 0:  # 只显示大于0的值
                # 简化后的逻辑
                if y >= 500000:
                    formatted_text = '>500k'
                elif y > 1000:
                    formatted_text = EngFormatter(y)
                else:
                    formatted_text = f'{int(y)}'
                
                formatted_text = formatted_text.replace(' ', '')
                ax.text(x, z, formatted_text, 
                    ha='center', va='bottom',
                    fontsize=(14 if len(formatted_text) > 4 else 14),
                    fontweight='medium',
                    color=COLORS[idx])

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=20, rotation=0)
    ax.set_ylabel("E2E Runtime(s)")
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylim(-MINIMAL, 900000)
    ax.set_xlabel("Cluster Size")

    ax.legend(framealpha=0.3, prop={"size": 18}, labelspacing=0.2, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig('./output/scale_end-to-end_dense.pdf')
