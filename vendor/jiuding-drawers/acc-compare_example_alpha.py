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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                gridspec_kw={'height_ratios': [3, 1]})



    x_labels = ["GPT-13B", "Llama-70B", "GPT-175B", "Mixtral-8x7B"]
    
    # 柱状图数据
    ydata = [
        [2982, 3236, 4162, 1983, "ground_truth"],
        [3125, 3114, 0, 1902, "HyperSim"],
        [847, 1587, 1120, 606, "SimAI"],
        [1414, 5399, 1888, 1069, "Astra-Sim"],
    ]
    
    # 假设的折线图数据（例如：效率或百分比）
    line_data = [85, 92, 78, 88]  # 单位：%

    ax1.grid(True, alpha=0.5, axis='y')
    
    # 上部：柱状图
    for idx in range(len(ydata)):
        x_positions = [i - 0.36 + idx * (0.95/len(ydata)) for i in range(len(x_labels))]
        bar_values = ydata[idx][:4]
        
        ax1.bar(x_positions, bar_values, width=0.23,
                hatch=HATCHS[idx], edgecolor=COLORS[idx],
                color="none", linewidth=2, label=ydata[idx][-1])

    ax1.set_ylabel("Time (ms)")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 下部：折线图
    ax2.plot(x_labels, line_data, marker='o', linewidth=2, markersize=8)
    ax2.set_ylabel("Efficiency (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()