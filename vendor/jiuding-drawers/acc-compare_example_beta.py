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
    fig, ax1 = plt.subplots(figsize=(10, 6))

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
    
    # 绘制柱状图
    for idx in range(0, len(ydata)):
        x_positions = [i - 0.36 + idx * (0.95 / len(ydata)) for i in range(0, len(x_labels))]
        bar_values = ydata[idx][0:len(x_labels)]
        
        hdl = ax1.bar(
            x_positions,
            bar_values,
            align="center",
            hatch=HATCHS[idx],
            color="none",
            edgecolor=COLORS[idx],
            linewidth=2,
            width=0.23,
            label=ydata[idx][-1],
        )
        
        # 在柱子上添加数值
        for x, y in zip(x_positions, bar_values):
            if y > 0:
                ax1.text(x, y, f'{y}', 
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold',
                        color=COLORS[idx])

    ax1.set_xticks(np.arange(len(x_labels)))
    ax1.set_xticklabels(x_labels, fontsize=16, rotation=10)
    ax1.set_ylabel("Time (ms)", fontsize=16)
    ax1.set_ylim(0, max([max(row[:4]) for row in ydata]) * 1.1)
    
    # 创建第二个Y轴用于折线图
    ax2 = ax1.twinx()
    
    # 绘制折线图
    # 注意：x位置需要调整，通常放在每个组的中心
    line_x_positions = np.arange(len(x_labels))  # 每个模型组的中心位置
    line = ax2.plot(line_x_positions, line_data, 
                    marker='o', markersize=10, linewidth=3,
                    color='black', label='Efficiency (%)',
                    markerfacecolor='white', markeredgecolor='black', markeredgewidth=2)
    
    # 在折线点添加数值
    for i, (x, y) in enumerate(zip(line_x_positions, line_data)):
        ax2.text(x, y, f'{y}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax2.set_ylabel("Efficiency (%)", fontsize=16, color='black')
    ax2.set_ylim(0, 100)  # 百分比范围
    ax2.tick_params(axis='y', labelcolor='black')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # 调整图例位置（放在下方）
    ax1.legend(lines1 + lines2, labels1 + labels2, 
               framealpha=0.3, prop={"size": 14}, 
               draggable=True, ncols=3,
               loc='upper center',
               bbox_to_anchor=(0.5, -0.15))
    
    plt.title("Performance Comparison with Efficiency", fontsize=18, pad=20)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()