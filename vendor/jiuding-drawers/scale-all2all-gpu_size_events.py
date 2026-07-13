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
    # no ground truth
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
    # 创建一个新的折线图
    fig = plt.figure()
    ax = fig.add_subplot()

    x_labels = ["128", "256", "512"]
    line_data = [
        [1064930, 3196260, 12936000, "HyperSim Events"],
        [102649317356,689773309937,3810840499393, "SimAI Events"],
    ]
    
    # 使用与柱状图相同的颜色
    colors = COLORS[:len(line_data)]  # 取前3个颜色
    markers = ['o', 's', '^']  # 圆形、方形、三角形
    formatter = ticker.LogFormatterSciNotation()

    ax.grid(True, alpha=0.5)
    
    for idx in range(0, len(line_data)):
        # 计算x位置（与柱状图保持一致）
        x_positions = np.arange(len(x_labels))
        line_values = line_data[idx][0:len(x_labels)]
        
        # 绘制折线
        hdl = ax.plot(
            x_positions,
            line_values,
            marker=markers[idx],  # 不同形状
            markersize=10,  # 标记大小
            linewidth=2,  # 线宽
            color=colors[idx],  # 相同颜色
            label=line_data[idx][-1],  # 标签
        )
        
        # 在每个数据点上添加数值标签
        # for x, y in zip(x_positions, line_values):
        #     if y > 0:
        #         # 使用工程格式显示
        #         formatted_text = formatter(y)
        #         ax.text(x, y, formatted_text, 
        #                ha='center', va='bottom',  # 水平居中，垂直底部对齐
        #                fontsize=18, fontweight='medium',
        #                color=colors[idx])

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=20, rotation=0)
    ax.set_ylabel("Number of Events")
    ax.set_yscale("log")  # 使用对数坐标，因为数值范围很大
    ax.set_xlabel("Cluster Size")

    # 设置Y轴格式
    ax.yaxis.set_major_formatter(formatter)
    
    # 设置合适的Y轴范围
    all_values = []
    for data in line_data:
        all_values.extend(data[:3])
    min_val = min(all_values)
    max_val = max(all_values)
    ax.set_ylim(min_val * 0.08, max_val * 120)
    ax.set_xlabel("Cluster Size")
    
    # 图例 - 与柱状图保持相同样式
    ax.legend(framealpha=0.2, prop={"size": 18}, draggable=True, ncols=2, labelspacing=0.2, loc="upper left")
    # plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    plt.tight_layout()
    plt.savefig('./output/scale_all2all_gpu_events.pdf')
    plt.show()