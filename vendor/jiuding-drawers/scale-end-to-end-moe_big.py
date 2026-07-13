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
    fig = plt.figure()
    ax = fig.add_subplot()

    x_labels = ["256", "512", "1024", "2048", "4096"]
    line_data = [
        [244+814.932, 255+822.9883575, 260+ 1773.896, 275+1655.83, 280+1710.92, "TP8+PP8+EP8"],
        # [235+24, 239+83, 243+67, 260+ 2076.82, 275+1976.14, 0, "MoE"],
    ]
    
    colors = COLORS[:len(line_data)] 
    markers = ['o', 's', '^'] 
    formatter = ticker.ScalarFormatter()

    ax.grid(True, alpha=0.5)
    
    for idx in range(0, len(line_data)):
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
    ax.set_xticklabels(x_labels, fontsize=20, rotation=10)
    ax.set_ylabel("E2E Runtime(s)")
    # ax.set_yscale("log")
    ax.set_xlabel("Cluster Size")

    ax.yaxis.set_major_formatter(formatter)
    
    ax.set_ylim(500, 2500)

    # 图例 - 与柱状图保持相同样式
    ax.legend(framealpha=0.2, prop={"size": 18},  labelspacing=0.2, loc="upper left")
    # ax.legend(framealpha=0.2, prop={"size": 18}, draggable=True, ncols=2, labelspacing=0.2, loc="upper left")
    # plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    plt.tight_layout()
    plt.savefig('./output/scale_end-to-end_moe_big.pdf')
    plt.show()