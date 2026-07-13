import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

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
        "figure.figsize": (10, 6),  # 从 (10, 4.5) 改为 (10, 6)，增加高度
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

    x_labels = ["256 GPUs", "512 GPUs", "1024 GPUs"]
    
    # 数据格式：[256卡数据, 512卡数据, 1024卡数据, 标签名]
    ydata = [
        [110+1.7, 115+0.7, 120+0.38, "HyperSim_autoswitch"],
        [110+89, 115+100, 120+118, "HyperSim"],
        [123403, 412498, 800000, "SimAI"],
        [500000, 500000, 800000, "Astra-sim"],
    ]

    MINIMAL = 50000  # 零刻度线的偏移量
    
    # 使用固定的科学计数法格式，指数为5
    formatter = ticker.FuncFormatter(lambda x, pos: f'{x/1e5:.1f}')
    
    EngFormatter = ticker.EngFormatter(places=0)

    ax.grid(True, alpha=0.5)
    for idx in range(0, len(ydata)):
        x_positions = [i - 0.36 + idx * (0.75 / len(ydata)) for i in range(0, len(x_labels))]
        bar_values = ydata[idx][0:len(x_labels)]
        bar_values_shift = [x + MINIMAL for x in bar_values]
        hdl = ax.bar(
            x_positions,
            bar_values_shift,
            bottom=-MINIMAL,  # 关键：使用负的MINIMAL作为bottom
            align="center",
            hatch=HATCHS[idx],
            color="none",
            edgecolor=COLORS[idx],
            linewidth=2,
            width=0.18,
            label=ydata[idx][-1],
        )
        for x, y, z in zip(x_positions, bar_values, bar_values_shift):
            if y > 0:  # 只显示大于0的值
                # 在柱子顶部中央显示原始数值
                if y > 1000:
                    formatted_text = EngFormatter(y)
                else:
                    formatted_text = f'{int(y)}'
                
                # 对于大于50K的数据，在数字前面加上>
                if y >= 500000:
                    formatted_text = '>' + formatted_text
                
                ax.text(x, z, formatted_text, 
                       ha='center', va='bottom',  # 水平居中，垂直底部对齐
                       fontsize=10, fontweight='medium',
                       color=COLORS[idx])  # 使用和柱子相同的颜色

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=20, rotation=0)
    
    # 设置y轴标签，明确显示单位为10^5
    ax.set_ylabel("Time (ms) " + r"$\times 10^5$", fontsize=FONT_SIZE_LG)
    ax.yaxis.set_major_formatter(formatter)
    
    # 设置更多的y轴刻度
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2*1e5))  # 每隔2e5一个主刻度
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5e5))  # 每隔0.5e5一个次刻度
    
    ax.set_xlabel("Cluster Size", fontsize=FONT_SIZE_LG, labelpad=12)
    
    # 根据数据范围自动设置y轴上限
    max_value = max([max(ydata[i][0:len(x_labels)]) for i in range(len(ydata))])
    ax.set_ylim(-MINIMAL, max_value * 1.2)  # y轴从-MINIMAL开始

    legend = ax.legend(framealpha=0.2, prop={"size": 18}, ncol=2, labelspacing=0.2, loc="upper left")
    legend.set_draggable(True)
    
    # 使用 tight_layout 并设置 pad 参数
    plt.tight_layout(pad=1.5)
    
    # 使用 bbox_inches='tight' 保存，防止左侧标签被截断
    plt.savefig('./output/scale_e2e_dense.pdf', bbox_inches='tight', pad_inches=0.1)
    
    plt.show()
