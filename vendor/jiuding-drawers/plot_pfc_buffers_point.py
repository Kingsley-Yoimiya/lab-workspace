#!/usr/bin/env python3
"""Plot simplified HyperSim piecewise curve (from key points) and ns-3 reference,
with selected intersections annotated."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BYTES_PER_KILOBYTE = 1024.0


def find_intersections(xs, ys, y_level):
    """
    在折线 (xs, ys) 与水平线 y = y_level 之间，找到所有交点 (x, y_level)。
    """
    points = []
    for i in range(len(xs) - 1):
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[i + 1], ys[i + 1]

        # 完全在同一侧且没有点正好落在 y_level 上，不相交
        if (y1 - y_level) * (y2 - y_level) > 0:
            continue

        # 端点在水平线上
        if y1 == y_level:
            points.append((x1, y_level))
        if y2 == y_level:
            points.append((x2, y_level))
            continue

        # 一上一下：线性插值
        if (y1 - y_level) * (y2 - y_level) < 0:
            x = x1 + (x2 - x1) * (y_level - y1) / (y2 - y1)
            points.append((x, y_level))

    return points


def plot_egress_piecewise(output_base: Path):
    # ---- 1. HyperSim 数据（单位：ms / kB）----
    HyperSim_points_all = [
        (0.0,   0.0),
        (0.900, 4.1),
        (1.184, 13857.5),
        (1.468, 0.0),
        (1.624, 4.1),
        (1.748, 6038.7),
        (1.908, 6034.6),
        (2.032, 0.0),
        (2.570, 0.0),
    ]
    jd_times_ms_all = [t for (t, _) in HyperSim_points_all]
    jd_kb_all = [q for (_, q) in HyperSim_points_all]

    # 手动标注的 HyperSim 关键点
    HyperSim_points_to_annotate = [
        (1.184, 13857.5),
        (1.624, 0.0),
        (1.748, 6038.7),
        (1.908, 6034.6),
    ]

    # ---- 2. ns-3 数据（单位：ms / bytes -> ms / kB）----
    ref_points_ms_bytes_all = [
        (0.0, 0),
        (0.9, 0.0),
        (1.18, 14_000_000),  # 14MB
        (1.46, 0),
        (1.64, 0),
        (1.76, 6_000_000),   # 6MB
        (1.92, 6_000_000),
        (2.04, 0),
        (2.57, 0),
    ]
    ref_times_ms_all = [t for (t, _) in ref_points_ms_bytes_all]
    ref_kb_all = [b / BYTES_PER_KILOBYTE for (_, b) in ref_points_ms_bytes_all]

    # 手动标注的 ns-3 关键点
    ref_points_to_annotate = [
        (1.18, 14_000_000 / BYTES_PER_KILOBYTE),
        (1.64, 0.0),
        (1.76, 6_000_000 / BYTES_PER_KILOBYTE),
        (1.92, 6_000_000 / BYTES_PER_KILOBYTE),
    ]

    # ---- 3. 交点计算 + 按规则筛选 ----
    XOFF = 4000.0
    XON = 1000.0

    jd_xoff_all = find_intersections(jd_times_ms_all, jd_kb_all, XOFF)
    jd_xon_all = find_intersections(jd_times_ms_all, jd_kb_all, XON)
    ns3_xoff_all = find_intersections(ref_times_ms_all, ref_kb_all, XOFF)
    ns3_xon_all = find_intersections(ref_times_ms_all, ref_kb_all, XON)

    # 与 Xoff 的交点：保留 1,3,5,...（奇数索引）
    jd_xoff_selected = [p for idx, p in enumerate(jd_xoff_all, start=1) if idx % 2 == 1]
    ns3_xoff_selected = [p for idx, p in enumerate(ns3_xoff_all, start=1) if idx % 2 == 1]

    # 与 Xon 的交点：保留 2,4,6,...（偶数索引）
    jd_xon_selected = [p for idx, p in enumerate(jd_xon_all, start=1) if idx % 2 == 0]
    ns3_xon_selected = [p for idx, p in enumerate(ns3_xon_all, start=1) if idx % 2 == 0]

    # ---- 4. 画图 ----
    fig, ax = plt.subplots(figsize=(12, 5.5))

    # 更深的颜色
    jd_color = "#003366"      # 深海军蓝
    ns3_color = "#a65300"     # 深焦橙

    ax.tick_params(axis="x", labelsize=17)
    ax.tick_params(axis="y", labelsize=16)

    # HyperSim 折线
    ax.plot(
        jd_times_ms_all,
        jd_kb_all,
        color=jd_color,
        linewidth=1.8,
        linestyle="--",
        marker="o",
        markersize=6,
        label="HyperSim",
    )

    # NS-3 折线
    ax.plot(
        ref_times_ms_all,
        ref_kb_all,
        color=ns3_color,
        linewidth=1.6,
        linestyle="--",
        marker=None,
        label="NS-3",
    )

    # Xoff / Xon 阈值线
    ax.axhline(
        y=XOFF,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="Xoff (4000 kB)",
    )
    ax.axhline(
        y=XON,
        color="green",
        linestyle="--",
        linewidth=1.2,
        label="Xon (1000 kB)",
    )

    ax.set_xlabel("Time (ms)", fontsize=16)
    ax.set_ylabel("Queue occupancy (kB)", fontsize=16)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    yticks = [0, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(v) for v in yticks], fontsize=15)

    ax.set_xlim(left=0.6, right=2.6)
    ax.set_ylim(bottom=-500, top=14500)

    # ---- 5. 文本格式函数：普通数字 ----
    def fmt_xy_plain(t, q):
        # t: 两位小数, q: 取整
        return f"({t:.2f}, {q:.0f})"

    # ---- 6. 原始关键点标注 ----
    for (t, q) in HyperSim_points_to_annotate:
        if abs(t - 1.184) < 1e-3:
            xytext = (8, -12)
            ha, va = "left", "top"
        elif abs(t - 1.624) < 1e-3:
            xytext = (6, 8)        # x 轴附近，右上
            ha, va = "left", "bottom"
        elif abs(t - 1.748) < 1e-3:
            xytext = (-8, -10)
            ha, va = "right", "top"
        elif abs(t - 1.908) < 1e-3:
            xytext = (8, 8)
            ha, va = "left", "bottom"
        else:
            xytext = (6, 6)
            ha, va = "left", "bottom"

        ax.annotate(
            fmt_xy_plain(t, q),
            xy=(t, q),
            xytext=xytext,
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",      # 坐标文字加粗
            color=jd_color,
            ha=ha,
            va=va,
            arrowprops=dict(
                arrowstyle="->",
                color=jd_color,
                lw=1.0,
                shrinkA=2,
                shrinkB=2,
            ),
        )

    for (t, q) in ref_points_to_annotate:
        if abs(t - 1.18) < 1e-3:
            xytext = (-8, -12)
            ha, va = "right", "top"
        elif abs(t - 1.64) < 1e-3:
            xytext = (-8, 8)       # x 轴附近，左上
            ha, va = "right", "bottom"
        elif abs(t - 1.76) < 1e-3:
            xytext = (-8, 8)
            ha, va = "right", "bottom"
        elif abs(t - 1.92) < 1e-3:
            xytext = (8, -8)
            ha, va = "left", "top"
        else:
            xytext = (6, -6)
            ha, va = "left", "top"

        ax.annotate(
            fmt_xy_plain(t, q),
            xy=(t, q),
            xytext=xytext,
            textcoords="offset points",
            fontsize=13,
            fontweight="bold",      # 坐标文字加粗
            color=ns3_color,
            ha=ha,
            va=va,
            arrowprops=dict(
                arrowstyle="->",
                color=ns3_color,
                lw=1.0,
                shrinkA=2,
                shrinkB=2,
            ),
        )

    # ---- 7. 交点标注 ----

    # HyperSim ∩ Xoff（蓝）
    for x, y in jd_xoff_selected:
        ax.scatter(x, y, color=jd_color, s=25)

        xytext = (4, 6)            # 默认右上
        ha, va = "left", "bottom"

        if 1.70 <= x <= 1.73:      # 约 1.71/1.72
            xytext = (4, -10)      # 右下
            ha, va = "left", "top"
        elif 1.73 < x <= 1.78:     # 约 1.75/1.76
            xytext = (-4, 6)       # 左上
            ha, va = "right", "bottom"

        ax.annotate(
            fmt_xy_plain(x, y),
            xy=(x, y),
            xytext=xytext,
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
            color=jd_color,
            ha=ha,
            va=va,
            arrowprops=dict(
                arrowstyle="->",
                color=jd_color,
                lw=0.8,
                shrinkA=2,
                shrinkB=2,
            ),
        )

    # HyperSim ∩ Xon（蓝）
    for x, y in jd_xon_selected:
        ax.scatter(x, y, color=jd_color, s=25)

        xytext = (4, -10)          # 默认右下
        ha, va = "left", "top"

        if 1.40 <= x <= 1.48:
            xytext = (-4, -10)     # 左下
            ha, va = "right", "top"
        elif 2.00 <= x <= 2.04:
            xytext = (4, 6)        # 右上
            ha, va = "left", "bottom"

        ax.annotate(
            fmt_xy_plain(x, y),
            xy=(x, y),
            xytext=xytext,
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
            color=jd_color,
            ha=ha,
            va=va,
            arrowprops=dict(
                arrowstyle="->",
                color=jd_color,
                lw=0.8,
                shrinkA=2,
                shrinkB=2,
            ),
        )

    # NS-3 ∩ Xoff（橙）
    for x, y in ns3_xoff_selected:
        ax.scatter(x, y, color=ns3_color, s=25)

        xytext = (-4, 6)           # 默认左上
        ha, va = "right", "bottom"

        if 1.70 <= x <= 1.73:
            if abs(x - 1.72) < 0.01:
                xytext = (-18, -10)  # (1.72,4000) 左下，离线远一点
            else:
                xytext = (-4, -10)   # 其它 1.7x：左下
            ha, va = "right", "top"
        elif 1.73 < x <= 1.78:
            xytext = (4, 6)        # 右上
            ha, va = "left", "bottom"

        ax.annotate(
            fmt_xy_plain(x, y),
            xy=(x, y),
            xytext=xytext,
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
            color=ns3_color,
            ha=ha,
            va=va,
            arrowprops=dict(
                arrowstyle="->",
                color=ns3_color,
                lw=0.8,
                shrinkA=2,
                shrinkB=2,
            ),
        )

    # NS-3 ∩ Xon（橙）
    for x, y in ns3_xon_selected:
        ax.scatter(x, y, color=ns3_color, s=25)

        xytext = (-4, -10)         # 默认左下
        ha, va = "right", "top"

        # 单独处理 (1.44, 1000)：放到点的左上方
        if abs(x - 1.44) < 0.01:
            xytext = (-18, 6)      # 左上
            ha, va = "right", "bottom"
        elif 1.40 <= x <= 1.48:
            xytext = (4, -10)      # 其它 1.4x：右下
            ha, va = "left", "top"
        elif 2.00 <= x <= 2.04:
            xytext = (-4, 6)       # 2.0x 段：左上
            ha, va = "right", "bottom"

        ax.annotate(
            fmt_xy_plain(x, y),
            xy=(x, y),
            xytext=xytext,
            textcoords="offset points",
            fontsize=13,
            fontweight="bold",
            color=ns3_color,
            ha=ha,
            va=va,
            arrowprops=dict(
                arrowstyle="->",
                color=ns3_color,
                lw=0.8,
                shrinkA=2,
                shrinkB=2,
            ),
        )

    ax.legend(fontsize=17)
    plt.tight_layout()

    # ---- 8. 同时输出 PNG 和 PDF ----
    output_base = output_base.with_suffix("")  # 去掉传入的扩展名（如果有）
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")

    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to {png_path} and {pdf_path}")


def main():
    # 只给“基名”，后面函数里会生成 .png 和 .pdf
    output_base = Path("./output/buffer_visualization")
    plot_egress_piecewise(output_base)


if __name__ == "__main__":
    main()
