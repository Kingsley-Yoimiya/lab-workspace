#!/usr/bin/env python3
"""默认出图样式：大字号、去顶右边框、y 点线网格、SVG 输出。

源自 jiuding-drawers 规范，现作为本仓库默认画图标准；文件名勿再加 jiuding 后缀。
"""
from __future__ import annotations

from pathlib import Path

SECONDARY_COLOR = "#808080"
FONT_SIZE_SM = 18
FONT_SIZE_MD = 20
FONT_SIZE_LG = 22

# tab 色板（与 jiuding 一致；跳过 tab:blue 作主系列起点可选）
COLORS = [
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "tab:blue",
]
HATCHES = ["//", "xx", "\\\\", "oo", "..", "+", "O", "*"]


def apply_plot_style(figsize: tuple[float, float] = (8, 4.5)) -> None:
    """默认出图样式（九鼎抽屉规范，作为默认不另加后缀）。"""
    apply_jiuding_style_impl(figsize)


def apply_jiuding_style(figsize: tuple[float, float] = (8, 4.5)) -> None:
    """兼容旧名。"""
    apply_plot_style(figsize)


def apply_jiuding_style_impl(figsize: tuple[float, float] = (8, 4.5)) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": figsize,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": FONT_SIZE_SM,
            "axes.titlesize": FONT_SIZE_LG,
            "axes.labelsize": FONT_SIZE_LG,
            "font.family": "sans-serif",
            "font.sans-serif": [
                # 中文报告优先 CJK 无衬线；字号/排版仍对齐 jiuding（Calibri 系大字号）
                "PingFang SC",
                "Heiti SC",
                "STHeiti",
                "Calibri",
                "Arial",
                "Helvetica Neue",
                "Arial Unicode MS",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
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
            "grid.linewidth": 1.0,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "svg.fonttype": "none",  # 文本可编辑，不转 path
        }
    )


def style_axes(ax) -> None:
    ax.grid(True, axis="y", linestyle=":", alpha=0.55, linewidth=1.0, color=SECONDARY_COLOR)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SECONDARY_COLOR)


def save_fig(fig, path: Path | str, *, also_png: bool = False) -> Path:
    """保存为 SVG（主交付）；可选顺带 PNG 预览。"""
    import matplotlib.pyplot as plt

    path = Path(path)
    if path.suffix.lower() != ".svg":
        path = path.with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    if also_png:
        fig.savefig(
            path.with_suffix(".png"),
            bbox_inches="tight",
            pad_inches=0.15,
            facecolor="white",
            dpi=160,
        )
    plt.close(fig)
    return path


def hatch_bar_kwargs(idx: int, *, width: float = 0.24) -> dict:
    """空心+描边+hatch，贴近 jiuding 柱状图。"""
    return {
        "hatch": HATCHES[idx % len(HATCHES)],
        "color": "none",
        "edgecolor": COLORS[idx % len(COLORS)],
        "linewidth": 2.0,
        "width": width,
    }


# ── 密网格 / 热力图专用（全局 18–22pt 会把格子字撑爆）────────────────
ANNOT_FS_DENSE = 6.5      # 8×16 / 16×16 格内数字
ANNOT_FS_MED = 7.0        # 8×10 一类中等密度
ANNOT_FS_SPARSE = 12      # 4×4 相关矩阵
TICK_FS_DENSE = 10
LABEL_FS_DENSE = 12
TITLE_FS_COMPACT = 14


def style_heatmap_axes(ax, *, tick_fs: float = TICK_FS_DENSE, label_fs: float = LABEL_FS_DENSE) -> None:
    """热力图：关网格、去顶右边框、缩小刻度与轴标题字。"""
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SECONDARY_COLOR)
    ax.tick_params(axis="both", labelsize=tick_fs, colors="#444444")
    ax.xaxis.label.set_size(label_fs)
    ax.yaxis.label.set_size(label_fs)
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_color("#333333")
    if ax.title.get_text():
        ax.title.set_size(TITLE_FS_COMPACT)


def short_host_label(host: str) -> str:
    """压缩超长 pod/hostname，避免热力图/排序条轴标签占满半张图。

    目标形态尽量短：`m0` / `w12`（master→m，worker→w）。
    """
    import re

    if not host:
        return "?"
    h = host.split(".")[0]
    for prefix in (
        "yushan-muxi-card-screen-128-cp-copy-",
        "yushan-muxi-card-screen-",
        "whj4stu-copy-copy-copy-",
        "huawei-8node-copy-",
        "huawei-8node-",
        "ascend-",
    ):
        if h.startswith(prefix):
            h = h[len(prefix) :]
            break
    # master-N / worker-N → mN / wN
    m = re.search(r"(?:^|-)(?:master|worker)-(\d+)$", h, re.I)
    if m:
        role = "m" if re.search(r"master-\d+$", h, re.I) else "w"
        return f"{role}{m.group(1)}"
    m2 = re.search(r"(?:^|-)((?:master|worker)-\d+)$", h, re.I)
    if m2:
        tok = m2.group(1).lower()
        return ("m" if tok.startswith("master") else "w") + tok.split("-")[1]
    return h


def natural_host_key(host: str) -> tuple:
    """host 自然序：master 优先，再按编号（避免 w10 插在 w1 后）。"""
    import re

    s = short_host_label(host)
    m = re.match(r"([mw])(\d+)$", s, re.I)
    if m:
        role = 0 if m.group(1).lower() == "m" else 1
        return (role, int(m.group(2)))
    m2 = re.match(r"(master|worker)-(\d+)$", s, re.I)
    if m2:
        role = 0 if m2.group(1).lower() == "master" else 1
        return (role, int(m2.group(2)))
    return (2, s)


def annot_fontsize_for_grid(nrows: int, ncols: int) -> float:
    cells = max(1, nrows * ncols)
    if cells >= 128:
        return ANNOT_FS_DENSE
    if cells >= 48:
        return ANNOT_FS_MED
    return ANNOT_FS_SPARSE
