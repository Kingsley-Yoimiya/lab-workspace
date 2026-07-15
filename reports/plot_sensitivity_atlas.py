#!/usr/bin/env python3
"""绘制资源干扰 × victim 的吞吐、长尾、非线性和条件差异图谱。

Example:
  python3 reports/plot_sensitivity_atlas.py \
    --analysis /path/to/atlas_analysis.json \
    --out-dir /path/to/figs
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports"))
from plot_style import (  # noqa: E402
    ANNOT_FS_MED,
    apply_plot_style,
    save_fig,
    style_heatmap_axes,
)


INJECTORS = ("cube", "vector", "hbm_mte", "hbm_vector", "small_ops")
WORKLOADS = ("gemm", "attention", "norm", "elementwise", "block", "transformer")
PANELS = (
    ("small", "periodic"),
    ("small", "poisson"),
    ("large", "periodic"),
    ("large", "poisson"),
)
INJECTOR_LABELS = ("Cube", "Vector", "HBM/MTE", "HBM+Vector", "Small ops")
WORKLOAD_LABELS = ("GEMM", "Attention", "LayerNorm", "Elementwise", "MLP Block", "Transformer")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _matrix(rows: list[dict], field: str, profile: str, pattern: str) -> list[list[float]]:
    index = {
        (row["inject_kind"], row["workload"], row["profile"], row["pattern"]): row
        for row in rows
    }
    return [
        [index[injector, workload, profile, pattern][field] for injector in INJECTORS]
        for workload in WORKLOADS
    ]


def _annot_color(value: float, vmin: float, vmax: float, cmap_name: str) -> str:
    """按色块相对亮度选黑/白字，避免深色格子上黑字读不清。"""
    import matplotlib as mpl

    span = vmax - vmin
    norm = 0.0 if span <= 0 else (value - vmin) / span
    red, green, blue, _alpha = mpl.colormaps[cmap_name](norm)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.48 else "#1a1a1a"


def _heatmap_4(
    rows: list[dict],
    field: str,
    title: str,
    colorbar_label: str,
    output: Path,
    *,
    cmap: str,
) -> None:
    matrices = [_matrix(rows, field, profile, pattern) for profile, pattern in PANELS]
    values = [value for matrix in matrices for row in matrix for value in row]
    vmin, vmax = min(values), max(values)
    apply_plot_style((17, 11.5))
    fig = plt.figure()
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1, 1, 0.045),
        wspace=0.26,
        hspace=0.34,
    )
    axes = [
        fig.add_subplot(grid[row_index, col_index])
        for row_index in range(2)
        for col_index in range(2)
    ]
    colorbar_axis = fig.add_subplot(grid[:, 2])
    image = None
    for axis, matrix, (profile, pattern) in zip(axes, matrices, PANELS):
        image = axis.imshow(
            matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_xticks(range(len(INJECTORS)), INJECTOR_LABELS, rotation=25, ha="right")
        axis.set_yticks(range(len(WORKLOADS)), WORKLOAD_LABELS)
        axis.set_title(f"{profile} / {pattern}")
        style_heatmap_axes(axis)
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                text = f"{value:.1f}" if abs(value) >= 1 else f"{value:.2f}"
                axis.text(
                    col_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=ANNOT_FS_MED,
                    color=_annot_color(value, vmin, vmax, cmap),
                )
    fig.suptitle(title, fontsize=22, y=0.975)
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(colorbar_label, fontsize=16, labelpad=14)
    colorbar.ax.tick_params(labelsize=14)
    fig.subplots_adjust(left=0.085, right=0.92, bottom=0.10, top=0.91)
    save_fig(fig, output)


def _condition_delta(rows: list[dict], output: Path) -> None:
    index = {
        (row["inject_kind"], row["workload"], row["profile"], row["pattern"]): row
        for row in rows
    }
    shape_matrix = []
    pattern_matrix = []
    field = "throughput_drop_pct_at_target_0_5_or_max"
    for workload in WORKLOADS:
        shape_row = []
        pattern_row = []
        for injector in INJECTORS:
            shape_row.append(
                st.mean(
                    index[injector, workload, "large", pattern][field]
                    - index[injector, workload, "small", pattern][field]
                    for pattern in ("periodic", "poisson")
                )
            )
            pattern_row.append(
                st.mean(
                    index[injector, workload, profile, "poisson"][field]
                    - index[injector, workload, profile, "periodic"][field]
                    for profile in ("small", "large")
                )
            )
        shape_matrix.append(shape_row)
        pattern_matrix.append(pattern_row)

    limit = max(
        abs(value)
        for matrix in (shape_matrix, pattern_matrix)
        for row in matrix
        for value in row
    )
    apply_plot_style((16.5, 6.8))
    fig = plt.figure()
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1, 1, 0.045),
        wspace=0.28,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    colorbar_axis = fig.add_subplot(grid[0, 2])
    images = []
    for axis, matrix, title in zip(
        axes,
        (shape_matrix, pattern_matrix),
        ("Large − Small 敏感度差", "Poisson − Periodic 敏感度差"),
    ):
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        images.append(image)
        axis.set_xticks(range(len(INJECTORS)), INJECTOR_LABELS, rotation=25, ha="right")
        axis.set_yticks(range(len(WORKLOADS)), WORKLOAD_LABELS)
        axis.set_title(title)
        style_heatmap_axes(axis)
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                axis.text(
                    col_index,
                    row_index,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=ANNOT_FS_MED,
                    color="black",
                )
    fig.suptitle("Shape 与随机 burst 对吞吐敏感度的改变", fontsize=22, y=0.97)
    colorbar = fig.colorbar(images[-1], cax=colorbar_axis)
    colorbar.set_label("50% 剂量下降差（百分点）", fontsize=16, labelpad=14)
    colorbar.ax.tick_params(labelsize=14)
    fig.subplots_adjust(left=0.085, right=0.92, bottom=0.17, top=0.84)
    save_fig(fig, output)


def main() -> int:
    args = _parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    rows = analysis["rows"]
    if len(rows) != 120:
        raise SystemExit(f"error: expected 120 atlas rows, got {len(rows)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _heatmap_4(
        rows,
        "throughput_drop_pct_at_target_0_5_or_max",
        "50% 资源干扰下的主进程吞吐敏感度",
        "主进程吞吐下降（%）",
        args.out_dir / "sensitivity_atlas_throughput.svg",
        cmap="YlOrRd",
    )
    _heatmap_4(
        rows,
        "max_dose_p99_amplification",
        "50% 资源干扰下的 step p99 长尾放大",
        "p99 / 无干扰 p99（倍）",
        args.out_dir / "sensitivity_atlas_tail.svg",
        # 浅→深青绿序列：与吞吐 YlOrRd 区分，且避免 magma 深紫黑底
        cmap="YlGnBu",
    )
    _heatmap_4(
        rows,
        "linear_max_residual_relative_to_peak",
        "剂量曲线偏离线性的程度",
        "最大线性残差 / 最大下降",
        args.out_dir / "sensitivity_atlas_nonlinearity.svg",
        cmap="PuBuGn",
    )
    _condition_delta(rows, args.out_dir / "sensitivity_atlas_conditions.svg")
    print(f"output_dir: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
