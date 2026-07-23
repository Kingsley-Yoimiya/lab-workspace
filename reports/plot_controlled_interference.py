#!/usr/bin/env python3
"""绘制可控 NPU 干扰的实际剂量与 victim 吞吐下降曲线。

Example:
  python3 reports/plot_controlled_interference.py \
    --result-dir /path/to/controlled-interference-run \
    --out-dir /path/to/figs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports"))
from plot_style import COLORS, apply_plot_style, save_fig, style_axes  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--filename",
        default="controlled_interference_dose_response.svg",
        help="输出 SVG 文件名",
    )
    return parser.parse_args()


def _load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"error: missing summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = _parse_args()
    series = [
        (
            "Cube sidecar → GEMM 主进程",
            _load(args.result_dir / "cube_gemm.summary.json"),
            COLORS[0],
            "o",
        ),
        (
            "HBM/MTE copy → MLP Block",
            _load(args.result_dir / "hbm_mte_block.summary.json"),
            COLORS[2],
            "s",
        ),
    ]

    apply_plot_style((11.5, 5.0))
    fig, axes = plt.subplots(1, 2)
    for label, summary, color, marker in series:
        points = summary["points"]
        targets = [100.0 * point["target_duty"] for point in points]
        actual = [
            0.0
            if point["target_duty"] == 0
            else 100.0 * point["sidecar_busy_wall_ratio_median"]
            for point in points
        ]
        drops = [point["victim_throughput_drop_pct"] for point in points]
        axes[0].plot(
            targets,
            actual,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=8,
            label=label,
        )
        axes[1].plot(
            targets,
            drops,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=8,
            label=label,
        )

    axes[0].plot([0, 50], [0, 50], color="tab:gray", linestyle="--", linewidth=1.5, label="目标=实测")
    axes[0].set_title("控制精度")
    axes[0].set_xlabel("目标 sidecar busy 比例 (%)")
    axes[0].set_ylabel("实测同步小批次 busy 比例 (%)")
    axes[0].set_xlim(-1, 52)
    axes[0].set_ylim(-1, 55)
    axes[0].legend(frameon=False, fontsize=13)

    axes[1].set_title("剂量—响应")
    axes[1].set_xlabel("目标 sidecar busy 比例 (%)")
    axes[1].set_ylabel("主进程吞吐下降 (%)")
    axes[1].set_xlim(-1, 52)
    axes[1].legend(frameon=False, fontsize=13)

    for axis in axes:
        style_axes(axis)
    fig.suptitle("预热常驻 sidecar 的可控 NPU 资源干扰", fontsize=22)
    fig.tight_layout()
    output = save_fig(fig, args.out_dir / args.filename)
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
