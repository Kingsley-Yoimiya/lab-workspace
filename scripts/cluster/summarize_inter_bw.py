#!/usr/bin/env python3
"""汇总机内/机间 P2P 带宽探针 JSONL，输出摘要表 + 可选 SVG。"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_REPORTS = Path(__file__).resolve().parents[2] / "reports"
if str(_REPORTS) not in sys.path:
    sys.path.insert(0, str(_REPORTS))

from plot_style import apply_plot_style, hatch_bar_kwargs, save_fig, style_axes


def load_rows(paths: list[Path], prefer_recv: bool = True) -> list[dict]:
    rows = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("record") != "hccl_inter_bw":
                continue
            role = r.get("role")
            if role == "pp_a":
                rows.append(r)
            elif role == "both" and int(r.get("rank", -1)) == int(r.get("src", -2)):
                rows.append(r)
            elif prefer_recv and role == "recv":
                rows.append(r)
            elif (not prefer_recv) and role == "send":
                rows.append(r)
    return rows


def summarize(rows: list[dict]) -> dict:
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = (r["kind"], int(r["nbytes"]))
        buckets[key].append(float(r["bw_GBps"]))

    summary = {}
    for (kind, nbytes), vals in sorted(buckets.items()):
        summary[f"{kind}@{nbytes}"] = {
            "kind": kind,
            "nbytes": nbytes,
            "n": len(vals),
            "median_GBps": statistics.median(vals),
            "mean_GBps": statistics.mean(vals),
            "min_GBps": min(vals),
            "max_GBps": max(vals),
            "p10_GBps": sorted(vals)[max(0, int(0.1 * len(vals)) - 1)] if vals else 0,
            "p90_GBps": sorted(vals)[min(len(vals) - 1, int(0.9 * len(vals)))] if vals else 0,
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="probe.jsonl.rank*.jsonl or merged")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--plot", default="")
    args = ap.parse_args()

    paths = []
    for spec in args.inputs:
        p = Path(spec)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.jsonl")))
        else:
            paths.append(p)
    rows = load_rows(paths)
    summary = summarize(rows)

    print(f"loaded rows={len(rows)} (prefer recv for uni)")
    print(f"{'kind':<6} {'size':>10} {'n':>4} {'med':>8} {'mean':>8} {'min':>8} {'max':>8} GB/s")
    for k, s in summary.items():
        size = s["nbytes"]
        size_s = f"{size/1024**2:.0f}M" if size >= 1024**2 else f"{size/1024:.0f}K"
        print(
            f"{s['kind']:<6} {size_s:>10} {s['n']:>4} "
            f"{s['median_GBps']:8.2f} {s['mean_GBps']:8.2f} "
            f"{s['min_GBps']:8.2f} {s['max_GBps']:8.2f}"
        )

    # 机间/机内比
    for nbytes in sorted({s["nbytes"] for s in summary.values()}):
        intra = summary.get(f"intra@{nbytes}")
        inter = summary.get(f"inter@{nbytes}")
        if intra and inter and intra["median_GBps"] > 0:
            ratio = inter["median_GBps"] / intra["median_GBps"]
            print(
                f"ratio inter/intra @{nbytes}: "
                f"{inter['median_GBps']:.2f}/{intra['median_GBps']:.2f} = {ratio:.3f}"
            )

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps({"n_rows": len(rows), "summary": summary, "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.out_json}")

    if args.out_md:
        lines = [
            "# 机内 vs 机间 P2P 带宽探针摘要",
            "",
            f"- 样本数（uni 取 recv 侧）: {len(rows)}",
            "",
            "| kind | size | n | median GB/s | mean | min | max |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for s in summary.values():
            size = s["nbytes"]
            size_s = f"{size/1024**2:.0f}M" if size >= 1024**2 else f"{size/1024:.0f}K"
            lines.append(
                f"| {s['kind']} | {size_s} | {s['n']} | {s['median_GBps']:.2f} | "
                f"{s['mean_GBps']:.2f} | {s['min_GBps']:.2f} | {s['max_GBps']:.2f} |"
            )
        Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.out_md}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib missing, skip plot")
            return
        apply_plot_style()
        kinds = sorted({s["kind"] for s in summary.values()})
        sizes = sorted({s["nbytes"] for s in summary.values()})
        x = range(len(sizes))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, kind in enumerate(kinds):
            ys = [
                summary.get(f"{kind}@{sz}", {}).get("median_GBps", 0) for sz in sizes
            ]
            ax.bar([xi + (i - 0.5) * width for xi in x], ys,
                   **hatch_bar_kwargs(i, width=width), label=kind)
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [f"{sz/1024**2:.0f}M" if sz >= 1024**2 else f"{sz/1024:.0f}K" for sz in sizes]
        )
        ax.set_ylabel("median bandwidth (GB/s)")
        ax.set_xlabel("message size")
        ax.set_title("Intra vs Inter HCCL P2P bandwidth")
        ax.legend()
        style_axes(ax)
        fig.tight_layout()
        plot_path = Path(args.plot)
        save_fig(fig, plot_path)
        print(f"wrote {plot_path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
