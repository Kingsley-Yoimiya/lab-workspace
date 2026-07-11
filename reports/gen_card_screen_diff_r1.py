#!/usr/bin/env python3
"""CARD_SCREEN Track A R1：按 (B,M,N,K) 切片生成 diff-first 图与 stats JSON。

只产出图与 stats，不写叙事 markdown（由 parent 报告负责）。
规范见 reports/research/viz_diff_first_norm.md。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

REPORTS = Path(__file__).resolve().parent
DEFAULT_FIG_DIR = REPORTS / "rounds" / "card_screen_diff_r1_figs"
TOP_K = 10


def shape_key(s: dict[str, Any]) -> str:
    return (
        f"B{s['B']}_M{s['M']}_N{s['N']}_K{s['K']}"
        f"_{s.get('layout', 'NN')}_{s.get('dtype', 'bf16')}"
    )


def shape_label(s: dict[str, Any]) -> str:
    return (
        f"(B,M,N,K)=({s['B']},{s['M']},{s['N']},{s['K']}) "
        f"{s.get('layout', 'NN')}/{s.get('dtype', 'bf16')}"
    )


def short_host(host: str) -> str:
    return host.replace("huawei-8node-copy-", "")


def safe_slug(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def rel_deviation(value: float, median: float) -> float:
    if median == 0:
        return 0.0
    return (value - median) / median * 100.0


def compute_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    return {
        "n": int(arr.size),
        "mean": mean,
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "cv_pct": float(np.std(arr, ddof=0) / mean * 100) if mean else 0.0,
    }


def load_bnmk_samples(paths: list[Path]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("record") != "gemm_bnmk_sample":
                    continue
                if obj.get("tflops") is None:
                    continue
                samples.append(obj)
    return samples


def group_by_shape(
    samples: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        groups[shape_key(s)].append(s)
    return dict(groups)


def plot_heatmap(samples: list[dict[str, Any]], median: float,
                 title: str, out: Path) -> None:
    hosts = sorted({s["host"] for s in samples})
    devices = sorted({s["device"] for s in samples})
    matrix = np.full((len(hosts), len(devices)), np.nan)
    for s in samples:
        hi = hosts.index(s["host"])
        di = devices.index(s["device"])
        matrix[hi, di] = rel_deviation(float(s["tflops"]), median)

    fig, ax = plt.subplots(figsize=(max(8, len(devices) * 0.7),
                                    max(4, len(hosts) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-5, vmax=5)
    ax.set_xticks(range(len(devices)))
    ax.set_xticklabels([str(d) for d in devices])
    ax.set_yticks(range(len(hosts)))
    ax.set_yticklabels([short_host(h) for h in hosts], fontsize=8)
    ax.set_xlabel("device")
    ax.set_ylabel("host")
    ax.set_title(f"{title}\nhost×device 相对中位数偏差 (%)")
    for i in range(len(hosts)):
        for j in range(len(devices)):
            v = matrix[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6,
                        color="black" if abs(v) < 3 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sorted_bars(samples: list[dict[str, Any]], median: float,
                     title: str, out: Path) -> None:
    ordered = sorted(samples, key=lambda s: float(s["tflops"]))
    vals = [float(s["tflops"]) for s in ordered]
    labels = [f"{short_host(s['host'])}:d{s['device']}" for s in ordered]
    fig, ax = plt.subplots(figsize=(max(10, len(vals) * 0.12), 5))
    ax.bar(range(len(vals)), vals, color="#4C78A8", width=0.9)
    ax.axhline(median, color="#E45756", linestyle="--", linewidth=1.5,
               label=f"median={median:.1f}")
    ax.set_ylabel("TFLOPS")
    ax.set_title(f"{title}\n按卡升序 TFLOPS")
    step = max(1, len(labels) // 16)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)],
                       rotation=60, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def topk_rows(samples: list[dict[str, Any]], median: float,
              ascending: bool, k: int = TOP_K) -> list[dict[str, Any]]:
    ordered = sorted(samples, key=lambda s: float(s["tflops"]),
                     reverse=not ascending)[:k]
    rows = []
    for s in ordered:
        v = float(s["tflops"])
        rows.append({
            "host": s["host"],
            "device": s["device"],
            "tflops": v,
            "rel_med_pct": rel_deviation(v, median),
        })
    return rows


def process_shape(key: str, samples: list[dict[str, Any]],
                  fig_dir: Path) -> dict[str, Any]:
    slug = safe_slug(key)
    vals = [float(s["tflops"]) for s in samples]
    stats = compute_stats(vals)
    median = stats["median"]
    label = shape_label(samples[0])

    heat_name = f"heatmap_host_device_relmed_{slug}.png"
    bar_name = f"bar_sorted_cards_{slug}.png"
    plot_heatmap(samples, median, label, fig_dir / heat_name)
    plot_sorted_bars(samples, median, label, fig_dir / bar_name)

    return {
        "shape_key": key,
        "B": samples[0]["B"],
        "M": samples[0]["M"],
        "N": samples[0]["N"],
        "K": samples[0]["K"],
        "layout": samples[0].get("layout", "NN"),
        "dtype": samples[0].get("dtype", "bf16"),
        "metric": "tflops",
        "stats": stats,
        "topk_slow": topk_rows(samples, median, ascending=True),
        "topk_fast": topk_rows(samples, median, ascending=False),
        "figures": {
            "heatmap": heat_name,
            "bar_sorted": bar_name,
        },
    }


def resolve_jsonl_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.jsonl:
        for p in args.jsonl:
            path = Path(p)
            if path.is_file():
                paths.append(path)
    if args.data_dir:
        root = Path(args.data_dir)
        paths.extend(sorted(root.glob(args.glob)))
    # de-dupe while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate per-BNMK-shape diff-first plots + stats JSON")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Directory containing JSONL files")
    ap.add_argument("--glob", type=str, default="*.jsonl",
                    help="Glob under --data-dir (default: *.jsonl)")
    ap.add_argument("--jsonl", nargs="*", default=None,
                    help="Explicit JSONL file paths")
    ap.add_argument("--fig-dir", type=str, default=str(DEFAULT_FIG_DIR),
                    help="Output directory for figures + stats.json")
    args = ap.parse_args()

    paths = resolve_jsonl_paths(args)
    if not paths:
        raise SystemExit(
            "No JSONL inputs. Pass --data-dir and/or --jsonl pointing to "
            "files that contain record=gemm_bnmk_sample lines.")

    samples = load_bnmk_samples(paths)
    if not samples:
        raise SystemExit(
            f"No gemm_bnmk_sample rows with tflops in {len(paths)} file(s).")

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    groups = group_by_shape(samples)
    all_stats: dict[str, Any] = {
        "n_samples": len(samples),
        "n_shapes": len(groups),
        "inputs": [str(p) for p in paths],
        "shapes": {},
    }
    for key in sorted(groups):
        all_stats["shapes"][key] = process_shape(key, groups[key], fig_dir)

    stats_path = fig_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print(f"shapes={len(groups)} samples={len(samples)}")
    print(f"wrote {stats_path}")
    for key, meta in all_stats["shapes"].items():
        print(f"  {key}: heatmap={meta['figures']['heatmap']} "
              f"bar={meta['figures']['bar_sorted']} "
              f"cv={meta['stats']['cv_pct']:.3f}%")


if __name__ == "__main__":
    main()
