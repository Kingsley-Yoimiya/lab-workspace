#!/usr/bin/env python3
"""HCCL P2P cluster R0 出图（仅图 + stats.json，无叙事 md）。

输入: P2P JSONL（record=hccl_p2p），含 src/dst/nbytes/avg_s/bw_GBps/lat_us/ok/...
输出: reports/rounds/hccl_cluster_r0_figs/
  - heatmap_host_host_lat.png / heatmap_host_host_bw.png
  - heatmap_rank_rank_lat.png（稀疏边）
  - bar_slow_edges_topk.png
  - stats.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parents[1]  # lab-workspace
DEFAULT_FIG_DIR = REPO / "reports" / "rounds" / "hccl_cluster_r0_figs"
# 默认扫本机 logs 下最新 hccl-cluster-r0-* /results
DEFAULT_LOG_GLOB = REPO.parent.parent / "logs"


def _short_host(host: str) -> str:
    return host.replace("huawei-8node-copy-", "").replace(".huawei-8node-copy", "")


def load_records(data_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = sorted(data_dir.glob("**/*.jsonl"))
    if not paths and data_dir.is_file():
        paths = [data_dir]
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rec = obj.get("record")
                if rec is not None and rec != "hccl_p2p":
                    continue
                if "src" not in obj or "dst" not in obj:
                    continue
                records.append(obj)
    return records


def prefer_role(records: list[dict[str, Any]], role: str = "recv") -> list[dict[str, Any]]:
    """同一 (src,dst,nbytes) 优先取 recv（带 ok 校验）；否则取任意。"""
    buckets: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for r in records:
        key = (int(r["src"]), int(r["dst"]), int(r["nbytes"]))
        buckets[key].append(r)
    out: list[dict[str, Any]] = []
    for _k, group in buckets.items():
        chosen = None
        for g in group:
            if g.get("role") == role:
                chosen = g
                break
        if chosen is None:
            chosen = group[0]
        out.append(chosen)
    return out


def compute_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "cv_pct": 0}
    mean = float(np.mean(arr))
    return {
        "mean": mean,
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "cv_pct": float(np.std(arr, ddof=0) / mean * 100) if mean else 0.0,
    }


def rel_dev(value: float, median: float) -> float:
    if median == 0:
        return 0.0
    return (value - median) / median * 100.0


def rank_to_host(records: list[dict[str, Any]]) -> dict[int, str]:
    """用发送/接收侧 host 推断 rank→host（local_rank + 出现过的 host）。"""
    # 优先: 记录里 rank 字段对应 host
    mapping: dict[int, str] = {}
    for r in records:
        if "rank" in r and "host" in r:
            mapping[int(r["rank"])] = str(r["host"])
    if mapping:
        return mapping
    # fallback: 假设 16 卡/节点，按 src 的 host 不够；用 world 内出现的 host 列表
    return mapping


def infer_host_for_rank(
    rank: int,
    rank_host: dict[int, str],
    records: list[dict[str, Any]],
) -> str:
    if rank in rank_host:
        return rank_host[rank]
    # 从边记录猜：若某条边的 src==rank 且 role=send，host 即该 rank
    for r in records:
        if int(r.get("src", -1)) == rank and r.get("role") == "send" and "host" in r:
            return str(r["host"])
        if int(r.get("dst", -1)) == rank and r.get("role") == "recv" and "host" in r:
            return str(r["host"])
    return f"rank{rank}"


def pick_nbytes(records: list[dict[str, Any]], prefer_small: bool) -> int | None:
    sizes = sorted({int(r["nbytes"]) for r in records})
    if not sizes:
        return None
    return sizes[0] if prefer_small else sizes[-1]


def plot_host_host_heatmap(
    edges: list[dict[str, Any]],
    rank_host: dict[int, str],
    metric: str,
    title: str,
    out: Path,
    relative: bool = True,
) -> str:
    hosts = sorted(
        {infer_host_for_rank(int(e["src"]), rank_host, edges) for e in edges}
        | {infer_host_for_rank(int(e["dst"]), rank_host, edges) for e in edges}
    )
    # 聚合 host×host：同 host 对取均值
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for e in edges:
        hs = infer_host_for_rank(int(e["src"]), rank_host, edges)
        hd = infer_host_for_rank(int(e["dst"]), rank_host, edges)
        buckets[(hs, hd)].append(float(e[metric]))

    vals = [float(np.mean(v)) for v in buckets.values()]
    med = float(np.median(vals)) if vals else 0.0

    n = len(hosts)
    matrix = np.full((n, n), np.nan)
    for (hs, hd), vs in buckets.items():
        i, j = hosts.index(hs), hosts.index(hd)
        v = float(np.mean(vs))
        matrix[i, j] = rel_dev(v, med) if relative else v

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))
    if relative:
        lim = max(5.0, float(np.nanmax(np.abs(matrix))) if np.any(~np.isnan(matrix)) else 5.0)
        # 延迟：正偏差=更慢=红；带宽：正偏差=更快，用 RdYlGn 时带宽取反显示「慢」
        if metric == "bw_GBps":
            display = -matrix  # 低于中位 → 正（红）
            cmap = "RdYlGn_r"
        else:
            display = matrix
            cmap = "RdYlGn_r"
        im = ax.imshow(display, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
        cbar_label = "相对中位数偏差 (%)"
    else:
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        cbar_label = metric
        display = matrix

    labels = [_short_host(h) for h in hosts]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("dst host")
    ax.set_ylabel("src host")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            v = display[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:+.1f}" if relative else f"{v:.2f}",
                        ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name


def plot_rank_rank_heatmap(
    edges: list[dict[str, Any]],
    metric: str,
    title: str,
    out: Path,
) -> str:
    ranks = sorted({int(e["src"]) for e in edges} | {int(e["dst"]) for e in edges})
    idx = {r: i for i, r in enumerate(ranks)}
    n = len(ranks)
    matrix = np.full((n, n), np.nan)
    for e in edges:
        matrix[idx[int(e["src"])], idx[int(e["dst"])]] = float(e[metric])

    fig, ax = plt.subplots(figsize=(max(6, n * 0.12 + 4), max(5, n * 0.12 + 3)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    step = max(1, n // 16)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(ranks[i]) for i in ticks], fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(ranks[i]) for i in ticks], fontsize=7)
    ax.set_xlabel("dst rank")
    ax.set_ylabel("src rank")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name


def plot_slow_edges_topk(
    edges: list[dict[str, Any]],
    rank_host: dict[int, str],
    metric: str,
    topk: int,
    title: str,
    out: Path,
    higher_is_worse: bool,
) -> tuple[str, list[dict[str, Any]]]:
    sorted_edges = sorted(edges, key=lambda e: float(e[metric]), reverse=higher_is_worse)
    top = sorted_edges[:topk]
    labels = []
    vals = []
    rows = []
    med = float(np.median([float(e[metric]) for e in edges])) if edges else 0.0
    for e in top:
        src, dst = int(e["src"]), int(e["dst"])
        hs = _short_host(infer_host_for_rank(src, rank_host, edges))
        hd = _short_host(infer_host_for_rank(dst, rank_host, edges))
        labels.append(f"{src}→{dst}\n{hs}→{hd}")
        vals.append(float(e[metric]))
        rows.append({
            "src": src,
            "dst": dst,
            "src_host": infer_host_for_rank(src, rank_host, edges),
            "dst_host": infer_host_for_rank(dst, rank_host, edges),
            metric: float(e[metric]),
            "relmed_pct": rel_dev(float(e[metric]), med),
            "ok": bool(e.get("ok", True)),
            "nbytes": int(e["nbytes"]),
        })

    fig, ax = plt.subplots(figsize=(max(8, topk * 0.55), 5))
    colors = ["#E45756" if (v >= med if higher_is_worse else v <= med) else "#4C78A8" for v in vals]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(med, color="#333", linestyle="--", linewidth=1, label=f"median={med:.3g}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name, rows


def find_default_data_dir() -> Path | None:
    root = DEFAULT_LOG_GLOB
    if not root.exists():
        return None
    cands = sorted(root.glob("hccl-cluster-r0-*/results"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None, help="含 P2P JSONL 的目录")
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    data_dir = args.data_dir or find_default_data_dir()
    if data_dir is None or not Path(data_dir).exists():
        raise SystemExit(
            "未找到数据目录。请传 --data-dir <path>，"
            "或先跑 run_hccl_p2p_128.sh 产出 logs/hccl-cluster-r0-*/results"
        )
    data_dir = Path(data_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = load_records(data_dir)
    if not raw:
        raise SystemExit(f"无 P2P 记录: {data_dir}")
    edges_all = prefer_role(raw, role="recv")
    rank_host = rank_to_host(raw)

    nbytes_lat = pick_nbytes(edges_all, prefer_small=True)
    nbytes_bw = pick_nbytes(edges_all, prefer_small=False)
    edges_lat = [e for e in edges_all if int(e["nbytes"]) == nbytes_lat] if nbytes_lat else []
    edges_bw = [e for e in edges_all if int(e["nbytes"]) == nbytes_bw] if nbytes_bw else []

    figs: list[str] = []
    topk_lat: list[dict] = []
    topk_bw: list[dict] = []

    if edges_lat:
        figs.append(plot_host_host_heatmap(
            edges_lat, rank_host, "lat_us",
            f"host×host 延迟相对中位数偏差\n(nbytes={nbytes_lat})",
            fig_dir / "heatmap_host_host_lat.png",
        ))
        figs.append(plot_rank_rank_heatmap(
            edges_lat, "lat_us",
            f"rank×rank 延迟 (us)\n(nbytes={nbytes_lat})",
            fig_dir / "heatmap_rank_rank_lat.png",
        ))
        name, topk_lat = plot_slow_edges_topk(
            edges_lat, rank_host, "lat_us", args.topk,
            f"慢边 Top{args.topk}（延迟 lat_us，nbytes={nbytes_lat}）",
            fig_dir / "bar_slow_edges_topk_lat.png",
            higher_is_worse=True,
        )
        figs.append(name)

    if edges_bw:
        figs.append(plot_host_host_heatmap(
            edges_bw, rank_host, "bw_GBps",
            f"host×host 带宽相对中位数偏差（慢=红）\n(nbytes={nbytes_bw})",
            fig_dir / "heatmap_host_host_bw.png",
        ))
        figs.append(plot_rank_rank_heatmap(
            edges_bw, "bw_GBps",
            f"rank×rank 带宽 (GB/s)\n(nbytes={nbytes_bw})",
            fig_dir / "heatmap_rank_rank_bw.png",
        ))
        name, topk_bw = plot_slow_edges_topk(
            edges_bw, rank_host, "bw_GBps", args.topk,
            f"慢边 Top{args.topk}（带宽 bw_GBps 升序最慢，nbytes={nbytes_bw}）",
            fig_dir / "bar_slow_edges_topk_bw.png",
            higher_is_worse=False,
        )
        figs.append(name)

    ok_rate = sum(1 for e in edges_all if e.get("ok", True)) / len(edges_all) if edges_all else 0.0
    stats = {
        "data_dir": str(data_dir),
        "n_raw_lines": len(raw),
        "n_edges_dedup": len(edges_all),
        "world_size": max((int(e.get("world_size", 0)) for e in edges_all), default=0),
        "nbytes_lat": nbytes_lat,
        "nbytes_bw": nbytes_bw,
        "ok_rate": ok_rate,
        "lat_us": compute_stats([float(e["lat_us"]) for e in edges_lat]) if edges_lat else {},
        "bw_GBps": compute_stats([float(e["bw_GBps"]) for e in edges_bw]) if edges_bw else {},
        "topk_slow_lat": topk_lat,
        "topk_slow_bw": topk_bw,
        "figures": figs,
    }
    stats_path = fig_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {stats_path}")
    for f in figs:
        print(f"  fig: {f}")


if __name__ == "__main__":
    main()
