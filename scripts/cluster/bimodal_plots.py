#!/usr/bin/env python3
"""bimodal_plots.py — HCCL 双峰探测数据可视化 + straggler rank 分析

用法（pod 里跑）:
  python3 bimodal_plots.py --out-dir /afs/.../plots
     --dir-128 /afs/.../bimodal-128-primitive-*/
     --dir-512 /afs/.../bimodal-512-primitive-*/
     --dir-1024 /afs/.../bimodal-1024-primitive-*/

输出（PNG，每张独立文件）：
  fig1_histograms_by_scale.png   — 3x2 面板：128/512/1024 × [1M / 64M all_reduce] 直方图
  fig2_p50_p99_by_scale.png      — p50/p90/p99/max 曲线，横轴 msg size
  fig3_barrier_scale.png         — barrier 分布箱线图 128/512/1024
  fig4_straggler_heatmap.png     — rank × iter 热力图（1024 卡 all_reduce @1M）
  fig5_straggler_leaderboard.png — 1024 卡 rank 慢度排行（top slow ranks）
  fig6_rank_stability.png        — rank 慢度稳定性（跨不同 op 是否总慢）

同时输出 straggler_report.md：
  - top 20 慢 rank，跨多个 op 的 mean/p90
  - 是否有集中的 host（同节点/超节点内）
"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_scale(dirpath: Path):
    """返回 {(op, size): {rank: [host_ns iters...], ...}}"""
    per_rank = defaultdict(lambda: defaultdict(list))
    hosts = {}  # rank -> host
    for f in dirpath.glob("*.periter.jsonl"):
        m = re.search(r"rank(\d+)\.periter", f.name)
        if not m: continue
        rank = int(m.group(1))
        with f.open() as fp:
            for line in fp:
                line = line.strip()
                if not line.startswith("{"): continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("record") != "hccl_bench_iter": continue
                if d.get("profiled"): continue
                op = d.get("op")
                nb = d.get("nbytes", 0)
                shape = d.get("shape")
                key = (op, str(shape) if shape else str(nb))
                per_rank[key][rank].append(d.get("coll_host_ns"))
                if rank not in hosts:
                    hosts[rank] = d.get("host", "?")
    return per_rank, hosts


def fmt_size(nb_str):
    try: nb = int(nb_str)
    except: return nb_str
    if nb == 0: return "-"
    if nb >= 1024**2: return f"{nb//1024**2}M"
    if nb >= 1024: return f"{nb//1024}K"
    return f"{nb}B"


def all_iters(per_rank_op):
    """把 {rank: [iters]} 摊平成一个 list"""
    out = []
    for r, arr in per_rank_op.items():
        out.extend([x for x in arr if x is not None])
    return out


def plot_histograms_by_scale(data, out):
    """3x2 面板：128/512/1024 × [all_reduce @1M / @64M]"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    scales = [(128, data[128]), (512, data[512]), (1024, data[1024])]
    sizes_to_plot = [("all_reduce", "1048576", "1M"),
                     ("all_reduce", "67108864", "64M")]
    for row, (scale, pr) in enumerate(scales):
        for col, (op, size_key, size_label) in enumerate(sizes_to_plot):
            ax = axes[row, col]
            arr = all_iters(pr.get((op, size_key), {}))
            if not arr:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue
            arr_ms = np.array(arr) / 1e6  # ns → ms
            # 剪 p99 之后为 outlier（否则灾难长尾把主体压扁）
            p99 = np.percentile(arr_ms, 99)
            main = arr_ms[arr_ms <= p99 * 1.2]
            ax.hist(main, bins=80, color="#3b82f6", alpha=0.7, edgecolor="black", linewidth=0.3)
            p50 = np.median(arr_ms)
            ax.axvline(p50, color="red", linestyle="--", linewidth=1, label=f"p50={p50:.2f}ms")
            ax.axvline(p99, color="orange", linestyle="--", linewidth=1, label=f"p99={p99:.2f}ms")
            ax.set_title(f"{scale} cards | {op} @ {size_label}\nN={len(arr)}", fontsize=11)
            ax.set_xlabel("iter time (ms)")
            ax.set_ylabel("count")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
    fig.suptitle("HCCL all_reduce iter time distribution — bimodal / long tail\n(each iter's host wall time, per rank per iter)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out}")


def plot_p_curves(data, out):
    """p50/p90/p99/max 随 msg size 变化，三个 scale 一图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    sizes_map = {"4K": 4096, "64K": 65536, "1M": 1048576, "64M": 67108864, "248M": 260046848}
    colors = {128: "#059669", 512: "#2563eb", 1024: "#dc2626"}
    for scale, pr in [(128, data[128]), (512, data[512]), (1024, data[1024])]:
        p50s, p90s, p99s = [], [], []
        xs = []
        for label, size in sizes_map.items():
            arr = all_iters(pr.get(("all_reduce", str(size)), {}))
            if not arr: continue
            arr = np.array(arr) / 1e6  # ms
            p50s.append(np.percentile(arr, 50))
            p90s.append(np.percentile(arr, 90))
            p99s.append(np.percentile(arr, 99))
            xs.append(size)
        if xs:
            c = colors[scale]
            ax.plot(xs, p50s, "o-", color=c, label=f"{scale}c p50")
            ax.plot(xs, p90s, "^--", color=c, alpha=0.7, label=f"{scale}c p90")
            ax.plot(xs, p99s, "s:", color=c, alpha=0.5, label=f"{scale}c p99")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("message size (bytes)")
    ax.set_ylabel("all_reduce iter time (ms)")
    ax.set_title("all_reduce iter time vs msg size, by scale\n(log-log; p50 solid, p90 dashed, p99 dotted)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out}")


def plot_barrier_box(data, out):
    """barrier 分布箱线图 128/512/1024"""
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = []
    labels = []
    for scale, pr in [(128, data[128]), (512, data[512]), (1024, data[1024])]:
        arr = all_iters(pr.get(("barrier", "0"), {}))
        if arr:
            arr = np.array(arr) / 1e6
            # clip 到 p99 * 1.5 避免灾难长尾压扁
            p99 = np.percentile(arr, 99)
            arr_clip = arr[arr <= p99 * 1.5]
            xs.append(arr_clip)
            labels.append(f"{scale} cards\np50={np.percentile(arr,50):.2f}ms\np99={p99:.1f}ms")
    ax.boxplot(xs, labels=labels, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#93c5fd"),
               medianprops=dict(color="red", linewidth=2))
    ax.set_ylabel("barrier iter time (ms, clipped @1.5×p99)")
    ax.set_title("barrier iter time distribution vs scale\n(box = IQR, whiskers = 5-95%, red = median)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out}")


def plot_straggler_heatmap(pr_1024, hosts, out):
    """1024 卡 rank × iter 热力图 (all_reduce @1M)"""
    key = ("all_reduce", "1048576")
    d = pr_1024.get(key, {})
    if not d:
        print(f"[skip] {out}: no data")
        return
    n_iter = max(len(v) for v in d.values())
    n_rank = len(d)
    ranks_sorted = sorted(d.keys())
    mat = np.full((n_rank, n_iter), np.nan)
    for i, r in enumerate(ranks_sorted):
        arr = d[r]
        arr = np.array([x if x is not None else np.nan for x in arr]) / 1e6  # ms
        mat[i, :len(arr)] = arr
    fig, ax = plt.subplots(figsize=(14, 10))
    # clip 到 p99 让主体分辨
    p99 = np.nanpercentile(mat, 99)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                   vmin=np.nanpercentile(mat, 1), vmax=p99, interpolation="nearest")
    ax.set_xlabel("iter")
    ax.set_ylabel("rank (0..1023)")
    ax.set_title(f"1024 卡 all_reduce @1M — 每 iter 每 rank 时长 (ms, clipped @p99={p99:.1f}ms)\n"
                 f"红色 = 慢；stable slow row → straggler rank")
    plt.colorbar(im, ax=ax, label="iter time (ms)")
    # 每 128 rank 一条参考线（节点边界）
    for hostline in range(128, n_rank, 128):
        ax.axhline(hostline - 0.5, color="black", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out}")


def plot_straggler_leaderboard(pr_1024, hosts, out):
    """1024 卡按 rank 排序 mean iter time，找 straggler"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    # 上图：all_reduce @1M 各 rank mean
    key = ("all_reduce", "1048576")
    d = pr_1024.get(key, {})
    rank_stats = []
    for r, arr in d.items():
        arr = np.array([x for x in arr if x is not None]) / 1e6
        if len(arr) < 30: continue
        rank_stats.append((r, np.mean(arr), np.percentile(arr, 90), np.max(arr)))
    rank_stats.sort(key=lambda x: x[1])  # by mean
    ranks_o = [x[0] for x in rank_stats]
    means = [x[1] for x in rank_stats]
    p90s = [x[2] for x in rank_stats]
    maxes = [x[3] for x in rank_stats]

    ax = axes[0]
    ax.plot(range(len(means)), means, "-", color="#059669", label="mean", alpha=0.8, linewidth=0.5)
    ax.plot(range(len(p90s)), p90s, "-", color="#f59e0b", label="p90", alpha=0.5, linewidth=0.5)
    ax.plot(range(len(maxes)), maxes, ".", color="#dc2626", label="max", markersize=1)
    ax.set_yscale("log")
    ax.set_xlabel("rank (sorted by mean, slowest right)")
    ax.set_ylabel("iter time (ms)")
    ax.set_title(f"1024 卡 all_reduce @1M — 每 rank mean/p90/max\n（右端是稳定慢的 rank）")
    ax.legend()
    ax.grid(alpha=0.3)

    # 下图：top-20 慢 rank 按 mean（从高到低）
    ax = axes[1]
    top20 = sorted(rank_stats, key=lambda x: -x[1])[:20]
    ranks_top = [x[0] for x in top20]
    means_top = [x[1] for x in top20]
    p90s_top = [x[2] for x in top20]
    idx = np.arange(len(top20))
    w = 0.35
    ax.bar(idx - w/2, means_top, w, label="mean", color="#059669")
    ax.bar(idx + w/2, p90s_top, w, label="p90", color="#f59e0b")
    labels = [f"r{r}\n{hosts.get(r,'?').replace('yjr-1024-0729-','')}" for r in ranks_top]
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("iter time (ms)")
    ax.set_title("Top-20 慢 rank（按 mean）")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out}")


def plot_rank_stability(pr_1024, hosts, out, out_md):
    """rank 慢度在不同 op 上是否稳定：如果 rank r 在 all_reduce 慢，在 barrier 也慢吗？"""
    ops_of_interest = [
        ("all_reduce", "4096", "AR@4K"),
        ("all_reduce", "1048576", "AR@1M"),
        ("all_reduce", "67108864", "AR@64M"),
        ("all_reduce", "260046848", "AR@248M"),
        ("barrier", "0", "barrier"),
    ]
    rank_means = {}  # rank -> {op_label: mean}
    for op, size, label in ops_of_interest:
        d = pr_1024.get((op, size), {})
        for r, arr in d.items():
            arr = np.array([x for x in arr if x is not None]) / 1e6
            if len(arr) < 30: continue
            rank_means.setdefault(r, {})[label] = np.mean(arr)

    # top-30 慢 rank（按 AR@1M mean 降序）
    ranked = sorted(rank_means.keys(), key=lambda r: -rank_means.get(r, {}).get("AR@1M", 0))[:30]
    labels = [x[2] for x in ops_of_interest]
    mat = np.zeros((len(ranked), len(labels)))
    for i, r in enumerate(ranked):
        for j, l in enumerate(labels):
            mat[i, j] = rank_means.get(r, {}).get(l, np.nan)

    fig, ax = plt.subplots(figsize=(12, 10))
    # normalize per column (z-score) 让不同 op 可比
    col_med = np.nanmedian(mat, axis=0)
    col_std = np.nanstd(mat, axis=0)
    z = (mat - col_med) / np.where(col_std > 0, col_std, 1)
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels([f"r{r} ({hosts.get(r,'?').replace('yjr-1024-0729-','')})" for r in ranked], fontsize=7)
    ax.set_title("Top-30 慢 rank（按 AR@1M）在各 op 上的相对慢度（z-score）\n红=显著慢；跨 op 稳定红 → 结构性慢 rank")
    plt.colorbar(im, ax=ax, label="z-score (0=median)")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out}")

    # md 报告
    lines = ["# 1024 卡 straggler rank 分析", ""]
    lines.append("## Top-30 慢 rank（按 all_reduce @1M mean）")
    lines.append("| rank | host | AR@4K | AR@1M | AR@64M | AR@248M | barrier |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in ranked:
        h = hosts.get(r, "?").replace("yjr-1024-0729-", "")
        row = [f"r{r}", h]
        for _, _, l in ops_of_interest:
            v = rank_means.get(r, {}).get(l)
            row.append(f"{v:.2f}" if v else "-")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## 稳定慢 rank（跨 3+ op 都是 top-30 慢）")
    per_op_top = {}
    for _, _, l in ops_of_interest:
        s = sorted(rank_means.keys(), key=lambda r: -rank_means.get(r, {}).get(l, 0))[:30]
        per_op_top[l] = set(s)
    from collections import Counter
    hits = Counter()
    for l, ranks_s in per_op_top.items():
        for r in ranks_s:
            hits[r] += 1
    stable = [r for r, c in hits.most_common() if c >= 3]
    lines.append(f"共 {len(stable)} 个 rank 在 ≥3 个 op 上位列 top-30 慢：")
    for r in stable[:30]:
        h = hosts.get(r, "?").replace("yjr-1024-0729-", "")
        lines.append(f"- rank {r} @ {h}: hit {hits[r]} ops top-30")

    lines.append("\n## Host-level 汇总（哪个节点最慢）")
    host_ranks_slow = defaultdict(list)
    for r in ranked:
        h = hosts.get(r, "?").replace("yjr-1024-0729-", "")
        host_ranks_slow[h].append(r)
    lines.append(f"top-30 慢 rank 的 host 分布：")
    for h, rs in sorted(host_ranks_slow.items(), key=lambda x: -len(x[1])):
        lines.append(f"- {h}: {len(rs)} 个 slow rank ({rs[:8]}...)")

    Path(out_md).write_text("\n".join(lines))
    print(f"[write] {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-128", required=True)
    ap.add_argument("--dir-512", required=True)
    ap.add_argument("--dir-1024", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)

    print(f"[load] 128 卡 from {args.dir_128}")
    pr128, hosts128 = load_scale(Path(args.dir_128))
    print(f"[load] 512 卡 from {args.dir_512}")
    pr512, hosts512 = load_scale(Path(args.dir_512))
    print(f"[load] 1024 卡 from {args.dir_1024}")
    pr1024, hosts1024 = load_scale(Path(args.dir_1024))
    data = {128: pr128, 512: pr512, 1024: pr1024}

    plot_histograms_by_scale(data, outd / "fig1_histograms_by_scale.png")
    plot_p_curves(data, outd / "fig2_p50_p99_by_scale.png")
    plot_barrier_box(data, outd / "fig3_barrier_scale.png")
    plot_straggler_heatmap(pr1024, hosts1024, outd / "fig4_straggler_heatmap.png")
    plot_straggler_leaderboard(pr1024, hosts1024, outd / "fig5_straggler_leaderboard.png")
    plot_rank_stability(pr1024, hosts1024,
                        outd / "fig6_rank_stability.png",
                        outd / "straggler_report.md")
    print("== DONE ==")


if __name__ == "__main__":
    main()
