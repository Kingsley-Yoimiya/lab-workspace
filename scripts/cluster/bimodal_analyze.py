#!/usr/bin/env python3
"""bimodal_analyze.py — 双峰检测分析器

用法（在 pod 里跑，或本机 mac 上跑）：
  python3 bimodal_analyze.py /path/to/hccl-bimodal-*/  [--out /path/to/report.md]

输出：
  - per-(op, size) 时延统计：p10/p50/p90/p99、min/max
  - 双峰检测：Hartigans's dip 简化版（用 quantile-mode gap 判定）
  - 直方图 ASCII 打印（20 bins）
  - 生成 markdown 报告
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_iters(dirpath: Path):
    """加载所有 rank 的 periter.jsonl，返回按 (op, size_or_shape) 分组的 host_ns 列表。"""
    groups = defaultdict(list)
    files = list(dirpath.glob("*.periter.jsonl"))
    print(f"[load] found {len(files)} rank jsonl files")
    for f in files:
        with f.open() as fp:
            for line in fp:
                line = line.strip()
                if not line.startswith("{"): continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("record") != "hccl_bench_iter": continue
                if d.get("profiled"): continue  # msprof 的 iter 排除
                op = d.get("op")
                shape = d.get("shape") or d.get("nbytes")
                key = (op, str(shape))
                groups[key].append(d.get("coll_host_ns"))
    return groups


def _percentiles(arr, pcts):
    arr = sorted(x for x in arr if x is not None)
    n = len(arr)
    if n == 0: return {p: None for p in pcts}
    return {p: arr[min(n - 1, int(round(p / 100.0 * (n - 1))))] for p in pcts}


def _detect_bimodal(arr, min_iters=30):
    """简易双峰检测：
       用 (p90-p50)/(p50-p10) 作为"尾部相对头部拉伸"指标，>2.5 视为有双峰。
       同时看 mode-gap = argmax_2 - argmax_1 是否 > 1σ。
    """
    if len(arr) < min_iters:
        return {"bimodal": None, "reason": f"too_few_iters {len(arr)}"}
    p = _percentiles(arr, [10, 25, 50, 75, 90, 99])
    ratio = (p[90] - p[50]) / max(1, p[50] - p[10])
    iqr = p[75] - p[25]
    sd = statistics.stdev(arr) if len(arr) > 1 else 0
    # 分箱找 modes
    lo, hi = min(arr), max(arr)
    nbins = 20
    if hi <= lo:
        return {"bimodal": False, "ratio": ratio, "sd": sd, "reason": "no_spread"}
    binw = (hi - lo) / nbins
    counts = [0] * nbins
    for x in arr:
        b = min(nbins - 1, int((x - lo) / binw))
        counts[b] += 1
    # find peak locations
    peaks = []
    for i in range(nbins):
        left = counts[i - 1] if i > 0 else 0
        right = counts[i + 1] if i < nbins - 1 else 0
        if counts[i] > left and counts[i] > right and counts[i] >= max(counts) * 0.15:
            peaks.append((i, counts[i]))
    # 有 2+ 峰且相隔 >= 3 bin (>= 15% range) 视为双峰
    is_bimodal = False
    peak_gap = 0
    if len(peaks) >= 2:
        top2 = sorted(peaks, key=lambda x: -x[1])[:2]
        peak_gap = abs(top2[0][0] - top2[1][0])
        is_bimodal = peak_gap >= 3
    return {
        "bimodal": is_bimodal,
        "ratio_p90_p10": round(ratio, 2),
        "peaks_count": len(peaks),
        "peak_gap_bins": peak_gap,
        "counts": counts,
        "hist_lo": lo,
        "hist_hi": hi,
    }


def _histogram_ascii(counts, width=40):
    if not counts: return "(no data)"
    m = max(counts)
    if m == 0: return "(all zero)"
    lines = []
    for c in counts:
        bar_len = int((c / m) * width)
        lines.append("|" + "█" * bar_len + " " * (width - bar_len) + f" {c}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="results dir with *.periter.jsonl")
    ap.add_argument("--out", default=None, help="output markdown path (default stdout)")
    ap.add_argument("--min-iters", type=int, default=30, help="min iters per group for bimodal check")
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.exists():
        raise SystemExit(f"dir not found: {d}")

    groups = load_iters(d)
    print(f"[analyze] {len(groups)} (op,shape) groups")

    lines = [f"# 双峰检测报告 — {d.name}", ""]
    lines.append(f"共 {len(groups)} 个 (op, shape) 组")
    lines.append("")
    lines.append("| op | shape/size | N | p10 | p50 | p90 | p99 | bimodal? | ratio_p90/p10 | peaks |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    findings = []
    for (op, shape), arr in sorted(groups.items()):
        arr = [x for x in arr if x is not None]
        n = len(arr)
        if n == 0: continue
        p = _percentiles(arr, [10, 50, 90, 99])
        b = _detect_bimodal(arr, min_iters=args.min_iters)
        bm = "✅" if b.get("bimodal") is True else ("❌" if b.get("bimodal") is False else "?")
        p10_us = f"{p[10]/1e3:.1f}" if p[10] else "-"
        p50_us = f"{p[50]/1e3:.1f}" if p[50] else "-"
        p90_us = f"{p[90]/1e3:.1f}" if p[90] else "-"
        p99_us = f"{p[99]/1e3:.1f}" if p[99] else "-"
        lines.append(f"| {op} | {shape} | {n} | {p10_us}μs | {p50_us}μs | {p90_us}μs | {p99_us}μs | {bm} | {b.get('ratio_p90_p10','?')} | {b.get('peaks_count','?')} |")
        if b.get("bimodal") is True:
            findings.append((op, shape, b, arr))

    lines.append("")
    if findings:
        lines.append(f"## 🚨 检出 {len(findings)} 组双峰")
        for (op, shape, b, arr) in findings:
            lines.append(f"\n### {op} @ {shape}")
            lines.append(f"- N={len(arr)}, ratio_p90/p10={b['ratio_p90_p10']}, peak_gap_bins={b['peak_gap_bins']}")
            lo_us = b['hist_lo'] / 1e3
            hi_us = b['hist_hi'] / 1e3
            lines.append(f"- range: {lo_us:.1f} — {hi_us:.1f} μs")
            lines.append("```")
            lines.append(_histogram_ascii(b['counts']))
            lines.append("```")
    else:
        lines.append("## ✅ 未检出显著双峰")

    out = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(out)
        print(f"[write] {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
