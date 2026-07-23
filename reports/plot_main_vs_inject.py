#!/usr/bin/env python3
"""出图：不同干扰对主进程吞吐/尾延迟的影响（main-vs-inject 补实验）。"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports"))
from plot_style import COLORS, HATCHES, apply_plot_style, save_fig  # noqa: E402

RES = Path("/Users/yinjinrun/Codespace/myportal/results/npu-dev-1/20260715_140931-main-vs-inject-d11")
OUT = Path("/Users/yinjinrun/Codespace/myportal/results/npu-dev-1/figs")
OUT.mkdir(parents=True, exist_ok=True)


def _load_abba(stem: str) -> dict[str, dict]:
    rows = [json.loads(l) for l in (RES / f"{stem}.jsonl").read_text().splitlines() if l.strip()]
    out = {}
    for r in rows:
        if r.get("record") != "factor_summary":
            continue
        out[r["factor"]] = r
    return out


def _window_quantiles(stem: str) -> dict[str, dict]:
    rows = [json.loads(l) for l in (RES / f"{stem}.jsonl").read_text().splitlines() if l.strip()]
    wins = [r for r in rows if r.get("record") == "window"]
    out = {}
    for factor in ["placebo", "cube", "vector", "hbm_mte"]:
        off = [r for r in wins if r["factor"] == factor and not r["inject_on"]]
        on = [r for r in wins if r["factor"] == factor and r["inject_on"]]
        if not off:
            continue

        def med(key, rs):
            return st.median([r[key] for r in rs]) if rs else float("nan")

        out[factor] = {
            "off_p50": med("iter_ms_p50", off),
            "on_p50": med("iter_ms_p50", on) if on else med("iter_ms_p50", off),
            "off_p90": med("iter_ms_p90", off),
            "on_p90": med("iter_ms_p90", on) if on else med("iter_ms_p90", off),
            "off_p95": med("iter_ms_p95", off),
            "on_p95": med("iter_ms_p95", on) if on else med("iter_ms_p95", off),
            "thru_drop": (
                (1 - med("iters_per_s", on) / med("iters_per_s", off)) * 100.0 if on else 0.0
            ),
        }
    return out


def _observed_contention(pts: list[dict]) -> dict:
    """从 step wall time 识别真正的争用生效区间，而不是沿用进程控制标记。"""
    pre = [r["iter_ms"] for r in pts if r.get("inject_state") == "pre"]
    baseline = st.median(pre)
    mad = st.median([abs(x - baseline) for x in pre])
    threshold = baseline + max(2.0, 10.0 * mad)
    burst_idx = [
        i
        for i, r in enumerate(pts)
        if r.get("inject_state") == "burst" and r["iter_ms"] > threshold
    ]
    if not burst_idx:
        return {}

    # 要求连续 3 步越过阈值，避免把单个自然长尾误判为 sidecar 生效。
    onset_idx = burst_idx[0]
    for i in burst_idx:
        if i + 2 < len(pts) and all(pts[j]["iter_ms"] > threshold for j in range(i, i + 3)):
            onset_idx = i
            break
    end_idx = burst_idx[-1]
    onset_s = pts[onset_idx]["t_rel_s"]
    end_s = pts[end_idx]["t_rel_s"] + pts[end_idx]["iter_ms"] / 1000.0
    active = [r for r in pts if onset_s <= r["t_rel_s"] < end_s]
    vals = [r["iter_ms"] for r in active]
    return {
        "baseline_ms": baseline,
        "threshold_ms": threshold,
        "onset_s": onset_s,
        "end_s": end_s,
        "active_s": end_s - onset_s,
        "active_iters_per_s": len(active) / (end_s - onset_s),
        "active_iter_ms_p50": st.median(vals),
        "active_iter_ms_p90": float(np.percentile(vals, 90)),
    }


def plot_thru_drop() -> Path:
    apply_plot_style((10, 4.8))
    fig, ax = plt.subplots()
    factors = ["cube", "vector", "hbm_mte"]
    labels = ["Cube GEMM", "Vector FMA", "HBM/MTE copy"]
    gemm = _load_abba("abba_gemm")
    block = _load_abba("abba_block")
    x = np.arange(len(factors))
    w = 0.36
    gvals = [gemm[f]["main_slowdown_pct"] for f in factors]
    bvals = [block[f]["main_slowdown_pct"] for f in factors]
    b1 = ax.bar(x - w / 2, gvals, w, label="主进程 GEMM", color=COLORS[0], hatch=HATCHES[0], edgecolor="black")
    b2 = ax.bar(x + w / 2, bvals, w, label="主进程 MLP Block", color=COLORS[1], hatch=HATCHES[1], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("主进程吞吐下降 (%)")
    ax.set_xlabel("同卡干扰类型（第二进程，duty=1）")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(gvals + bvals) * 1.25)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=14)
    fig.tight_layout()
    return save_fig(fig, OUT / "main_vs_inject_thru_drop.svg")


def plot_tail() -> Path:
    apply_plot_style((10, 4.8))
    fig, axes = plt.subplots(1, 2, sharey=False)
    for ax, stem, title in [
        (axes[0], "abba_gemm", "GEMM 主进程"),
        (axes[1], "abba_block", "MLP Block 主进程"),
    ]:
        q = _window_quantiles(stem)
        factors = ["cube", "vector", "hbm_mte"]
        labels = ["Cube", "Vector", "HBM/MTE"]
        x = np.arange(len(factors))
        w = 0.36
        off90 = [q[f]["off_p90"] for f in factors]
        on90 = [q[f]["on_p90"] for f in factors]
        ax.bar(x - w / 2, off90, w, label="干扰关 p90", color=COLORS[7], hatch=HATCHES[4], edgecolor="black")
        ax.bar(x + w / 2, on90, w, label="干扰开 p90", color=COLORS[2], hatch=HATCHES[2], edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("iter_ms p90")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=14)
    fig.tight_layout()
    return save_fig(fig, OUT / "main_vs_inject_tail_p90.svg")


def plot_timeline_block() -> Path:
    """区分 sidecar 控制区间与实测争用生效区间。"""
    apply_plot_style((11, 5.2))
    rows = [
        json.loads(l)
        for l in (RES / "timeline_block.timeline.jsonl").read_text().splitlines()
        if l.strip()
    ]
    fig, axes = plt.subplots(2, 1, sharex=False)
    for ax, factor, color in [
        (axes[0], "cube", COLORS[0]),
        (axes[1], "hbm_mte", COLORS[2]),
    ]:
        pts = [r for r in rows if r.get("factor") == factor and r.get("repeat") == 0]
        t = [r["t_rel_s"] for r in pts]
        y = [r["iter_ms"] for r in pts]
        ax.plot(t, y, color=color, lw=1.0, alpha=0.85)
        burst = [r for r in pts if r.get("inject_state") == "burst"]
        if burst:
            ax.axvline(
                8.0,
                color="tab:gray",
                linestyle=":",
                linewidth=1.8,
                label="发起 sidecar 进程",
            )
            ax.axvspan(
                burst[0]["t_rel_s"],
                burst[-1]["t_rel_s"] + burst[-1]["iter_ms"] / 1000.0,
                color="tab:gray",
                alpha=0.10,
                label="脚本标记为 burst",
            )
        observed = _observed_contention(pts)
        if observed:
            ax.axvspan(
                observed["onset_s"],
                observed["end_s"],
                color=color,
                alpha=0.20,
                label="实测争用生效",
            )
        ax.set_ylabel("单步墙钟时间 (ms)")
        ax.set_title(f"MLP Block 主进程 · 干扰={factor}")
        ax.legend(frameon=False, loc="upper left", fontsize=13)
    axes[-1].set_xlabel("相对时间 (s)")
    fig.tight_layout()
    return save_fig(fig, OUT / "main_vs_inject_timeline_block.svg")


def main() -> None:
    paths = [plot_thru_drop(), plot_tail(), plot_timeline_block()]
    # write compact analysis
    analysis = {
        "run_id": "20260715_140931-main-vs-inject-d11",
        "phys_device": 11,
        "abba": {
            "gemm": _load_abba("abba_gemm"),
            "block": _load_abba("abba_block"),
            "quantiles": {
                "gemm": _window_quantiles("abba_gemm"),
                "block": _window_quantiles("abba_block"),
            },
        },
        "timeline_thru_drop_note": (
            "原 on/burst 窗混入 sidecar 冷启动；p50 几乎不动主要是窗口混合，"
            "不能直接归因于设备时间片。应以观测到的 step 长尾起点作为实际生效时刻。"
        ),
    }
    # timeline thru from summaries
    tl = {}
    for stem in ["timeline_gemm", "timeline_block"]:
        tl[stem] = []
        for l in (RES / f"{stem}.jsonl").read_text().splitlines():
            r = json.loads(l)
            if r.get("record") != "timeline_summary":
                continue
            pre, bur = r["pre"], r["burst"]
            tl[stem].append(
                {
                    "factor": r["factor"],
                    "repeat": r["repeat"],
                    "thru_drop_pct": (1 - bur["iters_per_s"] / pre["iters_per_s"]) * 100,
                    "pre_p90": pre["iter_ms_p90"],
                    "burst_p90": bur["iter_ms_p90"],
                    "post_p90": r["post"]["iter_ms_p90"],
                }
            )
    analysis["timeline"] = tl
    block_rows = [
        json.loads(l)
        for l in (RES / "timeline_block.timeline.jsonl").read_text().splitlines()
        if l.strip()
    ]
    analysis["timeline_block_observed_contention"] = []
    for factor in ["cube", "hbm_mte"]:
        for repeat in [0, 1]:
            pts = [
                r
                for r in block_rows
                if r.get("factor") == factor and r.get("repeat") == repeat
            ]
            observed = _observed_contention(pts)
            analysis["timeline_block_observed_contention"].append(
                {"factor": factor, "repeat": repeat, **observed}
            )
    (RES / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    print("WROTE", RES / "analysis.json")
    for p in paths:
        print("WROTE", p)


if __name__ == "__main__":
    main()
