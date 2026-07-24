#!/usr/bin/env python3
"""QP-ECMP 实验本机聚合 + SUMMARY。

读 <root>/raw/qp<QP>/rep<rep>/*.rank*.jsonl（各 pod tar 回来的每-rank 结果），
按 QP 臂聚合带宽/尾延迟，出 SUMMARY.md（BW/尾延迟 vs QP 表 + ECMP 判据）。

无 AFS：数据已回拉本机，聚合纯本地跑（对标 jump validate.py 的 schema 校验，
但去掉 master-runs-validate 耦合）。

两种口径:
  - all_reduce（nccl_bench 记录）: 取 bus_bw_GBps_global_max（全局最慢 rank 口径），
    尾延迟从 iters_s_global_max 取 p50/p95/p99。
  - incast（nccl_incast 记录）: 从 root(role=recv_root) 取 agg_bw_GBps + p50/p99。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics


def _pctile(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return s[max(0, math.ceil(p * len(s)) - 1)]


def _load_jsonl(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        pass
    return out


def _collect_rep(rep_dir, world, mode):
    """聚合单个 (QP,rep) 目录 → 一条 rep 级指标 dict（或 None）。"""
    files = sorted(glob.glob(os.path.join(rep_dir, "*.rank*.jsonl")))
    recs = []
    for p in files:
        recs.extend(_load_jsonl(p))
    if not recs:
        return None

    rank_files = len(files)
    if mode == "incast":
        roots = [r for r in recs if r.get("role") == "recv_root"]
        if not roots:
            return {"rank_files": rank_files, "records": len(recs), "valid": False,
                    "note": "no recv_root record"}
        # 可能多个 size；这里按 size 聚合，取最大 size 那档为主指标
        by_size = {}
        for r in roots:
            by_size.setdefault(r["nbytes"], []).append(r)
        main_nb = max(by_size)
        rr = by_size[main_nb][0]
        iters = rr.get("iters_s", [])
        return {
            "rank_files": rank_files, "records": len(recs), "valid": True,
            "nbytes": main_nb,
            "agg_bw_GBps": rr.get("agg_bw_GBps"),
            "per_sender_bw_GBps": rr.get("per_sender_bw_GBps"),
            "n_senders": rr.get("n_senders"),
            "p50_ms": (_pctile(iters, 0.50) or 0) * 1e3,
            "p95_ms": (_pctile(iters, 0.95) or 0) * 1e3,
            "p99_ms": (_pctile(iters, 0.99) or 0) * 1e3,
        }

    # all_reduce / 其它集合通信
    bench = [r for r in recs if r.get("record") in ("nccl_bench", "nccl_torch_bench")
             or "bus_bw_GBps_global_max" in r]
    if not bench:
        return {"rank_files": rank_files, "records": len(recs), "valid": False,
                "note": "no bench record"}
    by_size = {}
    for r in bench:
        by_size.setdefault(r.get("nbytes"), []).append(r)
    main_nb = max(k for k in by_size if k is not None)
    rows = by_size[main_nb]
    # global_max 向量各 rank 一致；取第一条即可
    base = rows[0]
    gvec = base.get("iters_s_global_max", [])
    ranks = sorted(r.get("rank", -1) for r in rows)
    complete = (rank_files == world and ranks == list(range(world)))
    return {
        "rank_files": rank_files, "records": len(recs), "valid": True,
        "complete": complete, "nbytes": main_nb,
        "bus_bw_GBps": base.get("bus_bw_GBps_global_max"),
        "alg_bw_GBps": base.get("alg_bw_GBps_global_max"),
        "p50_ms": (statistics.median(gvec) * 1e3) if gvec else None,
        "p95_ms": (_pctile(gvec, 0.95) or 0) * 1e3 if gvec else None,
        "p99_ms": (_pctile(gvec, 0.99) or 0) * 1e3 if gvec else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--mode", default="all_reduce")
    ap.add_argument("--arms", default="default,1,2,4,8,16")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bw_key = "agg_bw_GBps" if args.mode == "incast" else "bus_bw_GBps"

    per_arm = {}
    for qp in arms:
        reps = []
        for rep in range(1, args.repeats + 1):
            rep_dir = os.path.join(args.root, "raw", f"qp{qp}", f"rep{rep}")
            m = _collect_rep(rep_dir, args.world, args.mode)
            if m:
                reps.append(m)
        per_arm[qp] = reps

    # 每臂：分离「干净跑」vs「stall 跑」。大规模同步尾部 stall（07-19 报告头号候选）
    # 会让单跑带宽塌到个位数，与 QP 效应混淆。用 p99 阈值分类，QP 效应只在干净跑里看。
    STALL_P99_MS = float(os.environ.get("STALL_P99_MS", "100"))
    rows = []
    for qp in arms:
        reps = [r for r in per_arm[qp] if r and r.get("valid")]
        if not reps:
            rows.append({"qp": qp, "n_valid": 0, "n_clean": 0, "n_stall": 0})
            continue
        clean_bw, stall_bw, all_p99 = [], [], []
        for r in reps:
            bw = r.get(bw_key); p99 = r.get("p99_ms")
            if p99 is not None:
                all_p99.append(p99)
            if bw is None:
                continue
            if p99 is not None and p99 >= STALL_P99_MS:
                stall_bw.append(bw)
            else:
                clean_bw.append(bw)
        rows.append({
            "qp": qp, "n_valid": len(reps),
            "n_clean": len(clean_bw), "n_stall": len(stall_bw),
            # 干净跑带宽分布（QP 真实效应看这个）
            "clean_bw_med": statistics.median(clean_bw) if clean_bw else None,
            "clean_bw_min": min(clean_bw) if clean_bw else None,
            "clean_bw_max": max(clean_bw) if clean_bw else None,
            # 全体（含 stall）带宽中位，供对照
            "all_bw_med": statistics.median([r[bw_key] for r in reps if r.get(bw_key) is not None])
                          if any(r.get(bw_key) is not None for r in reps) else None,
            "stall_rate": len(stall_bw) / len(reps) if reps else None,
            "p99_med": statistics.median(all_p99) if all_p99 else None,
            "rank_files": reps[0].get("rank_files"),
            "nbytes": reps[0].get("nbytes"),
        })

    # ECMP 判据（两条独立信号，都基于跨 rep 统计，避免单跑 stall 抽奖噪声）：
    #  (a) 干净跑带宽 vs QP：最低 QP 基线 → 最高 QP 臂（按 QP 数值，非按带宽），
    #      ECMP 假设是"带宽随 QP 上升"，故必须最低比最高，不能拿"最优臂"自比。
    #  (b) stall 率 vs QP：若高 QP 显著降低 stall 发生率，本身即 ECMP 极化的旁证。
    def _qp_val(q):
        # default 视为库默认（未知，排在 QP=1 之前作最低基线候选）
        return -1 if q == "default" else int(q)
    clean_rows = [r for r in rows if r.get("clean_bw_med") is not None]
    clean_rows_sorted = sorted(clean_rows, key=lambda r: _qp_val(r["qp"]))
    # 基线优先用 QP=1（真实单 QP，比 default 语义明确）；无则用最低 QP 臂
    base_row = next((r for r in clean_rows_sorted if r["qp"] == "1"), None) \
        or (clean_rows_sorted[0] if clean_rows_sorted else None)
    # 对比臂 = QP 数值最高的干净臂
    top_row = clean_rows_sorted[-1] if clean_rows_sorted else None
    # 峰值臂（带宽最高，仅供上下文，不作 ECMP 判据）
    peak_row = max(clean_rows, key=lambda r: r["clean_bw_med"]) if clean_rows else None
    # stall 率趋势
    low_qp_stall = next((r["stall_rate"] for r in rows if r["qp"] in ("1", "default") and r.get("stall_rate") is not None), None)
    hi_qps = [r for r in rows if r["qp"] not in ("1", "default") and r.get("stall_rate") is not None]
    hi_qp_stall = statistics.median([r["stall_rate"] for r in hi_qps]) if hi_qps else None

    verdict_lines = []
    if args.repeats < 3:
        verdict_lines.append(f"⚠️ **repeats={args.repeats} 偏少，stall 抽奖未充分平均，判据仅供参考**（建议 ≥3）。")
    if base_row and top_row and top_row["qp"] != base_row["qp"] and base_row.get("clean_bw_med"):
        gain = top_row["clean_bw_med"] / base_row["clean_bw_med"]
        head = (f"干净跑带宽 QP={base_row['qp']}→{base_row['clean_bw_med']:.1f} vs "
                f"QP={top_row['qp']}→{top_row['clean_bw_med']:.1f} GB/s（{gain:.2f}×）")
        if gain >= 1.5:
            verdict_lines.append(f"**(a) 支持 ECMP**：{head}。多子流打散上行救回带宽。")
        elif gain >= 1.15:
            verdict_lines.append(f"**(a) 弱支持**：{head}，有响应但幅度有限。")
        else:
            verdict_lines.append(f"**(a) 带宽层面不支持**：{head}，干净跑带宽不随 QP 上升。")
        if peak_row and peak_row["qp"] not in (base_row["qp"], top_row["qp"]):
            verdict_lines.append(f"（峰值出现在 QP={peak_row['qp']}→{peak_row['clean_bw_med']:.1f} GB/s，"
                                 f"非单调，不符合 ECMP「越多 QP 越好」的预期。）")
    else:
        verdict_lines.append("**(a)** 干净跑样本不足或臂不够，无法比较带宽 vs QP。")
    if low_qp_stall is not None and hi_qp_stall is not None:
        verdict_lines.append(f"**(b) stall 率**：低 QP={low_qp_stall:.0%} vs 高 QP 中位={hi_qp_stall:.0%}。"
                             + ("高 QP 显著降 stall → ECMP 旁证。" if hi_qp_stall + 0.15 < low_qp_stall
                                else "stall 率不随 QP 明显下降。"))
    verdict_lines.append("综合：若 (a)(b) 均无 QP 响应，则塌陷主因更可能是**大规模同步尾部 stall**"
                         "（报告 §4 头号候选），而非 ECMP 哈希极化——此时 QP 调优救不了，"
                         "方向转向 stall 根因（MCCL 运行时/启动销毁、共享 fabric 瞬时事件）。")
    verdict = "\n\n".join(verdict_lines)

    # 出 SUMMARY.md
    unit = "agg_bw(GB/s)" if args.mode == "incast" else "bus_bw(GB/s)"
    lines = [
        f"# QP-ECMP 实验汇总 — {os.path.basename(args.root)}",
        "",
        f"- 口径: **{args.mode}**  world={args.world}  repeats={args.repeats}  "
        f"stall 阈值 p99≥{STALL_P99_MS:.0f}ms",
        f"- **干净跑**带宽反映 QP 真实效应；**stall 率**反映尾部不稳定发生频率。两者分开看。",
        "",
        f"| QP | reps | 干净/stall | {unit} 干净跑(min/中位/max) | 全体中位 | stall率 | p99中位(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["n_valid"] == 0:
            lines.append(f"| {r['qp']} | 0 | — | — | — | — | — |")
            continue
        def _f(x, p=1): return f"{x:.{p}f}" if x is not None else "—"
        clean = (f"{_f(r['clean_bw_min'])}/{_f(r['clean_bw_med'])}/{_f(r['clean_bw_max'])}"
                 if r.get("clean_bw_med") is not None else "—(全 stall)")
        sr = f"{r['stall_rate']:.0%}" if r.get("stall_rate") is not None else "—"
        lines += [f"| {r['qp']} | {r['n_valid']} | {r['n_clean']}/{r['n_stall']} | {clean} "
                  f"| {_f(r.get('all_bw_med'))} | {sr} | {_f(r.get('p99_med'))} |"]
    lines += ["", "## ECMP 判据", "", verdict, ""]

    # 数据完整性提示
    incomplete = [r["qp"] for r in rows
                  if r.get("rank_files") is not None and r["rank_files"] != args.world]
    if incomplete:
        lines += [f"> ⚠️ 以下 QP 臂 rank_files ≠ world={args.world}（可能有节点崩/收集缺失）: "
                  f"{', '.join(incomplete)}。SIGSEGV 场景下若 jsonl 齐全仍可用；否则复测。", ""]

    out = os.path.join(args.root, "SUMMARY.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
