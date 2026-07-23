#!/usr/bin/env python3
"""根据预注册 trials 与复测结果分类异常。"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials-jsonl", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--global-median", type=float, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    trials = [json.loads(x) for x in args.trials_jsonl.read_text().splitlines() if x.strip()]
    results: dict[tuple[int, int], dict] = {}
    errors: list[str] = []
    for path in sorted(args.results_dir.glob("batch_*/pair_*.json")):
        lines = [x for x in path.read_text().splitlines() if x.strip()]
        if len(lines) != 1:
            errors.append(f"{path}:lines={len(lines)}")
            continue
        rec = json.loads(lines[0])
        results[(rec["round"], rec["slot"])] = rec

    # attach measured
    by_edge: dict[tuple[int, int], dict] = defaultdict(lambda: {"original": [], "reverse": [], "meta": None})
    measured_trials = []
    for t in trials:
        key = (t["batch_id"], t["batch_slot"])
        rec = results.get(key)
        row = dict(t)
        if rec is None:
            row["status"] = "missing"
            row["bw_GBps"] = None
            errors.append(f"missing_trial={t['trial_id']}")
        else:
            row["status"] = "ok"
            row["bw_GBps"] = rec["bw_GBps"]
            row["lat_us"] = rec["lat_us"]
            row["pattern_ok"] = rec.get("pattern_ok")
            if (
                rec.get("src_index") != t["src_index"]
                or rec.get("dst_index") != t["dst_index"]
                or rec.get("hca") != "xscale_0"
                or not rec.get("pattern_ok")
            ):
                errors.append(f"schema_mismatch_trial={t['trial_id']}")
        measured_trials.append(row)
        edge = (t["unordered_a"], t["unordered_b"])
        by_edge[edge]["meta"] = {
            "role": t["role"],
            "original_bw_GBps": t["original_bw_GBps"],
            "source_round": t["source_round"],
            "source_slot": t["source_slot"],
            "src_pod_orig": None,
            "dst_pod_orig": None,
        }
        if t["direction"] == "original" and t["repeat"] == 1:
            by_edge[edge]["meta"]["src_pod_orig"] = t["src_pod"]
            by_edge[edge]["meta"]["dst_pod_orig"] = t["dst_pod"]
        if row["bw_GBps"] is not None:
            by_edge[edge][t["direction"]].append(row["bw_GBps"])

    slow_thr = 0.75 * args.global_median
    recover_thr = 0.90 * args.global_median
    control_drift_thr = 0.10  # relative to median

    classifications = []
    for edge, blob in sorted(by_edge.items()):
        meta = blob["meta"]
        orig = blob["original"]
        rev = blob["reverse"]
        role = meta["role"]
        orig_med = statistics.median(orig) if orig else None
        rev_med = statistics.median(rev) if rev else None
        all_vals = orig + rev
        all_med = statistics.median(all_vals) if all_vals else None
        label = "unclassified"
        detail = ""
        if role == "control":
            if all_med is None:
                label = "control_missing"
            elif abs(all_med - args.global_median) / args.global_median > control_drift_thr:
                label = "negative_control_drift"
                detail = f"retest_median={all_med:.3f} vs global={args.global_median:.3f}"
            else:
                label = "negative_control_stable"
                detail = f"retest_median={all_med:.3f}"
        else:
            # worst pair classification
            orig_slow = bool(orig) and statistics.median(orig) < slow_thr
            rev_slow = bool(rev) and statistics.median(rev) < slow_thr
            orig_ok = bool(orig) and statistics.median(orig) >= recover_thr
            rev_ok = bool(rev) and statistics.median(rev) >= recover_thr
            if orig_slow and rev_slow:
                label = "stable_bidirectional_slow"
                detail = f"orig_med={orig_med:.3f} rev_med={rev_med:.3f}"
            elif (orig_slow and rev_ok) or (rev_slow and orig_ok):
                label = "directional_slow"
                detail = f"orig_med={orig_med:.3f} rev_med={rev_med:.3f}"
            elif all_med is not None and all_med >= recover_thr:
                label = "original_sporadic_recovery"
                detail = f"original={meta['original_bw_GBps']:.3f} retest_med={all_med:.3f}"
            elif all_med is not None and all_med < slow_thr:
                label = "stable_bidirectional_slow"
                detail = f"mixed_but_overall_med={all_med:.3f}"
            else:
                label = "partial_or_unstable"
                detail = f"orig_med={orig_med} rev_med={rev_med}"
        classifications.append(
            {
                "unordered_a": edge[0],
                "unordered_b": edge[1],
                "role": role,
                "label": label,
                "detail": detail,
                "original_bw_GBps": meta["original_bw_GBps"],
                "source_round": meta["source_round"],
                "source_slot": meta["source_slot"],
                "src_pod": meta.get("src_pod_orig"),
                "dst_pod": meta.get("dst_pod_orig"),
                "original_repeats_GBps": orig,
                "reverse_repeats_GBps": rev,
                "n_repeats": len(all_vals),
            }
        )

    # node margin anomaly: nodes appearing often in still-slow retests
    node_slow_hits: dict[int, int] = defaultdict(int)
    for c in classifications:
        if c["label"] in ("stable_bidirectional_slow", "directional_slow"):
            node_slow_hits[c["unordered_a"]] += 1
            node_slow_hits[c["unordered_b"]] += 1
    node_margin_anomaly = [
        {"node_index": k, "slow_pair_hits": v}
        for k, v in sorted(node_slow_hits.items(), key=lambda x: -x[1])
        if v >= 2
    ]
    for n in node_margin_anomaly:
        # annotate classification list conceptually
        pass

    rail_swap_list = [
        {
            "unordered_a": c["unordered_a"],
            "unordered_b": c["unordered_b"],
            "src_pod": c["src_pod"],
            "dst_pod": c["dst_pod"],
            "source_round": c["source_round"],
            "source_slot": c["source_slot"],
            "label": c["label"],
            "original_bw_GBps": c["original_bw_GBps"],
            "original_repeats_GBps": c["original_repeats_GBps"],
            "reverse_repeats_GBps": c["reverse_repeats_GBps"],
            "suggested_rails": ["xscale_1", "xscale_2", "xscale_3"],
            "note": "本轮不跑其他rail；仅清单",
        }
        for c in classifications
        if c["label"] in ("stable_bidirectional_slow", "directional_slow")
    ]

    counts = defaultdict(int)
    for c in classifications:
        counts[c["label"]] += 1

    out = {
        "valid": not errors,
        "errors": errors,
        "n_trials_expected": len(trials),
        "n_trials_measured": sum(1 for t in measured_trials if t["status"] == "ok"),
        "global_median_GBps": args.global_median,
        "thresholds": {
            "slow_lt": slow_thr,
            "recover_ge": recover_thr,
            "control_drift_rel": control_drift_thr,
        },
        "label_counts": dict(counts),
        "classifications": classifications,
        "node_margin_anomaly": node_margin_anomaly,
        "rail_swap_retest_list": rail_swap_list,
        "measured_trials": measured_trials,
    }
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")

    lines = [
        "# W2.2 pair 异常复测分类",
        "",
        f"- 期望 trials：{len(trials)}；测得：{out['n_trials_measured']}",
        f"- 全局中位带宽：{args.global_median:.3f} GB/s",
        f"- 慢阈值：<{slow_thr:.3f}；恢复阈值：≥{recover_thr:.3f}",
        "",
        "## 标签计数（按独立无向 pair）",
        "",
    ]
    for k, v in sorted(counts.items()):
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## 逐 pair", ""]
    for c in classifications:
        lines.append(
            f"- r{c['source_round']}s{c['source_slot']} `{c['label']}` "
            f"orig={c['original_bw_GBps']:.3f} "
            f"fwd={c['original_repeats_GBps']} rev={c['reverse_repeats_GBps']} "
            f"— {c['detail']}"
        )
    if node_margin_anomaly:
        lines += ["", "## 节点边际异常（复测仍慢 pair 命中≥2）", ""]
        for n in node_margin_anomaly:
            lines.append(f"- node {n['node_index']}: hits={n['slow_pair_hits']}")
    if rail_swap_list:
        lines += ["", "## 换 rail 复测清单（本轮不执行）", ""]
        for r in rail_swap_list:
            lines.append(
                f"- r{r['source_round']}s{r['source_slot']} `{r['label']}` "
                f"rails={','.join(r['suggested_rails'])}"
            )
    args.out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"valid": out["valid"], "label_counts": out["label_counts"], "rail_swap": len(rail_swap_list)}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
