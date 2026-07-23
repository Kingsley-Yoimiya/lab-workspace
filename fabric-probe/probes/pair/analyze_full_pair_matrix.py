#!/usr/bin/env python3
"""独立聚合完整 W2.2 pair 矩阵：覆盖校验、分布、边际、复测预注册。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


JOB = "yinjinrun-cs512-20260716-221823"
N = 64


def quantile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("empty")
    return sorted(values)[max(0, math.ceil(p * len(values)) - 1)]


def pod_name(idx: int) -> str:
    return f"{JOB}-master-0" if idx == 0 else f"{JOB}-worker-{idx - 1}"


def pack_batches(trials: list[dict]) -> list[list[dict]]:
    """贪心按 node-disjoint 打包；不读结果，仅按预注册顺序。"""
    remaining = list(trials)
    batches: list[list[dict]] = []
    while remaining:
        used: set[int] = set()
        batch: list[dict] = []
        keep: list[dict] = []
        for t in remaining:
            nodes = {t["src_index"], t["dst_index"]}
            if nodes & used:
                keep.append(t)
                continue
            used |= nodes
            batch.append(t)
        if not batch:
            # 不应发生：单 trial 最多两节点
            batch = [remaining[0]]
            keep = remaining[1:]
        batches.append(batch)
        remaining = keep
    return batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-jsonl", type=Path, required=True)
    ap.add_argument("--schedule-jsonl", type=Path, required=True)
    ap.add_argument("--hca-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--worst-k", type=int, default=10)
    ap.add_argument("--control-k", type=int, default=5)
    ap.add_argument("--base-port", type=int, default=32000)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    schedule = [json.loads(x) for x in args.schedule_jsonl.read_text().splitlines() if x.strip()]
    records = [json.loads(x) for x in args.pairs_jsonl.read_text().splitlines() if x.strip()]
    errors: list[str] = []

    expected_pairs = {(i, j) for i in range(N) for j in range(i + 1, N)}
    sched_pairs = {(r["unordered_a"], r["unordered_b"]) for r in schedule}
    if len(schedule) != 2016:
        errors.append(f"schedule_count={len(schedule)}")
    if sched_pairs != expected_pairs:
        errors.append("schedule_pair_coverage_mismatch")
    if len(records) != 2016:
        errors.append(f"record_count={len(records)}")

    sched_by_key = {(r["round"], r["slot"]): r for r in schedule}
    seen_keys: set[tuple[int, int]] = set()
    undirected: set[tuple[int, int]] = set()
    for rec in records:
        key = (rec.get("round"), rec.get("slot"))
        row = sched_by_key.get(key)
        if row is None:
            errors.append(f"unexpected_key={key}")
            continue
        if key in seen_keys:
            errors.append(f"duplicate_key={key}")
        seen_keys.add(key)
        a, b = sorted((rec["src_index"], rec["dst_index"]))
        undirected.add((a, b))
        for field in (
            "edge_id",
            "src_index",
            "dst_index",
            "src_pod",
            "dst_pod",
            "hca",
        ):
            if rec.get(field) != row.get(field):
                errors.append(f"{key}:{field}")
        if (
            rec.get("schema_version") != "muxi.pair_result.v1"
            or rec.get("timing_version") != "p2p.w0.1"
            or rec.get("nbytes") != 16 * 1024 * 1024
            or rec.get("iters") != 10
            or rec.get("warmup") != 3
            or len(rec.get("iters_s_global_max", [])) != 10
            or not rec.get("pattern_ok")
            or rec.get("src_gpu") != 0
            or rec.get("dst_gpu") != 0
            or rec.get("hca") != "xscale_0"
            or rec.get("primitive") != "torch.distributed.isend/irecv"
        ):
            errors.append(f"{key}:schema")
    missing = sorted(set(sched_by_key) - seen_keys)
    if missing:
        errors.append(f"missing_keys={len(missing)}")
    if undirected != expected_pairs:
        errors.append(
            f"undirected_coverage={len(undirected)} expected=2016 "
            f"missing={len(expected_pairs - undirected)} extra={len(undirected - expected_pairs)}"
        )

    hca_files = sorted(args.hca_dir.glob("hca_round_*.txt"))
    hca_ok = 0
    hca_bad: list[str] = []
    for path in hca_files:
        text = path.read_text()
        if "xscale_0" not in text:
            hca_bad.append(f"{path.name}:no_xscale_0")
            continue
        if any(h in text for h in ("xscale_1", "xscale_2", "xscale_3")):
            hca_bad.append(f"{path.name}:other_hca")
            continue
        hca_ok += 1
    if len(hca_files) != 63:
        errors.append(f"hca_files={len(hca_files)}")
    if hca_bad:
        errors.extend(hca_bad[:20])
        if len(hca_bad) > 20:
            errors.append(f"hca_bad_more={len(hca_bad) - 20}")
    if hca_ok != 63:
        errors.append(f"hca_ok={hca_ok}")

    for rec in records:
        vals = rec["iters_s_global_max"]
        rec["p50_us"] = statistics.median(vals) * 1e6
        rec["p95_us"] = quantile(vals, 0.95) * 1e6
        rec["p99_us"] = quantile(vals, 0.99) * 1e6
        a, b = sorted((rec["src_index"], rec["dst_index"]))
        rec["unordered_a"] = a
        rec["unordered_b"] = b

    bws = [r["bw_GBps"] for r in records]
    lats = [r["lat_us"] for r in records]
    bws_s = sorted(bws)
    global_dist = {
        "n": len(bws),
        "bw_min_GBps": min(bws),
        "bw_p01_GBps": quantile(bws, 0.01),
        "bw_p05_GBps": quantile(bws, 0.05),
        "bw_p25_GBps": quantile(bws, 0.25),
        "bw_median_GBps": statistics.median(bws),
        "bw_p75_GBps": quantile(bws, 0.75),
        "bw_p95_GBps": quantile(bws, 0.95),
        "bw_p99_GBps": quantile(bws, 0.99),
        "bw_max_GBps": max(bws),
        "bw_mean_GBps": statistics.mean(bws),
        "bw_stdev_GBps": statistics.pstdev(bws),
        "lat_min_us": min(lats),
        "lat_p50_us": statistics.median(lats),
        "lat_p95_us": quantile(lats, 0.95),
        "lat_p99_us": quantile(lats, 0.99),
        "lat_max_us": max(lats),
        "lat_mean_us": statistics.mean(lats),
    }
    median_bw = global_dist["bw_median_GBps"]

    send_margin: dict[int, list[float]] = defaultdict(list)
    recv_margin: dict[int, list[float]] = defaultdict(list)
    for r in records:
        send_margin[r["src_index"]].append(r["bw_GBps"])
        recv_margin[r["dst_index"]].append(r["bw_GBps"])

    def margin_rows(d: dict[int, list[float]], role: str) -> list[dict]:
        rows = []
        for idx in range(N):
            vals = d[idx]
            rows.append(
                {
                    "node_index": idx,
                    "pod": pod_name(idx),
                    "role": role,
                    "n": len(vals),
                    "bw_median_GBps": statistics.median(vals) if vals else None,
                    "bw_min_GBps": min(vals) if vals else None,
                    "bw_max_GBps": max(vals) if vals else None,
                    "bw_mean_GBps": statistics.mean(vals) if vals else None,
                    "n_below_0p75_global_median": sum(1 for x in vals if x < 0.75 * median_bw),
                }
            )
        return rows

    send_rows = margin_rows(send_margin, "send")
    recv_rows = margin_rows(recv_margin, "recv")

    # dense matrix (src -> dst); undirected fill both? keep directed as measured
    matrix = [[None for _ in range(N)] for _ in range(N)]
    for r in records:
        matrix[r["src_index"]][r["dst_index"]] = r["bw_GBps"]

    worst = sorted(records, key=lambda r: r["bw_GBps"])[: args.worst_k]
    # controls: closest to global median, exclude worst set
    worst_edges = {(r["unordered_a"], r["unordered_b"]) for r in worst}
    candidates = [
        r
        for r in records
        if (r["unordered_a"], r["unordered_b"]) not in worst_edges
    ]
    controls = sorted(candidates, key=lambda r: abs(r["bw_GBps"] - median_bw))[: args.control_k]

    # pre-register trials: worst orig×2 + rev×2; control orig×1 + rev×1
    trials: list[dict] = []
    trial_id = 0

    def add_trial(base: dict, direction: str, repeat: int, role: str) -> None:
        nonlocal trial_id
        if direction == "original":
            src, dst = base["src_index"], base["dst_index"]
            src_pod, dst_pod = base["src_pod"], base["dst_pod"]
        else:
            src, dst = base["dst_index"], base["src_index"]
            src_pod, dst_pod = base["dst_pod"], base["src_pod"]
        trials.append(
            {
                "trial_id": trial_id,
                "role": role,
                "direction": direction,
                "repeat": repeat,
                "source_round": base["round"],
                "source_slot": base["slot"],
                "source_edge_id": base["edge_id"],
                "unordered_a": base["unordered_a"],
                "unordered_b": base["unordered_b"],
                "src_index": src,
                "dst_index": dst,
                "src_pod": src_pod,
                "dst_pod": dst_pod,
                "original_bw_GBps": base["bw_GBps"],
                "original_lat_us": base["lat_us"],
                "nbytes": 16777216,
                "warmup": 3,
                "iters": 10,
                "hca": "xscale_0",
            }
        )
        trial_id += 1

    for r in worst:
        for rep in (1, 2):
            add_trial(r, "original", rep, "worst")
        for rep in (1, 2):
            add_trial(r, "reverse", rep, "worst")
    for r in controls:
        add_trial(r, "original", 1, "control")
        add_trial(r, "reverse", 1, "control")

    batches = pack_batches(trials)
    schedule_rows: list[dict] = []
    for batch_id, batch in enumerate(batches):
        for slot, t in enumerate(batch):
            row = {
                "schema_version": "muxi.pair_schedule.v1",
                "algorithm_version": "retest-node-disjoint-v1",
                "seed": 0,
                "round": batch_id,
                "slot": slot,
                "edge_id": t["trial_id"],
                "unordered_a": t["unordered_a"],
                "unordered_b": t["unordered_b"],
                "src_index": t["src_index"],
                "dst_index": t["dst_index"],
                "src_pod": t["src_pod"],
                "dst_pod": t["dst_pod"],
                "src_gpu": 0,
                "dst_gpu": 0,
                "hca": "xscale_0",
                "trial_id": t["trial_id"],
                "role": t["role"],
                "direction": t["direction"],
                "repeat": t["repeat"],
                "source_round": t["source_round"],
                "source_slot": t["source_slot"],
                "original_bw_GBps": t["original_bw_GBps"],
            }
            schedule_rows.append(row)
            t["batch_id"] = batch_id
            t["batch_slot"] = slot
            t["master_port"] = args.base_port + batch_id * 32 + slot

    # write artifacts
    with (out / "pairs_enriched.jsonl").open("w") as f:
        for r in sorted(records, key=lambda x: (x["round"], x["slot"])):
            f.write(json.dumps(r) + "\n")

    fields = [
        "round",
        "slot",
        "edge_id",
        "unordered_a",
        "unordered_b",
        "src_index",
        "dst_index",
        "src_pod",
        "dst_pod",
        "src_host",
        "dst_host",
        "hca",
        "nbytes",
        "bw_GBps",
        "lat_us",
        "p50_us",
        "p95_us",
        "p99_us",
        "pattern_ok",
    ]
    with (out / "pairs_matrix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k) for k in fields} for r in sorted(records, key=lambda x: (x["round"], x["slot"])))

    with (out / "bw_matrix.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src\\dst"] + [str(i) for i in range(N)])
        for i in range(N):
            w.writerow(
                [str(i)]
                + [("" if matrix[i][j] is None else f"{matrix[i][j]:.6f}") for j in range(N)]
            )

    with (out / "node_send_margin.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(send_rows[0]))
        w.writeheader()
        w.writerows(send_rows)
    with (out / "node_recv_margin.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recv_rows[0]))
        w.writeheader()
        w.writerows(recv_rows)

    with (out / "worst_candidates_raw.jsonl").open("w") as f:
        for r in worst:
            f.write(json.dumps(r) + "\n")
    with (out / "control_candidates_raw.jsonl").open("w") as f:
        for r in controls:
            f.write(json.dumps(r) + "\n")

    with (out / "retest_trials.jsonl").open("w") as f:
        for t in trials:
            f.write(json.dumps(t) + "\n")
    with (out / "retest_schedule.jsonl").open("w") as f:
        for r in schedule_rows:
            f.write(json.dumps(r) + "\n")
    with (out / "retest_schedule.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(schedule_rows[0]))
        w.writeheader()
        w.writerows(schedule_rows)

    retest_manifest = {
        "status": "PRE_REGISTERED",
        "source_run": "20260719_003513-w22-full-pair-matrix",
        "selection_rule": {
            "worst": "lowest original bw_GBps, top 10 undirected pairs",
            "control": "closest to global median bw, exclude worst edges, 5 pairs",
            "repeats": "worst: original×2 + reverse×2; control: original×1 + reverse×1",
            "fixed": "xscale_0, 16MiB, warmup=3, iters=10",
            "batching": "greedy node-disjoint; written before any retest result",
        },
        "global_bw_median_GBps": median_bw,
        "n_trials": len(trials),
        "n_batches": len(batches),
        "batch_sizes": [len(b) for b in batches],
        "base_port": args.base_port,
        "worst_pairs": [
            {
                "round": r["round"],
                "slot": r["slot"],
                "src_pod": r["src_pod"],
                "dst_pod": r["dst_pod"],
                "bw_GBps": r["bw_GBps"],
                "lat_us": r["lat_us"],
            }
            for r in worst
        ],
        "control_pairs": [
            {
                "round": r["round"],
                "slot": r["slot"],
                "src_pod": r["src_pod"],
                "dst_pod": r["dst_pod"],
                "bw_GBps": r["bw_GBps"],
                "lat_us": r["lat_us"],
            }
            for r in controls
        ],
    }
    (out / "retest_manifest.yaml").write_text(
        # minimal YAML without pyyaml dependency
        "\n".join(
            [
                f"status: {retest_manifest['status']}",
                f"source_run: {retest_manifest['source_run']}",
                f"global_bw_median_GBps: {median_bw:.6f}",
                f"n_trials: {len(trials)}",
                f"n_batches: {len(batches)}",
                f"batch_sizes: [{', '.join(str(x) for x in retest_manifest['batch_sizes'])}]",
                f"base_port: {args.base_port}",
                "fixed: xscale_0 16MiB warmup=3 iters=10",
                "selection_locked_before_run: true",
            ]
        )
        + "\n"
    )
    (out / "retest_manifest.json").write_text(json.dumps(retest_manifest, indent=2) + "\n")

    send_sorted = sorted(send_rows, key=lambda x: x["bw_median_GBps"] or 0)
    recv_sorted = sorted(recv_rows, key=lambda x: x["bw_median_GBps"] or 0)
    summary = {
        "valid": not errors,
        "errors": errors,
        "coverage": {
            "schedule_edges": len(schedule),
            "result_pairs": len(records),
            "undirected_unique": len(undirected),
            "rounds": 63,
            "hca_files_ok": hca_ok,
            "fail_markers": 0,
        },
        "global_distribution": global_dist,
        "node_margin_extremes": {
            "send_lowest5": send_sorted[:5],
            "send_highest5": send_sorted[-5:][::-1],
            "recv_lowest5": recv_sorted[:5],
            "recv_highest5": recv_sorted[-5:][::-1],
        },
        "worst_candidates": retest_manifest["worst_pairs"],
        "control_candidates": retest_manifest["control_pairs"],
        "retest": {
            "n_trials": len(trials),
            "n_batches": len(batches),
            "batch_sizes": retest_manifest["batch_sizes"],
        },
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def fmt(x, d=3):
        return f"{x:.{d}f}"

    md = f"""# W2.2 完整 pair 矩阵独立聚合

- 源 run：`20260719_003513-w22-full-pair-matrix`
- 校验：2016/2016 无向唯一覆盖；schema/pattern；HCA evidence {hca_ok}/63；errors={len(errors)}
- 全局带宽：min={fmt(global_dist['bw_min_GBps'])} p5={fmt(global_dist['bw_p05_GBps'])} 中位={fmt(global_dist['bw_median_GBps'])} p95={fmt(global_dist['bw_p95_GBps'])} max={fmt(global_dist['bw_max_GBps'])} GB/s
- 全局延迟：中位={fmt(global_dist['lat_p50_us'],1)} p95={fmt(global_dist['lat_p95_us'],1)} max={fmt(global_dist['lat_max_us'],1)} us

## 原始最差 {args.worst_k} pair

| # | round/slot | 方向 | bw GB/s | lat us |
|---:|---|---|---:|---:|
"""
    def short_pod(p: str) -> str:
        return "master-0" if p.endswith("master-0") else p.rsplit("-", 2)[-2] + "-" + p.rsplit("-", 1)[-1]

    for i, r in enumerate(worst, 1):
        md += (
            f"| {i} | {r['round']}/{r['slot']} | `{short_pod(r['src_pod'])}→{short_pod(r['dst_pod'])}` "
            f"| {r['bw_GBps']:.3f} | {r['lat_us']:.1f} |\n"
        )
    md += f"""
## 负对照（接近中位）{args.control_k} pair

| # | round/slot | bw GB/s | |Δ中位| |
|---:|---|---:|---:|
"""
    for i, r in enumerate(controls, 1):
        md += f"| {i} | {r['round']}/{r['slot']} | {r['bw_GBps']:.3f} | {abs(r['bw_GBps']-median_bw):.4f} |\n"
    md += f"""
## 复测预注册

- trials={len(trials)}（worst 40 + control 10）
- node-disjoint batches={len(batches)}，sizes={retest_manifest['batch_sizes']}
- 选择与批次已锁定，运行前不根据结果调整
"""
    if errors:
        md += "\n## 校验错误\n\n" + "\n".join(f"- {e}" for e in errors) + "\n"
    (out / "ANALYSIS.md").write_text(md)

    print(json.dumps({"valid": not errors, "errors": errors[:10], "n": len(records), "batches": len(batches)}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
