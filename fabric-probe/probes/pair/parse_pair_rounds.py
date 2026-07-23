#!/usr/bin/env python3
"""校验前若干round的pair结果并输出JSONL/CSV/中文摘要。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def quantile(values: list[float], p: float) -> float:
    return sorted(values)[math.ceil(p * len(values)) - 1]


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--round-ids", default="")
    parser.add_argument("--pair-limit", type=int, default=32)
    parser.add_argument("--expected-iters", type=int, default=10)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args()
    selected_rounds = (
        [int(x) for x in args.round_ids.split(",") if x.strip()]
        if args.round_ids
        else list(range(args.rounds))
    )

    schedule = [
        json.loads(x)
        for x in args.schedule.read_text().splitlines()
        if x.strip()
        and json.loads(x)["round"] in selected_rounds
        and json.loads(x)["slot"] < args.pair_limit
    ]
    expected = {(x["round"], x["slot"]): x for x in schedule}
    records: list[dict] = []
    errors: list[str] = []
    for path in sorted(args.results_dir.glob("round_*/*.json")):
        lines = [x for x in path.read_text().splitlines() if x.strip()]
        if len(lines) != 1:
            errors.append(f"{path.name}:lines={len(lines)}")
            continue
        record = json.loads(lines[0])
        records.append(record)

    seen: set[tuple[int, int]] = set()
    for record in records:
        key = (record.get("round"), record.get("slot"))
        row = expected.get(key)
        if row is None:
            errors.append(f"unexpected={key}")
            continue
        if key in seen:
            errors.append(f"duplicate={key}")
        seen.add(key)
        for field in ("edge_id", "src_index", "dst_index", "src_pod", "dst_pod", "hca"):
            if record.get(field) != row.get(field):
                errors.append(f"{key}:{field}")
        if (
            record.get("schema_version") != "muxi.pair_result.v1"
            or record.get("timing_version") != "p2p.w0.1"
            or record.get("nbytes") != 16 * 1024 * 1024
            or record.get("iters") != args.expected_iters
            or len(record.get("iters_s_global_max", [])) != args.expected_iters
            or not record.get("pattern_ok")
            or record.get("src_gpu") != 0
            or record.get("dst_gpu") != 0
        ):
            errors.append(f"{key}:schema")
    missing = sorted(set(expected) - seen)
    if missing:
        errors.append(f"missing={missing}")

    for record in records:
        values = record["iters_s_global_max"]
        record["p50_us"] = statistics.median(values) * 1e6
        record["p95_us"] = quantile(values, 0.95) * 1e6
        record["p99_us"] = quantile(values, 0.99) * 1e6

    bw_values = [x["bw_GBps"] for x in records]
    latency_values = [x["lat_us"] for x in records]
    bw_median = statistics.median(bw_values) if bw_values else 0
    latency_median = statistics.median(latency_values) if latency_values else 0
    outliers = [
        {
            "round": x["round"],
            "slot": x["slot"],
            "src_pod": x["src_pod"],
            "dst_pod": x["dst_pod"],
            "bw_GBps": x["bw_GBps"],
            "lat_us": x["lat_us"],
            "reason": "bw<0.75*median"
            if x["bw_GBps"] < 0.75 * bw_median
            else "lat>1.5*median",
        }
        for x in records
        if x["bw_GBps"] < 0.75 * bw_median or x["lat_us"] > 1.5 * latency_median
    ]
    rounds: dict[str, dict] = {}
    for round_id in selected_rounds:
        rows = [x for x in records if x["round"] == round_id]
        bws = [x["bw_GBps"] for x in rows]
        lats = [x["lat_us"] for x in rows]
        rounds[str(round_id)] = {
            "pairs": len(rows),
            "bw_median_GBps": statistics.median(bws) if bws else None,
            "bw_min_GBps": min(bws) if bws else None,
            "bw_max_GBps": max(bws) if bws else None,
            "lat_median_us": statistics.median(lats) if lats else None,
            "lat_p95_us": quantile(lats, 0.95) if lats else None,
            "lat_max_us": max(lats) if lats else None,
        }
    summary = {
        "valid": not errors,
        "errors": errors,
        "rounds_requested": len(selected_rounds),
        "round_ids": selected_rounds,
        "pairs": len(records),
        "rounds": rounds,
        "overall_bw_median_GBps": bw_median,
        "overall_bw_min_GBps": min(bw_values) if bw_values else None,
        "overall_bw_max_GBps": max(bw_values) if bw_values else None,
        "overall_lat_median_us": latency_median,
        "outliers": outliers,
        "analysis_limit": "two rounds validate scheduling/probe only; no clustering or topology naming",
    }

    with args.jsonl.open("w") as f:
        for record in sorted(records, key=lambda x: (x["round"], x["slot"])):
            f.write(json.dumps(record) + "\n")
    fields = [
        "round",
        "slot",
        "edge_id",
        "src_index",
        "dst_index",
        "src_pod",
        "dst_pod",
        "src_host",
        "dst_host",
        "src_gpu",
        "dst_gpu",
        "hca",
        "nbytes",
        "bw_GBps",
        "lat_us",
        "p50_us",
        "p95_us",
        "p99_us",
        "pattern_ok",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: x.get(k) for k in fields} for x in sorted(records, key=lambda x: (x["round"], x["slot"])))
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    round_lines = [
        f"| {rid} | {value['pairs']} | {fmt(value['bw_median_GBps'])} | "
        f"{fmt(value['bw_min_GBps'])}–{fmt(value['bw_max_GBps'])} | "
        f"{fmt(value['lat_median_us'], 1)} | {fmt(value['lat_p95_us'], 1)} |"
        for rid, value in rounds.items()
    ]
    outlier_lines = (
        [
            f"- round{o['round']} slot{o['slot']} `{o['src_pod']}→{o['dst_pod']}`："
            f"{o['bw_GBps']:.3f} GB/s，{o['lat_us']:.1f} us，{o['reason']}"
            for o in outliers
        ]
        or ["- 无"]
    )
    args.summary_md.write_text(
        f"""# Muxi W2.2 P2P 前两轮验证

| round | pair数 | 带宽中位(GB/s) | 带宽范围(GB/s) | 延迟中位(us) | 延迟p95(us) |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(round_lines)}

## 明显离群

{chr(10).join(outlier_lines)}

## 控制语义

- 正控制：16MiB单向 `torch.distributed.isend/irecv` 完成且pattern校验通过，
  MCCL使用single xscale_0，证明probe路径可运行。
- 负控制：每轮32个node-disjoint pair并发，每节点只参与一次，用于降低probe
  自身端点争用；它不是“网络绝无共享”的证明。

## 解释边界

仅执行2/63轮，目的只是验证完美匹配调度、MCCL P2P可运行、schema完整和
观察明显离群。不能据此聚类、恢复物理路径或命名leaf/spine。
"""
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
