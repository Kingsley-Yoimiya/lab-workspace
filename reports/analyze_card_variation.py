#!/usr/bin/env python3
"""昇腾/沐曦 128 卡体质：逐卡真实 CV / 百分位 / 按 host 聚簇统计。

给 WITHIN_CLUSTER_CARD_VARIATION 报告提供**本报告自己算出来**的证据，
不再只是引用别的报告已成文的定性结论。

用法：
    python3 reports/analyze_card_variation.py
输出：
    reports/rounds/within_cluster_variation_stats.json（逐指标统计 + 按host聚簇 + 逐卡离群榜）
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../random-thing
ASCEND_JSONL = REPO_ROOT / "logs/card-fillgap-20260711_140301/results/constitution128.merged.jsonl"
MUXI_JSONL = REPO_ROOT / "logs/muxi-constitution-20260711_232400-muxi-constitution128/results/constitution128.merged.jsonl"
OUT_JSON = Path(__file__).resolve().parent / "rounds" / "within_cluster_variation_stats.json"

METRICS = [
    ("func_tflops", "单卡方阵 GEMM 短窗吞吐"),
    ("sustained_tflops", "连续烤机后可持续吞吐（末窗）"),
    ("cube_vector_tflops", "GEMM+epilogue 端到端吞吐"),
    ("hbm_gbps", "访存+轻算混合带宽代理"),
    ("mte_gbps", "纯搬运带宽代理"),
    ("vector_gflops", "逐元素 FMA 吞吐"),
    ("sfu_gflops", "一元特殊函数吞吐代理"),
    ("scalar_elems_per_s", "长依赖串行链吞吐（跨栈不可比倍速）"),
    ("power_w", "负载末功耗"),
    ("board_temp_c", "负载态板温"),
    ("aicore_util_pct", "主计算核占用率"),
    ("launch_sync_p50_us", "空设备 sync 往返延迟 p50"),
    ("launch_host_overhead_p50_us", "host 侧发射开销 p50"),
    ("launch_burst_p50_us", "连续 64 核 burst 总时延 p50"),
]


def load_cards(path: Path) -> list[dict[str, Any]]:
    cards = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("record") == "card":
                cards.append(obj)
    return cards


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def metric_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    ordered = sorted(values)
    mean = statistics.mean(ordered)
    std = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    cv = (std / mean * 100.0) if mean else 0.0
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "mean": mean,
        "std": std,
        "cv_pct": cv,
        "min": ordered[0],
        "max": ordered[-1],
        "p5": _percentile(ordered, 5),
        "p95": _percentile(ordered, 95),
        "range_pct_of_median": ((ordered[-1] - ordered[0]) / statistics.median(ordered) * 100.0)
        if statistics.median(ordered)
        else 0.0,
    }


def host_cluster_stats(cards: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """按 host 分组求均值，再看 host 均值相对全局中位的偏差，找整节点级簇。"""
    by_host: dict[str, list[float]] = {}
    for c in cards:
        v = c.get(key)
        if isinstance(v, (int, float)):
            by_host.setdefault(c["host"], []).append(float(v))
    if not by_host:
        return {}
    all_vals = [v for vs in by_host.values() for v in vs]
    global_median = statistics.median(all_vals)
    host_means = {h: statistics.mean(vs) for h, vs in by_host.items()}
    rel = {h: (m - global_median) / global_median * 100.0 if global_median else 0.0 for h, m in host_means.items()}
    worst_low = sorted(rel.items(), key=lambda kv: kv[1])[:3]
    worst_high = sorted(rel.items(), key=lambda kv: -kv[1])[:3]
    n_cards = {h: len(vs) for h, vs in by_host.items()}
    return {
        "global_median": global_median,
        "n_hosts": len(by_host),
        "host_rel_pct": {h: round(v, 2) for h, v in rel.items()},
        "n_cards_per_host": n_cards,
        "worst_low_hosts": [(h, round(v, 2), n_cards[h]) for h, v in worst_low],
        "worst_high_hosts": [(h, round(v, 2), n_cards[h]) for h, v in worst_high],
    }


def outlier_cards(cards: list[dict[str, Any]], key: str, *, low_pct_threshold: float = -8.0, top_n: int = 8) -> list[dict[str, Any]]:
    """找相对全局中位偏低超过阈值的卡（默认关注偏慢一侧，慢卡运维价值更高）。"""
    vals = [(c["host"], c.get("device"), c.get(key)) for c in cards if isinstance(c.get(key), (int, float))]
    if not vals:
        return []
    median = statistics.median(v for _, _, v in vals)
    if not median:
        return []
    rel = [(h, d, v, (v - median) / median * 100.0) for h, d, v in vals]
    rel.sort(key=lambda t: t[3])
    out = [
        {"host": h, "device": d, "value": v, "rel_pct": round(r, 2)}
        for h, d, v, r in rel
        if r <= low_pct_threshold
    ][:top_n]
    return out


def verdict_by_host(cards: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for c in cards:
        h = c["host"]
        v = c.get("verdict", "?")
        out.setdefault(h, {}).setdefault(v, 0)
        out[h][v] += 1
    return out


def analyze(label: str, path: Path) -> dict[str, Any]:
    cards = load_cards(path)
    result: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "n_cards": len(cards),
        "n_hosts": len(set(c["host"] for c in cards)),
        "verdict_counts": {},
        "metrics": {},
        "host_cluster": {},
        "slow_outliers": {},
        "verdict_by_host": verdict_by_host(cards),
    }
    for c in cards:
        v = c.get("verdict", "?")
        result["verdict_counts"][v] = result["verdict_counts"].get(v, 0) + 1

    for key, _desc in METRICS:
        vals = [float(c[key]) for c in cards if isinstance(c.get(key), (int, float))]
        st = metric_stats(vals)
        if not st:
            continue
        result["metrics"][key] = st
        result["host_cluster"][key] = host_cluster_stats(cards, key)
        # HBM/MTE/矩阵算力关注偏慢；功耗/温度关注偏高，这里统一先看偏低侧（慢/低）
        result["slow_outliers"][key] = outlier_cards(cards, key)

    return result


def main() -> None:
    ascend = analyze("ascend", ASCEND_JSONL)
    muxi = analyze("muxi", MUXI_JSONL)

    print(f"[ascend] n_cards={ascend['n_cards']} n_hosts={ascend['n_hosts']} verdict={ascend['verdict_counts']}")
    print(f"[muxi]   n_cards={muxi['n_cards']} n_hosts={muxi['n_hosts']} verdict={muxi['verdict_counts']}")
    print()
    print(f"{'metric':<28}{'ascend CV%':>12}{'muxi CV%':>12}{'ascend range%':>16}{'muxi range%':>14}")
    for key, _desc in METRICS:
        a = ascend["metrics"].get(key, {})
        m = muxi["metrics"].get(key, {})
        print(
            f"{key:<28}"
            f"{a.get('cv_pct', float('nan')):>12.2f}"
            f"{m.get('cv_pct', float('nan')):>12.2f}"
            f"{a.get('range_pct_of_median', float('nan')):>16.1f}"
            f"{m.get('range_pct_of_median', float('nan')):>14.1f}"
        )

    print("\n[ascend] host 级 HBM 偏差最低 3 个节点:", ascend["host_cluster"].get("hbm_gbps", {}).get("worst_low_hosts"))
    print("[muxi]   host 级 HBM 偏差最低 3 个节点:", muxi["host_cluster"].get("hbm_gbps", {}).get("worst_low_hosts"))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps({"ascend": ascend, "muxi": muxi}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
