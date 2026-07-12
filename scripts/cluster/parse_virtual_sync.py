#!/usr/bin/env python3
"""虚拟同步重构：从独立时间戳事后模拟同步屏障。

输入：step_times_rank*.jsonl（含 t0/t1 或 ms）
输出：
  - gap_summary.json / gap_by_scale.csv
  - 实验0：real vs virtual 对比
  - 实验1：子集规模 8/16/32/64/128 的 gap 趋势
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load_ranks(run_dir: Path) -> dict[int, list[dict]]:
    by: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(run_dir.glob("step_times_rank*.jsonl")):
        for line in path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rank = int(rec.get("global_rank", rec.get("rank", -1)))
            if rank < 0:
                continue
            by[rank].append(rec)
    for r in by:
        by[r].sort(key=lambda x: int(x.get("iter", 0)))
    return by


def _ms(rec: dict) -> float:
    if "ms" in rec:
        return float(rec["ms"])
    if "t0" in rec and "t1" in rec:
        return (float(rec["t1"]) - float(rec["t0"])) * 1000.0
    raise KeyError("no ms/t0/t1")


def gap_from_step_ms(
    by_rank: dict[int, list[dict]], ranks: list[int], drop_first: int = 50
) -> dict | None:
    """真实同步场景：每 iter 直接用各 rank 的 step ms。"""
    by_iter: dict[int, list[float]] = defaultdict(list)
    for r in ranks:
        for rec in by_rank.get(r, []):
            it = int(rec.get("iter", 0))
            if it <= drop_first:
                continue
            by_iter[it].append(_ms(rec))
    return _gap_from_iter_lists(by_iter, n_ranks=len(ranks))


def gap_virtual_barrier(
    by_rank: dict[int, list[dict]], ranks: list[int], drop_first: int = 50
) -> dict | None:
    """虚拟同步：累计墙钟完成时间取 max，逐步差分。

    对每张卡构造累计完成时刻 C_r[i] = t1 of iter i（若无 t1，则用
    该卡相对起点的累计 ms）。
    虚拟同步步耗时 V[i] = max_r C_r[i] - max_r C_r[i-1]。
    同时给出「若用 step ms 直接取 max-median」作为对照。
    """
    # 优先墙钟累计
    cum: dict[int, list[float]] = {}
    step_ms: dict[int, list[float]] = {}
    for r in ranks:
        rows = [x for x in by_rank.get(r, []) if int(x.get("iter", 0)) > 0]
        rows.sort(key=lambda x: int(x["iter"]))
        if not rows:
            return None
        if "t1" in rows[0]:
            # 相对本卡第一条的 t0 对齐，减少跨节点绝对时钟偏差影响
            t_base = float(rows[0]["t0"])
            cum[r] = [float(x["t1"]) - t_base for x in rows]
        else:
            acc = 0.0
            cs = []
            for x in rows:
                acc += _ms(x) / 1000.0
                cs.append(acc)
            cum[r] = cs
        step_ms[r] = [_ms(x) for x in rows]

    n = min(len(cum[r]) for r in ranks)
    if n <= drop_first + 2:
        return None

    virt_ms: list[float] = []
    raw_gaps: list[float] = []
    raw_meds: list[float] = []
    for i in range(drop_first, n):
        c_now = [cum[r][i] for r in ranks]
        c_prev = [cum[r][i - 1] for r in ranks] if i > 0 else [0.0] * len(ranks)
        v = (max(c_now) - max(c_prev)) * 1000.0
        virt_ms.append(v)
        sms = sorted(step_ms[r][i] for r in ranks)
        med = statistics.median(sms)
        raw_gaps.append(sms[-1] - med)
        raw_meds.append(med)

    # 虚拟屏障下「有效 step」相对中位独立 step 的超额
    # 用 virt_ms - median(step_ms) 作为差距代理
    gaps = [v - m for v, m in zip(virt_ms, raw_meds)]
    return {
        "n_ranks": len(ranks),
        "n_iters": len(virt_ms),
        "method": "virtual_barrier",
        "gap_med_ms": statistics.median(gaps) if gaps else None,
        "gap_mean_ms": statistics.mean(gaps) if gaps else None,
        "virt_step_med_ms": statistics.median(virt_ms) if virt_ms else None,
        "indep_med_med_ms": statistics.median(raw_meds) if raw_meds else None,
        "raw_max_minus_med_med_ms": statistics.median(raw_gaps) if raw_gaps else None,
        "virt_p99_ms": _pct(sorted(virt_ms), 99) if virt_ms else None,
    }


def _pct(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _gap_from_iter_lists(by_iter: dict[int, list[float]], n_ranks: int) -> dict | None:
    gaps: list[float] = []
    meds: list[float] = []
    maxes: list[float] = []
    for it in sorted(by_iter):
        vals = sorted(by_iter[it])
        if len(vals) < max(2, n_ranks // 2):
            continue
        med = statistics.median(vals)
        gaps.append(vals[-1] - med)
        meds.append(med)
        maxes.append(vals[-1])
    if not gaps:
        return None
    return {
        "n_ranks": n_ranks,
        "n_iters": len(gaps),
        "method": "real_step_ms",
        "gap_med_ms": statistics.median(gaps),
        "gap_mean_ms": statistics.mean(gaps),
        "med_med_ms": statistics.median(meds),
        "max_med_ms": statistics.median(maxes),
        "p99_p50": (
            _pct(sorted(maxes), 99) / statistics.median(meds)
            if statistics.median(meds) > 0
            else None
        ),
    }


def sample_subsets(
    all_ranks: list[int],
    sizes: list[int],
    n_random: int,
    seed: int,
) -> dict[int, list[list[int]]]:
    """每个规模：按节点连续块 + 跨节点交错块 + 随机子集。"""
    rng = random.Random(seed)
    # 假设 8 卡/节点，rank = node*8 + local
    by_node: dict[int, list[int]] = defaultdict(list)
    for r in all_ranks:
        by_node[r // 8].append(r)
    nodes = sorted(by_node)
    out: dict[int, list[list[int]]] = {s: [] for s in sizes}
    for s in sizes:
        need_nodes = max(1, math.ceil(s / 8))
        # 物理节点连续块（起点滑动）
        if need_nodes <= len(nodes):
            for start in range(0, max(1, len(nodes) - need_nodes + 1), max(1, need_nodes)):
                chunk = []
                for n in nodes[start : start + need_nodes]:
                    chunk.extend(sorted(by_node[n]))
                out[s].append(sorted(chunk)[:s])
                if len(out[s]) >= 3:
                    break
        # 跨节点交错：每节点取前 k 卡
        if need_nodes <= len(nodes):
            per = max(1, s // need_nodes)
            chunk = []
            for n in nodes[:need_nodes]:
                chunk.extend(sorted(by_node[n])[:per])
            out[s].append(sorted(chunk)[:s])
        # 随机
        for _ in range(n_random):
            if len(all_ranks) >= s:
                out[s].append(sorted(rng.sample(all_ranks, s)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="vsync 根目录（含 real/ 与 independent/ 或单目录）")
    ap.add_argument("--drop-first", type=int, default=50)
    ap.add_argument("--sizes", default="8,16,32,64,128")
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = args.root
    out_dir = args.out or (root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

    results: dict = {"root": str(root)}

    # 实验0：若存在 real + independent 子目录
    real_dir = root / "exp0_real"
    virt_dir = root / "exp0_virtual"
    if real_dir.is_dir() and virt_dir.is_dir():
        real = load_ranks(real_dir)
        virt = load_ranks(virt_dir)
        rr = sorted(real)
        vr = sorted(virt)
        g_real = gap_from_step_ms(real, rr, args.drop_first)
        g_virt = gap_virtual_barrier(virt, vr, args.drop_first)
        g_virt_raw = gap_from_step_ms(virt, vr, args.drop_first)
        results["exp0"] = {
            "real": g_real,
            "virtual_barrier": g_virt,
            "virtual_raw_max_med": g_virt_raw,
            "n_real_ranks": len(rr),
            "n_virt_ranks": len(vr),
        }
        # 吻合判据：量级同阶（比值在 0.3~3）
        ok = None
        if g_real and g_virt and g_real["gap_med_ms"] and g_virt["gap_med_ms"]:
            ratio = g_virt["gap_med_ms"] / max(g_real["gap_med_ms"], 1e-9)
            ok = 0.3 <= ratio <= 3.0
            results["exp0"]["gap_ratio_virt_over_real"] = ratio
            results["exp0"]["calibrated_ok"] = ok

    # 实验1：128 卡独立目录
    exp1_dir = root / "exp1_independent" if (root / "exp1_independent").is_dir() else root
    if (exp1_dir / "step_times_rank000.jsonl").exists() or list(
        exp1_dir.glob("step_times_rank*.jsonl")
    ):
        by = load_ranks(exp1_dir)
        all_ranks = sorted(by)
        subsets = sample_subsets(all_ranks, sizes, args.n_random, args.seed)
        rows = []
        for s in sizes:
            gap_list = []
            for subset in subsets[s]:
                g = gap_virtual_barrier(by, subset, args.drop_first)
                if g and g["gap_med_ms"] is not None:
                    gap_list.append(g["gap_med_ms"])
                    rows.append(
                        {
                            "scale": s,
                            "n_ranks": len(subset),
                            "gap_med_ms": g["gap_med_ms"],
                            "virt_step_med_ms": g["virt_step_med_ms"],
                            "indep_med_med_ms": g["indep_med_med_ms"],
                            "ranks": ",".join(map(str, subset[:8]))
                            + ("..." if len(subset) > 8 else ""),
                        }
                    )
            results.setdefault("exp1", {})[str(s)] = {
                "n_subsets": len(gap_list),
                "gap_med_of_meds_ms": statistics.median(gap_list) if gap_list else None,
                "gap_mean_of_meds_ms": statistics.mean(gap_list) if gap_list else None,
                "gap_std_of_meds_ms": statistics.pstdev(gap_list) if len(gap_list) > 1 else 0.0,
            }
        csv_path = out_dir / "gap_by_scale.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "scale",
                    "n_ranks",
                    "gap_med_ms",
                    "virt_step_med_ms",
                    "indep_med_med_ms",
                    "ranks",
                ],
            )
            w.writeheader()
            for row in rows:
                w.writerow(row)
        results["exp1_csv"] = str(csv_path)
        results["exp1_n_ranks_total"] = len(all_ranks)

    summary = out_dir / "gap_summary.json"
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"WROTE {summary}")


if __name__ == "__main__":
    main()
