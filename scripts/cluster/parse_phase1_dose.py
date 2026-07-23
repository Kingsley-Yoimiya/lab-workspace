#!/usr/bin/env python3
"""Phase1 剂量结果汇总：dose_trial / sentinel card+meta → dose_table.json

用法:
  # 从已有 dose_trial 行汇总
  python3 parse_phase1_dose.py --trials results/*/dose_trials.jsonl \\
      --out results/dose_table.json

  # 从单次 sentinel + meta 发出一条 dose_trial（需 baseline 或同文件内 dose=0）
  python3 parse_phase1_dose.py --emit-trial \\
      --card results/R/sentinel.*.jsonl --meta results/R/meta.json \\
      --baseline-metric 297.2 >> dose_trials.jsonl

  # 目录批处理：每个子目录含 meta.json + sentinel*.jsonl
  python3 parse_phase1_dose.py --from-dirs results/phase1-* --out dose_table.json
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


# factor → (metric_field, higher_is_better)
FACTOR_METRIC: dict[str, tuple[str, bool]] = {
    "cube": ("func_tflops", True),
    "vector": ("vector_gflops", True),
    "hbm_mte": ("mte_gbps", True),
    "cpu": ("launch_host_overhead_p50_us", False),
    "placebo": ("func_tflops", True),
}

# 交叉指标（写入 cross_metrics，不参与主 drop 聚合主键）
CROSS_METRICS = (
    "func_tflops",
    "hbm_gbps",
    "sustained_tflops",
    "vector_gflops",
    "mte_gbps",
    "launch_host_overhead_p50_us",
    "launch_sync_p50_us",
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _iter_jsonl_paths(patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat, recursive=True))
        if not matches and Path(pat).is_file():
            matches = [pat]
        for m in matches:
            p = Path(m)
            if p.is_file():
                out.append(p)
    return out


def _first_card(rows: list[dict]) -> dict | None:
    for r in rows:
        if r.get("record") == "card":
            return r
    return None


def _find_sentinel(dir_path: Path) -> Path | None:
    cands = sorted(dir_path.glob("sentinel*.jsonl"))
    return cands[0] if cands else None


def _metric_drop_pct(baseline: float, value: float, higher_is_better: bool) -> float:
    if baseline == 0:
        return 0.0
    if higher_is_better:
        return (baseline - value) / baseline * 100.0
    # 延迟类：升高 = 性能变差
    return (value - baseline) / baseline * 100.0


def _resolve_metric(factor: str, meta: dict) -> tuple[str, bool]:
    factor = meta.get("factor") or factor
    if factor in FACTOR_METRIC:
        return FACTOR_METRIC[factor]
    # 未知因素默认看 func_tflops
    return "func_tflops", True


def emit_trial(
    card: dict,
    meta: dict,
    baseline_metric: float | None = None,
    baselines: dict[str, float] | None = None,
) -> dict:
    """拼一条 record=dose_trial。"""
    factor = str(meta.get("factor") or meta.get("inject_kind") or "cube")
    metric_name, hib = _resolve_metric(factor, meta)
    value = card.get(metric_name)
    if value is None and metric_name == "mte_gbps":
        value = card.get("dma_copy_gbps")
        metric_name = "mte_gbps" if value is not None else metric_name

    base = baseline_metric
    if base is None and baselines is not None:
        base = baselines.get(factor)
    drop = None
    if base is not None and value is not None:
        drop = _metric_drop_pct(float(base), float(value), hib)

    cross: dict[str, Any] = {}
    for k in CROSS_METRICS:
        if k == metric_name:
            continue
        v = card.get(k)
        if v is None and k == "mte_gbps":
            v = card.get("dma_copy_gbps")
        if v is None:
            continue
        entry: dict[str, Any] = {"value": v}
        if baselines and factor in baselines and k == metric_name:
            pass
        # 对交叉指标若有同名 baseline 字典扩展：baselines 仅主因素；交叉 drop 可选
        cross[k] = entry

    trial = {
        "record": "dose_trial",
        "run_id": meta.get("run_id"),
        "phase": meta.get("phase", 1),
        "phys_device": meta.get("phys_device"),
        "visible_device": meta.get("visible_device", 0),
        "factor": factor,
        "dose_label": meta.get("dose_label"),
        "inject_kind": meta.get("inject_kind", factor),
        "inject_mode": meta.get("inject_mode", "process"),
        "inject_params": meta.get("inject_params") or {},
        "placebo": bool(meta.get("placebo", False)),
        "sentinel_probe": metric_name,
        "metric_name": metric_name,
        "metric_baseline_med": base,
        "metric_value": value,
        "metric_drop_pct": drop,
        "cross_metrics": cross,
        "temp_c": card.get("hotspot_temp_c") or card.get("health_temp_c") or card.get("board_temp_c"),
        "power_w": card.get("power_w") or card.get("health_power_w"),
        "host": card.get("host"),
        "device": card.get("device"),
        "ok": value is not None,
    }
    return trial


def collect_trials_from_dirs(dirs: list[Path]) -> list[dict]:
    """扫描目录：meta.json + sentinel*.jsonl → dose_trial；先收集 baseline(dose 0/placebo)。"""
    metas: list[tuple[Path, dict, dict]] = []
    for d in dirs:
        meta_path = d / "meta.json"
        sent = _find_sentinel(d)
        if not meta_path.is_file() or sent is None:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        card = _first_card(_load_jsonl(sent))
        if card is None:
            continue
        metas.append((d, meta, card))

    # 每 factor 取 dose_label in {0, off, placebo} 或 placebo=true 的中位作 baseline
    by_factor_base: dict[str, list[float]] = {}
    for _, meta, card in metas:
        factor = str(meta.get("factor") or meta.get("inject_kind") or "")
        label = str(meta.get("dose_label", ""))
        is_base = (
            bool(meta.get("placebo"))
            or label in ("0", "off", "placebo", "zero")
            or float((meta.get("inject_params") or {}).get("duty") or 1) <= 0.0
        )
        if not is_base:
            continue
        metric_name, _ = _resolve_metric(factor, meta)
        val = card.get(metric_name)
        if val is None and metric_name == "mte_gbps":
            val = card.get("dma_copy_gbps")
        if val is None:
            continue
        by_factor_base.setdefault(factor, []).append(float(val))

    baselines = {
        f: statistics.median(vs) for f, vs in by_factor_base.items() if vs
    }

    trials: list[dict] = []
    for _, meta, card in metas:
        trials.append(emit_trial(card, meta, baselines=baselines))
    return trials


def aggregate_dose_table(trials: list[dict]) -> dict:
    """factor → dose_label → {median_drop_pct, params, n, ...}"""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for t in trials:
        if t.get("record") and t.get("record") != "dose_trial":
            continue
        factor = str(t.get("factor") or "")
        label = str(t.get("dose_label") or "")
        if not factor or not label:
            continue
        buckets.setdefault((factor, label), []).append(t)

    table: dict[str, dict[str, Any]] = {}
    for (factor, label), rows in sorted(buckets.items()):
        drops = [float(r["metric_drop_pct"]) for r in rows if r.get("metric_drop_pct") is not None]
        # params：取众数/首条非空 inject_params
        params = {}
        for r in rows:
            p = r.get("inject_params") or {}
            if p:
                params = p
                break
        values = [r.get("metric_value") for r in rows if r.get("metric_value") is not None]
        entry = {
            "median_drop_pct": statistics.median(drops) if drops else None,
            "mean_drop_pct": statistics.mean(drops) if drops else None,
            "params": params,
            "n": len(rows),
            "n_with_drop": len(drops),
            "metric_name": rows[0].get("metric_name"),
            "median_metric_value": statistics.median(values) if values else None,
            "median_baseline": rows[0].get("metric_baseline_med"),
        }
        table.setdefault(factor, {})[label] = entry
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse Phase1 dose calibration results")
    ap.add_argument("--trials", nargs="*", default=[], help="dose_trial jsonl 路径/glob")
    ap.add_argument("--from-dirs", nargs="*", default=[], help="含 meta.json+sentinel 的目录")
    ap.add_argument("--card", default="", help="sentinel jsonl（配合 --emit-trial）")
    ap.add_argument("--meta", default="", help="meta.json（配合 --emit-trial）")
    ap.add_argument("--baseline-metric", type=float, default=None)
    ap.add_argument("--emit-trial", action="store_true", help="打印单条 dose_trial JSON 到 stdout")
    ap.add_argument("--out", default="", help="写出 dose_table.json")
    ap.add_argument("--write-trials", default="", help="把收集到的 dose_trial 写入该 jsonl")
    args = ap.parse_args()

    trials: list[dict] = []

    if args.emit_trial:
        if not args.card or not args.meta:
            ap.error("--emit-trial 需要 --card 与 --meta")
        card_paths = _iter_jsonl_paths([args.card])
        if not card_paths:
            raise SystemExit(f"no card jsonl: {args.card}")
        card = _first_card(_load_jsonl(card_paths[0]))
        if card is None:
            raise SystemExit(f"no record=card in {card_paths[0]}")
        meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
        trial = emit_trial(card, meta, baseline_metric=args.baseline_metric)
        print(json.dumps(trial, ensure_ascii=False))
        return

    if args.trials:
        for p in _iter_jsonl_paths(args.trials):
            for row in _load_jsonl(p):
                if row.get("record") == "dose_trial":
                    trials.append(row)

    if args.from_dirs:
        dirs: list[Path] = []
        for pat in args.from_dirs:
            for m in sorted(glob.glob(pat)):
                p = Path(m)
                if p.is_dir():
                    dirs.append(p)
                elif p.is_file() and p.name == "meta.json":
                    dirs.append(p.parent)
        # 也接受直接给父目录：扫一层子目录
        extra: list[Path] = []
        for d in list(dirs):
            if (d / "meta.json").is_file():
                continue
            for sub in sorted(d.iterdir()):
                if sub.is_dir() and (sub / "meta.json").is_file():
                    extra.append(sub)
        dirs.extend(extra)
        trials.extend(collect_trials_from_dirs(dirs))

    if not trials:
        print("no dose_trial rows found", file=sys.stderr)
        raise SystemExit(1)

    if args.write_trials:
        out_t = Path(args.write_trials)
        out_t.parent.mkdir(parents=True, exist_ok=True)
        with out_t.open("w", encoding="utf-8") as f:
            for t in trials:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"wrote {len(trials)} trials → {out_t}", file=sys.stderr)

    table = aggregate_dose_table(trials)
    text = json.dumps(table, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(text, encoding="utf-8")
        print(f"wrote dose_table → {out_p}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
