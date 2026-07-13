#!/usr/bin/env python3
"""在跳板/集群侧生成 Markdown 简报（无 matplotlib）。"""
from __future__ import annotations

import csv
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

AFS = Path("/afs-a3-241ceshi-shared/montyyin")
OUT = AFS / "results" / "reports" / "offline_20260713"
PEAK = 292.79

# 复用解析逻辑（内联最小版，避免 import 路径）
RE_TFLOP = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)", re.I)


def steady_tflop(rows: list[dict], drop_first: int = 2) -> float | None:
    vals = [r["tflops"] for r in rows if r["iter"] > drop_first]
    return statistics.median(vals) if vals else None


def parse_mfu_logs(scale_dir: Path) -> float | None:
    text = ""
    for f in sorted(scale_dir.glob("*.log")):
        text += f.read_text(errors="ignore") + "\n"
    rows = []
    for line in text.splitlines():
        m = RE_TFLOP.search(line)
        mi = re.search(r"iteration\s+(\d+)\s*/\s*(\d+)", line, re.I)
        if m and mi:
            rows.append({"iter": int(mi.group(1)), "tflops": float(m.group(1))})
    return steady_tflop(rows)


def load_gap_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def moe_mfu_table() -> str:
    stamps = {
        "32": "20260712_181247",
        "64": "20260712_181247",
        "96": "20260712_221912",
    }
    lines = ["| N | stamp | TFLOP/s/GPU (median) | MFU% |", "|---|-------|----------------------|------|"]
    for n, stamp in stamps.items():
        d = AFS / "results" / "mfu_moe_scale" / stamp / f"scale_{n}"
        t = parse_mfu_logs(d) if d.is_dir() else None
        mfu = (t / PEAK * 100) if t else None
        lines.append(f"| {n} | {stamp} | {t or '—'} | {f'{mfu:.1f}' if mfu else '—'} |")
    return "\n".join(lines)


def gap_table_from_root(root: Path, label: str) -> str:
    csvp = root / "gap_vs_n.csv"
    if not csvp.exists():
        # try parse inline if only scale dirs exist
        return f"### {label}\n\n（无 gap_vs_n.csv，见 {root}）\n"
    rows = load_gap_csv(csvp)
    lines = [f"### {label}", "", "| N | gap_med_ms | median_step_ms |", "|---|------------|----------------|"]
    for r in rows:
        lines.append(
            f"| {r.get('world_npu','')} | {r.get('gap_median_ms','')} | {r.get('median_step_ms','')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %z")
    parts = [
        f"# 华为 A3 离线收口简报",
        f"",
        f"> 生成时间：{ts}",
        f"> 路径：`{OUT}`",
        f"",
        f"## MoE 弱扩展 MFU（Phase1 汇总）",
        f"",
        moe_mfu_table(),
        f"",
        gap_table_from_root(AFS / "results" / "dense_failslow" / "20260713_001230", "Dense FailSlow 固定 GBS"),
        gap_table_from_root(AFS / "results" / "dense_failslow_gbsprop" / "20260713_071316", "Dense FailSlow GBS∝DP"),
    ]
    # latest moe failslow
    moe_roots = sorted((AFS / "results" / "moe_failslow").glob("20260713_*"), reverse=True)
    if moe_roots:
        parts.append(gap_table_from_root(moe_roots[0], f"MoE FailSlow ({moe_roots[0].name})"))
    report = OUT / "SUMMARY.md"
    report.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"WROTE {report}")


if __name__ == "__main__":
    main()
