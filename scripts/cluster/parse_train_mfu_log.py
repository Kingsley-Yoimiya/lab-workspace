#!/usr/bin/env python3
"""从 Megatron/MindSpeed 训练日志抽取稳态 TFLOP/s/GPU 与 MFU。"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# 常见日志形态：
# iteration        3/5 | ... | elapsed time per iteration (ms): 123.4 | throughput per GPU (TFLOP/s/GPU): 170.5 | ...
RE_ITER = re.compile(
    r"iteration\s+(\d+)/\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)",
    re.IGNORECASE | re.DOTALL,
)
RE_TFLOP = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)", re.I)
RE_MS = re.compile(r"elapsed time per iteration \(ms\):\s*([\d.]+)", re.I)


def parse_text(text: str) -> list[dict]:
    rows: list[dict] = []
    # 按行扫，避免跨行 DOTALL 误匹配
    for line in text.splitlines():
        if "iteration" not in line.lower() and "tflop" not in line.lower():
            continue
        m_iter = re.search(r"iteration\s+(\d+)\s*/\s*(\d+)", line, re.I)
        m_t = RE_TFLOP.search(line)
        m_ms = RE_MS.search(line)
        if not (m_iter and m_t):
            continue
        rows.append(
            {
                "iter": int(m_iter.group(1)),
                "iters_total": int(m_iter.group(2)),
                "ms": float(m_ms.group(1)) if m_ms else None,
                "tflops_per_gpu": float(m_t.group(1)),
            }
        )
    return rows


def steady(rows: list[dict], drop_first: int = 1) -> dict | None:
    if not rows:
        return None
    use = rows[drop_first:] if len(rows) > drop_first else rows
    vals = [r["tflops_per_gpu"] for r in use]
    mss = [r["ms"] for r in use if r.get("ms") is not None]
    return {
        "n": len(vals),
        "tflops_mean": statistics.fmean(vals),
        "tflops_median": statistics.median(vals),
        "tflops_min": min(vals),
        "tflops_max": max(vals),
        "ms_mean": statistics.fmean(mss) if mss else None,
        "iters": use,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="日志文件或目录")
    ap.add_argument("--peak", type=float, default=292.79, help="单卡 peak TFLOPS")
    ap.add_argument("--drop-first", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    texts: list[str] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            for f in sorted(path.rglob("*.log")):
                texts.append(f.read_text(errors="ignore"))
        elif path.is_file():
            texts.append(path.read_text(errors="ignore"))
        else:
            print(f"WARN missing {path}", file=sys.stderr)

    rows = parse_text("\n".join(texts))
    st = steady(rows, drop_first=args.drop_first)
    out = {
        "n_iters_parsed": len(rows),
        "peak_tflops": args.peak,
        "all_iters": rows,
        "steady": st,
    }
    if st:
        out["mfu"] = st["tflops_mean"] / args.peak if args.peak > 0 else None
        out["mfu_pct"] = (out["mfu"] * 100.0) if out["mfu"] is not None else None

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        if not st:
            print("NO_METRICS")
            return 1
        print(
            f"steady_tflops={st['tflops_mean']:.3f} "
            f"median={st['tflops_median']:.3f} "
            f"mfu={out['mfu_pct']:.2f}% "
            f"n={st['n']} peak={args.peak}"
        )
    return 0 if st else 1


if __name__ == "__main__":
    raise SystemExit(main())
