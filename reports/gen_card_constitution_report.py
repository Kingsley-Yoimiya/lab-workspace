#!/usr/bin/env python3
"""128 卡 constitution 分布报告（兼容入口）。

实际逻辑在 plot_card_constitution.py；本文件保留旧 CLI。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from plot_card_constitution import discover_jsonl, generate


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 constitution128 分布报告")
    ap.add_argument("--data-dir", type=Path,
                    help="含 JSONL 的 logs/results 目录（递归搜索 *.jsonl）")
    ap.add_argument("--jsonl", type=Path, action="append", default=[],
                    help="单个 JSONL 文件，可重复")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "rounds")
    args = ap.parse_args()

    paths: list[Path] = list(args.jsonl)
    if args.data_dir:
        paths.extend(discover_jsonl(args.data_dir))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("未找到 JSONL：请指定 --data-dir 或 --jsonl")

    report_path, fig_dir = generate(paths, args.out_dir)
    print(f"wrote {report_path}")
    if fig_dir.exists():
        print(f"figs  {fig_dir}")


if __name__ == "__main__":
    main()
