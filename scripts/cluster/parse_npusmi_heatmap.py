#!/usr/bin/env python3
"""解析 npu-smi info 多节点采样日志 → 功耗/AICore 均值 CSV（Block C）。"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

RE_POWER = re.compile(
    r"\|\s*(\d+)\s+Ascend\S+\s+\|\s+\S+\s+\|\s+([\d.]+|-)\s+(\d+)\s+"
)
RE_CHIP = re.compile(
    r"\|\s*(\d+)\s+(\d+)\s+\|\s+\S+\s+\|\s+(\d+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)"
)


def parse_file(path: Path) -> list[dict]:
    samples: list[dict] = []
    pending_power: float | None = None
    for line in path.read_text(errors="ignore").splitlines():
        mp = RE_POWER.search(line)
        if mp:
            pending_power = None if mp.group(2) == "-" else float(mp.group(2))
            continue
        mc = RE_CHIP.search(line)
        if not mc:
            continue
        samples.append(
            {
                "phy_id": int(mc.group(2)),
                "power_w": pending_power,
                "aicore_pct": int(mc.group(3)),
                "hbm_used": int(mc.group(6)),
                "hbm_total": int(mc.group(7)),
            }
        )
        pending_power = None
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    power: dict[tuple[int, int], list[float]] = defaultdict(list)
    aicore: dict[tuple[int, int], list[float]] = defaultdict(list)
    logs = sorted(args.root.glob("*.log"))
    node_names = [p.stem for p in logs]
    for ni, p in enumerate(logs):
        for s in parse_file(p):
            key = (ni, int(s["phy_id"]))
            if s["power_w"] is not None:
                power[key].append(float(s["power_w"]))
            aicore[key].append(float(s["aicore_pct"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "node_idx", "node", "phy_id",
                "power_w_mean", "aicore_pct_mean", "n_power", "n_aicore",
            ],
        )
        w.writeheader()
        for ni, name in enumerate(node_names):
            for phy in range(16):
                pk = (ni, phy)
                pw, ac = power.get(pk) or [], aicore.get(pk) or []
                w.writerow(
                    {
                        "node_idx": ni,
                        "node": name,
                        "phy_id": phy,
                        "power_w_mean": (sum(pw) / len(pw)) if pw else "",
                        "aicore_pct_mean": (sum(ac) / len(ac)) if ac else "",
                        "n_power": len(pw),
                        "n_aicore": len(ac),
                    }
                )
    print(f"nodes={len(node_names)} samples_ok → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
