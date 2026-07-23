#!/usr/bin/env python3
"""生成偶数节点无向 all-pairs 的 round-robin 完美匹配调度。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ALGORITHM_VERSION = "circle-v1-direction-sha256-v1"


def pod_names(n: int, job: str) -> list[str]:
    if n < 2:
        raise ValueError("n must be >= 2")
    return [f"{job}-master-0"] + [f"{job}-worker-{i}" for i in range(n - 1)]


def _direction(seed: int, round_id: int, a: int, b: int) -> bool:
    lo, hi = sorted((a, b))
    digest = hashlib.sha256(f"{seed}:{round_id}:{lo}:{hi}".encode()).digest()
    return bool(digest[0] & 1)


def generate_schedule(n: int, seed: int, job: str) -> list[dict]:
    if n % 2:
        raise ValueError("n must be even")
    names = pod_names(n, job)
    ring = list(range(n))
    rows: list[dict] = []
    edge_id = 0
    for round_id in range(n - 1):
        for slot in range(n // 2):
            a, b = ring[slot], ring[n - 1 - slot]
            lo, hi = sorted((a, b))
            reverse = _direction(seed, round_id, lo, hi)
            src, dst = (hi, lo) if reverse else (lo, hi)
            rows.append(
                {
                    "schema_version": "muxi.pair_schedule.v1",
                    "algorithm_version": ALGORITHM_VERSION,
                    "seed": seed,
                    "round": round_id,
                    "slot": slot,
                    "edge_id": edge_id,
                    "unordered_a": lo,
                    "unordered_b": hi,
                    "src_index": src,
                    "dst_index": dst,
                    "src_pod": names[src],
                    "dst_pod": names[dst],
                    "src_gpu": 0,
                    "dst_gpu": 0,
                    "hca": "xscale_0",
                }
            )
            edge_id += 1
        ring = [ring[0], ring[-1], *ring[1:-1]]
    validate_schedule(rows, n)
    return rows


def validate_schedule(rows: list[dict], n: int) -> None:
    if len(rows) != n * (n - 1) // 2:
        raise ValueError(f"edge_count={len(rows)}")
    pairs = [(r["unordered_a"], r["unordered_b"]) for r in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate unordered pair")
    expected = {(i, j) for i in range(n) for j in range(i + 1, n)}
    if set(pairs) != expected:
        raise ValueError("pair coverage mismatch")
    for round_id in range(n - 1):
        round_rows = [r for r in rows if r["round"] == round_id]
        nodes = [x for r in round_rows for x in (r["src_index"], r["dst_index"])]
        if len(round_rows) != n // 2 or sorted(nodes) != list(range(n)):
            raise ValueError(f"round {round_id} is not a perfect matching")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--job", default="yinjinrun-cs512-20260716-221823")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    rows = generate_schedule(args.nodes, args.seed, args.job)
    with args.jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"SCHEDULE_VALID nodes={args.nodes} rounds={args.nodes-1} "
        f"pairs_per_round={args.nodes//2} edges={len(rows)} "
        f"seed={args.seed} algorithm={ALGORITHM_VERSION}"
    )


if __name__ == "__main__":
    main()
