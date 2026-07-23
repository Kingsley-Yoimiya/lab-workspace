#!/usr/bin/env python3
"""本地、无 GPU：nccl_torch_bench 带宽公式 / timing 字段 / JSONL schema。"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from nccl_torch_bench_metrics import (
    alg_bw_gbps,
    assert_timing_contract_notes,
    build_bench_record,
    bus_bw_factor,
    bus_bw_gbps,
    elementwise_max,
    mean_or_zero,
    parse_bytes_list,
    validate_bench_record,
)


class TestParseBytes(unittest.TestCase):
    def test_suffixes(self) -> None:
        self.assertEqual(parse_bytes_list("1K,2M,1G"), [1024, 2 * 1024**2, 1024**3])
        self.assertEqual(parse_bytes_list("256"), [256])


class TestBusBwFormula(unittest.TestCase):
    def test_all_reduce_factor(self) -> None:
        # N=8 → 2*(7/8)=1.75
        self.assertAlmostEqual(bus_bw_factor("all_reduce", 8), 1.75)
        self.assertAlmostEqual(bus_bw_factor("all_reduce", 512), 2.0 * 511 / 512)

    def test_ag_rs_factor(self) -> None:
        self.assertAlmostEqual(bus_bw_factor("all_gather", 8), 7 / 8)
        self.assertAlmostEqual(bus_bw_factor("reduce_scatter", 8), 7 / 8)

    def test_broadcast_factor(self) -> None:
        self.assertEqual(bus_bw_factor("broadcast", 8), 1.0)

    def test_bw_values(self) -> None:
        # 1e9 bytes / 1s → alg=1 GB/s；all_reduce N=2 → bus=1
        self.assertAlmostEqual(alg_bw_gbps(1_000_000_000, 1.0), 1.0)
        self.assertAlmostEqual(bus_bw_gbps("all_reduce", 2, 1_000_000_000, 1.0), 1.0)
        self.assertAlmostEqual(bus_bw_gbps("all_reduce", 8, 1_000_000_000, 1.0), 1.75)


class TestTimingAggregation(unittest.TestCase):
    def test_elementwise_max(self) -> None:
        rows = [
            [0.10, 0.20, 0.15],
            [0.12, 0.18, 0.30],
            [0.11, 0.25, 0.14],
        ]
        self.assertEqual(elementwise_max(rows), [0.12, 0.25, 0.30])

    def test_global_max_not_polluted_by_local_mean(self) -> None:
        local = [0.10, 0.10, 0.10]
        slower = [0.20, 0.20, 0.20]
        gmax = elementwise_max([local, slower])
        self.assertEqual(gmax, slower)
        self.assertAlmostEqual(mean_or_zero(local), 0.10)
        self.assertAlmostEqual(mean_or_zero(gmax), 0.20)
        # 若误用 local 平均算吞吐会高估 2 倍
        nbytes = 1_000_000_000
        self.assertAlmostEqual(alg_bw_gbps(nbytes, mean_or_zero(local)), 10.0)
        self.assertAlmostEqual(alg_bw_gbps(nbytes, mean_or_zero(gmax)), 5.0)


class TestRecordSchema(unittest.TestCase):
    def test_build_and_validate(self) -> None:
        local = [0.01, 0.02, 0.015]
        gmax = [0.02, 0.025, 0.02]
        rec = build_bench_record(
            op="all_reduce",
            world_size=8,
            rank=3,
            host="node-a",
            local_rank=3,
            nbytes=256 * 1024**2,
            dtype="fp32",
            iters_s_local=local,
            iters_s_global_max=gmax,
        )
        self.assertEqual(rec["bw_basis"], "global_max")
        self.assertEqual(rec["timing_version"], "w0.1")
        self.assertEqual(rec["iters_s_local"], local)
        self.assertEqual(rec["iters_s_global_max"], gmax)
        self.assertEqual(rec["n_iters"], 3)
        # 旧字段对齐 global_max
        self.assertAlmostEqual(rec["avg_s"], rec["avg_s_global_max"])
        self.assertAlmostEqual(rec["bus_bw_GBps"], rec["bus_bw_GBps_global_max"])
        self.assertGreater(rec["alg_bw_GBps_local"], rec["alg_bw_GBps_global_max"])
        errs = validate_bench_record(rec)
        self.assertEqual(errs, [])
        # JSON 可序列化
        line = json.dumps(rec)
        back = json.loads(line)
        self.assertEqual(validate_bench_record(back), [])

    def test_validate_rejects_global_lt_local(self) -> None:
        rec = build_bench_record(
            op="all_reduce",
            world_size=4,
            rank=0,
            host="h",
            local_rank=0,
            nbytes=1024,
            dtype="fp16",
            iters_s_local=[0.2],
            iters_s_global_max=[0.1],
        )
        errs = validate_bench_record(rec)
        self.assertTrue(any("global_max" in e for e in errs))

    def test_timing_contract_notes(self) -> None:
        notes = list(assert_timing_contract_notes())
        self.assertIn("stop_clock_before_result_reduction", notes)
        self.assertIn("global_bw_from_max_rank_time", notes)
        self.assertIn("keep_per_iter_raw_times", notes)


class TestLauncherImportSurface(unittest.TestCase):
    """确保 bench 入口可被 py_compile / import 路径发现（不加载 torch）。"""

    def test_metrics_module_path(self) -> None:
        self.assertTrue((_SCRIPT_DIR / "nccl_torch_bench.py").is_file())
        self.assertTrue((_SCRIPT_DIR / "nccl_torch_bench_metrics.py").is_file())


if __name__ == "__main__":
    unittest.main()
