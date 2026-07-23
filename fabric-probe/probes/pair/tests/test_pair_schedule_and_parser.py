from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "run_muxi_pair_rounds_jump.sh"
sys.path.insert(0, str(ROOT))
from generate_round_robin import ALGORITHM_VERSION, generate_schedule, validate_schedule


class TestPairSchedule(unittest.TestCase):
    def test_driver_does_not_expand_unbound_local(self) -> None:
        source = DRIVER.read_text()
        self.assertIn('local round_id="$1"\n  local round_dir=', source)
        self.assertNotIn('local round_id="$1" round_dir=', source)
        self.assertIn("timeout 150 bash '$AFS_OUT/code/round_", source)
        self.assertNotIn("setsid nohup bash /tmp/${RUN_ID}.r${current_round}", source)
        self.assertIsNone(
            re.search(
                r'local (?:rank|idx)="\\$1"[^\\n]*pod="\\$\\{(?:pods|round_pods)\\[\\$(?:rank|idx)\\]\\}"',
                source,
            )
        )
        self.assertIn("/--master_port=$port/ && /$RUN_ID/", source)
        self.assertIn("kill -KILL -- -\\$g", source)

    def test_complete_perfect_matching(self) -> None:
        rows = generate_schedule(64, 202607182345, "test-job")
        validate_schedule(rows, 64)
        self.assertEqual(len(rows), 2016)
        self.assertEqual({x["algorithm_version"] for x in rows}, {ALGORITHM_VERSION})
        for round_id in range(63):
            nodes = [
                node
                for row in rows
                if row["round"] == round_id
                for node in (row["src_index"], row["dst_index"])
            ]
            self.assertEqual(sorted(nodes), list(range(64)))
        pairs = {(x["unordered_a"], x["unordered_b"]) for x in rows}
        self.assertEqual(len(pairs), 2016)

    def test_direction_is_reproducible(self) -> None:
        a = generate_schedule(64, 123, "test-job")
        b = generate_schedule(64, 123, "test-job")
        c = generate_schedule(64, 124, "test-job")
        self.assertEqual(a, b)
        self.assertNotEqual(
            [(x["src_index"], x["dst_index"]) for x in a],
            [(x["src_index"], x["dst_index"]) for x in c],
        )

    def test_pair_local_rendezvous_plan(self) -> None:
        rows = [x for x in generate_schedule(64, 321, "test-job") if x["round"] == 0]
        plans = []
        for row in rows:
            port = 31000 + row["slot"]
            master_addr = f"{row['src_pod']}.test-job"
            plans.extend(
                [
                    (row["slot"], 0, row["src_pod"], master_addr, port),
                    (row["slot"], 1, row["dst_pod"], master_addr, port),
                ]
            )
        self.assertEqual(len({x[4] for x in plans}), 32)
        for slot in range(32):
            pair = [x for x in plans if x[0] == slot]
            self.assertEqual({x[1] for x in pair}, {0, 1})
            self.assertEqual(pair[0][3], pair[1][3])
            src = next(x for x in pair if x[1] == 0)
            self.assertEqual(src[2] + ".test-job", src[3])

    def test_first_two_rounds_parser(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            schedule = generate_schedule(64, 456, "test-job")
            schedule_path = root / "schedule.jsonl"
            schedule_path.write_text("\n".join(json.dumps(x) for x in schedule) + "\n")
            results = root / "results"
            for row in schedule:
                if row["round"] >= 2:
                    continue
                out = results / f"round_{row['round']}" / f"pair_{row['slot']}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                times = [0.001] * 10
                record = {
                    "schema_version": "muxi.pair_result.v1",
                    "timing_version": "p2p.w0.1",
                    "round": row["round"],
                    "slot": row["slot"],
                    "edge_id": row["edge_id"],
                    "src_index": row["src_index"],
                    "dst_index": row["dst_index"],
                    "src_pod": row["src_pod"],
                    "dst_pod": row["dst_pod"],
                    "src_host": row["src_pod"],
                    "dst_host": row["dst_pod"],
                    "src_gpu": 0,
                    "dst_gpu": 0,
                    "hca": "xscale_0",
                    "nbytes": 16 * 1024 * 1024,
                    "iters": 10,
                    "iters_s_global_max": times,
                    "avg_s_global_max": 0.001,
                    "bw_GBps": 16.777216,
                    "lat_us": 1000.0,
                    "pattern_ok": True,
                }
                out.write_text(json.dumps(record) + "\n")
            cmd = [
                "python3",
                str(ROOT / "parse_pair_rounds.py"),
                "--schedule",
                str(schedule_path),
                "--results-dir",
                str(results),
                "--rounds",
                "2",
                "--expected-iters",
                "10",
                "--jsonl",
                str(root / "pairs.jsonl"),
                "--csv",
                str(root / "pairs.csv"),
                "--summary-json",
                str(root / "summary.json"),
                "--summary-md",
                str(root / "SUMMARY.md"),
            ]
            subprocess.run(cmd, check=True)
            summary = json.loads((root / "summary.json").read_text())
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["pairs"], 64)
            self.assertEqual(summary["rounds"]["0"]["pairs"], 32)
            self.assertEqual(summary["rounds"]["1"]["pairs"], 32)

    def test_empty_round_reports_schema_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            schedule = generate_schedule(64, 789, "test-job")
            schedule_path = root / "schedule.jsonl"
            schedule_path.write_text("\n".join(json.dumps(x) for x in schedule) + "\n")
            (root / "results").mkdir()
            cmd = [
                "python3",
                str(ROOT / "parse_pair_rounds.py"),
                "--schedule",
                str(schedule_path),
                "--results-dir",
                str(root / "results"),
                "--rounds",
                "1",
                "--jsonl",
                str(root / "pairs.jsonl"),
                "--csv",
                str(root / "pairs.csv"),
                "--summary-json",
                str(root / "summary.json"),
                "--summary-md",
                str(root / "SUMMARY.md"),
            ]
            result = subprocess.run(cmd)
            self.assertNotEqual(result.returncode, 0)
            summary = json.loads((root / "summary.json").read_text())
            self.assertFalse(summary["valid"])
            self.assertEqual(summary["pairs"], 0)
            self.assertIn("—", (root / "SUMMARY.md").read_text())


if __name__ == "__main__":
    unittest.main()
