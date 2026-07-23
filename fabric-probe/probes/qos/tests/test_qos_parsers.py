from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestQosParsers(unittest.TestCase):
    def test_visibility_evidence_layers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw"
            raw.mkdir()
            for i in range(3):
                record = {
                    "pod": f"pod{i}",
                    "library_string_hits": [
                        {
                            "environment_names": [
                                "MCCL_IB_TC",
                                "MCCL_ALGO",
                                "MCCL_PROTO",
                                "MCCL_IB_QPS_PER_CONNECTION",
                            ],
                            "semantic_strings": ["MCCL_IB_TC", "tclass", "ibv_modify_qp"],
                        }
                    ],
                    "mccl_log_hits": [
                        {"lines": ["MCCL INFO MCCL_IB_TC set by environment to 128."]}
                    ],
                    "xscale_sysfs": [
                        {
                            "kind": "counter_directory",
                            "exists": False,
                            "readable": False,
                        }
                        for _ in range(8)
                    ],
                    "xscale_counter_files": [],
                    "commands": {
                        name: None
                        for name in (
                            "perfquery",
                            "rdma",
                            "ethtool",
                            "ibstat",
                            "iblinkinfo",
                            "devlink",
                            "xscale",
                            "xscale_tool",
                        )
                    },
                    "filesystem_visibility": {"/sys/kernel/debug": []},
                }
                (raw / f"pod{i}.json").write_text(json.dumps(record))
            cmd = [
                "python3",
                str(ROOT / "parse_muxi_visibility.py"),
                "--raw-dir",
                str(raw),
                "--summary-json",
                str(root / "summary.json"),
                "--parameters-csv",
                str(root / "params.csv"),
                "--summary-md",
                str(root / "SUMMARY.md"),
            ]
            subprocess.run(cmd, check=True)
            summary = json.loads((root / "summary.json").read_text())
            self.assertTrue(summary["tc_evidence"]["environment_variable_read"]["proven"])
            self.assertFalse(summary["tc_evidence"]["qp_traffic_class_set"]["proven"])
            self.assertFalse(summary["tc_evidence"]["wire_dscp_ecn"]["proven"])
            self.assertEqual(summary["xscale_counter_directories"]["existing"], 0)

    def test_netdev_counter_not_data_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            before = root / "before"
            after = root / "after"
            before.mkdir()
            after.mkdir()
            fields = {
                "rx_bytes": 1000,
                "tx_bytes": 1000,
                "rx_packets": 10,
                "tx_packets": 10,
                "rx_errors": 0,
                "tx_errors": 0,
                "rx_dropped": 0,
                "tx_dropped": 0,
            }
            for i in range(2):
                a = {
                    "pod": f"pod{i}",
                    "interfaces": {f"net{j}": dict(fields) for j in range(1, 5)},
                }
                b = json.loads(json.dumps(a))
                for values in b["interfaces"].values():
                    values["rx_bytes"] += 260
                    values["tx_bytes"] += 210
                    values["rx_packets"] += 5
                    values["tx_packets"] += 5
                (before / f"pod{i}.json").write_text(json.dumps(a))
                (after / f"pod{i}.json").write_text(json.dumps(b))
            cmd = [
                "python3",
                str(ROOT / "parse_netdev_counter_delta.py"),
                "--before-dir",
                str(before),
                "--after-dir",
                str(after),
                "--expected-pods",
                "2",
                "--payload-bytes",
                str(16 * 1024 * 1024 * 13),
                "--json",
                str(root / "delta.json"),
                "--csv",
                str(root / "delta.csv"),
                "--summary-md",
                str(root / "SUMMARY.md"),
            ]
            subprocess.run(cmd, check=True)
            summary = json.loads((root / "delta.json").read_text())
            self.assertTrue(summary["valid"])
            self.assertFalse(summary["roce_data_byte_sensitive"])
            self.assertFalse(summary["error_drop_delta_nonzero"])


if __name__ == "__main__":
    unittest.main()
