from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSER = ROOT / "parse_muxi_inventory.py"
SCHEMA = ROOT / "schema" / "muxi.inventory.v1.schema.json"


def raw_record(pod: str, rate: int = 200) -> str:
    lines = [
        "META\tschema_version\tmuxi.inventory.v1",
        f"META\tpod\t{pod}",
        f"META\thost\t{pod}",
        "META\tpod_ip\t10.0.0.1",
        "META\tcollected_at\t2026-07-18T00:00:00+00:00",
        "META\ttool_lldpctl\tunavailable",
        "META\ttool_ip\tunavailable",
    ]
    for i in range(4):
        dev = f"xscale_{i}"
        lines.extend(
            [
                f"RAIL\t{dev}\tpresent\ttrue",
                f"RAIL\t{dev}\tpci_path\t/sys/devices/0000:0{i}:00.0",
                f"RAIL\t{dev}\tpci_bdf\t0000:0{i}:00.0",
                f"RAIL\t{dev}\tstate\t4: ACTIVE",
                f"RAIL\t{dev}\tphys_state\t5: LinkUp",
                f"RAIL\t{dev}\trate\t{rate} Gb/sec",
                f"RAIL\t{dev}\tactive_mtu\t5: 4096",
                f"RAIL\t{dev}\tactive_mtu_source\tsysfs",
                f"RAIL\t{dev}\tgid_index5\t0000:0000:0000:0000:0000:ffff:0a00:0001",
                f"RAIL\t{dev}\tgid_index5_netdev\tnet{i+1}",
                f"RAIL\t{dev}\tdevice_netdevs\tnet{i+1}",
            ]
        )
    lines.extend(
        [
            "SECTION\tproc_net_dev\tBEGIN",
            "net1: 100 1 0 0 0 0 0 0 200 2 0 0 0 0 0 0",
            "SECTION\tproc_net_dev\tEND",
            "SECTION\tmx_smi_topo_n\tBEGIN",
            "GPU0 NIC0 PIX",
            "SECTION\tmx_smi_topo_n\tEND",
            "SECTION\tibv_devinfo\tBEGIN",
            "hca_id: xscale_0",
            "SECTION\tibv_devinfo\tEND",
            "PROBE_STATUS\tOK",
        ]
    )
    return "\n".join(lines) + "\n"


class TestMuxiInventoryParser(unittest.TestCase):
    def test_valid_inventory_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw"
            raw.mkdir()
            (raw / "pod0.raw.log").write_text(raw_record("pod0"))
            (raw / "pod1.raw.log").write_text(raw_record("pod1"))
            cmd = [
                "python3",
                str(PARSER),
                "--raw-dir",
                str(raw),
                "--expected-nodes",
                "2",
                "--jsonl",
                str(root / "inventory.jsonl"),
                "--csv",
                str(root / "rails.csv"),
                "--summary-json",
                str(root / "summary.json"),
                "--summary-md",
                str(root / "SUMMARY.md"),
            ]
            subprocess.run(cmd, check=True)
            summary = json.loads((root / "summary.json").read_text())
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["nodes"], 2)
            self.assertEqual(summary["rails"], 8)
            self.assertEqual(summary["rate_gbps_counts"], {"200": 8})
            with (root / "rails.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 8)
            records = [json.loads(x) for x in (root / "inventory.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(len(records[0]["rails"]), 4)
            schema = json.loads(SCHEMA.read_text())
            self.assertEqual(schema["properties"]["schema_version"]["const"], "muxi.inventory.v1")

    def test_rate_outlier_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw"
            raw.mkdir()
            (raw / "pod0.raw.log").write_text(raw_record("pod0", rate=100))
            cmd = [
                "python3",
                str(PARSER),
                "--raw-dir",
                str(raw),
                "--expected-nodes",
                "1",
                "--jsonl",
                str(root / "inventory.jsonl"),
                "--csv",
                str(root / "rails.csv"),
                "--summary-json",
                str(root / "summary.json"),
                "--summary-md",
                str(root / "SUMMARY.md"),
            ]
            result = subprocess.run(cmd)
            self.assertEqual(result.returncode, 0)
            summary = json.loads((root / "summary.json").read_text())
            self.assertIn("pod0", summary["outliers"])


if __name__ == "__main__":
    unittest.main()
