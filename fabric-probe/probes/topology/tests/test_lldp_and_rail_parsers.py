#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from muxi_lldp_listen import parse_lldp_frame  # noqa: E402
from parse_muxi_lldp import aggregate as agg_lldp  # noqa: E402
from parse_muxi_logical_rail import aggregate as agg_rail  # noqa: E402


def _tlv(t: int, value: bytes) -> bytes:
    head = (t << 9) | len(value)
    return struct.pack("!H", head) + value


def build_lldp_frame() -> bytes:
    dst = bytes.fromhex("0180c200000e")
    src = bytes.fromhex("9025f2bd1f53")
    eth = struct.pack("!H", 0x88CC)
    chassis = _tlv(1, bytes([4]) + bytes.fromhex("aabbccddeeff"))
    port = _tlv(2, bytes([7]) + b"Ethernet1/1")
    ttl = _tlv(3, struct.pack("!H", 120))
    name = _tlv(5, b"sw-test-01")
    desc = _tlv(6, b"Vendor NOS")
    end = _tlv(0, b"")
    return dst + src + eth + chassis + port + ttl + name + desc + end


class TestLldpParser(unittest.TestCase):
    def test_parse_basic_tlvs(self) -> None:
        frame = build_lldp_frame()
        parsed = parse_lldp_frame(frame)
        self.assertEqual(parsed["chassis_id"], "aa:bb:cc:dd:ee:ff")
        self.assertEqual(parsed["port_id"], "Ethernet1/1")
        self.assertEqual(parsed["ttl"], 120)
        self.assertEqual(parsed["system_name"], "sw-test-01")
        self.assertIn("raw_hex", parsed)

    def test_aggregate_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pod = root / "master-0"
            (pod / "net1").mkdir(parents=True)
            summary = {
                "pod": "master-0",
                "any_lldp": True,
                "cap": {"cap_net_raw_effective": True, "af_packet_open": True},
                "ifaces": [
                    {
                        "iface": "net1",
                        "status": "OK",
                        "duration_s": 70,
                        "requested_duration_s": 70,
                        "raw_frames_seen": 1,
                        "lldp_frames": 1,
                    }
                ],
            }
            (pod / "summary.json").write_text(json.dumps(summary))
            frame = {
                "chassis_id": "aa:bb:cc:dd:ee:ff",
                "port_id": "Ethernet1/1",
                "system_name": "sw-test-01",
                "ttl": 120,
                "management_addresses": [],
                "vlan_tlvs": [],
                "src_mac": "90:25:f2:bd:1f:53",
            }
            (pod / "net1" / "frames.jsonl").write_text(json.dumps(frame) + "\n")
            agg = agg_lldp([pod])
            self.assertTrue(agg["any_lldp_observed"])
            self.assertEqual(len(agg["mapping_rows"]), 1)
            self.assertEqual(agg["mapping_rows"][0]["switch_port"], "Ethernet1/1")


class TestLogicalRailAggregate(unittest.TestCase):
    def test_four_subnets(self) -> None:
        records = []
        for i in range(2):
            rails = []
            for r, subnet in enumerate(
                ["172.23.0.0/22", "172.24.0.0/22", "172.25.0.0/22", "172.26.0.0/22"]
            ):
                rails.append(
                    {
                        "device": f"xscale_{r}",
                        "logical_netdev": f"net{r+1}",
                        "ipv4": f"172.{23+r}.168.{10+i}",
                        "subnet": subnet,
                        "pci_bdf": f"0000:0{r}:00.0",
                        "gid_index5": "x",
                        "ifindex": str(r + 2),
                        "rate": "200 Gb/sec",
                        "active_mtu": "4096",
                        "netdev_mac": "5c:6a:ec:00:00:01",
                        "netdev_mac_oui": "5c:6a:ec",
                        "dominant_neigh_mac": f"90:25:f2:00:00:0{r}",
                        "dominant_neigh_mac_oui": "90:25:f2",
                        "default_gateway": None,
                    }
                )
            records.append(
                {
                    "pod": f"pod-{i}",
                    "host": f"host-{i}",
                    "rails": rails,
                    "routes": [
                        {
                            "iface": "eth0",
                            "destination": "0.0.0.0",
                            "gateway": "10.120.0.1",
                        }
                    ],
                }
            )
        agg = agg_rail(records)
        self.assertTrue(agg["facts"]["four_distinct_logical_subnets"])
        self.assertEqual(agg["rail_rows"], 8)


if __name__ == "__main__":
    unittest.main()
