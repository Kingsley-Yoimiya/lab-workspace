#!/usr/bin/env python3
"""聚合 64×4 rail 逻辑清单：子网、gateway MAC、地址规律。"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_records(raw_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(raw_dir.glob("*.json")):
        records.append(json.loads(path.read_text()))
    return records


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    subnet_by_rail: dict[str, Counter[str]] = defaultdict(Counter)
    neigh_mac_by_rail: dict[str, Counter[str]] = defaultdict(Counter)
    ipv4_by_rail: dict[str, list[str]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for rec in records:
        for rail in rec.get("rails", []):
            device = rail.get("device")
            subnet = rail.get("subnet") or "unknown"
            neigh = rail.get("dominant_neigh_mac") or "unknown"
            ipv4 = rail.get("ipv4") or "unknown"
            subnet_by_rail[device][subnet] += 1
            neigh_mac_by_rail[device][neigh] += 1
            if ipv4 != "unknown":
                ipv4_by_rail[device].append(ipv4)
            rows.append(
                {
                    "pod": rec.get("pod"),
                    "host": rec.get("host"),
                    "device": device,
                    "logical_netdev": rail.get("logical_netdev"),
                    "ipv4": ipv4,
                    "gid_index5": rail.get("gid_index5"),
                    "subnet": subnet,
                    "pci_bdf": rail.get("pci_bdf"),
                    "ifindex": rail.get("ifindex"),
                    "rate": rail.get("rate"),
                    "active_mtu": rail.get("active_mtu"),
                    "netdev_mac": rail.get("netdev_mac"),
                    "netdev_mac_oui": rail.get("netdev_mac_oui"),
                    "dominant_neigh_mac": neigh,
                    "dominant_neigh_mac_oui": rail.get("dominant_neigh_mac_oui"),
                    "default_gateway": rail.get("default_gateway") or "none",
                    "evidence_level": "host_static",
                    "switch_chassis": "unknown",
                    "switch_port": "unknown",
                }
            )

    # 事实判断：四逻辑子网 vs 同子网
    unique_subnets = sorted({r["subnet"] for r in rows if r["subnet"] != "unknown"})
    eth0_default_gw = Counter()
    for rec in records:
        for route in rec.get("routes", []):
            if route.get("iface") == "eth0" and route.get("destination") == "0.0.0.0":
                eth0_default_gw[route.get("gateway")] += 1

    # 地址第三/四字节分布（规律）
    octet_patterns = {}
    for device, ips in ipv4_by_rail.items():
        thirds = Counter()
        for ip in ips:
            try:
                parts = ip.split(".")
                thirds[parts[2]] += 1
            except Exception:
                continue
        octet_patterns[device] = {
            "count": len(ips),
            "third_octet_top": thirds.most_common(5),
            "unique_third_octets": len(thirds),
        }

    facts = {
        "distinct_rail_subnets": unique_subnets,
        "rail_count_expected_logic": 4,
        "four_distinct_logical_subnets": len(unique_subnets) == 4
        and all(s.startswith("172.") for s in unique_subnets),
        "same_subnet_across_rails": len(unique_subnets) == 1,
        "per_rail_dominant_neigh_mac": {
            d: macs.most_common(3) for d, macs in sorted(neigh_mac_by_rail.items())
        },
        "per_rail_subnet_counts": {
            d: dict(c) for d, c in sorted(subnet_by_rail.items())
        },
        "eth0_default_gateway_counts": dict(eth0_default_gw),
        "octet_patterns": octet_patterns,
        "interpretation": (
            "观测到四个互不相同的逻辑 IPv4 子网（每 rail 一个），且每 rail 的邻居 MAC "
            "高度集中，符合「逻辑 rail / 逻辑子网隔离 + 网关/代理 ARP」；"
            "不能据此声称四个物理独立交换平面。"
            if len(unique_subnets) == 4
            else "子网模式与「四逻辑子网」不完全一致，见 per_rail_subnet_counts。"
        ),
    }
    return {
        "schema_version": "muxi.logical_rail.v1",
        "nodes": len(records),
        "rail_rows": len(rows),
        "rows": rows,
        "facts": facts,
        "valid": len(records) >= 1 and len(rows) == len(records) * 4,
    }


def write_outputs(agg: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logical_rail_aggregate.json").write_text(
        json.dumps({k: v for k, v in agg.items() if k != "rows"}, indent=2, ensure_ascii=False)
        + "\n"
    )
    fields = [
        "pod",
        "host",
        "device",
        "logical_netdev",
        "ipv4",
        "gid_index5",
        "subnet",
        "pci_bdf",
        "ifindex",
        "rate",
        "active_mtu",
        "netdev_mac",
        "netdev_mac_oui",
        "dominant_neigh_mac",
        "dominant_neigh_mac_oui",
        "default_gateway",
        "switch_chassis",
        "switch_port",
        "evidence_level",
    ]
    with (out_dir / "host_rail_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in agg["rows"]:
            w.writerow({k: row.get(k, "unknown") for k in fields})
    with (out_dir / "host_rail_table.jsonl").open("w") as f:
        for row in agg["rows"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    facts = agg["facts"]
    md = f"""# 逻辑 rail / 子网事实

- 节点：{agg['nodes']}
- rail 行：{agg['rail_rows']}
- 互异逻辑子网：`{facts['distinct_rail_subnets']}`
- 四逻辑子网：{facts['four_distinct_logical_subnets']}
- 同子网跨 rail：{facts['same_subnet_across_rails']}
- 每 rail 子网计数：`{facts['per_rail_subnet_counts']}`
- 每 rail 主导邻居 MAC：`{facts['per_rail_dominant_neigh_mac']}`
- eth0 默认网关：`{facts['eth0_default_gateway_counts']}`

## 解释

{facts['interpretation']}

证据等级：主机静态（sysfs /proc/net/route / ARP / GID）。
不得把逻辑隔离直接称为物理独立平面。
"""
    (out_dir / "LOGICAL_RAIL_SUMMARY.md").write_text(md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.raw_dir)
    agg = aggregate(records)
    write_outputs(agg, args.out_dir)
    print(
        json.dumps(
            {
                "nodes": agg["nodes"],
                "valid": agg["valid"],
                "subnets": agg["facts"]["distinct_rail_subnets"],
                "four_logical_subnets": agg["facts"]["four_distinct_logical_subnets"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
