#!/usr/bin/env python3
"""解析 muxi_inventory_probe.sh 原始日志并做严格 inventory schema 校验。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "muxi.inventory.v1"
REQUIRED_RAILS = [f"xscale_{i}" for i in range(4)]
REQUIRED_RAIL_FIELDS = {
    "present",
    "pci_path",
    "pci_bdf",
    "state",
    "phys_state",
    "rate",
    "active_mtu",
    "active_mtu_source",
    "gid_index5",
    "gid_index5_netdev",
    "device_netdevs",
}


def _signature(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _first_number(value: str) -> int | None:
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def _largest_number(value: str) -> int | None:
    values = [int(x) for x in re.findall(r"\d+", value)]
    return max(values) if values else None


def _parse_proc_net(lines: list[str]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for line in lines:
        if ":" not in line:
            continue
        iface, values = line.split(":", 1)
        columns = values.split()
        if len(columns) < 16 or not all(x.isdigit() for x in columns[:16]):
            continue
        nums = [int(x) for x in columns[:16]]
        result[iface.strip()] = {
            "rx_bytes": nums[0],
            "rx_packets": nums[1],
            "tx_bytes": nums[8],
            "tx_packets": nums[9],
        }
    return result


def parse_raw(path: Path) -> dict[str, Any]:
    meta: dict[str, str] = {}
    rails: dict[str, dict[str, Any]] = {}
    sections: dict[str, list[str]] = {}
    section: str | None = None
    status = ""
    for raw in path.read_text(errors="replace").splitlines():
        parts = raw.split("\t")
        if len(parts) >= 3 and parts[0] == "META":
            meta[parts[1]] = "\t".join(parts[2:])
        elif len(parts) >= 4 and parts[0] == "RAIL":
            rails.setdefault(parts[1], {})[parts[2]] = "\t".join(parts[3:])
        elif len(parts) == 3 and parts[0] == "SECTION":
            if parts[2] == "BEGIN":
                section = parts[1]
                sections[section] = []
            elif parts[2] == "END":
                section = None
        elif len(parts) == 2 and parts[0] == "PROBE_STATUS":
            status = parts[1]
        elif section is not None:
            sections[section].append(raw)

    errors: list[str] = []
    if meta.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version")
    for field in ("pod", "host", "pod_ip", "collected_at"):
        if not meta.get(field):
            errors.append(f"meta.{field}")
    if status != "OK":
        errors.append(f"probe_status={status or 'missing'}")

    normalized_rails: list[dict[str, Any]] = []
    for dev in REQUIRED_RAILS:
        values = rails.get(dev, {})
        missing = sorted(REQUIRED_RAIL_FIELDS - values.keys())
        if missing:
            errors.append(f"{dev}.missing={','.join(missing)}")
        gid = values.get("gid_index5", "")
        gid_valid = bool(gid and gid not in {"::", "0:0:0:0:0:0:0:0"})
        rail = {
            "device": dev,
            **values,
            "state_active": "ACTIVE" in values.get("state", "").upper(),
            "phys_link_up": "LINKUP" in values.get("phys_state", "").upper(),
            "rate_gbps": _first_number(values.get("rate", "")),
            "active_mtu_bytes": _largest_number(values.get("active_mtu", "")),
            "gid_index5_valid": gid_valid,
        }
        normalized_rails.append(rail)
        if values.get("present") != "true":
            errors.append(f"{dev}.not_present")
        if not rail["state_active"]:
            errors.append(f"{dev}.state")
        if not rail["phys_link_up"]:
            errors.append(f"{dev}.phys_state")
        if not rail["gid_index5_valid"]:
            errors.append(f"{dev}.gid5")
        if not values.get("gid_index5_netdev"):
            errors.append(f"{dev}.gid5_netdev")
        if not values.get("pci_bdf"):
            errors.append(f"{dev}.pci_bdf")

    topo_raw = "\n".join(sections.get("mx_smi_topo_n", []))
    ibv_raw = "\n".join(sections.get("ibv_devinfo", []))
    layout_key = [
        (r["device"], r.get("pci_bdf"), r.get("gid_index5_netdev"), r.get("device_netdevs"))
        for r in normalized_rails
    ]
    topo_normalized = "\n".join(
        " ".join(x.split())
        for x in topo_raw.splitlines()
        if x.strip() and not x.lstrip().startswith("Timestamp")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_file": path.name,
        "pod": meta.get("pod"),
        "host": meta.get("host"),
        "pod_ip": meta.get("pod_ip"),
        "collected_at": meta.get("collected_at"),
        "tools": {k[5:]: v for k, v in meta.items() if k.startswith("tool_")},
        "rails": normalized_rails,
        "proc_net_dev": _parse_proc_net(sections.get("proc_net_dev", [])),
        "mx_smi_topo_n_raw": topo_raw,
        "topo_signature": _signature(topo_normalized),
        "pci_layout_signature": _signature(layout_key),
        "ibv_devinfo_available": bool(ibv_raw and "UNAVAILABLE" not in ibv_raw),
        "validation_errors": errors,
    }


def aggregate(records: list[dict[str, Any]], expected_nodes: int) -> dict[str, Any]:
    errors: list[str] = []
    pods = [r.get("pod") for r in records]
    if len(records) != expected_nodes:
        errors.append(f"nodes={len(records)} expected={expected_nodes}")
    if len(set(pods)) != len(pods):
        errors.append("duplicate_pod")
    for record in records:
        if record["validation_errors"]:
            errors.append(f"{record['pod']}:{';'.join(record['validation_errors'])}")

    layout_counts = Counter(r["pci_layout_signature"] for r in records)
    topo_counts = Counter(r["topo_signature"] for r in records)
    majority_layout = layout_counts.most_common(1)[0][0] if layout_counts else None
    majority_topo = topo_counts.most_common(1)[0][0] if topo_counts else None
    outliers: dict[str, list[str]] = {}
    for record in records:
        reasons = list(record["validation_errors"])
        if (
            majority_layout
            and layout_counts[record["pci_layout_signature"]] == 1
            and record["pci_layout_signature"] != majority_layout
        ):
            reasons.append("pci_layout_signature")
        if (
            majority_topo
            and topo_counts[record["topo_signature"]] == 1
            and record["topo_signature"] != majority_topo
        ):
            reasons.append("topo_signature")
        for rail in record["rails"]:
            if rail["rate_gbps"] != 200:
                reasons.append(f"{rail['device']}.rate={rail['rate_gbps']}")
            if rail["active_mtu_bytes"] != 4096:
                reasons.append(f"{rail['device']}.mtu={rail['active_mtu_bytes']}")
        if reasons:
            outliers[str(record["pod"])] = sorted(set(reasons))

    rails = [rail for record in records for rail in record["rails"]]
    tool_names = sorted({name for record in records for name in record["tools"]})
    tool_status_counts = {
        name: dict(Counter(record["tools"].get(name, "missing") for record in records))
        for name in tool_names
    }
    interface_counts = Counter(
        iface for record in records for iface in record["proc_net_dev"].keys()
    )
    gid5_mapping_counts = Counter(
        f"{rail['device']}->{rail.get('gid_index5_netdev')}" for rail in rails
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not errors,
        "errors": errors,
        "nodes": len(records),
        "rails": len(rails),
        "active_rails": sum(x["state_active"] for x in rails),
        "link_up_rails": sum(x["phys_link_up"] for x in rails),
        "gid5_valid_rails": sum(x["gid_index5_valid"] for x in rails),
        "gid5_ipv4_mapped_rails": sum(":ffff:" in x.get("gid_index5", "") for x in rails),
        "rate_gbps_counts": dict(sorted(Counter(x["rate_gbps"] for x in rails).items(), key=str)),
        "active_mtu_counts": dict(
            sorted(Counter(x["active_mtu_bytes"] for x in rails).items(), key=str)
        ),
        "gid5_netdev_counts": dict(
            sorted(Counter(x.get("gid_index5_netdev") for x in rails).items(), key=str)
        ),
        "gid5_device_netdev_counts": dict(sorted(gid5_mapping_counts.items())),
        "proc_net_interface_counts": dict(sorted(interface_counts.items())),
        "tool_status_counts": tool_status_counts,
        "pci_layout_groups": dict(layout_counts),
        "topo_groups": dict(topo_counts),
        "ibv_devinfo_available_nodes": sum(x["ibv_devinfo_available"] for x in records),
        "outliers": outliers,
        "physical_topology_naming": "forbidden_without_ground_truth",
    }


def write_outputs(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    jsonl_path: Path,
    csv_path: Path,
    summary_json_path: Path,
    summary_md_path: Path,
) -> None:
    with jsonl_path.open("w") as f:
        for record in sorted(records, key=lambda x: str(x["pod"])):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with csv_path.open("w", newline="") as f:
        fields = [
            "pod",
            "host",
            "pod_ip",
            "device",
            "pci_bdf",
            "state",
            "phys_state",
            "rate_gbps",
            "active_mtu_bytes",
            "active_mtu_source",
            "gid_index5",
            "gid_index5_netdev",
            "device_netdevs",
            "pci_layout_signature",
            "topo_signature",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda x: str(x["pod"])):
            for rail in record["rails"]:
                writer.writerow(
                    {
                        "pod": record["pod"],
                        "host": record["host"],
                        "pod_ip": record["pod_ip"],
                        "pci_layout_signature": record["pci_layout_signature"],
                        "topo_signature": record["topo_signature"],
                        **{k: rail.get(k) for k in fields if k in rail},
                    }
                )
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    outlier_lines = (
        [f"- `{pod}`：{', '.join(reasons)}" for pod, reasons in summary["outliers"].items()]
        or ["- 无"]
    )
    md = f"""# Muxi W2.1 低风险 inventory

- 节点：{summary['nodes']}
- xscale rail：{summary['rails']}
- ACTIVE / LinkUp / GID5有效：{summary['active_rails']} / {summary['link_up_rails']} / {summary['gid5_valid_rails']}
- rate分布：`{summary['rate_gbps_counts']}`
- active_mtu分布：`{summary['active_mtu_counts']}`
- GID5 netdev分布：`{summary['gid5_netdev_counts']}`
- GID5 IPv4-mapped：{summary['gid5_ipv4_mapped_rails']} / {summary['rails']}
- GID5 device→netdev：`{summary['gid5_device_netdev_counts']}`
- /proc/net/dev接口覆盖：`{summary['proc_net_interface_counts']}`
- 已知工具可用性：`{summary['tool_status_counts']}`
- ibv_devinfo可用节点：{summary['ibv_devinfo_available_nodes']}
- PCI布局组：`{summary['pci_layout_groups']}`
- GPU↔NIC topo签名组：`{summary['topo_groups']}`

## 离群节点

{chr(10).join(outlier_lines)}

## 解释边界

这里只验证端点inventory、静态链路字段和亲和签名。LLDP、ip、ethtool、
traceroute不可用时不安装；没有交换机ground truth，禁止把任何分组命名为
leaf/spine或具体物理故障域。
"""
    summary_md_path.write_text(md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--expected-nodes", type=int, default=64)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args()
    records = [parse_raw(path) for path in sorted(args.raw_dir.glob("*.raw.log"))]
    summary = aggregate(records, args.expected_nodes)
    write_outputs(records, summary, args.jsonl, args.csv, args.summary_json, args.summary_md)
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
