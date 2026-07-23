#!/usr/bin/env python3
"""聚合 LLDP 监听结果：host/rail → switch chassis/port 映射。"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_pod_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def aggregate(pod_dirs: list[Path]) -> dict[str, Any]:
    mappings: list[dict[str, Any]] = []
    iface_stats: list[dict[str, Any]] = []
    chassis_hosts: dict[str, set[str]] = defaultdict(set)
    chassis_ports: dict[str, set[str]] = defaultdict(set)
    any_lldp = False
    for pod_dir in pod_dirs:
        summary_path = pod_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_pod_summary(summary_path)
        pod = summary.get("pod") or pod_dir.name
        if summary.get("any_lldp"):
            any_lldp = True
        for iface_res in summary.get("ifaces", []):
            iface = iface_res.get("iface")
            frames = iface_res.get("lldp_frames", 0)
            iface_stats.append(
                {
                    "pod": pod,
                    "iface": iface,
                    "status": iface_res.get("status"),
                    "duration_s": iface_res.get("duration_s"),
                    "requested_duration_s": iface_res.get("requested_duration_s"),
                    "raw_frames_seen": iface_res.get("raw_frames_seen"),
                    "lldp_frames": frames,
                    "cap_net_raw_effective": summary.get("cap", {}).get("cap_net_raw_effective"),
                    "af_packet_open": summary.get("cap", {}).get("af_packet_open"),
                }
            )
            # 读 jsonl 取最新稳定邻居
            jsonl = pod_dir / str(iface) / "frames.jsonl"
            best = None
            if jsonl.exists():
                for line in jsonl.read_text().splitlines():
                    if not line.strip():
                        continue
                    best = json.loads(line)
            if best and frames:
                chassis = best.get("chassis_id") or "unknown"
                port = best.get("port_id") or "unknown"
                mappings.append(
                    {
                        "pod": pod,
                        "iface": iface,
                        "switch_chassis": chassis,
                        "switch_port": port,
                        "system_name": best.get("system_name"),
                        "system_description": best.get("system_description"),
                        "port_description": best.get("port_description"),
                        "ttl": best.get("ttl"),
                        "management_addresses": best.get("management_addresses"),
                        "vlan_tlvs": best.get("vlan_tlvs"),
                        "src_mac": best.get("src_mac"),
                        "evidence_level": "lldp",
                    }
                )
                chassis_hosts[chassis].add(pod)
                chassis_ports[chassis].add(str(port))

    switch_summary = [
        {
            "switch_chassis": chassis,
            "attached_pods": len(hosts),
            "unique_ports": len(ports),
            "pods": sorted(hosts),
            "ports": sorted(ports),
        }
        for chassis, hosts in sorted(chassis_hosts.items())
        for ports in [chassis_ports[chassis]]
    ]
    zero_pods = sorted(
        {
            s["pod"]
            for s in iface_stats
            if s["iface"] in {"net1", "net2", "net3", "net4"} and (s.get("lldp_frames") or 0) == 0
        }
    )
    return {
        "schema_version": "muxi.lldp.v1",
        "any_lldp_observed": any_lldp,
        "pods_with_results": len(pod_dirs),
        "mapping_rows": mappings,
        "iface_stats": iface_stats,
        "switch_summary": switch_summary,
        "pods_with_zero_lldp_on_rails": zero_pods,
        "conclusion": (
            "lldp_visible_in_pod_netns"
            if any_lldp
            else "lldp_not_visible_in_pod_netns_not_proof_of_absent_on_wire"
        ),
        "frame_count_total": sum(s.get("lldp_frames") or 0 for s in iface_stats),
    }


def write_outputs(agg: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lldp_aggregate.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False) + "\n"
    )
    # CSV-like JSONL for mapping
    with (out_dir / "host_rail_switch_map.jsonl").open("w") as f:
        for row in agg["mapping_rows"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    md = [
        "# LLDP 监听聚合",
        "",
        f"- 结论：`{agg['conclusion']}`",
        f"- 任意接口见到 LLDP：{agg['any_lldp_observed']}",
        f"- 总 LLDP 帧：{agg['frame_count_total']}",
        f"- 映射行数：{len(agg['mapping_rows'])}",
        f"- rail 上 0 帧 pod 数：{len(agg['pods_with_zero_lldp_on_rails'])}",
        "",
        "## 交换机汇总（仅 LLDP 可见时）",
        "",
    ]
    if agg["switch_summary"]:
        for s in agg["switch_summary"]:
            md.append(
                f"- chassis `{s['switch_chassis']}`：pods={s['attached_pods']} ports={s['unique_ports']}"
            )
    else:
        md.append("- 无（pod 网络命名空间未观察到 LLDP）")
    md.append("")
    md.append("## 解释边界")
    md.append("")
    md.append(
        "若帧数为 0：证明的是「在给定接口/CAP/时长下 pod netns 不可见」，"
        "不是「网络没有 LLDP」。"
    )
    (out_dir / "LLDP_SUMMARY.md").write_text("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lldp-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    pod_dirs = sorted(p for p in args.lldp_root.iterdir() if p.is_dir() and (p / "summary.json").exists())
    agg = aggregate(pod_dirs)
    write_outputs(agg, args.out_dir)
    print(json.dumps({k: agg[k] for k in ("any_lldp_observed", "conclusion", "frame_count_total", "pods_with_results")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
