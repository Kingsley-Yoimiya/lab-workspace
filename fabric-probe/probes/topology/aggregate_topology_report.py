#!/usr/bin/env python3
"""生成中文三层拓扑报告 + host×rail 总表 + 网络组请求清单。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def merge_tables(
    logical_rows: list[dict[str, Any]],
    lldp_rows: list[dict[str, Any]],
    pod_node: dict[str, str],
) -> list[dict[str, Any]]:
    lldp_index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in lldp_rows:
        iface = row.get("iface")
        # net1→xscale_0 ...
        rail = {
            "net1": "xscale_0",
            "net2": "xscale_1",
            "net3": "xscale_2",
            "net4": "xscale_3",
            "eth0": "eth0",
        }.get(str(iface))
        if rail:
            lldp_index[(row.get("pod"), rail if rail != "eth0" else "eth0")] = row
            if rail != "eth0":
                lldp_index[(row.get("pod"), rail)] = row

    out = []
    for row in logical_rows:
        key = (row.get("pod"), row.get("device"))
        lldp = lldp_index.get(key) or lldp_index.get((row.get("pod"), row.get("logical_netdev")))
        merged = dict(row)
        merged["k8s_node"] = pod_node.get(str(row.get("pod")), "unknown")
        if lldp:
            merged["switch_chassis"] = lldp.get("switch_chassis") or "unknown"
            merged["switch_port"] = lldp.get("switch_port") or "unknown"
            merged["switch_system_name"] = lldp.get("system_name") or "unknown"
            merged["evidence_level"] = "lldp"
        else:
            merged["switch_chassis"] = "unknown"
            merged["switch_port"] = "unknown"
            merged["switch_system_name"] = "unknown"
            # 保留 host_static
            merged.setdefault("evidence_level", "host_static")
        out.append(merged)
    return out


def write_report(
    out_dir: Path,
    control: dict[str, Any],
    logical: dict[str, Any],
    lldp: dict[str, Any],
    table: list[dict[str, Any]],
    prior_notes: list[str],
) -> None:
    facts = logical.get("facts") or {}
    md: list[str] = []
    md.append("# 沐曦物理 / 逻辑 / 行为三层拓扑报告")
    md.append("")
    md.append(f"- 作业：`{control.get('job', 'yinjinrun-cs512-20260716-221823')}`")
    md.append(f"- 身份：`{control.get('identity_user', 'yinjinrun.p')}`")
    md.append(f"- 控制面 pod/node：{control.get('pod_count')} pods / {control.get('node_count')} nodes")
    md.append(f"- 逻辑 rail 行：{logical.get('rail_rows', len(table))}")
    md.append(
        f"- LLDP：`{lldp.get('conclusion', 'unknown')}`；总帧={lldp.get('frame_count_total', 0)}"
    )
    md.append("")
    md.append("## 证据等级约定")
    md.append("")
    md.append("| 等级 | 含义 |")
    md.append("|---|---|")
    md.append("| 直接配置 | Kubernetes/CNI/注解或厂商配置快照原文 |")
    md.append("| LLDP | pod 内 AF_PACKET 实际收到并解析的邻居 TLV |")
    md.append("| 主机静态 | sysfs /proc/net/GID/ARP 等只读主机状态 |")
    md.append("| 行为推断 | pair/collective 性能聚类等行为证据 |")
    md.append("| 未知 | 当前权限或观测窗口下无法证实 |")
    md.append("")
    md.append("## A. 物理直连事实")
    md.append("")
    if lldp.get("any_lldp_observed"):
        md.append("### LLDP 可见（证据等级：LLDP）")
        md.append("")
        for s in lldp.get("switch_summary") or []:
            md.append(
                f"- chassis `{s['switch_chassis']}`：接入 pod={s['attached_pods']}，"
                f"唯一 port={s['unique_ports']}"
            )
        md.append("")
        md.append("不得把未经验证的设备角色命名为 leaf/spine。")
    else:
        md.append("### LLDP 在 pod 网络命名空间不可见（证据等级：主机静态 + 负结果）")
        md.append("")
        md.append(
            "已在 CAP_NET_RAW / AF_PACKET 可用前提下，对 eth0 与 net1..4 做只读监听。"
            "未见 EtherType 0x88cc 帧。"
        )
        md.append("")
        md.append("**这证明的是「pod netns 不可见」，不是「网络没 LLDP」。**")
        md.append("")
        sample = (lldp.get("iface_stats") or [])[:5]
        if sample:
            md.append("抽样监听证据：")
            for s in sample:
                md.append(
                    f"- `{s.get('pod')}` `{s.get('iface')}`：时长={s.get('duration_s')}s，"
                    f"raw={s.get('raw_frames_seen')}，lldp={s.get('lldp_frames')}，"
                    f"CAP_NET_RAW={s.get('cap_net_raw_effective')}"
                )
        md.append("")
        md.append("可从主机静态旁证的直连相关事实：")
        md.append("")
        md.append(
            f"- 每 rail 主导邻居 MAC（多为网关/代理 ARP）：`{facts.get('per_rail_dominant_neigh_mac')}`"
            " （证据等级：主机静态）"
        )
        md.append(
            "- NIC 永久 MAC OUI 见总表 `netdev_mac_oui`；不能据此反推交换机型号。"
            " （证据等级：主机静态）"
        )
        md.append("- 交换机管理地址 / chassis / 物理端口：**未知**（缺 LLDP 与交换机控制面）")
    md.append("")
    md.append("## B. 逻辑 rail / IP 子网")
    md.append("")
    md.append(f"- 互异逻辑子网：`{facts.get('distinct_rail_subnets')}` （主机静态）")
    md.append(f"- 四逻辑子网：{facts.get('four_distinct_logical_subnets')} （主机静态）")
    md.append(f"- 同子网跨 rail：{facts.get('same_subnet_across_rails')} （主机静态）")
    md.append(f"- 每 rail 子网计数：`{facts.get('per_rail_subnet_counts')}`")
    md.append(f"- eth0 默认网关分布：`{facts.get('eth0_default_gateway_counts')}`")
    md.append("")
    md.append(facts.get("interpretation") or "")
    md.append("")
    md.append("控制面与数据面角色（逻辑区分，非物理平面声明）：")
    md.append("")
    md.append("- `eth0`：Kubernetes pod 控制面 / 默认路由 （直接配置 + 主机静态）")
    md.append("- `net1..4` ↔ `xscale_0..3`：RoCE 数据面逻辑 rail （主机静态）")
    md.append("- Multus / SR-IOV / CNI 注解：见 `control_plane.json` 过滤字段 （直接配置或未知）")
    md.append("")
    md.append("## C. 行为 rail 分组（不得命名 leaf/spine）")
    md.append("")
    md.append(
        "前序 pair/collective 实验只产生**行为推断**层证据，可单独画「慢 pair / rail 掩码」"
        "分组图，但**禁止**把行为聚类命名为 leaf/spine 或物理故障域。"
    )
    md.append("")
    for note in prior_notes:
        md.append(f"- {note}")
    md.append("")
    md.append("## D. 为何 2016 pair 矩阵在对称 Clos+ECMP 下无法唯一反演物理拓扑")
    md.append("")
    md.append(
        "在对称 Clos 中，任意一对 host 的多 rail / 多 ECMP 路径在带宽与时延上高度同构；"
        "pair 矩阵观测到的是端到端吞吐/尾时延的行为签名，而不是唯一的物理边集合。"
    )
    md.append("")
    md.append("具体不可逆原因：")
    md.append("")
    md.append("1. **ECMP 多路径**：同一逻辑 pair 的报文可哈希到不同上行；重复测量不保证同路径。")
    md.append("2. **对称性**：同 leaf 内与跨 leaf 的多跳路径在无拥塞时性能可重叠，缺少独特指纹。")
    md.append("3. **四 rail 逻辑并行**：端到端慢可能来自任一 rail、共享上行或端侧调度，矩阵单元不是物理链路。")
    md.append("4. **缺直连锚点**：无 LLDP/交换机 port counter 时，行为边无法对齐到 chassis/port。")
    md.append("5. **复测恢复**：前序最差 pair 复测恢复，说明矩阵噪声/间歇拥塞大于稳定拓扑割。")
    md.append("")
    md.append("因此 pair 矩阵最多支持行为分组假设，不能唯一反演物理拓扑。")
    md.append("")
    md.append("## E. 控制面元数据摘要（直接配置）")
    md.append("")
    # 从 pods 提炼 boson / network-status / logical_switch
    logical_switches = {}
    nic_device = None
    nic_host = None
    hostnetwork = set()
    for pod in control.get("pods") or []:
        hostnetwork.add(bool(pod.get("hostNetwork")))
        ann = pod.get("annotations_filtered") or {}
        if nic_device is None and ann.get("boson.sensecore.cn/nic-device"):
            nic_device = ann.get("boson.sensecore.cn/nic-device")
        if nic_host is None and ann.get("boson.sensecore.cn/nic-host"):
            nic_host = ann.get("boson.sensecore.cn/nic-host")
        for k, v in ann.items():
            if k.endswith("logical_switch"):
                logical_switches[str(v)] = logical_switches.get(str(v), 0) + 1
    zone_counts = {}
    cluster_counts = {}
    access_mode = {}
    sriov = {}
    for node in control.get("nodes") or []:
        lab = node.get("labels_filtered") or {}
        z = lab.get("topology.sensecore.cn/zone", "unknown")
        zone_counts[z] = zone_counts.get(z, 0) + 1
        c = lab.get("topology.sensecore.cn/boson-rdma-network-cluster", "unknown")
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
        a = lab.get("resource.sensecore.cn/boson-nic-access-mode", "unknown")
        access_mode[a] = access_mode.get(a, 0) + 1
        s = lab.get("metax-tech.com/gpu.sriov", "unknown")
        sriov[s] = sriov.get(s, 0) + 1
    md.append(f"- `hostNetwork`：`{sorted(hostnetwork)}`")
    md.append(f"- 逻辑交换机 UUID（`*.logical_switch`）计数：`{logical_switches}`")
    md.append(f"- `boson.sensecore.cn/nic-device`：`{nic_device}`")
    md.append(f"- `boson.sensecore.cn/nic-host`（宿主机网卡名）：`{nic_host}`")
    md.append(f"- zone：`{zone_counts}`；RDMA cluster：`{cluster_counts}`")
    md.append(f"- NIC access-mode：`{access_mode}`；`metax-tech.com/gpu.sriov`：`{sriov}`")
    md.append(
        "- `k8s.v1.cni.cncf.io/network-status`：eth0=`kube-ovn`（控制面）；"
        "net1..4=`prod-boson/<logical_switch_uuid>`（数据面）"
    )
    md.append(
        "- 注解中部分 `network_type=geneve` / `pod_nic_type=veth-pair` 与 "
        "`boson-nic-access-mode=PF`、`nic-host=eth10..13` 并存；"
        "前者更像 OVN/控制面元数据，**不能**单独用来否定 PF RoCE 数据面。"
    )
    md.append("")
    md.append(f"- pod 注解键（过滤后）：`{sorted((control.get('pod_annotation_keys') or {}).keys())}`")
    md.append(f"- node label 键（过滤后）：`{sorted((control.get('node_label_keys') or {}).keys())}`")
    md.append("")
    md.append("完整内容见 `control_plane/control_plane.json`（已脱敏，无 token/cert）。")
    md.append("")
    md.append("## F. 仍未知 / 下一步访问")
    md.append("")
    md.append("| 未知项 | 需要的访问 |")
    md.append("|---|---|")
    md.append("| host×rail→交换机 chassis/port | 宿主机 netns LLDP，或交换机 LLDP neighbor |")
    md.append("| leaf/spine 角色与接线 | 网络组正式拓扑 / 配线表（直接配置） |")
    md.append("| VLAN / LAG / MLAG / ECMP hash | 交换机只读配置 |")
    md.append("| DSCP/PCP trust、PFC/ETS/ECN | 交换机 QoS 配置与 counter |")
    md.append("| xscale RoCE 硬件 counter | 驱动/宿主机/厂商 API |")
    md.append("")
    md.append("## G. 可直接让网络组执行的厂商中立命令清单")
    md.append("")
    md.append("请网络组在业务端口与上联上导出（字段级，非具体厂商 CLI）：")
    md.append("")
    md.append("1. **LLDP neighbor**：local if ↔ remote chassis/port/system name/mgmt")
    md.append("2. **interface description / admin+oper status / speed / FEC**")
    md.append("3. **VLAN / port-channel / MLAG 成员与角色**")
    md.append("4. **L3 / BGP / ECMP**：neighbor、前缀、ECMP 成员、hash 字段")
    md.append("5. **DSCP/PCP trust map** → priority / PG / ETS bandwidth")
    md.append("6. **PFC**：enable bitmap、headroom、xon/xoff threshold")
    md.append("7. **ECN / WRED**：min/max/probability")
    md.append("8. **queue/buffer counters**：bytes/packets、occupancy、drop、PFC pause、ECN mark、CNP")
    md.append("9. **host 端口映射**：`hostname / PCI BDF / switch / port / rail`")
    md.append("")
    md.append("返回 CSV/文本即可；不要只给架构示意图。")
    md.append("")
    md.append("## H. host×rail 总表")
    md.append("")
    md.append("见同目录 `host_rail_topology_table.csv`；缺项为 `unknown`。")
    md.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "沐曦物理逻辑行为三层拓扑报告.md").write_text("\n".join(md) + "\n")

    fields = [
        "pod",
        "k8s_node",
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
        "switch_system_name",
        "evidence_level",
    ]
    with (out_dir / "host_rail_topology_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in table:
            w.writerow({k: row.get(k, "unknown") for k in fields})
    with (out_dir / "host_rail_topology_table.jsonl").open("w") as f:
        for row in table:
            f.write(json.dumps({k: row.get(k, "unknown") for k in fields}, ensure_ascii=False) + "\n")

    # 网络组请求独立短文
    req = """# 沐曦交换机侧只读取证请求（厂商中立）

目标：恢复 64 节点 × 4 rail 的物理直连与 QoS/转发配置事实。

请导出以下字段（任意厂商 CLI/API，返回文本或 CSV）：

1. LLDP neighbor（local if ↔ remote chassis/port/system/mgmt）
2. interface description / status / speed / FEC
3. VLAN / port-channel / MLAG
4. L3/BGP/ECMP hash 字段与成员
5. DSCP/PCP trust map、priority/PG/ETS
6. PFC enable/headroom/threshold
7. ECN/WRED
8. queue/buffer counters（含 PFC/ECN/CNP/drop）
9. host↔switch port 映射（hostname、PCI BDF、rail）

对齐作业：`yinjinrun-cs512-20260716-221823`，身份 `yinjinrun.p`。
"""
    (out_dir / "网络组交换机取证请求.md").write_text(req)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    control = load_json(args.work_dir / "control_plane" / "control_plane.json")
    logical = load_json(args.work_dir / "logical" / "logical_rail_aggregate.json")
    # rows live in csv/jsonl
    logical_rows = []
    jsonl = args.work_dir / "logical" / "host_rail_table.jsonl"
    if jsonl.exists():
        logical_rows = [json.loads(x) for x in jsonl.read_text().splitlines() if x.strip()]
    lldp = load_json(args.work_dir / "lldp" / "lldp_aggregate.json")
    lldp_rows = []
    lmap = args.work_dir / "lldp" / "host_rail_switch_map.jsonl"
    if lmap.exists():
        lldp_rows = [json.loads(x) for x in lmap.read_text().splitlines() if x.strip()]
    # enrich logical facts if aggregate stripped rows
    if not logical.get("facts"):
        facts_path = args.work_dir / "logical" / "logical_rail_aggregate.json"
        logical = load_json(facts_path)
    pod_node = {
        p.get("pod"): p.get("node")
        for p in control.get("pods") or []
        if p.get("pod")
    }
    table = merge_tables(logical_rows, lldp_rows, pod_node)
    prior_notes = [
        "W2 inventory：xscale_i→net{i+1} 64/64，rate 200G，MTU 4096（主机静态）",
        "W2 pair 2016：无稳定双向慢链路；最差 pair 复测恢复（行为推断）",
        "W4 rail 掩码：dual01/dual23/quad 差异未形成可重复物理拓扑结论（行为推断）",
        "禁止把行为聚类命名为 leaf/spine",
    ]
    write_report(args.out_dir, control, logical, lldp, table, prior_notes)
    # SUMMARY
    summary = {
        "control_pods": control.get("pod_count"),
        "control_nodes": control.get("node_count"),
        "logical_subnets": (logical.get("facts") or {}).get("distinct_rail_subnets"),
        "four_logical_subnets": (logical.get("facts") or {}).get("four_distinct_logical_subnets"),
        "lldp_conclusion": lldp.get("conclusion"),
        "lldp_frames": lldp.get("frame_count_total"),
        "table_rows": len(table),
        "known_switch_mappings": sum(
            1 for r in table if r.get("switch_chassis") not in (None, "unknown")
        ),
    }
    (args.out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.out_dir / "SUMMARY.md").write_text(
        "# Topology facts SUMMARY\n\n"
        + "\n".join(f"- {k}: `{v}`" for k, v in summary.items())
        + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
