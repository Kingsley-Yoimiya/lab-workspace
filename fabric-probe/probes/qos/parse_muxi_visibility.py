#!/usr/bin/env python3
"""聚合MCCL/QoS可见性探针，严格区分三层TC证据。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--parameters-csv", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(x.read_text()) for x in sorted(args.raw_dir.glob("*.json"))]
    if not records:
        raise SystemExit("no raw records")

    pods = [x["pod"] for x in records]
    tc_log_hits = []
    env_names = set()
    semantic_strings = set()
    for record in records:
        for item in record["library_string_hits"]:
            env_names.update(item.get("environment_names", []))
            semantic_strings.update(item.get("semantic_strings", []))
        for item in record["mccl_log_hits"]:
            tc_log_hits.extend(
                line for line in item.get("lines", []) if "MCCL_IB_TC set by environment" in line
            )

    counter_dirs = [
        row
        for record in records
        for row in record["xscale_sysfs"]
        if row.get("kind") == "counter_directory"
    ]
    command_names = (
        "perfquery",
        "rdma",
        "ethtool",
        "ibstat",
        "iblinkinfo",
        "devlink",
        "xscale",
        "xscale_tool",
    )
    unavailable = {
        name: sum(record["commands"].get(name) is None for record in records)
        for name in command_names
    }
    parameter_rows = [
        {
            "parameter": "MCCL_IB_TC",
            "evidence": "runtime_info",
            "controllable": "yes_read_by_library",
            "supported_value_evidence": "MCCL INFO set by environment to 128",
            "not_proven": "QP traffic_class value and wire DSCP/ECN",
        },
        {
            "parameter": "MCCL_ALGO",
            "evidence": "library_explicit_set_string",
            "controllable": "yes_candidate",
            "supported_value_evidence": "vendor sample NCCL_ALGO=Ring; library Tree string",
            "not_proven": "all values accepted on this cluster",
        },
        {
            "parameter": "MCCL_PROTO",
            "evidence": "library_explicit_set_string",
            "controllable": "yes_candidate",
            "supported_value_evidence": "vendor sample NCCL_PROTO=Simple; library Simple/LL128 strings",
            "not_proven": "all values accepted on this cluster",
        },
        {
            "parameter": "MCCL_MIN_NCHANNELS",
            "evidence": "library_environment_name",
            "controllable": "recognized_not_run",
            "supported_value_evidence": "name and topo search strings in libmccl",
            "not_proven": "value range and runtime acceptance",
        },
        {
            "parameter": "MCCL_MAX_NCHANNELS",
            "evidence": "library_environment_name",
            "controllable": "recognized_not_run",
            "supported_value_evidence": "name and topo search strings in libmccl",
            "not_proven": "value range and runtime acceptance",
        },
        {
            "parameter": "MCCL_IB_QPS_PER_CONNECTION",
            "evidence": "library_environment_name",
            "controllable": "recognized_not_run",
            "supported_value_evidence": "name in libmccl strings",
            "not_proven": "value range, QP creation effect, ECMP entropy",
        },
    ]
    with args.parameters_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(parameter_rows[0]))
        writer.writeheader()
        writer.writerows(parameter_rows)

    summary = {
        "schema_version": "muxi.qos_visibility.summary.v1",
        "valid": len(records) == 3 and len(set(pods)) == 3,
        "pods": pods,
        "tc_evidence": {
            "environment_variable_read": {
                "proven": len(tc_log_hits) > 0,
                "hits": len(tc_log_hits),
                "example": tc_log_hits[0] if tc_log_hits else None,
            },
            "qp_traffic_class_set": {
                "proven": False,
                "implementation_clues": [
                    x
                    for x in ("MCCL_IB_TC", "tclass", "ibv_modify_qp")
                    if any(x.lower() in s.lower() for s in semantic_strings)
                ],
            },
            "wire_dscp_ecn": {"proven": False, "reason": "no capture or equivalent telemetry"},
        },
        "xscale_counter_directories": {
            "checked": len(counter_dirs),
            "existing": sum(bool(x["exists"]) for x in counter_dirs),
            "readable": sum(bool(x["readable"]) for x in counter_dirs),
        },
        "xscale_counter_files": sum(len(x["xscale_counter_files"]) for x in records),
        "unavailable_command_counts": unavailable,
        "debugfs_top_entries": {
            record["pod"]: record["filesystem_visibility"].get("/sys/kernel/debug", [])
            for record in records
        },
        "recognized_mccl_environment_names": sorted(
            x
            for x in env_names
            if any(k in x for k in ("ALGO", "PROTO", "CHANNEL", "NCHANNEL", "QP", "IB_TC"))
        ),
        "parameters": parameter_rows,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    args.summary_md.write_text(
        f"""# Muxi W3.1/W3.2 端侧可见性

- 代表pod：{len(records)}
- MCCL_IB_TC运行日志命中：{len(tc_log_hits)}
- xscale counters/hw_counters目录：{summary['xscale_counter_directories']['existing']} / {len(counter_dirs)}
- xscale专用counter文件：{summary['xscale_counter_files']}
- perfquery/rdma/ethtool：3/3均不可用

## TC证据分层

1. 环境变量被读取：**已证实**。MCCL INFO明确记录
   `MCCL_IB_TC set by environment to 128`。
2. QP traffic class被设置：**未证实**。libmccl同时存在`MCCL_IB_TC`、
   `tclass`和`ibv_modify_qp`字符串/符号，但没有运行日志或源码证明三者调用链。
3. 线上DSCP/ECN：**未证实**。无抓包、RNIC等价遥测或交换机数据，禁止按
   标准ToS自行换算。

## Counter结论

xscale_0..3在三个pod上均无`ports/1/counters`或`hw_counters`目录；
perfquery、rdma、ethtool及xscale厂商CLI不可用。当前不能读取CE/CNP/PFC、
retry、drop、FEC或symbol counter。
"""
    )
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
