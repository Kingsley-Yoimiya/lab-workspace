#!/usr/bin/env python3
"""只读快照net1..4通用netdev统计；不宣称等价于RoCE硬件counter。"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

FIELDS = (
    "rx_bytes",
    "tx_bytes",
    "rx_packets",
    "tx_packets",
    "rx_errors",
    "tx_errors",
    "rx_dropped",
    "tx_dropped",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    args = parser.parse_args()
    interfaces = {}
    for name in ("net1", "net2", "net3", "net4"):
        root = Path("/sys/class/net") / name / "statistics"
        interfaces[name] = {field: int((root / field).read_text()) for field in FIELDS}
    print(
        json.dumps(
            {
                "schema_version": "muxi.netdev_counter_snapshot.v1",
                "pod": args.pod,
                "host": socket.gethostname(),
                "timestamp_s": time.time(),
                "interfaces": interfaces,
            }
        )
    )


if __name__ == "__main__":
    main()
