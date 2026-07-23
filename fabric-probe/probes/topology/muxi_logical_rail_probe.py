#!/usr/bin/env python3
"""只读采集每 pod 的 eth0/net1..4 / xscale_0..3 逻辑拓扑事实。

不依赖 ip/ethtool/lldpctl；从 sysfs、/proc/net、IB GID 读取。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import struct
import time
from pathlib import Path
from typing import Any


NETDEVS = ["eth0", "net1", "net2", "net3", "net4"]
RAILS = ["xscale_0", "xscale_1", "xscale_2", "xscale_3"]


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _hex_ip(hexstr: str) -> str:
    # /proc/net/route 地址为主机字节序 uint32 的十六进制
    return socket.inet_ntoa(struct.pack("<L", int(hexstr, 16)))


def _gid_to_ipv4(gid: str) -> str | None:
    # IPv4-mapped: 0000:0000:0000:0000:0000:ffff:aabb:ccdd
    parts = gid.strip().lower().split(":")
    if len(parts) != 8:
        return None
    if parts[5] != "ffff":
        return None
    try:
        hi = int(parts[6], 16)
        lo = int(parts[7], 16)
    except ValueError:
        return None
    return f"{(hi >> 8) & 0xFF}.{hi & 0xFF}.{(lo >> 8) & 0xFF}.{lo & 0xFF}"


def get_ifaddrs_ipv4() -> dict[str, list[dict[str, str]]]:
    """尽量用 libc getifaddrs；失败则返回空，由 GID/route 补。"""
    result: dict[str, list[dict[str, str]]] = {}
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

        class sockaddr(ctypes.Structure):
            _fields_ = [("sa_family", ctypes.c_ushort), ("sa_data", ctypes.c_char * 14)]

        class sockaddr_in(ctypes.Structure):
            _fields_ = [
                ("sin_family", ctypes.c_ushort),
                ("sin_port", ctypes.c_uint16),
                ("sin_addr", ctypes.c_uint32),
                ("sin_zero", ctypes.c_char * 8),
            ]

        class ifaddrs(ctypes.Structure):
            pass

        ifaddrs._fields_ = [
            ("ifa_next", ctypes.POINTER(ifaddrs)),
            ("ifa_name", ctypes.c_char_p),
            ("ifa_flags", ctypes.c_uint),
            ("ifa_addr", ctypes.POINTER(sockaddr)),
            ("ifa_netmask", ctypes.POINTER(sockaddr)),
            ("ifa_ifu", ctypes.c_void_p),
            ("ifa_data", ctypes.c_void_p),
        ]

        getifaddrs = libc.getifaddrs
        getifaddrs.argtypes = [ctypes.POINTER(ctypes.POINTER(ifaddrs))]
        freeifaddrs = libc.freeifaddrs
        freeifaddrs.argtypes = [ctypes.POINTER(ifaddrs)]

        ptr = ctypes.POINTER(ifaddrs)()
        if getifaddrs(ctypes.byref(ptr)) != 0:
            return result
        cur = ptr
        while cur:
            name = cur.contents.ifa_name.decode() if cur.contents.ifa_name else ""
            addr_p = cur.contents.ifa_addr
            mask_p = cur.contents.ifa_netmask
            if addr_p and addr_p.contents.sa_family == 2:  # AF_INET
                sin = ctypes.cast(addr_p, ctypes.POINTER(sockaddr_in)).contents
                ip = socket.inet_ntoa(struct.pack("I", sin.sin_addr))
                prefix = None
                if mask_p and mask_p.contents.sa_family == 2:
                    msin = ctypes.cast(mask_p, ctypes.POINTER(sockaddr_in)).contents
                    mask = socket.inet_ntoa(struct.pack("I", msin.sin_addr))
                    bits = bin(struct.unpack("!I", socket.inet_aton(mask))[0]).count("1")
                    prefix = f"{mask}/{bits}"
                result.setdefault(name, []).append({"ipv4": ip, "netmask_prefix": prefix or "unknown"})
            cur = cur.contents.ifa_next
        freeifaddrs(ptr)
    except Exception as e:  # noqa: BLE001 — 只读探测需容忍
        result["_error"] = [{"ipv4": "", "netmask_prefix": str(e)}]
    return result


def parse_route() -> list[dict[str, Any]]:
    path = Path("/proc/net/route")
    rows = []
    text = _read_text(path)
    if not text:
        return rows
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        iface, dest, gateway, flags, _, _, metric, mask = parts[:8]
        rows.append(
            {
                "iface": iface,
                "destination": _hex_ip(dest),
                "gateway": _hex_ip(gateway),
                "flags": flags,
                "metric": int(metric),
                "mask": _hex_ip(mask),
                "prefix_len": bin(int(mask, 16)).count("1")
                if re.fullmatch(r"[0-9A-Fa-f]{8}", mask)
                else None,
            }
        )
    return rows


def parse_arp() -> list[dict[str, str]]:
    text = _read_text(Path("/proc/net/arp"))
    if not text:
        return []
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        rows.append(
            {
                "ip": parts[0],
                "hw_type": parts[1],
                "flags": parts[2],
                "mac": parts[3],
                "mask": parts[4],
                "device": parts[5],
            }
        )
    return rows


def collect_netdev(name: str, ifaddrs: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    root = Path("/sys/class/net") / name
    if not root.exists():
        return {"name": name, "present": False}
    device = root / "device"
    pci_bdf = None
    pci_path = None
    if device.exists():
        pci_path = str(device.resolve())
        # climb to PCI device
        p = device.resolve()
        for _ in range(6):
            if re.match(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$", p.name):
                pci_bdf = p.name
                break
            if p.parent == p:
                break
            p = p.parent
    permanent_mac = _read_text(root / "address")
    # some drivers expose addr_assign_type / address vs permanent
    return {
        "name": name,
        "present": True,
        "ifindex": _read_text(root / "ifindex"),
        "mtu": _read_text(root / "mtu"),
        "operstate": _read_text(root / "operstate"),
        "mac": permanent_mac,
        "addr_assign_type": _read_text(root / "addr_assign_type"),
        "pci_bdf": pci_bdf,
        "pci_path": pci_path,
        "ipv4_addrs": ifaddrs.get(name, []),
        "speed": _read_text(root / "speed"),
    }


def collect_rail(dev: str) -> dict[str, Any]:
    root = Path("/sys/class/infiniband") / dev
    if not root.exists():
        return {"device": dev, "present": False}
    device_path = str((root / "device").resolve()) if (root / "device").exists() else None
    pci_bdf = Path(device_path).name if device_path else None
    netdevs = []
    net_dir = root / "device" / "net"
    if net_dir.exists():
        netdevs = sorted(p.name for p in net_dir.iterdir())
    gid5 = _read_text(root / "ports/1/gids/5")
    gid4 = _read_text(root / "ports/1/gids/4")
    ndev5 = _read_text(root / "ports/1/gid_attrs/ndevs/5")
    ndev4 = _read_text(root / "ports/1/gid_attrs/ndevs/4")
    return {
        "device": dev,
        "present": True,
        "pci_bdf": pci_bdf,
        "pci_path": device_path,
        "state": _read_text(root / "ports/1/state"),
        "phys_state": _read_text(root / "ports/1/phys_state"),
        "rate": _read_text(root / "ports/1/rate"),
        "active_mtu": _read_text(root / "ports/1/active_mtu"),
        "gid_index4": gid4,
        "gid_index5": gid5,
        "gid4_ipv4": _gid_to_ipv4(gid4 or ""),
        "gid5_ipv4": _gid_to_ipv4(gid5 or ""),
        "gid_index4_netdev": ndev4,
        "gid_index5_netdev": ndev5,
        "device_netdevs": netdevs,
    }


def collect_oui(mac: str | None) -> str | None:
    if not mac or mac.count(":") != 5:
        return None
    return mac.lower()[0:8]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", default=os.environ.get("HOSTNAME", "unknown"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ifaddrs = get_ifaddrs_ipv4()
    routes = parse_route()
    arps = parse_arp()
    netdevs = [collect_netdev(n, ifaddrs) for n in NETDEVS]
    rails = [collect_rail(r) for r in RAILS]

    # 每 rail 关联 gateway MAC：同子网 ARP 中最高频 MAC
    rail_facts = []
    for rail in rails:
        ndev = rail.get("gid_index5_netdev") or (rail.get("device_netdevs") or [None])[0]
        ipv4 = rail.get("gid5_ipv4") or rail.get("gid4_ipv4")
        iface_routes = [r for r in routes if r["iface"] == ndev]
        iface_arps = [a for a in arps if a["device"] == ndev]
        mac_counts: dict[str, int] = {}
        for a in iface_arps:
            mac_counts[a["mac"]] = mac_counts.get(a["mac"], 0) + 1
        top_mac = None
        if mac_counts:
            top_mac = sorted(mac_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        local_route = next((r for r in iface_routes if r["gateway"] == "0.0.0.0"), None)
        default_via = next((r for r in iface_routes if r["destination"] == "0.0.0.0"), None)
        netdev_obj = next((n for n in netdevs if n["name"] == ndev), {})
        subnet = None
        if local_route:
            subnet = f"{local_route['destination']}/{local_route['prefix_len']}"
        rail_facts.append(
            {
                **rail,
                "logical_netdev": ndev,
                "ipv4": ipv4 or (netdev_obj.get("ipv4_addrs") or [{}])[0].get("ipv4"),
                "subnet": subnet,
                "route_mask": local_route["mask"] if local_route else None,
                "default_gateway": default_via["gateway"] if default_via else None,
                "arp_entries": len(iface_arps),
                "dominant_neigh_mac": top_mac,
                "dominant_neigh_mac_count": mac_counts.get(top_mac, 0) if top_mac else 0,
                "dominant_neigh_mac_oui": collect_oui(top_mac),
                "netdev_mac": netdev_obj.get("mac"),
                "netdev_mac_oui": collect_oui(netdev_obj.get("mac")),
                "ifindex": netdev_obj.get("ifindex"),
                "netdev_mtu": netdev_obj.get("mtu"),
                "evidence_level": "host_static",
            }
        )

    fib_head = _read_text(Path("/proc/net/fib_trie"))
    fib_sample = "\n".join((fib_head or "").splitlines()[:80])

    record = {
        "schema_version": "muxi.logical_rail.v1",
        "pod": args.pod,
        "host": socket.gethostname(),
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "control_plane_hint": {
            "eth0_role": "pod_control_plane",
            "xscale_role": "roce_data_plane",
            "note": "逻辑区分：eth0 承载默认路由/控制面；net1..4/xscale 为 RoCE 数据面。"
            "不宣称物理独立平面。",
        },
        "netdevs": netdevs,
        "rails": rail_facts,
        "routes": routes,
        "arp": arps,
        "fib_trie_sample": fib_sample,
        "ifaddrs_error": ifaddrs.get("_error"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    # 紧凑一行摘要便于并发日志
    summary = {
        "pod": args.pod,
        "rails": [
            {
                "device": r["device"],
                "netdev": r.get("logical_netdev"),
                "ipv4": r.get("ipv4"),
                "subnet": r.get("subnet"),
                "pci_bdf": r.get("pci_bdf"),
                "neigh_mac": r.get("dominant_neigh_mac"),
            }
            for r in rail_facts
        ],
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
