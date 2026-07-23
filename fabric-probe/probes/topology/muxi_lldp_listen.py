#!/usr/bin/env python3
"""只读 AF_PACKET LLDP 监听器：不发送帧、不改接口。

绑定指定 netdev，监听 EtherType 0x88cc，解析并输出 JSONL。
"""
from __future__ import annotations

import argparse
import json
import os
import select
import socket
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


ETH_P_LLDP = 0x88CC
ETH_P_ALL = 0x0003


def _mac(b: bytes) -> str:
    return ":".join(f"{x:02x}" for x in b)


def _tlv_iter(payload: bytes) -> list[tuple[int, bytes]]:
    out: list[tuple[int, bytes]] = []
    i = 0
    while i + 2 <= len(payload):
        head = struct.unpack("!H", payload[i : i + 2])[0]
        tlv_type = head >> 9
        tlv_len = head & 0x1FF
        i += 2
        if tlv_type == 0:
            break
        value = payload[i : i + tlv_len]
        if len(value) < tlv_len:
            break
        out.append((tlv_type, value))
        i += tlv_len
    return out


def parse_lldp_frame(frame: bytes) -> dict[str, Any]:
    """解析以太网帧中的 LLDP PDU。返回结构化字段 + raw hex。"""
    if len(frame) < 14:
        return {"error": "frame_too_short", "raw_hex": frame.hex()}
    dst = _mac(frame[0:6])
    src = _mac(frame[6:12])
    ethertype = struct.unpack("!H", frame[12:14])[0]
    offset = 14
    # 802.1Q VLAN tag
    vlan_id = None
    if ethertype == 0x8100 and len(frame) >= 18:
        tci = struct.unpack("!H", frame[14:16])[0]
        vlan_id = tci & 0x0FFF
        ethertype = struct.unpack("!H", frame[16:18])[0]
        offset = 18
    if ethertype != ETH_P_LLDP:
        return {
            "error": f"unexpected_ethertype=0x{ethertype:04x}",
            "dst_mac": dst,
            "src_mac": src,
            "raw_hex": frame.hex(),
        }
    payload = frame[offset:]
    result: dict[str, Any] = {
        "dst_mac": dst,
        "src_mac": src,
        "ethertype": f"0x{ethertype:04x}",
        "outer_vlan_id": vlan_id,
        "chassis_id": None,
        "chassis_id_subtype": None,
        "port_id": None,
        "port_id_subtype": None,
        "ttl": None,
        "port_description": None,
        "system_name": None,
        "system_description": None,
        "management_addresses": [],
        "vlan_tlvs": [],
        "tlvs": [],
        "raw_hex": frame.hex(),
    }
    for tlv_type, value in _tlv_iter(payload):
        entry: dict[str, Any] = {"type": tlv_type, "len": len(value)}
        if tlv_type == 1 and value:
            subtype = value[0]
            data = value[1:]
            result["chassis_id_subtype"] = subtype
            if subtype == 4 and len(data) == 6:
                result["chassis_id"] = _mac(data)
            else:
                result["chassis_id"] = data.decode("utf-8", "replace")
            entry["subtype"] = subtype
            entry["value"] = result["chassis_id"]
        elif tlv_type == 2 and value:
            subtype = value[0]
            data = value[1:]
            result["port_id_subtype"] = subtype
            if subtype == 3 and len(data) == 6:
                result["port_id"] = _mac(data)
            else:
                result["port_id"] = data.decode("utf-8", "replace")
            entry["subtype"] = subtype
            entry["value"] = result["port_id"]
        elif tlv_type == 3 and len(value) >= 2:
            result["ttl"] = struct.unpack("!H", value[:2])[0]
            entry["value"] = result["ttl"]
        elif tlv_type == 4:
            result["port_description"] = value.decode("utf-8", "replace")
            entry["value"] = result["port_description"]
        elif tlv_type == 5:
            result["system_name"] = value.decode("utf-8", "replace")
            entry["value"] = result["system_name"]
        elif tlv_type == 6:
            result["system_description"] = value.decode("utf-8", "replace")
            entry["value"] = result["system_description"]
        elif tlv_type == 8 and value:
            # IEEE 802.1AB management address
            addr_len = value[0]
            if 1 + addr_len <= len(value):
                addr_subtype = value[1]
                addr = value[2 : 1 + addr_len]
                mgmt: dict[str, Any] = {"subtype": addr_subtype}
                if addr_subtype == 1 and len(addr) == 4:
                    mgmt["address"] = ".".join(str(x) for x in addr)
                elif addr_subtype == 2 and len(addr) == 16:
                    mgmt["address"] = ":".join(f"{addr[i]:02x}{addr[i+1]:02x}" for i in range(0, 16, 2))
                else:
                    mgmt["address_hex"] = addr.hex()
                result["management_addresses"].append(mgmt)
                entry["value"] = mgmt
        elif tlv_type == 127 and len(value) >= 4:
            oui = value[0:3]
            subtype = value[3]
            data = value[4:]
            entry["oui"] = oui.hex()
            entry["subtype"] = subtype
            # 802.1 OUI 00-80-c2, subtype 1 = Port VLAN ID
            if oui == b"\x00\x80\xc2" and subtype == 1 and len(data) >= 2:
                vid = struct.unpack("!H", data[:2])[0]
                result["vlan_tlvs"].append({"kind": "port_vlan_id", "vlan_id": vid})
                entry["value"] = {"vlan_id": vid}
            elif oui == b"\x00\x80\xc2" and subtype == 3 and len(data) >= 2:
                # VLAN name: 2-byte VID + name
                vid = struct.unpack("!H", data[:2])[0]
                name = data[2:].decode("utf-8", "replace")
                result["vlan_tlvs"].append({"kind": "vlan_name", "vlan_id": vid, "name": name})
                entry["value"] = {"vlan_id": vid, "name": name}
            else:
                entry["value_hex"] = data.hex()
        else:
            entry["value_hex"] = value.hex()
        result["tlvs"].append(entry)
    return result


def listen_iface(iface: str, duration_s: float, out_path: str | None = None) -> dict[str, Any]:
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_LLDP))
    try:
        sock.bind((iface, ETH_P_LLDP))
    except OSError:
        # 某些环境需 ETH_P_ALL + 用户态过滤
        sock.close()
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))
        sock.bind((iface, ETH_P_ALL))
    sock.setblocking(False)
    deadline = time.time() + duration_s
    frames: list[dict[str, Any]] = []
    raw_count = 0
    lldp_count = 0
    started = time.time()
    fh = open(out_path, "w") if out_path else None
    try:
        while time.time() < deadline:
            remaining = max(0.05, deadline - time.time())
            r, _, _ = select.select([sock], [], [], min(1.0, remaining))
            if not r:
                continue
            try:
                data, addr = sock.recvfrom(65535)
            except BlockingIOError:
                continue
            raw_count += 1
            if len(data) >= 14:
                ethertype = struct.unpack("!H", data[12:14])[0]
                if ethertype == 0x8100 and len(data) >= 18:
                    ethertype = struct.unpack("!H", data[16:18])[0]
                if ethertype != ETH_P_LLDP:
                    continue
            parsed = parse_lldp_frame(data)
            if parsed.get("error"):
                continue
            lldp_count += 1
            rec = {
                "iface": iface,
                "received_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "monotonic": time.time(),
                "addr": str(addr),
                **parsed,
            }
            frames.append(rec)
            if fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
    finally:
        sock.close()
        if fh:
            fh.close()
    return {
        "iface": iface,
        "duration_s": round(time.time() - started, 3),
        "requested_duration_s": duration_s,
        "raw_frames_seen": raw_count,
        "lldp_frames": lldp_count,
        "unique_chassis": sorted(
            {f.get("chassis_id") for f in frames if f.get("chassis_id")}
        ),
        "unique_ports": sorted({f.get("port_id") for f in frames if f.get("port_id")}),
        "frames": frames,
    }


def check_cap_net_raw() -> dict[str, Any]:
    status: dict[str, Any] = {"cap_net_raw_bit": 13}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("CapEff:"):
                    hexval = line.split()[1]
                    status["CapEff"] = hexval
                    status["cap_net_raw_effective"] = bool(int(hexval, 16) & (1 << 13))
                    break
    except OSError as e:
        status["status_error"] = str(e)
    try:
        s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_LLDP))
        s.close()
        status["af_packet_open"] = True
    except OSError as e:
        status["af_packet_open"] = False
        status["af_packet_error"] = str(e)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only LLDP AF_PACKET listener")
    parser.add_argument("--ifaces", default="net1,net2,net3,net4,eth0")
    parser.add_argument("--duration", type=float, default=70.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pod", default=os.environ.get("HOSTNAME", "unknown"))
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="逐接口串行监听；默认多接口并行同一 TTL 窗口",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cap = check_cap_net_raw()
    with open(os.path.join(args.out_dir, "cap_check.json"), "w") as f:
        json.dump(cap, f, indent=2, ensure_ascii=False)
        f.write("\n")
    if not cap.get("af_packet_open"):
        summary = {
            "pod": args.pod,
            "status": "AF_PACKET_UNAVAILABLE",
            "cap": cap,
            "ifaces": [],
        }
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(json.dumps(summary, ensure_ascii=False))
        sys.exit(2)

    ifaces = [x.strip() for x in args.ifaces.split(",") if x.strip()]

    def _run_one(iface: str) -> dict[str, Any]:
        iface_dir = os.path.join(args.out_dir, iface)
        os.makedirs(iface_dir, exist_ok=True)
        out_jsonl = os.path.join(iface_dir, "frames.jsonl")
        try:
            res = listen_iface(iface, args.duration, out_jsonl)
            res["status"] = "OK"
        except OSError as e:
            res = {
                "iface": iface,
                "status": "BIND_FAIL",
                "error": str(e),
                "lldp_frames": 0,
                "raw_frames_seen": 0,
                "duration_s": 0,
                "requested_duration_s": args.duration,
                "frames": [],
            }
        slim = {k: v for k, v in res.items() if k != "frames"}
        slim["sample_frames"] = res.get("frames", [])[:5]
        with open(os.path.join(iface_dir, "result.json"), "w") as f:
            json.dump(slim, f, indent=2, ensure_ascii=False)
            f.write("\n")
        return slim

    results: list[dict[str, Any]] = []
    if args.sequential:
        for iface in ifaces:
            results.append(_run_one(iface))
    else:
        with ThreadPoolExecutor(max_workers=len(ifaces)) as pool:
            futs = {pool.submit(_run_one, iface): iface for iface in ifaces}
            for fut in as_completed(futs):
                results.append(fut.result())
        results.sort(key=lambda r: ifaces.index(r["iface"]) if r.get("iface") in ifaces else 99)

    summary = {
        "pod": args.pod,
        "status": "OK",
        "cap": cap,
        "ifaces": results,
        "any_lldp": any(r.get("lldp_frames", 0) > 0 for r in results),
        "total_lldp_frames": sum(r.get("lldp_frames", 0) for r in results),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "ifaces"}, ensure_ascii=False))
    print(
        json.dumps(
            [
                {
                    "iface": r["iface"],
                    "lldp_frames": r.get("lldp_frames"),
                    "status": r.get("status"),
                    "unique_chassis": r.get("unique_chassis"),
                    "unique_ports": r.get("unique_ports"),
                }
                for r in results
            ],
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
