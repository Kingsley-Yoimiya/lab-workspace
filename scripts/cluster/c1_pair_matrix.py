#!/usr/bin/env python3
"""C1: 16x16 TCP/ping connectivity matrix on MUXI job pods."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

OUT = Path(os.environ.get("C1_OUT", "/tmp/c1"))
JOB = os.environ.get("C1_JOB", "yushan-muxi-card-screen-128-cp-copy")
OUT.mkdir(parents=True, exist_ok=True)

names = [f"{JOB}-master-0"] + [f"{JOB}-worker-{i}" for i in range(15)]


def resolve(host: str):
    for h in (
        host,
        f"{host}.{JOB}",
        f"{host}.default.svc",
        f"{host}.default.svc.cluster.local",
    ):
        try:
            return socket.gethostbyname(h)
        except Exception:
            continue
    try:
        r = subprocess.run(
            ["getent", "hosts", host], capture_output=True, text=True, timeout=3
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.split()[0]
    except Exception:
        pass
    return None


ips = [resolve(n) for n in names]
(OUT / "hosts_resolved.json").write_text(
    json.dumps({"names": names, "ips": ips}, indent=2)
)

rows = []
matrix = []
for i, src in enumerate(names):
    row = []
    for j, _dst in enumerate(names):
        if i == j:
            cell = {"status": "self", "ms": 0.0, "ip": ips[i]}
        elif ips[j] is None:
            cell = {"status": "unresolved", "ms": None, "ip": None}
        else:
            t0 = time.time()
            status = "fail"
            err = ""
            port_ok = None
            for port in (22, 29500, 8080, 443):
                try:
                    s = socket.create_connection((ips[j], port), timeout=0.8)
                    s.close()
                    status = "ok"
                    port_ok = port
                    break
                except Exception as e:
                    err = str(e)[:120]
            if status != "ok":
                try:
                    r = subprocess.run(
                        ["ping", "-c", "1", "-W", "1", ips[j]],
                        capture_output=True,
                        text=True,
                        timeout=3,
                    )
                    if r.returncode == 0:
                        status = "ping_ok"
                except Exception as e:
                    err = err or str(e)[:120]
            ms = round((time.time() - t0) * 1000, 2)
            cell = {
                "status": status,
                "ms": ms,
                "ip": ips[j],
                "port": port_ok,
                "err": err,
            }
        row.append(cell)
        rows.append({"i": i, "j": j, "src": src, "dst": names[j], **cell})
    matrix.append(row)

fail_modes: dict[str, int] = {}
for r in rows:
    st = r["status"]
    if st in ("fail", "unresolved"):
        key = st
        err = r.get("err") or ""
        if "Connection refused" in err:
            key = "tcp_refused_all_ports"
        elif "timed out" in err.lower() or "Timeout" in err:
            key = "tcp_timeout"
        fail_modes[key] = fail_modes.get(key, 0) + 1

summary = {
    "n": 16,
    "ok": sum(1 for r in rows if r["status"] in ("ok", "ping_ok", "self")),
    "fail": sum(1 for r in rows if r["status"] in ("fail", "unresolved")),
    "l3_reachable_tcp_refused": sum(
        1 for r in rows if "Connection refused" in (r.get("err") or "")
    ),
    "fail_modes": fail_modes,
    "ips": ips,
    "names": names,
    "interpretation": (
        "对角 self=16；非对角若 Connection refused=L3 路由可达但容器无监听端口；"
        "若 timeout=黑洞。机间 RoCE 训练面仍按 roce_note 记 FAIL，不把 eth0 TCP 冒充 MFU。"
    ),
    "note": "TCP/ping 管理面；机间 RoCE 见 roce_note（预期 FAIL）",
}
(OUT / "pair_matrix.json").write_text(
    json.dumps({"summary": summary, "matrix": matrix, "flat": rows}, indent=2)
)
(OUT / "pair_summary.json").write_text(json.dumps(summary, indent=2))
bad = [r for r in rows if r["status"] in ("fail", "unresolved")]
(OUT / "bad_links.json").write_text(json.dumps(bad, indent=2))
(OUT / "roce_note.json").write_text(
    json.dumps(
        {
            "roce_inter_node": "expected_fail_proxy_connect",
            "roce_intra_node": "ok_prior_gate",
            "ref": "docs/muxi/muxi_ib_gate_20260712_gid4.md",
            "x1_x2": "deferred_until_roce_fixed",
        },
        indent=2,
    )
)
print("MATRIX_DONE", json.dumps(summary))
