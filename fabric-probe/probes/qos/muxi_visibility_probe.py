#!/usr/bin/env python3
"""只读扫描MCCL TC语义、xscale计数入口和可控参数证据。"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

POD = os.environ.get("PROBE_POD", "")
TOKENS = (
    "MCCL_IB_TC",
    "NCCL_IB_TC",
    "TRAFFIC_CLASS",
    "TRAFFIC CLASS",
    "DSCP",
    "ECN",
    "PFC",
    "CNP",
    "MCCL_ALGO",
    "MCCL_PROTO",
    "MCCL_NCHANNEL",
    "MCCL_MIN_NCHANNEL",
    "MCCL_MAX_NCHANNEL",
    "MCCL_IB_QPS",
    "MCCL_IB_QPS_PER_CONNECTION",
    "MCCL_MIN_NCHANNELS",
    "MCCL_MAX_NCHANNELS",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_MIN_NCHANNEL",
    "NCCL_MAX_NCHANNEL",
    "NCCL_IB_QPS",
    "IBV_MODIFY_QP",
    "IBV_QP_AV",
    "TCLASS",
    "GRH",
)
COUNTER_RE = re.compile(
    r"byte|packet|retry|cnp|ecn|pfc|pause|drop|discard|error|fec|symbol|counter",
    re.I,
)
ENV_RE = re.compile(r"\b(?:MCCL|NCCL)_[A-Z0-9_]{3,}\b")


def readable_file(path: Path, limit: int = 4096) -> dict:
    result = {
        "path": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK),
        "mode": None,
        "value": None,
        "error": None,
    }
    try:
        result["mode"] = oct(stat.S_IMODE(path.stat().st_mode))
        if path.is_file() and result["readable"] and path.stat().st_size <= limit:
            result["value"] = path.read_text(errors="replace").strip()
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
    return result


def scan_xscale() -> tuple[list[dict], list[dict]]:
    entries: list[dict] = []
    counters: list[dict] = []
    for dev in [f"xscale_{i}" for i in range(4)]:
        root = Path(f"/sys/class/infiniband/{dev}/ports/1")
        for rel in ("state", "phys_state", "rate", "active_mtu", "gids/5"):
            item = readable_file(root / rel)
            item["device"] = dev
            item["kind"] = "port_attribute"
            entries.append(item)
        for dirname in ("counters", "hw_counters"):
            directory = root / dirname
            entries.append(
                {
                    "device": dev,
                    "kind": "counter_directory",
                    "path": str(directory),
                    "exists": directory.is_dir(),
                    "readable": os.access(directory, os.R_OK),
                }
            )
            if directory.is_dir():
                for path in sorted(directory.iterdir()):
                    item = readable_file(path)
                    item["device"] = dev
                    item["kind"] = dirname
                    counters.append(item)
    return entries, counters


def scan_candidate_files() -> list[dict]:
    roots = [
        Path("/sys/kernel/debug"),
        Path("/proc/driver"),
        Path("/proc/net"),
        Path("/sys/class/net"),
    ]
    rows: list[dict] = []
    for root in roots:
        if not root.exists():
            rows.append({"root": str(root), "exists": False, "readable": False})
            continue
        rows.append(
            {"root": str(root), "exists": True, "readable": os.access(root, os.R_OK)}
        )
        try:
            for path in root.rglob("*"):
                if COUNTER_RE.search(path.name) and len(rows) < 200:
                    rows.append(readable_file(path))
        except Exception as exc:  # noqa: BLE001
            rows.append({"root": str(root), "walk_error": repr(exc)})
    return rows


def scan_opt_maca_text() -> list[dict]:
    root = Path("/opt/maca")
    hits: list[dict] = []
    if not root.exists():
        return hits
    allowed = {
        ".md",
        ".txt",
        ".rst",
        ".h",
        ".hpp",
        ".c",
        ".cc",
        ".cpp",
        ".py",
        ".sh",
        ".yaml",
        ".yml",
        ".json",
        ".conf",
    }
    for path in root.rglob("*"):
        try:
            if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
                continue
            if path.suffix.lower() not in allowed and path.suffix:
                continue
            data = path.read_bytes()
            if b"\x00" in data[:4096]:
                continue
            text = data.decode(errors="ignore")
        except Exception:
            continue
        matched = [
            line.strip()
            for line in text.splitlines()
            if any(token.lower() in line.lower() for token in TOKENS)
        ]
        if matched:
            hits.append({"path": str(path), "lines": [x[:500] for x in matched[:30]]})
            if len(hits) >= 50:
                break
    return hits


def scan_library_strings() -> list[dict]:
    strings = shutil.which("strings")
    if not strings:
        return []
    files = sorted(
        {
            p
            for pattern in ("*mccl*.so*", "*nccl*.so*")
            for p in Path("/opt/maca").rglob(pattern)
            if p.is_file()
        }
    )
    rows: list[dict] = []
    for path in files[:100]:
        try:
            proc = subprocess.run(
                [strings, str(path)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            envs = sorted(set(ENV_RE.findall(proc.stdout)))
            semantic = [
                line
                for line in proc.stdout.splitlines()
                if any(token.lower() in line.lower() for token in TOKENS)
            ]
            if envs or semantic:
                rows.append(
                    {
                        "path": str(path),
                        "environment_names": envs[:200],
                        "semantic_strings": [x[:500] for x in semantic[:50]],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            rows.append({"path": str(path), "error": repr(exc)})
        if len(rows) >= 30:
            break
    return rows


def scan_mccl_logs() -> list[dict]:
    root = Path("/root/mxlog/mccl")
    if not root.is_dir():
        return []
    rows: list[dict] = []
    files = sorted(root.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
    for path in files:
        try:
            lines = [
                line
                for line in path.read_text(errors="replace").splitlines()
                if any(token.lower() in line.lower() for token in TOKENS)
                or re.search(r"\b(Channel|Trees|Ring)\b", line)
            ]
            if lines:
                rows.append({"path": str(path), "lines": [x[:500] for x in lines[:80]]})
        except Exception as exc:  # noqa: BLE001
            rows.append({"path": str(path), "error": repr(exc)})
    return rows


def command_visibility() -> dict[str, str | None]:
    commands = (
        "perfquery",
        "rdma",
        "ethtool",
        "ibv_devinfo",
        "ibstat",
        "iblinkinfo",
        "mx-smi",
        "xscale",
        "xscale_tool",
        "devlink",
    )
    return {name: shutil.which(name) for name in commands}


def command_help_evidence() -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    patterns = re.compile(r"counter|packet|byte|retry|cnp|ecn|pfc|pause|roce|rdma|network", re.I)
    for name in ("mx-smi", "ibv_devinfo"):
        path = shutil.which(name)
        if not path:
            continue
        try:
            proc = subprocess.run(
                [path, "--help"], capture_output=True, text=True, timeout=15, check=False
            )
            rows[name] = [
                line[:500]
                for line in (proc.stdout + "\n" + proc.stderr).splitlines()
                if patterns.search(line)
            ][:100]
        except Exception as exc:  # noqa: BLE001
            rows[name] = [repr(exc)]
    return rows


def filesystem_visibility() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for root in (Path("/sys/kernel/debug"), Path("/proc/driver")):
        try:
            result[str(root)] = [str(x) for x in sorted(root.iterdir())[:200]]
        except Exception as exc:  # noqa: BLE001
            result[str(root)] = [repr(exc)]
    tools = []
    for root in (Path("/opt/maca"), Path("/opt/mxdriver/bin")):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and re.search(
                r"counter|roce|rdma|network|xscale|diag|debug", path.name, re.I
            ):
                tools.append(str(path))
            if len(tools) >= 500:
                break
    result["vendor_candidate_files"] = tools
    return result


def main() -> None:
    sysfs, counters = scan_xscale()
    result = {
        "schema_version": "muxi.qos_visibility.v1",
        "pod": POD,
        "host": os.uname().nodename,
        "commands": command_visibility(),
        "command_help_evidence": command_help_evidence(),
        "filesystem_visibility": filesystem_visibility(),
        "xscale_sysfs": sysfs,
        "xscale_counter_files": counters,
        "candidate_files": scan_candidate_files(),
        "opt_maca_text_hits": scan_opt_maca_text(),
        "library_string_hits": scan_library_strings(),
        "mccl_log_hits": scan_mccl_logs(),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
