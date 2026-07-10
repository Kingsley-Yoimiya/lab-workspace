#!/usr/bin/env python3
"""npu_telemetry_bench: 空载/满载遥测（对标 nvidia-smi dmon / DCGM）。

在单节点对指定 NPU 采样 AICore/HBM/温/功耗；满载阶段跑短 GEMM。
输出 JSONL 到 --out。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _npu_smi_sample() -> list[dict]:
    """Best-effort parse of `npu-smi info` into per-chip rows."""
    try:
        out = subprocess.check_output(["npu-smi", "info"], text=True, timeout=30)
    except Exception as e:
        return [{"error": str(e)}]
    rows: list[dict] = []
    # Keep raw block for offline parse; also try simple line scrape
    rows.append({"raw_head": "\n".join(out.splitlines()[:80]), "ts": time.time()})
    return rows


def _gemm_load(device: int, seconds: float, n: int = 4096) -> None:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    a = torch.randn(n, n, device=f"npu:{device}", dtype=torch.bfloat16)
    b = torch.randn(n, n, device=f"npu:{device}", dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        c = a @ b
        torch.npu.synchronize()
        _ = c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="all", help="comma list or all")
    ap.add_argument("--idle-s", type=float, default=60.0)
    ap.add_argument("--load-s", type=float, default=60.0)
    ap.add_argument("--sample-interval", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch_npu  # noqa: F401

    n = torch.npu.device_count()
    if args.devices == "all":
        devices = list(range(n))
    else:
        devices = [int(x) for x in args.devices.split(",")]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def sample_phase(phase: str, duration: float) -> None:
        t_end = time.time() + duration
        while time.time() < t_end:
            rec = {
                "record": "telemetry",
                "phase": phase,
                "ts": time.time(),
                "npu_smi": _npu_smi_sample(),
            }
            # per-device mem
            mem = []
            for d in devices:
                try:
                    free, total = torch.npu.mem_get_info(d)
                    mem.append(
                        {
                            "device": d,
                            "used_mib": (total - free) / 1024 / 1024,
                            "total_mib": total / 1024 / 1024,
                        }
                    )
                except Exception as e:
                    mem.append({"device": d, "error": str(e)})
            rec["mem"] = mem
            with out.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            time.sleep(args.sample_interval)

    # idle
    sample_phase("idle", args.idle_s)

    # load on all devices in parallel via threads
    import threading

    threads = [
        threading.Thread(target=_gemm_load, args=(d, args.load_s), daemon=True)
        for d in devices
    ]
    for t in threads:
        t.start()
    sample_phase("load", args.load_s)
    for t in threads:
        t.join(timeout=args.load_s + 30)

    with out.open("a") as f:
        f.write(json.dumps({"record": "done", "devices": devices}) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
