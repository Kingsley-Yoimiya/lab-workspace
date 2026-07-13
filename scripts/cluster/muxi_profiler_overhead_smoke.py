#!/usr/bin/env python3
"""Smoke: master-0 only, arms B/P/T, sync_L8, short iters."""
from __future__ import annotations

import os
import sys

# ensure /tmp has modules
sys.path.insert(0, "/tmp")
os.environ.setdefault("ITERS", "20")
os.environ.setdefault("WARMUP", "5")

from muxi_profiler_overhead_orch import (  # noqa: E402
    AFS,
    PODS,
    RECIPES,
    fire_arm,
    kill_bench,
    log,
    stop_mxsmi,
    upload_bench,
    wait_arm,
    ex,
)

def main() -> None:
    global AFS
    import muxi_profiler_overhead_orch as m
    # stamp afs
    import time
    m.AFS = f"/afs-a3-weight-share/yinjinrun.p/results/muxi-prof-overhead-{time.strftime('%Y%m%d_%H%M%S')}"
    log(f"SMOKE AFS={m.AFS}")
    upload_bench()
    pod = PODS[0]
    ex(pod, "python3 -c 'import probing' 2>/dev/null || pip install -q probing; command -v probing")
    recipe = {"name": "sync_L8", "mode": "real_sync", "layers": 8}
    for arm in ["B", "P", "T"]:
        out = f"{m.AFS}/{pod}/smoke/{arm}"
        log(f"SMOKE arm={arm}")
        fire_arm(pod, recipe, arm, out)
        ok = wait_arm(pod, out, need=8, timeout_s=900)
        summ = ex(pod, f"cat {out}/summary.json 2>/dev/null")
        log(f"SMOKE {arm} ok={ok}\n{summ}")
        kill_bench(pod)
    ex(pod, f"echo SMOKE_OK > {m.AFS}/SMOKE_DONE")
    log("SMOKE_DONE")

if __name__ == "__main__":
    main()
