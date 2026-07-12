#!/usr/bin/env python3
"""Jump-host Wave4 B1 delay-inject amplitude scan."""
from __future__ import annotations

import base64
import os
import subprocess
import time

KC = {"KUBECONFIG": "/root/.kube/config.muxi-mohe"}
AFS = "/afs-a3-weight-share/montyyin/results/muxi-day-20260713_002719"
BENCH = "/afs-a3-weight-share/montyyin/lab-workspace/scripts/cluster"
MASTER = "yushan-muxi-card-screen-128-cp-copy-master-0"
ITERS = 600


def ex(cmd: str, timeout: int = 180) -> str:
    p = subprocess.run(
        ["vcctl", "pod", "exec", MASTER, "--", "bash", "-lc", cmd],
        env={**os.environ, **KC},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = (p.stdout or "") + (p.stderr or "")
    lines = [
        l
        for l in out.splitlines()
        if not l.startswith("Defaulted") and not l.startswith("Found")
    ]
    return "\n".join(lines)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit()) or "0"


def run_one(tag: str, inject: int, dms: int, every: int, burst: int) -> None:
    out = f"{AFS}/B1_scan/{tag}"
    log(f"FIRE {tag} inject={inject} ms={dms} every={every} burst={burst}")
    ex(f"mkdir -p {out}; rm -f {out}/done_rank*.txt {out}/step_times_rank*.jsonl")
    ex(
        "me=$$; ps -eo pid=,args= | awk -v me=\"$me\" "
        "'$1==me{next} /[v]irtual_sync_bench/{print $1}' | "
        "while read p; do kill -9 $p 2>/dev/null; done; echo K"
    )
    time.sleep(2)
    fire_py = f"""
import os, subprocess
out = {out!r}
bench = {BENCH!r}
iters = {ITERS}
inject = {inject}
dms = {dms}
every = {every}
burst = {burst}
tag = {tag!r}
for g in range(8):
    env = os.environ.copy()
    env.update({{
      "PATH": "/opt/conda/bin:" + env.get("PATH", ""),
      "PYTHONUNBUFFERED": "1",
      "CUDA_HOME": "/opt/maca/tools/cu-bridge",
      "PYTHONPATH": bench + ":" + env.get("PYTHONPATH", ""),
      "CUDA_VISIBLE_DEVICES": str(g),
      "LOCAL_RANK": "0",
      "NODE_RANK": "0",
      "GPUS_PER_NODE": "8",
      "GLOBAL_RANK": str(g),
      "WORLD_SIZE": "8",
      "PP_SIZE": "4",
      "DELAY_INJECT": str(inject),
      "DELAY_STAGE": "1",
      "DELAY_MS": str(dms),
      "DELAY_EVERY": str(every),
      "DELAY_BURST": str(burst),
    }})
    logf = open(f"{{out}}/gpu{{g}}.log", "w")
    subprocess.Popen(
        [
            "python3",
            f"{{bench}}/virtual_sync_bench.py",
            "--mode",
            "independent",
            "--iters",
            str(iters),
            "--warmup",
            "10",
            "--hidden",
            "4096",
            "--seq",
            "2048",
            "--layers",
            "8",
            "--batch",
            "2",
            "--out-dir",
            out,
            "--tag",
            f"B1_{{tag}}",
        ],
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print("FIRED")
"""
    b64 = base64.b64encode(fire_py.encode()).decode()
    print(ex(f"echo {b64} | base64 -d > /tmp/fire_b1.py && python3 /tmp/fire_b1.py"))
    for _ in range(90):
        n = digits(ex(f"ls {out}/done_rank*.txt 2>/dev/null|wc -l"))
        steps = digits(ex(f"wc -l <{out}/step_times_rank000.jsonl 2>/dev/null"))
        log(f"{tag} done={n}/8 steps={steps}")
        if int(n) >= 8:
            break
        time.sleep(12)


def main() -> None:
    log("START_SCAN")
    run_one("baseline", 0, 0, 20, 1)
    run_one("ms80_e20_b3", 1, 80, 20, 3)
    run_one("ms200_e10_b5", 1, 200, 10, 5)
    run_one("ms400_e5_b3", 1, 400, 5, 3)
    ex(f"echo OK > {AFS}/B1_scan/DONE")
    log("B1_SCAN_DONE")


if __name__ == "__main__":
    main()
