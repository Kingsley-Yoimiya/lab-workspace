#!/usr/bin/env python3
"""Jump-host: MUXI 16×8 profiler overhead campaign.

Each node runs arms sequentially: B → P → Pplus → T → S(mxsmi) → NP(nccl plugin try).
Node groups use different load recipes (independent vs real_sync × layers).
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from pathlib import Path

KC = {"KUBECONFIG": "/root/.kube/config.muxi-mohe"}
JOB = "yushan-muxi-card-screen-128-cp-copy"
AFS = os.environ.get(
    "AFS_OUT",
    f"/afs-a3-weight-share/yinjinrun.p/results/muxi-prof-overhead-{time.strftime('%Y%m%d_%H%M%S')}",
)
BENCH_AFS = "/afs-a3-weight-share/yinjinrun.p/lab-workspace/scripts/cluster"
ITERS = int(os.environ.get("ITERS", "40"))
WARMUP = int(os.environ.get("WARMUP", "10"))

PODS = [f"{JOB}-master-0"] + [f"{JOB}-worker-{i}" for i in range(15)]

# 4 recipes × 4 replicas
RECIPES = [
    {"name": "indep_L8", "mode": "independent", "layers": 8},
    {"name": "sync_L8", "mode": "real_sync", "layers": 8},
    {"name": "sync_L4", "mode": "real_sync", "layers": 4},
    {"name": "indep_L12", "mode": "independent", "layers": 12},
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ex(pod: str, cmd: str, timeout: int = 300) -> str:
    p = subprocess.run(
        ["vcctl", "pod", "exec", pod, "--", "bash", "-lc", cmd],
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


def upload_bench() -> None:
    local = Path(__file__).resolve().parent / "profiler_overhead_bench.py"
    if not local.exists():
        # when running from /tmp on jump
        local = Path("/tmp/profiler_overhead_bench.py")
    b64 = base64.b64encode(local.read_bytes()).decode()
    r = ex(
        PODS[0],
        f"mkdir -p {BENCH_AFS} {AFS}; echo {b64} | base64 -d > {BENCH_AFS}/profiler_overhead_bench.py; "
        f"wc -c {BENCH_AFS}/profiler_overhead_bench.py; echo AFS={AFS} > {AFS}/ROOT.txt",
    )
    log(r)


def kill_bench(pod: str) -> None:
    ex(
        pod,
        "me=$$; ps -eo pid=,args= | awk -v me=\"$me\" "
        "'$1==me{next} /[p]rofiler_overhead_bench|[t]orchrun/{print $1}' | "
        "while read p; do kill -9 $p 2>/dev/null; done; echo K",
    )


def fire_arm(pod: str, recipe: dict, arm: str, out: str) -> None:
    """arm: B|P|Pplus|T|S|NP"""
    kill_bench(pod)
    time.sleep(1)
    ex(pod, f"mkdir -p {out}; rm -f {out}/done_rank*.txt {out}/step_times_rank*.jsonl {out}/summary.json")

    probing = "0"
    profiler = "none"
    extra_env = ""
    if arm == "P":
        probing = "1"
    elif arm == "Pplus":
        probing = "1"
        extra_env = "PROBING_TORCH_PROFILING=on"
    elif arm == "T":
        profiler = "torch"
    elif arm == "NP":
        probing = "1"
        extra_env = (
            "NCCL_PROFILER_PLUGIN=$(python3 -m probing.nccl --plugin-path 2>/dev/null || true)"
        )

    mode = recipe["mode"]
    layers = recipe["layers"]
    # S: same as B but start mx-smi sampler sidecar
    if arm == "S":
        ex(
            pod,
            f"nohup bash -c 'while true; do date -Is >> {out}/mxsmi.jsonl; "
            f"mx-smi -j 2>/dev/null | head -c 200000 >> {out}/mxsmi.jsonl; "
            f"echo >> {out}/mxsmi.jsonl; sleep 1; done' >{out}/mxsmi_loop.log 2>&1 & echo $! > {out}/mxsmi.pid; echo S_SIDE",
        )

    # For independent: 8 separate processes; for real_sync: torchrun
    if mode == "independent":
        fire = f"""
set +e
OUT={out}
export PATH=/opt/conda/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_HOME=/opt/maca/tools/cu-bridge
export PROBING={probing}
{('export ' + extra_env) if extra_env and '=' in extra_env and not extra_env.startswith('NCCL') else ''}
{('export ' + extra_env) if extra_env.startswith('PROBING_TORCH') else ''}
for g in $(seq 0 7); do
  nohup env CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 NODE_RANK=0 GPUS_PER_NODE=8 GLOBAL_RANK=$g \\
    PROBING=$PROBING \\
    python3 {BENCH_AFS}/profiler_overhead_bench.py \\
      --mode independent --iters {ITERS} --warmup {WARMUP} \\
      --hidden 4096 --seq 2048 --layers {layers} --batch 2 \\
      --profiler {profiler} --out-dir $OUT --tag {arm}_{recipe["name"]} \\
      >$OUT/gpu$g.log 2>&1 &
done
echo FIRED_INDEP PROBING=$PROBING
"""
    else:
        # torch.profiler only on rank0 via --profiler; probing via export for all ranks
        nccl_line = ""
        if arm == "NP":
            nccl_line = "export NCCL_PROFILER_PLUGIN=$(python3 -m probing.nccl --plugin-path 2>/dev/null || true)"
        torch_prof_line = ""
        if arm == "Pplus":
            torch_prof_line = "export PROBING_TORCH_PROFILING=on"
        fire = f"""
set +e
OUT={out}
cd /tmp
export PATH=/opt/conda/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_HOME=/opt/maca/tools/cu-bridge
export PROBING={probing}
{torch_prof_line}
{nccl_line}
# force probing site hook if installed
export PYTHONPATH=$(python3 -c 'import probing,os; print(os.path.dirname(os.path.dirname(probing.__file__)))' 2>/dev/null):$PYTHONPATH
nohup torchrun --standalone --nproc_per_node=8 \\
    {BENCH_AFS}/profiler_overhead_bench.py \\
      --mode real_sync --iters {ITERS} --warmup {WARMUP} \\
      --hidden 4096 --seq 2048 --layers {layers} --batch 2 \\
      --profiler {profiler} --out-dir $OUT --tag {arm}_{recipe["name"]} \\
      >$OUT/torchrun.log 2>&1 &
echo FIRED_SYNC PROBING=$PROBING
"""
    # write launcher via base64 to avoid quoting
    b64 = base64.b64encode(fire.encode()).decode()
    print(ex(pod, f"echo {b64} | base64 -d > /tmp/fire_prof.sh && bash /tmp/fire_prof.sh"))


def wait_arm(pod: str, out: str, need: int = 8, timeout_s: int = 1200) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        n = "".join(c for c in ex(pod, f"ls {out}/done_rank*.txt 2>/dev/null | wc -l") if c.isdigit()) or "0"
        steps = "".join(
            c
            for c in ex(pod, f"wc -l <{out}/step_times_rank000.jsonl 2>/dev/null")
            if c.isdigit()
        ) or "0"
        log(f"{pod.split('-')[-1]} {out.split('/')[-2:][-1]}/{out.split('/')[-1]} done={n}/{need} steps={steps}")
        if int(n) >= need:
            return True
        time.sleep(15)
    return False


def stop_mxsmi(pod: str, out: str) -> None:
    ex(pod, f"test -f {out}/mxsmi.pid && kill $(cat {out}/mxsmi.pid) 2>/dev/null; echo STOP_S")


def probing_accuracy(pod: str, out: str) -> None:
    # best-effort: list tables / count events if probing attached
    cmd = f"""
set +e
pid=$(pgrep -f profiler_overhead_bench | head -1)
if [[ -z "$pid" ]]; then echo NO_PID > {out}/probing_acc.txt; exit 0; fi
probing $pid query "SELECT name, count(*) AS c FROM python.trace_event GROUP BY name ORDER BY c DESC LIMIT 20" > {out}/probing_trace_counts.txt 2>&1 || true
probing $pid query "SELECT count(*) AS n FROM gpu.utilization" > {out}/probing_gpu_util.txt 2>&1 || true
probing $pid query "SELECT count(*) AS n FROM python.comm_collective" > {out}/probing_collective.txt 2>&1 || true
echo DONE_ACC > {out}/probing_acc.txt
"""
    # usually process already exited — skip or sample mid-run; write placeholder
    ex(pod, f"echo 'posthoc_skip' > {out}/probing_acc.txt")


def run_node(pod: str, recipe: dict) -> None:
    base = f"{AFS}/{pod}/{recipe['name']}"
    arms = ["B", "P", "Pplus", "T", "S", "NP"]
    for arm in arms:
        out = f"{base}/{arm}"
        log(f"FIRE {pod} recipe={recipe['name']} arm={arm}")
        fire_arm(pod, recipe, arm, out)
        need = 8
        ok = wait_arm(pod, out, need=need)
        if arm == "S":
            stop_mxsmi(pod, out)
        # pull summary snippet
        summ = ex(pod, f"cat {out}/summary.json 2>/dev/null | head -c 2000")
        (Path("/tmp") / f"muxi_prof_{pod}_{recipe['name']}_{arm}.json").write_text(
            summ or "{}", encoding="utf-8"
        )
        log(f"DONE {pod} {arm} ok={ok} summary_head={summ[:120].replace(chr(10),' ')}")
        kill_bench(pod)
        time.sleep(2)
    ex(pod, f"echo OK > {base}/NODE_DONE")


def main() -> None:
    log(f"START AFS={AFS}")
    upload_bench()
    # ensure probing present on all nodes (pip may be per-node image cache)
    for pod in PODS:
        ex(pod, "python3 -c 'import probing' 2>/dev/null || pip install -q probing; command -v probing; echo OK")
    # parallel fanout: one process wait chain per pod via background on jump
    # For simplicity sequential batches of 4 nodes to avoid jump overload
    for batch_start in range(0, 16, 4):
        batch = list(range(batch_start, min(batch_start + 4, 16)))
        procs = []
        for i in batch:
            pod = PODS[i]
            recipe = RECIPES[i // 4]
            # run in subprocess on jump
            script = f"""
import os,sys
sys.path.insert(0,'/tmp')
os.environ['AFS_OUT']='{AFS}'
from muxi_profiler_overhead_orch import run_node, RECIPES, PODS, log
run_node(PODS[{i}], RECIPES[{i // 4}])
"""
            # inline call instead
            log(f"SCHEDULE pod={pod} recipe={recipe['name']}")
        # Actually run batch sequentially within batch parallel using subprocess
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futs = [
                pool.submit(run_node, PODS[i], RECIPES[i // 4]) for i in batch
            ]
            for f in concurrent.futures.as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    log(f"ERR {e}")
    ex(PODS[0], f"echo COMPLETE > {AFS}/CAMPAIGN_DONE; ls {AFS} | wc -l")
    log("CAMPAIGN_DONE")


if __name__ == "__main__":
    main()
