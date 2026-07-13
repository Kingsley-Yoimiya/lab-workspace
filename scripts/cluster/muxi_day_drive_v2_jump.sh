#!/usr/bin/env bash
set +e
export KUBECONFIG=/root/.kube/config.muxi-mohe
AFS=/afs-a3-weight-share/yinjinrun.p/results/muxi-day-20260713_002719
JOB=yushan-muxi-card-screen-128-cp-copy
BENCH=/afs-a3-weight-share/yinjinrun.p/lab-workspace/scripts/cluster/virtual_sync_bench.py
LOG=/tmp/muxi_day_drive_v2.log
: >"$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

fire_virt_node() {
  local pod=$1 tag=$2
  local out=$AFS/$tag/exp0_virtual
  log "FIRE_VIRT $tag on $pod"
  vcctl pod exec "$pod" -- bash -lc "mkdir -p $out; rm -f $out/*"
  vcctl pod exec "$pod" -- bash -lc 'me=$$; ps -eo pid=,args= | awk -v me="$me" '\''$1==me{next}/[v]irtual_sync|[t]orchrun/{print $1}'\'' | while read p; do kill -9 $p; done; echo K'
  sleep 2
  # write a launcher script on the pod to avoid quoting hell
  vcctl pod exec "$pod" -- bash -lc "cat > /tmp/fire_virt_${tag}.sh <<'EOS'
#!/bin/bash
set +e
export PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1 CUDA_HOME=/opt/maca/tools/cu-bridge
OUT=$out
BENCH=$BENCH
for g in 0 1 2 3 4 5 6 7; do
  nohup env CUDA_VISIBLE_DEVICES=\$g LOCAL_RANK=0 NODE_RANK=0 GPUS_PER_NODE=8 GLOBAL_RANK=\$g \\
    python3 \$BENCH --mode independent --iters 3000 --warmup 30 \\
    --hidden 4096 --seq 2048 --layers 8 --batch 2 --out-dir \$OUT --tag ${tag}_v \\
    >\$OUT/gpu\$g.log 2>&1 &
done
sleep 2
ls -la \$OUT | head
ps -ef | grep virtual_sync | grep -v grep | wc -l
EOS
chmod +x /tmp/fire_virt_${tag}.sh && bash /tmp/fire_virt_${tag}.sh"
}

wait_virt() {
  local tag=$1 t=0 n=0
  local out=$AFS/$tag/exp0_virtual
  while [ $t -lt 80 ]; do
    n=$(vcctl pod exec ${JOB}-master-0 -- bash -lc "ls $out/done_rank*.txt 2>/dev/null|wc -l" | tr -dc 0-9)
    s=$(vcctl pod exec ${JOB}-master-0 -- bash -lc "wc -l <$out/step_times_rank000.jsonl 2>/dev/null||echo 0" | tr -dc 0-9)
    log "wait $tag done=${n:-0}/8 steps=${s:-0}"
    [ "${n:-0}" -ge 8 ] && return 0
    sleep 45
    t=$((t+1))
  done
  return 1
}

log START_V2
fire_virt_node ${JOB}-master-0 A1_master
fire_virt_node ${JOB}-worker-3 A1_worker3
fire_virt_node ${JOB}-worker-10 A1_worker10
wait_virt A1_master
wait_virt A1_worker3
wait_virt A1_worker10
vcctl pod exec ${JOB}-master-0 -- bash -lc "echo A1_DONE > $AFS/A1.done"
log A1_DONE

# A4 telem brief
log TELEM
vcctl pod exec ${JOB}-master-0 -- bash -lc "mkdir -p $AFS/A4/telemetry"
idx=0
for logic in master-0 worker-0 worker-1 worker-2 worker-3 worker-4 worker-5 worker-6 worker-7 worker-8 worker-9 worker-10 worker-11 worker-12 worker-13 worker-14; do
  vcctl pod exec ${JOB}-$logic -- bash -lc "nohup python3 -c 'import json,subprocess,time;o=\"$AFS/A4/telemetry/node${idx}.jsonl\";e=time.time()+900;n=$idx
while time.time()<e:
 t=time.time();r=subprocess.run([\"mx-smi\",\"-j\"],capture_output=True,text=True,timeout=15).stdout;i=r.find(\"{\");d=json.loads(r[i:]) if i>=0 else {};open(o,\"a\").write(json.dumps({\"ts\":t,\"node\":n,\"smi\":d})+\"\\n\");time.sleep(5)' >/dev/null 2>&1 & echo T" >/dev/null
  idx=$((idx+1))
done

# A2
log A2_FIRE
out1=$AFS/A2/exp1_independent
vcctl pod exec ${JOB}-master-0 -- bash -lc "mkdir -p $out1; rm -f $out1/*"
idx=0
for logic in master-0 worker-0 worker-1 worker-2 worker-3 worker-4 worker-5 worker-6 worker-7 worker-8 worker-9 worker-10 worker-11 worker-12 worker-13 worker-14; do
  pod=${JOB}-$logic
  log "A2 node $idx"
  vcctl pod exec $pod -- bash -lc 'me=$$; ps -eo pid=,args=|awk -v me="$me" '\''$1==me{next}/[v]irtual_sync/{print $1}'\''|while read p; do kill -9 $p; done; echo K'
  vcctl pod exec $pod -- bash -lc "cat > /tmp/fire_a2.sh <<EOS
#!/bin/bash
set +e
export PATH=/opt/conda/bin:\\\$PATH PYTHONUNBUFFERED=1 CUDA_HOME=/opt/maca/tools/cu-bridge
OUT=$out1
BENCH=$BENCH
NR=$idx
for g in 0 1 2 3 4 5 6 7; do
  gr=\\\$((NR*8+g))
  nohup env CUDA_VISIBLE_DEVICES=\\\$g LOCAL_RANK=0 NODE_RANK=\\\$NR GPUS_PER_NODE=8 GLOBAL_RANK=\\\$gr \\
    python3 \\\$BENCH --mode independent --iters 3000 --warmup 30 --hidden 4096 --seq 2048 --layers 8 --batch 2 --out-dir \\\$OUT --tag A2 \\
    >\\\$OUT/n\\\${NR}_g\\\$g.log 2>&1 &
done
echo FIRED
EOS
chmod +x /tmp/fire_a2.sh && bash /tmp/fire_a2.sh"
  idx=$((idx+1))
  sleep 0.5
done

t=0
while [ $t -lt 100 ]; do
  n=$(vcctl pod exec ${JOB}-master-0 -- bash -lc "ls $out1/done_rank*.txt 2>/dev/null|wc -l" | tr -dc 0-9)
  s=$(vcctl pod exec ${JOB}-master-0 -- bash -lc "wc -l <$out1/step_times_rank000.jsonl 2>/dev/null||echo 0" | tr -dc 0-9)
  log "A2 done=${n:-0}/128 steps=${s:-0}"
  [ "${n:-0}" -ge 128 ] && break
  sleep 45
  t=$((t+1))
done
vcctl pod exec ${JOB}-master-0 -- bash -lc "echo WAVE1_DONE > $AFS/WAVE1.done"
log WAVE1_DONE
