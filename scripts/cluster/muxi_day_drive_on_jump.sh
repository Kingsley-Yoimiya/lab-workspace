#!/usr/bin/env bash
# 在 ais 跳板上跑（vcctl 本地），避免本机嵌套 SSH 长跑被杀
set +e
export KUBECONFIG=/root/.kube/config.muxi-mohe
AFS=/afs-a3-weight-share/montyyin/results/muxi-day-20260713_002719
JOB=yushan-muxi-card-screen-128-cp-copy
MASTER=${JOB}-master-0
AFS_BENCH=/afs-a3-weight-share/montyyin/lab-workspace/scripts/cluster
LOG=/tmp/muxi_day_drive.log
: >"$LOG"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ex() { vcctl pod exec "$1" -- bash -lc "$2"; }
cdone() { ex "$MASTER" "ls $1/done_rank*.txt 2>/dev/null|wc -l" 2>/dev/null | tr -dc '0-9'; echo; }
csteps() { ex "$MASTER" "wc -l <$1/step_times_rank000.jsonl 2>/dev/null||echo 0" 2>/dev/null | tr -dc '0-9'; echo; }
wait8() {
  local d=$1 l=$2 t=0 n s
  while [ "$t" -lt 100 ]; do
    n=$(cdone "$d"); s=$(csteps "$d")
    log "$l done=${n:-0}/8 steps=${s:-0}"
    [ "${n:-0}" -ge 8 ] && return 0
    sleep 45
    t=$((t + 1))
  done
  return 1
}
killb() {
  ex "$1" 'me=$$; ps -eo pid=,args=|awk -v me="$me" '\''$1==me{next}/[t]orchrun|[v]irtual_sync/{print $1}'\''|while read p; do kill -9 $p; done; echo K'
}
firev() {
  local pod=$1 tag=$2 out=$AFS/$tag/exp0_virtual
  log "virt $tag"
  ex "$pod" "mkdir -p $out"
  killb "$pod"
  sleep 2
  ex "$pod" "rm -f $out/done_rank*.txt $out/step_times_rank*.jsonl; for g in \$(seq 0 7); do nohup env PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1 CUDA_HOME=/opt/maca/tools/cu-bridge CUDA_VISIBLE_DEVICES=\$g LOCAL_RANK=0 NODE_RANK=0 GPUS_PER_NODE=8 GLOBAL_RANK=\$g python3 $AFS_BENCH/virtual_sync_bench.py --mode independent --iters 3000 --warmup 30 --hidden 4096 --seq 2048 --layers 8 --batch 2 --out-dir $out --tag ${tag}_v >$out/gpu\$g.log 2>&1 & done; echo V"
}

log START
wait8 "$AFS/A1_master/exp0_real" mR
wait8 "$AFS/A1_worker3/exp0_real" w3R
wait8 "$AFS/A1_worker10/exp0_real" w10R
firev "$MASTER" A1_master
firev "${JOB}-worker-3" A1_worker3
firev "${JOB}-worker-10" A1_worker10
wait8 "$AFS/A1_master/exp0_virtual" mV
wait8 "$AFS/A1_worker3/exp0_virtual" w3V
wait8 "$AFS/A1_worker10/exp0_virtual" w10V
ex "$MASTER" "echo A1_DONE > $AFS/A1.done"
log A1_DONE

log telem
ex "$MASTER" "mkdir -p $AFS/A4/telemetry"
idx=0
for logic in master-0 worker-0 worker-1 worker-2 worker-3 worker-4 worker-5 worker-6 worker-7 worker-8 worker-9 worker-10 worker-11 worker-12 worker-13 worker-14; do
  ex "${JOB}-$logic" "nohup python3 -c 'import json,subprocess,time;out=\"$AFS/A4/telemetry/node${idx}.jsonl\";end=time.time()+1200;n=$idx
while time.time()<end:
 t=time.time();r=subprocess.run([\"mx-smi\",\"-j\"],capture_output=True,text=True,timeout=15).stdout;i=r.find(\"{\");d=json.loads(r[i:]) if i>=0 else {};open(out,\"a\").write(json.dumps({\"ts\":t,\"node\":n,\"smi\":d})+\"\\n\");time.sleep(5)
print(1)' >$AFS/A4/telemetry/n${idx}.log 2>&1 & echo T$idx" >/dev/null
  idx=$((idx + 1))
  sleep 0.2
done

out1=$AFS/A2/exp1_independent
ex "$MASTER" "mkdir -p $out1; rm -f $out1/step_times_rank*.jsonl $out1/done_rank*.txt $out1/*.log $out1/meta*.json"
idx=0
for logic in master-0 worker-0 worker-1 worker-2 worker-3 worker-4 worker-5 worker-6 worker-7 worker-8 worker-9 worker-10 worker-11 worker-12 worker-13 worker-14; do
  log "A2 $idx"
  killb "${JOB}-$logic"
  ex "${JOB}-$logic" "mkdir -p $out1; for g in \$(seq 0 7); do gr=\$(($idx*8+g)); nohup env PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1 CUDA_HOME=/opt/maca/tools/cu-bridge CUDA_VISIBLE_DEVICES=\$g LOCAL_RANK=0 NODE_RANK=$idx GPUS_PER_NODE=8 GLOBAL_RANK=\$gr python3 $AFS_BENCH/virtual_sync_bench.py --mode independent --iters 3000 --warmup 30 --hidden 4096 --seq 2048 --layers 8 --batch 2 --out-dir $out1 --tag A2 >$out1/n${idx}_g\$g.log 2>&1 & done; echo F$idx"
  idx=$((idx + 1))
  sleep 0.8
done

t=0
while [ "$t" -lt 120 ]; do
  n=$(cdone "$out1")
  s=$(csteps "$out1")
  log "A2 done=${n:-0}/128 steps=${s:-0}"
  [ "${n:-0}" -ge 128 ] && break
  sleep 45
  t=$((t + 1))
done
ex "$MASTER" "echo WAVE1_DONE > $AFS/WAVE1.done"
log WAVE1_DONE
