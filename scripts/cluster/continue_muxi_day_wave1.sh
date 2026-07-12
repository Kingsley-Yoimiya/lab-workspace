#!/usr/bin/env bash
# 抗 SSH 失败的 Wave1 续跑：等 A1 三节点 real→virt，再 A4+A2
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_FORCE_JUMP=1
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"
STAMP=$(cat /tmp/muxi_day_stamp.txt)
DAY_ROOT=$(cat /tmp/muxi_day_root.txt)
AFS_OUT=$(cat /tmp/muxi_day_afs.txt)
AFS_BENCH=/afs-a3-weight-share/montyyin/lab-workspace/scripts/cluster
ITERS=3000; WARMUP=30
JOB=$CLUSTER_JOB
MASTER=${JOB}-master-0
LOGF=$DAY_ROOT/continue_wave1.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGF"; }

wait_tag(){
  local tag=$1 mode=$2 need=8
  local dir="$AFS_OUT/$tag/exp0_$mode"
  local start=$SECONDS n
  while (( SECONDS-start < 3600 )); do
    n=$(cluster_pod_exec "$MASTER" "ls $dir/done_rank*.txt 2>/dev/null|wc -l" 2>/dev/null|tr -dc '0-9'||echo 0)
    n=${n:-0}
    steps=$(cluster_pod_exec "$MASTER" "wc -l <$dir/step_times_rank000.jsonl 2>/dev/null||echo 0" 2>/dev/null|tr -dc '0-9'||echo 0)
    log "  wait $tag/$mode done=$n/$need steps0=${steps:-0}"
    [[ "$n" -ge "$need" ]] && return 0
    sleep 45
  done
  return 1
}

fire_virt(){
  local pod=$1 tag=$2
  local out=$AFS_OUT/$tag/exp0_virtual
  cluster_pod_exec "$pod" "mkdir -p $out" || return 1
  cluster_pod_exec "$pod" 'set +e; me=$$; ps -eo pid=,args= | awk -v me="$me" '\''$1==me{next} /[t]orchrun|[v]irtual_sync/{print $1}'\'' | while read p; do kill -9 $p 2>/dev/null; done; echo K' || true
  sleep 2
  cluster_pod_exec "$pod" "
set +e
rm -f $out/done_rank*.txt $out/step_times_rank*.jsonl
for g in \$(seq 0 7); do
  nohup env PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1 CUDA_HOME=/opt/maca/tools/cu-bridge \
    CUDA_VISIBLE_DEVICES=\$g LOCAL_RANK=0 NODE_RANK=0 GPUS_PER_NODE=8 GLOBAL_RANK=\$g \
    python3 $AFS_BENCH/virtual_sync_bench.py --mode independent --iters $ITERS --warmup $WARMUP \
    --hidden 4096 --seq 2048 --layers 8 --batch 2 --out-dir $out --tag ${tag}_v \
    >$out/gpu\$g.log 2>&1 &
done
echo FIRED_VIRT
" || return 1
}

ensure_real(){
  local pod=$1 tag=$2 port=$3
  local out=$AFS_OUT/$tag/exp0_real
  local n
  n=$(cluster_pod_exec "$MASTER" "ls $out/done_rank*.txt 2>/dev/null|wc -l" 2>/dev/null|tr -dc '0-9'||echo 0)
  if [[ "${n:-0}" -ge 8 ]]; then log "$tag real already done"; return 0; fi
  alive=$(cluster_pod_exec "$pod" "ps -ef|grep '[t]orchrun'|wc -l" 2>/dev/null|tr -dc '0-9'||echo 0)
  if [[ "${alive:-0}" -gt 0 ]]; then log "$tag real running alive=$alive"; return 0; fi
  log "FIRE real $tag"
  cluster_pod_exec "$pod" "mkdir -p $out" || true
  cluster_pod_exec "$pod" "
set +e
rm -f $out/done_rank*.txt $out/step_times_rank*.jsonl
nohup env PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1 NCCL_IB_DISABLE=1 MCCL_IB_DISABLE=1 \
  CUDA_HOME=/opt/maca/tools/cu-bridge GPUS_PER_NODE=8 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$port \
  bash -c 'torchrun --standalone --nproc_per_node=8 $AFS_BENCH/virtual_sync_bench.py --mode real_sync --iters $ITERS --warmup $WARMUP --hidden 4096 --seq 2048 --layers 8 --batch 2 --out-dir $out --tag ${tag}_r >$out/torchrun.log 2>&1' &
echo FIRED
" || log "WARN fire $tag"
}

log "continue wave1 start"
ensure_real "$MASTER" A1_master 29511
ensure_real "${JOB}-worker-3" A1_worker3 29512
ensure_real "${JOB}-worker-10" A1_worker10 29513

wait_tag A1_master real || log WARN_master_real
wait_tag A1_worker3 real || log WARN_w3_real
wait_tag A1_worker10 real || log WARN_w10_real

fire_virt "$MASTER" A1_master
fire_virt "${JOB}-worker-3" A1_worker3
fire_virt "${JOB}-worker-10" A1_worker10
wait_tag A1_master virtual || log WARN_master_v
wait_tag A1_worker3 virtual || log WARN_w3_v
wait_tag A1_worker10 virtual || log WARN_w10_v
echo A1_DONE > $DAY_ROOT/A1.done
log A1_DONE

# A4 + A2
log "A4 telem"
AFS_TELEM=$AFS_OUT/A4/telemetry INTERVAL=5 DURATION=1200 bash $SCRIPT_DIR/sample_mx_telemetry_muxi.sh | tee -a $LOGF || log WARN_telem

out1=$AFS_OUT/A2/exp1_independent
cluster_pod_exec "$MASTER" "mkdir -p $out1; rm -f $out1/step_times_rank*.jsonl $out1/done_rank*.txt $out1/*.log $out1/meta_rank*.json" || true
pods=($MASTER); for i in $(seq 0 14); do pods+=(${JOB}-worker-$i); done
idx=0
for pod in "${pods[@]}"; do
  log "A2 fire node=$idx"
  cluster_pod_exec "$pod" 'set +e; me=$$; ps -eo pid=,args= | awk -v me="$me" '\''$1==me{next} /[v]irtual_sync/{print $1}'\'' | while read p; do kill -9 $p 2>/dev/null; done; echo K' || true
  cluster_pod_exec "$pod" "
set +e
mkdir -p $out1
for g in \$(seq 0 7); do
  gr=\$(( $idx * 8 + g ))
  nohup env PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1 CUDA_HOME=/opt/maca/tools/cu-bridge \
    CUDA_VISIBLE_DEVICES=\$g LOCAL_RANK=0 NODE_RANK=$idx GPUS_PER_NODE=8 GLOBAL_RANK=\$gr \
    python3 $AFS_BENCH/virtual_sync_bench.py --mode independent --iters $ITERS --warmup $WARMUP \
    --hidden 4096 --seq 2048 --layers 8 --batch 2 --out-dir $out1 --tag A2 \
    >$out1/node${idx}_gpu\$g.log 2>&1 &
done
echo FIRED $idx
" || log "WARN a2 $idx"
  idx=$((idx+1))
  sleep 0.8
done

start=$SECONDS
while (( SECONDS-start < 7200 )); do
  n=$(cluster_pod_exec "$MASTER" "ls $out1/done_rank*.txt 2>/dev/null|wc -l" 2>/dev/null|tr -dc '0-9'||echo 0)
  n=${n:-0}
  log "A2 done=$n/128"
  [[ "$n" -ge 128 ]] && break
  sleep 45
done
echo A2_DONE > $DAY_ROOT/A2.done
echo WAVE1_CLUSTER_DONE > $DAY_ROOT/WAVE1_CLUSTER.done
log WAVE1_CLUSTER_DONE
