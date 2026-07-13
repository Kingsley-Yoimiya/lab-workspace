#!/usr/bin/env bash
# 仅扇出 + 等待 exp1（128 卡独立负载）；SSH 失败不中止
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_FORCE_JUMP="${CLUSTER_FORCE_JUMP:-1}"
export CLUSTER_SSH_HOST_OVERRIDE="${CLUSTER_SSH_HOST_OVERRIDE:-ais-cf3e61a5}"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="${STAMP:?}"
AFS_OUT="${AFS_OUT:-/afs-a3-weight-share/yinjinrun.p/results/muxi-vsync-${STAMP}}"
AFS_BENCH="/afs-a3-weight-share/yinjinrun.p/lab-workspace/scripts/cluster"
out1="$AFS_OUT/exp1_independent"
ITERS="${ITERS:-1000}"; WARMUP="${WARMUP:-20}"
HIDDEN="${HIDDEN:-4096}"; SEQ="${SEQ:-2048}"; LAYERS="${LAYERS:-8}"; BATCH="${BATCH:-2}"
REPO_LOG="${REPO_LOG:-/Users/yinjinrun/random-thing/logs/muxi-vsync-${STAMP}}"
mkdir -p "$REPO_LOG"
LOGF="$REPO_LOG/exp1_fire.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGF"; }

JOB="$CLUSTER_JOB"
MASTER="${JOB}-master-0"
pods=("${JOB}-master-0")
for i in $(seq 0 14); do pods+=("${JOB}-worker-$i"); done

cluster_pod_exec "$MASTER" "mkdir -p '$out1'" || true

idx=0
for pod in "${pods[@]}"; do
  node_rank=$idx
  log "fire node=$node_rank pod=$pod"
  # 不在 fire 命令里写 virtual_sync 字样的 kill 模式；单独 kill
  cluster_pod_exec "$pod" 'set +e; me=$$; ps -eo pid=,args= | awk -v me="$me" '\''$1==me{next} /defunct/{next} /[v]irtual_sync_bench/{print $1}'\'' | while read -r p; do kill -9 "$p" 2>/dev/null || true; done; echo K' || log "WARN kill $pod"
  sleep 0.5
  if ! cluster_pod_exec "$pod" "
set +e
mkdir -p '$out1'
for g in \$(seq 0 7); do
  gr=\$(( $node_rank * 8 + g ))
  nohup env PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1 CUDA_HOME=/opt/maca/tools/cu-bridge \
    CUDA_VISIBLE_DEVICES=\$g LOCAL_RANK=0 NODE_RANK=$node_rank GPUS_PER_NODE=8 GLOBAL_RANK=\$gr \
    python3 $AFS_BENCH/virtual_sync_bench.py \
      --mode independent --iters $ITERS --warmup $WARMUP \
      --hidden $HIDDEN --seq $SEQ --layers $LAYERS --batch $BATCH \
      --out-dir $out1 --tag exp1 \
      >$out1/node${node_rank}_gpu\${g}.log 2>&1 &
done
echo FIRED $node_rank
"; then
    log "WARN fire failed node=$node_rank — retry once"
    sleep 2
    cluster_pod_exec "$pod" "
set +e
mkdir -p '$out1'
for g in \$(seq 0 7); do
  gr=\$(( $node_rank * 8 + g ))
  nohup env PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1 CUDA_HOME=/opt/maca/tools/cu-bridge \
    CUDA_VISIBLE_DEVICES=\$g LOCAL_RANK=0 NODE_RANK=$node_rank GPUS_PER_NODE=8 GLOBAL_RANK=\$gr \
    python3 $AFS_BENCH/virtual_sync_bench.py \
      --mode independent --iters $ITERS --warmup $WARMUP \
      --hidden $HIDDEN --seq $SEQ --layers $LAYERS --batch $BATCH \
      --out-dir $out1 --tag exp1 \
      >$out1/node${node_rank}_gpu\${g}.log 2>&1 &
done
echo FIRED_RETRY $node_rank
" || log "FAIL fire node=$node_rank"
  fi
  idx=$((idx + 1))
  sleep 1
done

log "all nodes fired; waiting for 128 done files"
start=$SECONDS
while (( SECONDS - start < ${EXP1_TIMEOUT:-3600} )); do
  n=$(cluster_pod_exec "$MASTER" "ls '$out1'/done_rank*.txt 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9' || echo 0)
  n=${n:-0}
  steps=$(cluster_pod_exec "$MASTER" "wc -l <'$out1'/step_times_rank000.jsonl 2>/dev/null || echo 0" 2>/dev/null | tr -dc '0-9' || echo 0)
  log "exp1 done=$n/128 steps0=${steps:-0}"
  [[ "$n" -ge 128 ]] && { log "EXP1_OK"; log "ORCH_DONE"; exit 0; }
  sleep 30
done
log "FAIL exp1 timeout done=$n"
exit 1
