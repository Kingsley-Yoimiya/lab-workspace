#!/usr/bin/env bash
# 续跑 / 稳健编排：不用 exec+tee 进程替换（nohup 下会挂）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_FORCE_JUMP="${CLUSTER_FORCE_JUMP:-1}"
export CLUSTER_SSH_HOST_OVERRIDE="${CLUSTER_SSH_HOST_OVERRIDE:-ais-cf3e61a5}"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="${STAMP:?set STAMP}"
AFS_OUT="${AFS_OUT:-/afs-a3-weight-share/montyyin/results/muxi-vsync-${STAMP}}"
AFS_BENCH="/afs-a3-weight-share/montyyin/lab-workspace/scripts/cluster"
ITERS="${ITERS:-1000}"
WARMUP="${WARMUP:-20}"
HIDDEN="${HIDDEN:-4096}"
SEQ="${SEQ:-2048}"
LAYERS="${LAYERS:-8}"
BATCH="${BATCH:-2}"
REPO_LOG="${REPO_LOG:-/Users/yinjinrun/random-thing/logs/muxi-vsync-${STAMP}}"
mkdir -p "$REPO_LOG"
LOGF="$REPO_LOG/continue.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGF"; }

JOB="$CLUSTER_JOB"
MASTER="${JOB}-master-0"
pods=("${JOB}-master-0")
for i in $(seq 0 14); do pods+=("${JOB}-worker-$i"); done

wait_done() {
  local dir="$1" need="$2" timeout="${3:-2400}" label="${4:-job}"
  local start=$SECONDS n
  while (( SECONDS - start < timeout )); do
    n=$(cluster_pod_exec "$MASTER" "ls '$dir'/done_rank*.txt 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9')
    n=${n:-0}
    log "  $label done=$n/$need"
    [[ "$n" -ge "$need" ]] && return 0
    sleep 20
  done
  log "FAIL $label timeout done=$n/$need"
  return 1
}

# --- exp0 real: 若已在跑则等；否则启动 ---
out_real="$AFS_OUT/exp0_real"
out_virt="$AFS_OUT/exp0_virtual"
cluster_pod_exec "$MASTER" "mkdir -p '$out_real' '$out_virt' '$AFS_OUT'"

n_real=$(cluster_pod_exec "$MASTER" "ls '$out_real'/done_rank*.txt 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9')
n_real=${n_real:-0}
alive=$(cluster_pod_exec "$MASTER" "ps -ef | grep -E '[t]orchrun.*virtual_sync|[v]irtual_sync_bench' | wc -l" 2>/dev/null | tr -dc '0-9')
alive=${alive:-0}
log "exp0_real done=$n_real alive=$alive"

if [[ "$n_real" -lt 8 ]]; then
  if [[ "$alive" -lt 1 ]]; then
    log "FIRE exp0_real"
    cluster_pod_exec "$MASTER" "
set -euo pipefail
source '$AFS_OUT/env_common.sh' 2>/dev/null || true
export PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0 MCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1 MCCL_IB_DISABLE=1
CU_BRIDGE_BIN=/opt/maca/tools/cu-bridge/bin
[[ -x \$CU_BRIDGE_BIN/cucc && ! -e \$CU_BRIDGE_BIN/nvcc ]] && ln -sfn \$CU_BRIDGE_BIN/cucc \$CU_BRIDGE_BIN/nvcc || true
export CUDA_HOME=/opt/maca/tools/cu-bridge
rm -f '$out_real'/done_rank*.txt
nohup bash -lc '
export PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0 MCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1 MCCL_IB_DISABLE=1
export CUDA_HOME=/opt/maca/tools/cu-bridge
export GPUS_PER_NODE=8 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501
torchrun --standalone --nproc_per_node=8 $AFS_BENCH/virtual_sync_bench.py \
  --mode real_sync --iters $ITERS --warmup $WARMUP \
  --hidden $HIDDEN --seq $SEQ --layers $LAYERS --batch $BATCH \
  --out-dir $out_real --tag exp0_real \
  >$out_real/torchrun.log 2>&1
echo REAL_DONE \$? >>$out_real/torchrun.log
' >/dev/null 2>&1 &
echo FIRED \$!
"
  else
    log "exp0_real already running, wait"
  fi
  wait_done "$out_real" 8 "${EXP0_TIMEOUT:-2400}" "exp0_real"
fi
log "EXP0_REAL_OK"

# --- exp0 virtual ---
n_virt=$(cluster_pod_exec "$MASTER" "ls '$out_virt'/done_rank*.txt 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9')
n_virt=${n_virt:-0}
if [[ "$n_virt" -lt 8 ]]; then
  log "KILL leftovers before exp0_virtual"
  cluster_pod_exec "$MASTER" 'set +e; me=$$; ps -eo pid=,args= | awk -v me="$me" '\''$1==me{next} /defunct/{next} /[t]orchrun|[v]irtual_sync_bench/{print $1}'\'' | while read -r p; do kill -9 "$p" 2>/dev/null || true; done; echo KILL_DONE'
  sleep 2
  log "FIRE exp0_virtual"
  cluster_pod_exec "$MASTER" "
set +e
rm -f '$out_virt'/done_rank*.txt '$out_virt'/step_times_rank*.jsonl
for g in \$(seq 0 7); do
  nohup env PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1 CUDA_HOME=/opt/maca/tools/cu-bridge \
    CUDA_VISIBLE_DEVICES=\$g LOCAL_RANK=0 NODE_RANK=0 GPUS_PER_NODE=8 GLOBAL_RANK=\$g \
    python3 $AFS_BENCH/virtual_sync_bench.py \
      --mode independent --iters $ITERS --warmup $WARMUP \
      --hidden $HIDDEN --seq $SEQ --layers $LAYERS --batch $BATCH \
      --out-dir $out_virt --tag exp0_virtual \
      >$out_virt/gpu\${g}.log 2>&1 &
done
sleep 1
echo FIRED_VIRT alive=\$(ps -ef | grep '[v]irtual_sync_bench' | wc -l)
"
  wait_done "$out_virt" 8 "${EXP0_TIMEOUT:-2400}" "exp0_virtual"
fi
log "EXP0_VIRTUAL_OK"

# --- exp1 ---
out1="$AFS_OUT/exp1_independent"
cluster_pod_exec "$MASTER" "mkdir -p '$out1'"
n1=$(cluster_pod_exec "$MASTER" "ls '$out1'/done_rank*.txt 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9')
n1=${n1:-0}
if [[ "$n1" -lt 128 ]]; then
  log "FIRE exp1 on 16 nodes"
  idx=0
  for pod in "${pods[@]}"; do
    node_rank=$idx
    # kill 与 fire 分开，避免命令行自匹配
    cluster_pod_exec "$pod" 'set +e; me=$$; ps -eo pid=,args= | awk -v me="$me" '\''$1==me{next} /defunct/{next} /[v]irtual_sync_bench/{print $1}'\'' | while read -r p; do kill -9 "$p" 2>/dev/null || true; done; echo KILL_OK' || true
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
echo FIRED_NODE $node_rank
" | tee -a "$LOGF"
    idx=$((idx + 1))
    sleep 0.3
  done
  wait_done "$out1" 128 "${EXP1_TIMEOUT:-3600}" "exp1"
fi
log "EXP1_OK"
log "AFS_OUT=$AFS_OUT"
log "ORCH_DONE"
