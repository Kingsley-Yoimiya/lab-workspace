#!/usr/bin/env bash
# MUXI Phase0 一天战役主编排（A1→A2+A4→A3，B1/B2/C1 由外层续调）
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_FORCE_JUMP="${CLUSTER_FORCE_JUMP:-1}"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="${STAMP:-$(cat /tmp/muxi_day_stamp.txt)}"
DAY_ROOT="${DAY_ROOT:-$(cat /tmp/muxi_day_root.txt)}"
AFS_OUT="${AFS_OUT:-$(cat /tmp/muxi_day_afs.txt)}"
AFS_BENCH="/afs-a3-weight-share/yinjinrun.p/lab-workspace/scripts/cluster"
ITERS="${ITERS:-3000}"
WARMUP="${WARMUP:-30}"
HIDDEN="${HIDDEN:-4096}"; SEQ="${SEQ:-2048}"; LAYERS="${LAYERS:-8}"; BATCH="${BATCH:-2}"
JOB="$CLUSTER_JOB"
MASTER="${JOB}-master-0"
LOGF="$DAY_ROOT/orchestrator.log"
mkdir -p "$DAY_ROOT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGF"; }

upload_scripts() {
  cluster_pod_exec "$MASTER" "mkdir -p '$AFS_BENCH' '$AFS_OUT'"
  for f in virtual_sync_bench.py delay_inject.py gpu_busy_preempt.py; do
    ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
      "$(_cluster_vcctl_prefix) pod exec -i ${MASTER} -- bash -c 'cat > ${AFS_BENCH}/$f && wc -c ${AFS_BENCH}/$f'" \
      < "$SCRIPT_DIR/$f" | tee -a "$LOGF"
  done
}

safe_kill() {
  local pod="$1"
  cluster_pod_exec "$pod" 'set +e; me=$$; ps -eo pid=,args= | awk -v me="$me" '\''$1==me{next} /defunct/{next} /[t]orchrun|[v]irtual_sync_bench|[g]pu_busy_preempt/{print $1}'\'' | while read -r p; do kill -9 "$p" 2>/dev/null || true; done; echo K' || true
}

wait_done() {
  local dir="$1" need="$2" timeout="${3:-3600}" label="${4:-job}"
  local start=$SECONDS n
  while (( SECONDS - start < timeout )); do
    n=$(cluster_pod_exec "$MASTER" "ls '$dir'/done_rank*.txt 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9' || echo 0)
    n=${n:-0}
    log "  $label done=$n/$need"
    [[ "$n" -ge "$need" ]] && return 0
    sleep 30
  done
  log "FAIL $label timeout done=$n"
  return 1
}

fire_exp0_on_pod() {
  local pod="$1" tag="$2"  # tag e.g. A1_master
  local out_real="$AFS_OUT/$tag/exp0_real"
  local out_virt="$AFS_OUT/$tag/exp0_virtual"
  log "A1 FIRE real on $pod -> $tag"
  cluster_pod_exec "$pod" "mkdir -p '$out_real' '$out_virt'"
  safe_kill "$pod"
  sleep 1
  cluster_pod_exec "$pod" "
set +e
export PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0 MCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1 MCCL_IB_DISABLE=1
export CUDA_HOME=/opt/maca/tools/cu-bridge
rm -f '$out_real'/done_rank*.txt '$out_real'/step_times_rank*.jsonl
nohup env PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1 \
  NCCL_IB_DISABLE=1 MCCL_IB_DISABLE=1 CUDA_HOME=/opt/maca/tools/cu-bridge \
  GPUS_PER_NODE=8 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 \
  bash -c 'torchrun --standalone --nproc_per_node=8 $AFS_BENCH/virtual_sync_bench.py \
    --mode real_sync --iters $ITERS --warmup $WARMUP \
    --hidden $HIDDEN --seq $SEQ --layers $LAYERS --batch $BATCH \
    --out-dir $out_real --tag ${tag}_real >$out_real/torchrun.log 2>&1' &
echo FIRED_REAL
"
}

fire_exp0_virt_on_pod() {
  local pod="$1" tag="$2"
  local out_virt="$AFS_OUT/$tag/exp0_virtual"
  log "A1 FIRE virt on $pod -> $tag"
  safe_kill "$pod"
  sleep 2
  cluster_pod_exec "$pod" "
set +e
rm -f '$out_virt'/done_rank*.txt '$out_virt'/step_times_rank*.jsonl
for g in \$(seq 0 7); do
  nohup env PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1 CUDA_HOME=/opt/maca/tools/cu-bridge \
    CUDA_VISIBLE_DEVICES=\$g LOCAL_RANK=0 NODE_RANK=0 GPUS_PER_NODE=8 GLOBAL_RANK=\$g \
    python3 $AFS_BENCH/virtual_sync_bench.py \
      --mode independent --iters $ITERS --warmup $WARMUP \
      --hidden $HIDDEN --seq $SEQ --layers $LAYERS --batch $BATCH \
      --out-dir $out_virt --tag ${tag}_virt \
      >$out_virt/gpu\${g}.log 2>&1 &
done
echo FIRED_VIRT
"
}

# --- main ---
log "DAY START STAMP=$STAMP AFS=$AFS_OUT"
upload_scripts

# A1: 三节点并行 real（错开 port），再各自 virt
declare -a A1_PODS=("$MASTER" "${JOB}-worker-3" "${JOB}-worker-10")
declare -a A1_TAGS=("A1_master" "A1_worker3" "A1_worker10")

log "WAVE1 A1 real_sync parallel"
for i in 0 1 2; do
  fire_exp0_on_pod "${A1_PODS[$i]}" "${A1_TAGS[$i]}"
  sleep 2
done
# wait all reals — poll each
for i in 0 1 2; do
  wait_done "$AFS_OUT/${A1_TAGS[$i]}/exp0_real" 8 3600 "A1_real_${A1_TAGS[$i]}" || log "WARN real incomplete ${A1_TAGS[$i]}"
done

log "WAVE1 A1 independent"
for i in 0 1 2; do
  fire_exp0_virt_on_pod "${A1_PODS[$i]}" "${A1_TAGS[$i]}"
  sleep 2
done
for i in 0 1 2; do
  wait_done "$AFS_OUT/${A1_TAGS[$i]}/exp0_virtual" 8 3600 "A1_virt_${A1_TAGS[$i]}" || log "WARN virt incomplete ${A1_TAGS[$i]}"
done
log "A1_DONE"
echo A1_DONE > "$DAY_ROOT/A1.done"

# A4 telemetry start then A2
log "WAVE1 A4 telemetry + A2 128"
AFS_TELEM="$AFS_OUT/A4/telemetry"
export AFS_TELEM INTERVAL=5 DURATION=$((ITERS * 1 + 600))
# rough duration: ~0.23s * 3000 ≈ 700s + buffer
DURATION=1200
bash "$SCRIPT_DIR/sample_mx_telemetry_muxi.sh" | tee -a "$LOGF"

# clear + fire exp1
out1="$AFS_OUT/A2/exp1_independent"
cluster_pod_exec "$MASTER" "mkdir -p '$out1'; rm -f '$out1'/step_times_rank*.jsonl '$out1'/done_rank*.txt '$out1'/*.log '$out1'/meta_rank*.json"
pods=("${JOB}-master-0"); for i in $(seq 0 14); do pods+=("${JOB}-worker-$i"); done
idx=0
for pod in "${pods[@]}"; do
  node_rank=$idx
  safe_kill "$pod"
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
      --out-dir $out1 --tag A2 \
      >$out1/node${node_rank}_gpu\${g}.log 2>&1 &
done
echo FIRED_NODE $node_rank
" || log "WARN fire node $node_rank"
  idx=$((idx + 1))
  sleep 0.8
done
wait_done "$out1" 128 7200 "A2_exp1" || log "WARN A2 incomplete"
log "A2_DONE"
echo A2_DONE > "$DAY_ROOT/A2.done"
log "WAVE1_CLUSTER_DONE"
echo WAVE1_CLUSTER_DONE > "$DAY_ROOT/WAVE1_CLUSTER.done"
