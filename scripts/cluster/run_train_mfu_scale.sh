#!/usr/bin/env bash
# MindSpeed 训练 MFU scale（修复远程变量展开）
# MODE=dense|moe  SCALES=16,32,64,128  TRAIN_ITERS=20
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

MODE="${MODE:-dense}"
SCALES="${SCALES:-16,32,64,128}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MEGATRON_ROOT="/afs-a3-241ceshi-shared/geruijun/Megatron-LM-0.12.3"
DATA_ROOT="/afs-a3-241ceshi-shared/geruijun"
RUN_ROOT="/afs-a3-241ceshi-shared/montyyin/logs/train-${MODE}-${STAMP}"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/train-${MODE}-${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/train.log") 2>&1

if [[ "$MODE" == "moe" ]]; then
  BASE_SCRIPT="examples/qwen3/PT_qwen3_30B_A3B.sh"
else
  BASE_SCRIPT="examples/qwen3/PT_qwen3_32B.sh"
fi

MASTER_ADDR="huawei-8node-copy-master-0.huawei-8node-copy"
MASTER_PORT="${MASTER_PORT:-24670}"
NPUS_PER_NODE="${NPUS_PER_NODE:-16}"

pod_for_rank() {
  local r="$1"
  if [[ "$r" -eq 0 ]]; then echo "${CLUSTER_JOB}-master-0"; else echo "${CLUSTER_JOB}-worker-$((r-1))"; fi
}

echo "==> MODE=$MODE BASE=$BASE_SCRIPT SCALES=$SCALES ITERS=$TRAIN_ITERS"
echo "==> RUN_ROOT=$RUN_ROOT LOG_DIR=$LOG_DIR"

# 上传通用 runner 到 AFS
RUNNER_LOCAL="$LOG_DIR/remote_train_one.sh"
cat > "$RUNNER_LOCAL" <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail
# args: megatron_root base_script run_dir rank nnodes master_addr master_port npus iters data_root
MEGATRON_ROOT="$1"
BASE_SCRIPT="$2"
RUN_DIR="$3"
RANK="$4"
NNODES="$5"
MASTER_ADDR="$6"
MASTER_PORT="$7"
NPUS="$8"
ITERS="$9"
DATA_ROOT="${10}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

export TOKENIZER_MODEL="${DATA_ROOT}/Qwen3-32B"
export VOCAB_FILE="${DATA_ROOT}/Qwen3-32B/vocab.json"
export DATA_PATH="${DATA_ROOT}/dataset/data_text_document"
mkdir -p /tmp/afs-grj-wrap
ln -sfn "${DATA_ROOT}/Qwen3-32B" /tmp/afs-grj-wrap/Qwen3-32B
ln -sfn "${DATA_ROOT}/dataset" /tmp/afs-grj-wrap/dataset
# 若脚本写死 /afs-grj，用 sed 改到真实路径
cd "$MEGATRON_ROOT"
mkdir -p "$RUN_DIR"
OUT_SH="$RUN_DIR/run_rank${RANK}.sh"
cp "$BASE_SCRIPT" "$OUT_SH"
sed -i "s|TOKENIZER_MODEL=/afs-grj/Qwen3-32B|TOKENIZER_MODEL=${TOKENIZER_MODEL}|g" "$OUT_SH"
sed -i "s|VOCAB_FILE=/afs-grj/Qwen3-32B/vocab.json|VOCAB_FILE=${VOCAB_FILE}|g" "$OUT_SH"
sed -i "s|DATA_PATH=\"/afs-grj/dataset/data_text_document\"|DATA_PATH=\"${DATA_PATH}\"|g" "$OUT_SH"
sed -i "s/^TRAIN_ITERS=.*/TRAIN_ITERS=${ITERS}/" "$OUT_SH"
sed -i "s/^NPUS_PER_NODE=.*/NPUS_PER_NODE=${NPUS}/" "$OUT_SH"
sed -i "s/^PROC_PER_NODE=.*/PROC_PER_NODE=${NPUS}/" "$OUT_SH"
sed -i "s/^GPUS_PER_NODE=.*/GPUS_PER_NODE=${NPUS}/" "$OUT_SH"
# 分布式：覆盖硬编码
sed -i "s/^WORLD_SIZE=.*/WORLD_SIZE=${NNODES}/" "$OUT_SH"
sed -i "s/^RANK=.*/RANK=${RANK}/" "$OUT_SH"
sed -i "s/^MASTER_ADDR=.*/MASTER_ADDR=${MASTER_ADDR}/" "$OUT_SH"
sed -i "s/^MASTER_PORT=.*/MASTER_PORT=${MASTER_PORT}/" "$OUT_SH"
# 日志目录
sed -i "s|^LOG_DIR=.*|LOG_DIR=\"${RUN_DIR}/\"|" "$OUT_SH" || true
sed -i "s|^tensorboard_dir=.*|tensorboard_dir=\"${RUN_DIR}/tb\"|" "$OUT_SH" || true
sed -i "s|^CKPT_SAVE_DIR=.*|CKPT_SAVE_DIR=\"${RUN_DIR}/ckpt\"|" "$OUT_SH" || true
sed -i "s|^TENSORBOARD_DIR=.*|TENSORBOARD_DIR=\"${RUN_DIR}/tb\"|" "$OUT_SH" || true

export WORLD_SIZE="$NNODES"
export RANK
export MASTER_ADDR
export MASTER_PORT
export NPUS_PER_NODE="$NPUS"

bash "$OUT_SH" 2>&1 | tee "${RUN_DIR}/rank${RANK}.log"
echo TRAIN_RANK_${RANK}_DONE
REMOTE

ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'mkdir -p $RUN_ROOT && cat > $RUN_ROOT/remote_train_one.sh && chmod +x $RUN_ROOT/remote_train_one.sh'" \
  < "$RUNNER_LOCAL"

run_scale() {
  local world_npu="$1"
  local nnodes=$((world_npu / NPUS_PER_NODE))
  local scale_dir="$RUN_ROOT/scale_${world_npu}"
  echo "==> scale=$world_npu nnodes=$nnodes"
  cluster_pod_exec "${CLUSTER_JOB}-master-0" "mkdir -p '$scale_dir'"
  local pids=() r=0
  while [[ "$r" -lt "$nnodes" ]]; do
    local pod logf
    pod="$(pod_for_rank "$r")"
    logf="$LOG_DIR/scale${world_npu}_rank${r}.log"
    ssh -o BatchMode=yes -o ConnectTimeout=30 "$CLUSTER_SSH_HOST" \
      "vcctl pod exec ${pod} -- bash -lc $(printf '%q' "bash $RUN_ROOT/remote_train_one.sh $MEGATRON_ROOT $BASE_SCRIPT $scale_dir $r $nnodes $MASTER_ADDR $MASTER_PORT $NPUS_PER_NODE $TRAIN_ITERS $DATA_ROOT")" \
      >"$logf" 2>&1 &
    pids+=("$!")
    r=$((r+1))
  done
  local fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  cluster_pod_exec "${CLUSTER_JOB}-master-0" \
    "grep -hEi 'mfu|throughput|tokens/s|tokens per|elapsed time per iteration|TFLOP|iteration' '$scale_dir'/*.log 2>/dev/null | tail -50 || true" \
    | tee "$LOG_DIR/scale${world_npu}_metrics.txt"
  [[ "$fail" -eq 0 ]] || echo "FAIL scale=$world_npu"
}

IFS=',' read -ra ARR <<< "$SCALES"
for s in "${ARR[@]}"; do
  run_scale "$s" || true
  MASTER_PORT=$((MASTER_PORT+1))
done

ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'tar -C $RUN_ROOT -cf - .' " \
  > "$LOG_DIR/results.tar" || true
mkdir -p "$LOG_DIR/results"
tar -xf "$LOG_DIR/results.tar" -C "$LOG_DIR/results" 2>/dev/null || true
echo "TRAIN_MFU_DONE mode=$MODE → $RUN_ROOT / $LOG_DIR"
