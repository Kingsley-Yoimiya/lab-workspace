#!/usr/bin/env bash
# MindSpeed 真训练 MFU scale（C0 解阻）
# MODE=dense|moe  SCALES=16,32,64,128  TRAIN_ITERS=20
# SOURCE=wrapper|pt  （默认 wrapper：muxi 模型配置的 Ascend 适配脚本）
#
# dense → wrappers/train_qwen3_8B_ascend.sh（真 dense；勿用 PT_qwen3_32B，其含 NUM_EXPERTS=128）
# moe   → wrappers/train_qwen3_30B_A3B_ascend.sh
# SOURCE=pt 时回退到 examples/qwen3/PT_*.sh，经 patch_pt_train.sh 修 /afs-grj 与 | tee
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

MODE="${MODE:-dense}"
SCALES="${SCALES:-16,32,64,128}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
SOURCE="${SOURCE:-wrapper}"
NPUS_PER_NODE="${NPUS_PER_NODE:-16}"
MASTER_ADDR="${MASTER_ADDR:-huawei-8node-copy-master-0.huawei-8node-copy}"
MASTER_PORT="${MASTER_PORT:-24670}"
SKIP_SAVE="${SKIP_SAVE:-1}"
SKIP_PROFILE="${SKIP_PROFILE:-1}"
# 并行/超参（透传给 wrapper）
TP="${TP:-2}"
PP="${PP:-2}"
MBS="${MBS:-1}"
GBS="${GBS:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SKIP_TB="${SKIP_TB:-1}"
CP="${CP:-1}"
EP="${EP:-1}"
ETP="${ETP:-1}"

MEGATRON_ROOT="${MEGATRON_ROOT:-/afs-a3-241ceshi-shared/geruijun/Megatron-LM-0.12.3}"
DATA_ROOT="${DATA_ROOT:-/afs-a3-241ceshi-shared/geruijun}"
AFS_WRAPPERS="${AFS_WRAPPERS:-/afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster/wrappers}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-/afs-a3-241ceshi-shared/montyyin/logs/train-${MODE}-${STAMP}}"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/train-${MODE}-${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/train.log") 2>&1

case "$MODE" in
  dense)
    WRAPPER_NAME="train_qwen3_8B_ascend.sh"
    # PT 回退：8B 脚本实际也带 NUM_EXPERTS，仅作语法/路径修补试验，不推荐当 dense
    PT_REL="examples/qwen3/PT_qwen3_8B.sh"
    ;;
  moe)
    WRAPPER_NAME="train_qwen3_30B_A3B_ascend.sh"
    PT_REL="examples/qwen3/PT_qwen3_30B_A3B.sh"
    ;;
  *)
    echo "MODE must be dense|moe, got: $MODE" >&2
    exit 2
    ;;
esac

if [[ "$MODE" == "dense" && "$SOURCE" == "pt" ]]; then
  echo "WARN: PT_qwen3_8B/32B 均含 NUM_EXPERTS=128（MoE）。dense 请用 SOURCE=wrapper（默认）。" >&2
fi

pod_for_rank() {
  local r="$1"
  if [[ "$r" -eq 0 ]]; then echo "${CLUSTER_JOB}-master-0"; else echo "${CLUSTER_JOB}-worker-$((r-1))"; fi
}

echo "==> MODE=$MODE SOURCE=$SOURCE SCALES=$SCALES ITERS=$TRAIN_ITERS NPUS_PER_NODE=$NPUS_PER_NODE"
echo "==> TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP MBS=$MBS GBS=$GBS SEQ=$SEQ_LENGTH"
echo "==> MEGATRON_ROOT=$MEGATRON_ROOT"
echo "==> RUN_ROOT=$RUN_ROOT LOG_DIR=$LOG_DIR"
echo "==> MASTER=${MASTER_ADDR}:${MASTER_PORT}"

# ---------- 上传 wrappers 到 AFS ----------
echo "==> 上传 wrappers → $AFS_WRAPPERS"
cluster_pod_exec "${CLUSTER_JOB}-master-0" "mkdir -p '$AFS_WRAPPERS'"
for f in train_qwen3_8B_ascend.sh train_qwen3_30B_A3B_ascend.sh patch_pt_train.sh; do
  ssh -o BatchMode=yes -o ConnectTimeout=30 "$CLUSTER_SSH_HOST" \
    "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'cat > ${AFS_WRAPPERS}/${f} && chmod +x ${AFS_WRAPPERS}/${f}'" \
    < "$SCRIPT_DIR/wrappers/$f"
done

# ---------- 远程单 rank launcher ----------
RUNNER_LOCAL="$LOG_DIR/remote_train_one.sh"
cat > "$RUNNER_LOCAL" <<'REMOTE'
#!/usr/bin/env bash
# 勿 source set_env.sh（其可能直接 exit）。用 bash -lc 跑 wrapper。
set -uo pipefail
MEGATRON_ROOT="$1"
SOURCE="$2"
MODE="$3"
BASE_REL="$4"
RUN_DIR="$5"
RANK="$6"
NNODES="$7"
MASTER_ADDR="$8"
MASTER_PORT="$9"
NPUS="${10}"
ITERS="${11}"
DATA_ROOT="${12}"
AFS_WRAPPERS="${13}"
SKIP_SAVE="${14}"
SKIP_PROFILE="${15}"

export PATH=/root/miniconda3/envs/llm_test/bin:${PATH:-}
export PYTHONPATH=/MindSpeed-LLM/MindSpeed:${PYTHONPATH:-}
export WORLD_SIZE="$NNODES"
export NNODES RANK
export NODE_RANK="$RANK"
export MASTER_ADDR MASTER_PORT
export NPUS_PER_NODE="$NPUS"
export GPUS_PER_NODE="$NPUS"
export DATA_ROOT RUN_DIR
export LOG_DIR="${RUN_DIR}/"
export TENSORBOARD_DIR="${RUN_DIR}/tb"
export CKPT_SAVE_DIR="${RUN_DIR}/ckpt"
export TRAIN_ITERS="$ITERS"
export SKIP_SAVE SKIP_PROFILE
export SKIP_TB="${SKIP_TB:-1}"
export TP="${TP:-2}" PP="${PP:-2}" CP="${CP:-1}" EP="${EP:-1}" ETP="${ETP:-1}"
export MBS="${MBS:-1}" GBS="${GBS:-128}" SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export HCCL_IF_BASE_PORT=$((MASTER_PORT + 2000))

mkdir -p "$RUN_DIR" "$LOG_DIR" "$TENSORBOARD_DIR" "$CKPT_SAVE_DIR"
cd "$MEGATRON_ROOT"

if [[ "$SOURCE" == "wrapper" ]]; then
  WRAP="$AFS_WRAPPERS/$BASE_REL"
  test -f "$WRAP"
  echo "[remote] wrapper=$WRAP TP=$TP PP=$PP MBS=$MBS GBS=$GBS"
  # bash -lc：登录壳带上 Ascend 环境，且隔离 set_env 的 exit
  bash -lc "export PATH=/root/miniconda3/envs/llm_test/bin:\$PATH
export PYTHONPATH=/MindSpeed-LLM/MindSpeed:\$PYTHONPATH
export WORLD_SIZE=$NNODES NNODES=$NNODES RANK=$RANK NODE_RANK=$RANK
export MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT
export NPUS_PER_NODE=$NPUS GPUS_PER_NODE=$NPUS
export DATA_ROOT=$DATA_ROOT RUN_DIR=$RUN_DIR LOG_DIR=$RUN_DIR/
export TENSORBOARD_DIR=$RUN_DIR/tb CKPT_SAVE_DIR=$RUN_DIR/ckpt
export TRAIN_ITERS=$ITERS SKIP_SAVE=$SKIP_SAVE SKIP_PROFILE=$SKIP_PROFILE SKIP_TB=\${SKIP_TB:-1}
export TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP MBS=$MBS GBS=$GBS SEQ_LENGTH=$SEQ_LENGTH
export HCCL_IF_BASE_PORT=\$((MASTER_PORT+2000))
cd $MEGATRON_ROOT
bash $WRAP" 2>&1 | tee "${RUN_DIR}/rank${RANK}.log"
  rc=${PIPESTATUS[0]}
else
  SRC="$MEGATRON_ROOT/$BASE_REL"
  OUT_SH="$RUN_DIR/run_rank${RANK}.sh"
  bash "$AFS_WRAPPERS/patch_pt_train.sh" \
    "$SRC" "$OUT_SH" \
    "$DATA_ROOT" "$ITERS" "$NPUS" "$NNODES" "$RANK" \
    "$MASTER_ADDR" "$MASTER_PORT" "$RUN_DIR"
  bash -lc "bash $OUT_SH" 2>&1 | tee -a "${RUN_DIR}/rank${RANK}.log"
  rc=${PIPESTATUS[0]}
fi
echo "TRAIN_RANK_${RANK}_DONE rc=$rc"
exit "$rc"
REMOTE

ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'mkdir -p $RUN_ROOT && cat > $RUN_ROOT/remote_train_one.sh && chmod +x $RUN_ROOT/remote_train_one.sh'" \
  < "$RUNNER_LOCAL"

if [[ "$SOURCE" == "wrapper" ]]; then
  BASE_ARG="$WRAPPER_NAME"
else
  BASE_ARG="$PT_REL"
fi

run_scale() {
  local world_npu="$1"
  if (( world_npu % NPUS_PER_NODE != 0 )); then
    echo "FAIL scale=$world_npu not divisible by NPUS_PER_NODE=$NPUS_PER_NODE"
    return 1
  fi
  local nnodes=$((world_npu / NPUS_PER_NODE))
  local scale_dir="$RUN_ROOT/scale_${world_npu}"
  echo "==> scale=$world_npu nnodes=$nnodes WORLD_SIZE(nnodes)=$nnodes"
  cluster_pod_exec "${CLUSTER_JOB}-master-0" "mkdir -p '$scale_dir'"

  # 每 rank：写 launcher + 用 python3 - subprocess 启动（避开 vcctl 长参/Usage）
  local r=0
  while [[ "$r" -lt "$nnodes" ]]; do
    local launch="$LOG_DIR/launch_scale${world_npu}_rank${r}.sh"
    cat > "$launch" <<EOF
#!/usr/bin/env bash
export TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP MBS=$MBS GBS=$GBS SEQ_LENGTH=$SEQ_LENGTH SKIP_TB=$SKIP_TB
exec bash $RUN_ROOT/remote_train_one.sh \\
  $MEGATRON_ROOT $SOURCE $MODE $BASE_ARG \\
  $scale_dir $r $nnodes $MASTER_ADDR $MASTER_PORT \\
  $NPUS_PER_NODE $TRAIN_ITERS $DATA_ROOT $AFS_WRAPPERS \\
  $SKIP_SAVE $SKIP_PROFILE
EOF
    ssh -o BatchMode=yes -o ConnectTimeout=30 "$CLUSTER_SSH_HOST" \
      "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'cat > ${scale_dir}/launch_rank${r}.sh && chmod +x ${scale_dir}/launch_rank${r}.sh'" \
      < "$launch"
    r=$((r + 1))
  done

  local pids=() r=0
  while [[ "$r" -lt "$nnodes" ]]; do
    local pod logf
    pod="$(pod_for_rank "$r")"
    logf="$LOG_DIR/scale${world_npu}_rank${r}.log"
    local py_boot="$LOG_DIR/pyboot_scale${world_npu}_rank${r}.py"
    cat > "$py_boot" <<EOF
import subprocess, sys
print("PY_START", flush=True)
p = subprocess.run(["bash", "${scale_dir}/launch_rank${r}.sh"])
print("PY_EXIT", p.returncode, flush=True)
sys.exit(p.returncode)
EOF
    ssh -o BatchMode=yes -o ConnectTimeout=30 "$CLUSTER_SSH_HOST" \
      "vcctl pod exec -i ${pod} -- python3 -" \
      < "$py_boot" >"$logf" 2>&1 &
    pids+=("$!")
    # 错开启动：8 节点时跳板更容易被打挂
    sleep 8
    r=$((r + 1))
  done
  local fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  # 拉回 rank 日志关键行
  cluster_pod_exec "${CLUSTER_JOB}-master-0" \
    "grep -hEi 'mfu|throughput|tokens/s|tokens per|elapsed time per iteration|TFLOP|iteration|DONE|Error|error|OOM' '$scale_dir'/*.log 2>/dev/null | tail -80 || true" \
    | tee "$LOG_DIR/scale${world_npu}_metrics.txt"
  # 也拷贝 AFS rank 日志到本地便于 parse
  ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'tar -C $scale_dir -cf - .' " \
    > "$LOG_DIR/scale${world_npu}_afs.tar" 2>/dev/null || true
  mkdir -p "$LOG_DIR/scale${world_npu}_afs"
  tar -xf "$LOG_DIR/scale${world_npu}_afs.tar" -C "$LOG_DIR/scale${world_npu}_afs" 2>/dev/null || true
  if grep -qiE 'out of memory|ChildFailedError|NPU out of memory' \
      "$LOG_DIR/scale${world_npu}_rank"*.log \
      "$LOG_DIR/scale${world_npu}_afs"/*.log 2>/dev/null; then
    fail=1
  fi
  [[ "$fail" -eq 0 ]] || { echo "FAIL scale=$world_npu"; return 1; }
  echo "OK scale=$world_npu"
}

IFS=',' read -ra ARR <<< "$SCALES"
for s in "${ARR[@]}"; do
  run_scale "$s" || true
  MASTER_PORT=$((MASTER_PORT + 1))
done

ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'tar -C $RUN_ROOT -cf - .' " \
  > "$LOG_DIR/results.tar" || true
mkdir -p "$LOG_DIR/results"
tar -xf "$LOG_DIR/results.tar" -C "$LOG_DIR/results" 2>/dev/null || true
echo "TRAIN_MFU_DONE mode=$MODE source=$SOURCE → $RUN_ROOT / $LOG_DIR"
