#!/usr/bin/env bash
# Muxi G9：单机 8 卡 tiny GPT 真训练冒烟（mock-data）
# 用法: AFS_OUT=... ./fire_train_tiny_muxi.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

AFS_OUT="${AFS_OUT:?set AFS_OUT}"
AFS_WRAPPERS="/afs-a3-weight-share/montyyin/lab-workspace/scripts/cluster/wrappers"
MASTER_PORT="${MASTER_PORT:-30111}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
GPUS="${GPUS_PER_NODE:-8}"
POD="${CLUSTER_POD}"
LOG_DIR="${LOG_DIR:-/Users/yinjinrun/random-thing/logs/muxi-train-tiny-fire}"
mkdir -p "$LOG_DIR"

echo "FIRE train-tiny AFS_OUT=$AFS_OUT port=$MASTER_PORT iters=$TRAIN_ITERS gpus=$GPUS"

cluster_pod_exec "$POD" "mkdir -p '$AFS_WRAPPERS' '$AFS_OUT'"
# 先上传 wrapper（与启动分离，避免 stdin 空文件）
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${POD} -- bash -c 'cat > ${AFS_WRAPPERS}/train_gpt_tiny_muxi.sh && chmod +x ${AFS_WRAPPERS}/train_gpt_tiny_muxi.sh && wc -c ${AFS_WRAPPERS}/train_gpt_tiny_muxi.sh'" \
  < "$SCRIPT_DIR/wrappers/train_gpt_tiny_muxi.sh" \
  >"$LOG_DIR/upload.log" 2>&1
echo "  uploaded: $(tr '\n' ' ' < "$LOG_DIR/upload.log")"

run_local="/tmp/run_train_tiny_muxi.sh"
run_body=$(cat <<EOF
#!/usr/bin/env bash
set -uo pipefail
export PATH="/opt/conda/bin:\${PATH:-/usr/bin}"
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0
export MCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
# nvcc shim（pod 重启后可能丢失）
CU_BRIDGE_BIN=/opt/maca/tools/cu-bridge/bin
if [[ -x "\$CU_BRIDGE_BIN/cucc" && ! -e "\$CU_BRIDGE_BIN/nvcc" ]]; then
  ln -sfn "\$CU_BRIDGE_BIN/cucc" "\$CU_BRIDGE_BIN/nvcc" || true
fi
export CUDA_HOME=/opt/maca/tools/cu-bridge
rm -f '$AFS_OUT/train.done' '$AFS_OUT/train.fail'
export RUN_DIR='$AFS_OUT'
export NNODES=1 NODE_RANK=0
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$MASTER_PORT
export GPUS_PER_NODE=$GPUS TRAIN_ITERS=$TRAIN_ITERS
bash '$AFS_WRAPPERS/train_gpt_tiny_muxi.sh' >'$AFS_OUT/train.outer.log' 2>&1
ec=\$?
if [[ \$ec -eq 0 ]] && grep -q TRAIN_TINY_MUXI_DONE '$AFS_OUT/train.log' 2>/dev/null; then
  echo OK >'$AFS_OUT/train.done'
else
  echo FAIL >'$AFS_OUT/train.fail'
  # torchrun 失败时 wrapper 的 tee 仍可能有部分日志
  [[ -f '$AFS_OUT/train.log' ]] || cp -f '$AFS_OUT/train.outer.log' '$AFS_OUT/train.log' 2>/dev/null || true
fi
exit \$ec
EOF
)

printf '%s\n' "$run_body" | ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${POD} -- bash -c \"cat > $run_local && chmod +x $run_local && wc -c $run_local\"" \
  >"$LOG_DIR/fire.log" 2>&1

ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec ${POD} -- bash -c \"setsid nohup bash $run_local </dev/null >/dev/null 2>&1 & echo STARTED \\\$!; sleep 2; pgrep -af 'train_gpt_tiny|pretrain_gpt|torchrun' | head -5\"" \
  >>"$LOG_DIR/fire.log" 2>&1

echo "  fired: $(tr '\n' ' ' < "$LOG_DIR/fire.log")"
echo "FIRE_DONE train-tiny AFS_OUT=$AFS_OUT"
