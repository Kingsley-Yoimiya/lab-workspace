#!/usr/bin/env bash
# Muxi MoE 多机 fire（缩小版 EP）
# 用法: AFS_OUT=... WORLD=8 EP=8 TP=1 PP=1 ./fire_train_moe_muxi.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

AFS_OUT="${AFS_OUT:?set AFS_OUT}"
WORLD="${WORLD:-8}"
NPROC="${DEVICES_PER_NODE:-8}"
nnodes=$((WORLD / NPROC))
TP="${TP:-1}"
PP="${PP:-1}"
EP="${EP:-8}"
GBS="${GBS:-64}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
MOE_TOPK="${MOE_TOPK:-2}"
MASTER_PORT="${MASTER_PORT:-30311}"
HOSTSET="${HOSTSET:-clean}"
NCCL_IB_HCA="${NCCL_IB_HCA:-xscale}"
MCCL_IB_HCA="${MCCL_IB_HCA:-xscale}"
AFS_WRAPPERS="/afs-a3-weight-share/yinjinrun.p/lab-workspace/scripts/cluster/wrappers"
LOG_DIR="${LOG_DIR:-/Users/yinjinrun/random-thing/logs/muxi-train-moe-fire}"
mkdir -p "$LOG_DIR"

HOSTFILE="$SCRIPT_DIR/hosts_${HOSTSET}.txt"
mapfile -t HOST_LOGIC < <(grep -v '^[[:space:]]*$' "$HOSTFILE" | grep -v '^#')
POD_NODES=()
for h in "${HOST_LOGIC[@]}"; do POD_NODES+=("${CLUSTER_JOB}-${h}"); done
[[ "$nnodes" -le "${#POD_NODES[@]}" ]] || { echo "FAIL nnodes"; exit 2; }
MASTER_ADDR="${MASTER_ADDR:-${POD_NODES[0]}.${CLUSTER_JOB}}"

echo "FIRE moe WORLD=$WORLD TP/PP/EP=$TP/$PP/$EP experts=$NUM_EXPERTS topk=$MOE_TOPK HOSTSET=$HOSTSET"

cluster_pod_exec "${POD_NODES[0]}" "mkdir -p '$AFS_WRAPPERS' '$AFS_OUT'"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${POD_NODES[0]} -- bash -c 'cat > ${AFS_WRAPPERS}/train_gpt_moe_muxi.sh && chmod +x ${AFS_WRAPPERS}/train_gpt_moe_muxi.sh && wc -c ${AFS_WRAPPERS}/train_gpt_moe_muxi.sh'" \
  < "$SCRIPT_DIR/wrappers/train_gpt_moe_muxi.sh" \
  >"$LOG_DIR/upload.log" 2>&1

r=0
while [[ "$r" -lt "$nnodes" ]]; do
  pod="${POD_NODES[$r]}"
  run_local="/tmp/run_train_moe_w${WORLD}_n${r}.sh"
  donef="$AFS_OUT/node_${r}.done"
  failf="$AFS_OUT/node_${r}.fail"
  run_body=$(cat <<EOF
#!/usr/bin/env bash
set -uo pipefail
export PATH="/opt/conda/bin:\${PATH:-/usr/bin}"
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0 MCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=$NCCL_IB_HCA MCCL_IB_HCA=$MCCL_IB_HCA
CU_BRIDGE_BIN=/opt/maca/tools/cu-bridge/bin
[[ -x "\$CU_BRIDGE_BIN/cucc" && ! -e "\$CU_BRIDGE_BIN/nvcc" ]] && ln -sfn "\$CU_BRIDGE_BIN/cucc" "\$CU_BRIDGE_BIN/nvcc" || true
export CUDA_HOME=/opt/maca/tools/cu-bridge
rm -f '$donef' '$failf'
export RUN_DIR='$AFS_OUT' NNODES=$nnodes NODE_RANK=$r
export MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT
export GPUS_PER_NODE=$NPROC TRAIN_ITERS=$TRAIN_ITERS
export TP=$TP PP=$PP EP=$EP GBS=$GBS SEQ_LENGTH=$SEQ_LENGTH MBS=1
export NUM_EXPERTS=$NUM_EXPERTS MOE_TOPK=$MOE_TOPK
bash '$AFS_WRAPPERS/train_gpt_moe_muxi.sh' >'$AFS_OUT/node_${r}.outer.log' 2>&1
ec=\$?
if [[ \$ec -eq 0 ]] && grep -q TRAIN_MOE_MUXI_DONE '$AFS_OUT/train.log' 2>/dev/null; then
  echo OK >'$donef'
else
  echo FAIL >'$failf'
  [[ -f '$AFS_OUT/train.log' ]] || cp -f '$AFS_OUT/node_${r}.outer.log' '$AFS_OUT/train.log' 2>/dev/null || true
fi
exit \$ec
EOF
)
  printf '%s\n' "$run_body" | ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec -i ${pod} -- bash -c \"cat > $run_local && chmod +x $run_local && wc -c $run_local\"" \
    >"$LOG_DIR/w${WORLD}_n${r}.fire.log" 2>&1
  ssh -o BatchMode=yes -o ConnectTimeout=30 "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec ${pod} -- bash -c \"setsid nohup bash $run_local </dev/null >/dev/null 2>&1 & echo STARTED \\\$!\"" \
    >>"$LOG_DIR/w${WORLD}_n${r}.fire.log" 2>&1
  echo "  fired rank=$r pod=$pod"
  r=$((r + 1)); sleep 1
done
echo "FIRE_DONE moe WORLD=$WORLD AFS_OUT=$AFS_OUT"
