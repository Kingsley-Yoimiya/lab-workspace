#!/usr/bin/env bash
# Muxi Dense 多机 fire（对标 fire_train_tiny_muxi，支持 NNODES/TP/PP/GBS）
# 用法:
#   AFS_OUT=... WORLD=8 TP=4 PP=2 GBS=2048 ./fire_train_dense_muxi.sh
#   AFS_OUT=... WORLD=16 TP=4 PP=2 HOSTSET=clean ./fire_train_dense_muxi.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

AFS_OUT="${AFS_OUT:?set AFS_OUT}"
WORLD="${WORLD:-8}"
NPROC="${DEVICES_PER_NODE:-8}"
nnodes=$((WORLD / NPROC))
TP="${TP:-4}"
PP="${PP:-2}"
GBS="${GBS:-2048}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
MASTER_PORT="${MASTER_PORT:-30211}"
HOSTSET="${HOSTSET:-full}"  # full|clean
NCCL_IB_HCA="${NCCL_IB_HCA:-xscale}"
MCCL_IB_HCA="${MCCL_IB_HCA:-xscale}"
AFS_WRAPPERS="/afs-a3-weight-share/montyyin/lab-workspace/scripts/cluster/wrappers"
LOG_DIR="${LOG_DIR:-/Users/yinjinrun/random-thing/logs/muxi-train-dense-fire}"
mkdir -p "$LOG_DIR"

HOSTFILE="$SCRIPT_DIR/hosts_${HOSTSET}.txt"
if [[ ! -f "$HOSTFILE" ]]; then
  echo "FAIL: missing $HOSTFILE" >&2
  exit 2
fi
mapfile -t HOST_LOGIC < <(grep -v '^[[:space:]]*$' "$HOSTFILE" | grep -v '^#')
POD_NODES=()
for h in "${HOST_LOGIC[@]}"; do
  POD_NODES+=("${CLUSTER_JOB}-${h}")
done
if [[ "$nnodes" -gt "${#POD_NODES[@]}" ]]; then
  echo "FAIL: need $nnodes nodes for world=$WORLD hostset=$HOSTSET (have ${#POD_NODES[@]})" >&2
  exit 2
fi
MASTER_ADDR="${MASTER_ADDR:-${POD_NODES[0]}.${CLUSTER_JOB}}"

echo "FIRE dense WORLD=$WORLD nnodes=$nnodes TP=$TP PP=$PP GBS=$GBS HOSTSET=$HOSTSET IB_HCA=$NCCL_IB_HCA"
echo "  AFS_OUT=$AFS_OUT MASTER=$MASTER_ADDR:$MASTER_PORT"

cluster_pod_exec "${POD_NODES[0]}" "mkdir -p '$AFS_WRAPPERS' '$AFS_OUT'"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${POD_NODES[0]} -- bash -c 'cat > ${AFS_WRAPPERS}/train_gpt_dense_muxi.sh && chmod +x ${AFS_WRAPPERS}/train_gpt_dense_muxi.sh && wc -c ${AFS_WRAPPERS}/train_gpt_dense_muxi.sh'" \
  < "$SCRIPT_DIR/wrappers/train_gpt_dense_muxi.sh" \
  >"$LOG_DIR/upload.log" 2>&1

r=0
while [[ "$r" -lt "$nnodes" ]]; do
  pod="${POD_NODES[$r]}"
  run_local="/tmp/run_train_dense_w${WORLD}_n${r}.sh"
  donef="$AFS_OUT/node_${r}.done"
  failf="$AFS_OUT/node_${r}.fail"
  run_body=$(cat <<EOF
#!/usr/bin/env bash
set -uo pipefail
export PATH="/opt/conda/bin:\${PATH:-/usr/bin}"
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0
export MCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=$NCCL_IB_HCA
export MCCL_IB_HCA=$MCCL_IB_HCA
export NCCL_DEBUG=\${NCCL_DEBUG:-WARN}
export MCCL_DEBUG=\${MCCL_DEBUG:-WARN}
CU_BRIDGE_BIN=/opt/maca/tools/cu-bridge/bin
if [[ -x "\$CU_BRIDGE_BIN/cucc" && ! -e "\$CU_BRIDGE_BIN/nvcc" ]]; then
  ln -sfn "\$CU_BRIDGE_BIN/cucc" "\$CU_BRIDGE_BIN/nvcc" || true
fi
export CUDA_HOME=/opt/maca/tools/cu-bridge
rm -f '$donef' '$failf'
export RUN_DIR='$AFS_OUT'
export NNODES=$nnodes NODE_RANK=$r
export MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT
export GPUS_PER_NODE=$NPROC TRAIN_ITERS=$TRAIN_ITERS
export TP=$TP PP=$PP GBS=$GBS SEQ_LENGTH=$SEQ_LENGTH MBS=1
bash '$AFS_WRAPPERS/train_gpt_dense_muxi.sh' >'$AFS_OUT/node_${r}.outer.log' 2>&1
ec=\$?
if [[ \$ec -eq 0 ]] && grep -q TRAIN_DENSE_MUXI_DONE '$AFS_OUT/train.log' 2>/dev/null; then
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
  r=$((r + 1))
  sleep 1
done
echo "FIRE_DONE dense WORLD=$WORLD AFS_OUT=$AFS_OUT"
