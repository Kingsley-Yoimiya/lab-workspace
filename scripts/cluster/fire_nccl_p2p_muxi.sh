#!/usr/bin/env bash
# Muxi NCCL P2P fire（对标 fire_nccl_scale_muxi；bench=nccl_p2p_bench.py）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

WORLD="${1:?usage: fire_nccl_p2p_muxi.sh <world> [master_port]}"
MASTER_PORT="${2:-${MASTER_PORT:-29631}}"
NPROC="${DEVICES_PER_NODE:-8}"
nnodes=$((WORLD / NPROC))
AFS_OUT="${AFS_OUT:?set AFS_OUT}"
AFS_SCRIPTS="/afs-a3-weight-share/montyyin/lab-workspace/scripts/cluster"
SIZES="${SIZES:-64K,16M}"
STRATEGIES="${STRATEGIES:-}"
MASTER_ADDR="${MASTER_ADDR:-${CLUSTER_JOB}-master-0.${CLUSTER_JOB}}"
LOG_DIR="${LOG_DIR:-/Users/yinjinrun/random-thing/logs/muxi-nccl-fire}"
mkdir -p "$LOG_DIR"

POD_NODES=("${CLUSTER_JOB}-master-0")
for i in $(seq 0 "$((CLUSTER_N_WORKERS - 1))"); do
  POD_NODES+=("${CLUSTER_JOB}-worker-${i}")
done

if [[ $((WORLD % NPROC)) -ne 0 ]] || [[ "$nnodes" -gt "${#POD_NODES[@]}" ]]; then
  echo "bad world=$WORLD"; exit 1
fi

out="$AFS_OUT/p2p_${WORLD}.jsonl"
echo "FIRE p2p scale=$WORLD nnodes=$nnodes port=$MASTER_PORT out=$out"

cluster_pod_exec "${CLUSTER_POD}" "mkdir -p '$AFS_SCRIPTS' '$AFS_OUT'"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${CLUSTER_POD} -- bash -c 'cat > $AFS_SCRIPTS/nccl_p2p_bench.py'" \
  < "$SCRIPT_DIR/nccl_p2p_bench.py"

r=0
while [[ "$r" -lt "$nnodes" ]]; do
  pod="${POD_NODES[$r]}"
  donef="$AFS_OUT/p2p_${WORLD}.node_${r}.done"
  failf="$AFS_OUT/p2p_${WORLD}.node_${r}.fail"
  rlog="$AFS_OUT/p2p_${WORLD}.node_${r}.log"
  run_local="/tmp/run_nccl_p2p_${WORLD}_node_${r}.sh"

  run_body=$(cat <<EOF
#!/usr/bin/env bash
export PATH="/opt/conda/bin:\${PATH:-/usr/bin}"
export PYTHONUNBUFFERED=1
# 沐曦多机：强制 eth0 做 NCCL/MCCL socket；IB/RoCE 用 xscale（verbs 可见；mlx5 无 ibv）
export NCCL_SOCKET_IFNAME=eth0
export MCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA="${NCCL_IB_HCA:-xscale}"
export MCCL_IB_HCA="${MCCL_IB_HCA:-xscale}"
export NCCL_DEBUG=\${NCCL_DEBUG:-WARN}
export MCCL_DEBUG=\${MCCL_DEBUG:-WARN}
export FORCE_ACTIVE_WAIT=\${FORCE_ACTIVE_WAIT:-2}
rm -f '$donef' '$failf'
cp -f '$AFS_SCRIPTS/nccl_p2p_bench.py' /tmp/nccl_p2p_bench.py
/opt/conda/bin/torchrun --nnodes=$nnodes --node_rank=$r --nproc_per_node=$NPROC \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  /tmp/nccl_p2p_bench.py --sizes '$SIZES' --strategies '$STRATEGIES' --out '$out' \
  >'$rlog' 2>&1
ec=\$?
if [[ \$ec -eq 0 ]]; then echo OK >'$donef'; else echo FAIL >'$failf'; fi
exit \$ec
EOF
)

  # 分两步：先 stdin 写 /tmp 脚本，再 setsid 启动（避免复合命令吃掉 stdin）
  printf '%s\n' "$run_body" | ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec -i ${pod} -- bash -c \"cat > $run_local && chmod +x $run_local && wc -c $run_local\"" \
    >"$LOG_DIR/p2p${WORLD}_noderank${r}.fire.log" 2>&1
  ssh -o BatchMode=yes -o ConnectTimeout=30 "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec ${pod} -- bash -c \"setsid nohup bash $run_local </dev/null >/dev/null 2>&1 & echo STARTED \\\$!; sleep 2; pgrep -af torchrun | head -3\"" \
    >>"$LOG_DIR/p2p${WORLD}_noderank${r}.fire.log" 2>&1
  echo "  fired node_rank=$r pod=$pod -> $(tr '\n' ' ' < "$LOG_DIR/p2p${WORLD}_noderank${r}.fire.log")"
  r=$((r + 1))
  sleep 2
done
echo "FIRE_DONE scale=$WORLD"
