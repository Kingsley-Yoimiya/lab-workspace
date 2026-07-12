#!/usr/bin/env bash
# Muxi MFU 微基准 fire（dense/moe；对标 run_mfu_bench_scale.sh，8 卡/节点）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

WORLD="${1:?usage: fire_mfu_bench_muxi.sh <world> [master_port]}"
MASTER_PORT="${2:-${MASTER_PORT:-30001}}"
MODE="${MODE:-dense}"
ITERS="${ITERS:-15}"
PEAK="${PEAK_TFLOPS:-279}"
NPROC="${DEVICES_PER_NODE:-8}"
nnodes=$((WORLD / NPROC))
AFS_OUT="${AFS_OUT:?set AFS_OUT}"
AFS_SCRIPTS="/afs-a3-weight-share/montyyin/lab-workspace/scripts/cluster"
MASTER_ADDR="${MASTER_ADDR:-${CLUSTER_JOB}-master-0.${CLUSTER_JOB}}"
LOG_DIR="${LOG_DIR:-/Users/yinjinrun/random-thing/logs/muxi-mfu-fire}"
mkdir -p "$LOG_DIR"

POD_NODES=("${CLUSTER_JOB}-master-0")
for i in $(seq 0 "$((CLUSTER_N_WORKERS - 1))"); do
  POD_NODES+=("${CLUSTER_JOB}-worker-${i}")
done

if [[ $((WORLD % NPROC)) -ne 0 ]] || [[ "$nnodes" -gt "${#POD_NODES[@]}" ]]; then
  echo "bad world=$WORLD"; exit 1
fi

out="$AFS_OUT/mfu_${MODE}_${WORLD}.jsonl"
echo "FIRE mfu mode=$MODE scale=$WORLD nnodes=$nnodes port=$MASTER_PORT peak=$PEAK"

cluster_pod_exec "${CLUSTER_POD}" "mkdir -p '$AFS_SCRIPTS' '$AFS_OUT'"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${CLUSTER_POD} -- bash -c 'cat > $AFS_SCRIPTS/mfu_train_bench_nccl.py'" \
  < "$SCRIPT_DIR/mfu_train_bench_nccl.py"

r=0
while [[ "$r" -lt "$nnodes" ]]; do
  pod="${POD_NODES[$r]}"
  donef="$AFS_OUT/mfu_${MODE}_${WORLD}.node_${r}.done"
  failf="$AFS_OUT/mfu_${MODE}_${WORLD}.node_${r}.fail"
  rlog="$AFS_OUT/mfu_${MODE}_${WORLD}.node_${r}.log"
  run_local="/tmp/run_mfu_${MODE}_${WORLD}_node_${r}.sh"
  run_body=$(cat <<EOF
#!/usr/bin/env bash
export PATH="/opt/conda/bin:\${PATH:-/usr/bin}"
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0
export MCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA="${NCCL_IB_HCA:-xscale}"
export MCCL_IB_HCA="${MCCL_IB_HCA:-xscale}"
rm -f '$donef' '$failf'
cp -f '$AFS_SCRIPTS/mfu_train_bench_nccl.py' /tmp/mfu_train_bench_nccl.py
/opt/conda/bin/torchrun --nnodes=$nnodes --node_rank=$r --nproc_per_node=$NPROC \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  /tmp/mfu_train_bench_nccl.py --mode '$MODE' --iters $ITERS --peak-tflops $PEAK --out '$out' \
  >'$rlog' 2>&1
ec=\$?
if [[ \$ec -eq 0 ]]; then echo OK >'$donef'; else echo FAIL >'$failf'; fi
exit \$ec
EOF
)
  printf '%s\n' "$run_body" | ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec -i ${pod} -- bash -c \"cat > $run_local && chmod +x $run_local && wc -c $run_local\"" \
    >"$LOG_DIR/mfu${MODE}${WORLD}_noderank${r}.fire.log" 2>&1
  ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec ${pod} -- bash -c \"setsid nohup bash $run_local </dev/null >/dev/null 2>&1 & echo STARTED \\\$!; sleep 2; pgrep -af torchrun | head -2\"" \
    >>"$LOG_DIR/mfu${MODE}${WORLD}_noderank${r}.fire.log" 2>&1
  echo "  fired node_rank=$r -> $(tr '\n' ' ' < "$LOG_DIR/mfu${MODE}${WORLD}_noderank${r}.fire.log")"
  r=$((r + 1))
  sleep 2
done
echo "FIRE_DONE mfu mode=$MODE world=$WORLD"
