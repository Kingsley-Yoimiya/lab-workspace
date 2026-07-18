#!/usr/bin/env bash
# Muxi NCCL P2P fire（对标 fire_nccl_scale_muxi；bench=nccl_p2p_bench.py）
# 多节点默认有界并行（CLUSTER_FANOUT_PARALLEL）；禁止改回 for+sleep 串行。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

WORLD="${1:?usage: fire_nccl_p2p_muxi.sh <world> [master_port]}"
MASTER_PORT="${2:-${MASTER_PORT:-29631}}"
NPROC="${DEVICES_PER_NODE:-8}"
nnodes=$((WORLD / NPROC))
AFS_OUT="${AFS_OUT:?set AFS_OUT}"
AFS_SCRIPTS="/afs-a3-weight-share/yinjinrun.p/lab-workspace/scripts/cluster"
SIZES="${SIZES:-64K,16M}"
STRATEGIES="${STRATEGIES:-}"
MASTER_ADDR="${MASTER_ADDR:-${CLUSTER_JOB}-master-0.${CLUSTER_JOB}}"
LOG_DIR="${LOG_DIR:-/Users/yinjinrun/random-thing/logs/muxi-nccl-fire}"
PARALLEL="${CLUSTER_FANOUT_PARALLEL:-16}"
mkdir -p "$LOG_DIR"

POD_NODES=("${CLUSTER_JOB}-master-0")
for i in $(seq 0 "$((CLUSTER_N_WORKERS - 1))"); do
  POD_NODES+=("${CLUSTER_JOB}-worker-${i}")
done

if [[ $((WORLD % NPROC)) -ne 0 ]] || [[ "$nnodes" -gt "${#POD_NODES[@]}" ]]; then
  echo "bad world=$WORLD"; exit 1
fi

out="$AFS_OUT/p2p_${WORLD}.jsonl"
echo "FIRE p2p scale=$WORLD nnodes=$nnodes port=$MASTER_PORT parallel=$PARALLEL out=$out"

cluster_pod_exec "${CLUSTER_POD}" "mkdir -p '$AFS_SCRIPTS' '$AFS_OUT'"
cluster_pod_exec_i "${CLUSTER_POD}" "cat > $AFS_SCRIPTS/nccl_p2p_bench.py" \
  < "$SCRIPT_DIR/nccl_p2p_bench.py"

fire_one_rank() {
  local r="$1"
  local pod="${POD_NODES[$r]}"
  local donef="$AFS_OUT/p2p_${WORLD}.node_${r}.done"
  local failf="$AFS_OUT/p2p_${WORLD}.node_${r}.fail"
  local rlog="$AFS_OUT/p2p_${WORLD}.node_${r}.log"
  local run_local="/tmp/run_nccl_p2p_${WORLD}_node_${r}.sh"
  local flog="$LOG_DIR/p2p${WORLD}_noderank${r}.fire.log"

  local run_body
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

  if ! printf '%s\n' "$run_body" | cluster_pod_exec_i "$pod" \
    "cat > $run_local && chmod +x $run_local && wc -c $run_local" \
    >"$flog" 2>&1; then
    echo "FAIL_UPLOAD node_rank=$r pod=$pod"
    return 1
  fi
  if ! cluster_pod_exec "$pod" \
    "setsid nohup bash $run_local </dev/null >/dev/null 2>&1 & echo STARTED \$!; sleep 1; pgrep -af torchrun | head -3" \
    >>"$flog" 2>&1; then
    echo "FAIL_START node_rank=$r pod=$pod"
    return 1
  fi
  echo "  fired node_rank=$r pod=$pod"
  return 0
}

FAIL_RANKS=()
ACTIVE=0
PIDS=()
RANK_OF_PID=()
r=0
while [[ "$r" -lt "$nnodes" ]]; do
  while [[ "$ACTIVE" -ge "$PARALLEL" ]]; do
    for i in "${!PIDS[@]}"; do
      if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
        rc=0
        wait "${PIDS[$i]}" || rc=$?
        if [[ "$rc" -ne 0 ]]; then
          FAIL_RANKS+=("${RANK_OF_PID[$i]}")
        fi
        unset "PIDS[$i]"
        unset "RANK_OF_PID[$i]"
        ACTIVE=$((ACTIVE - 1))
      fi
    done
    if [[ ${#PIDS[@]} -gt 0 ]]; then
      PIDS=("${PIDS[@]}")
      RANK_OF_PID=("${RANK_OF_PID[@]}")
    else
      PIDS=()
      RANK_OF_PID=()
    fi
    [[ "$ACTIVE" -ge "$PARALLEL" ]] && sleep 0.3
  done
  fire_one_rank "$r" &
  PIDS+=("$!")
  RANK_OF_PID+=("$r")
  ACTIVE=$((ACTIVE + 1))
  r=$((r + 1))
done

for i in "${!PIDS[@]}"; do
  rc=0
  wait "${PIDS[$i]}" || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    FAIL_RANKS+=("${RANK_OF_PID[$i]}")
  fi
done

if [[ ${#FAIL_RANKS[@]} -gt 0 ]]; then
  echo "FIRE_FAIL ranks=${FAIL_RANKS[*]} (ok=$((nnodes - ${#FAIL_RANKS[@]}))/$nnodes parallel=$PARALLEL)"
  printf '%s\n' "${FAIL_RANKS[@]}" >"$LOG_DIR/p2p${WORLD}_fail_ranks.txt"
  exit 1
fi
echo "FIRE_DONE scale=$WORLD nnodes=$nnodes parallel=$PARALLEL"
