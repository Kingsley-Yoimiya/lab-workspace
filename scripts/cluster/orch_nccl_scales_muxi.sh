#!/usr/bin/env bash
# 顺序 fire+poll 多个 NCCL scale（依赖 fire/poll_nccl_scale_muxi.sh）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

AFS_OUT="${AFS_OUT:?set AFS_OUT}"
LOG_DIR="${LOG_DIR:-/Users/yinjinrun/random-thing/logs/muxi-nccl-orch}"
mkdir -p "$LOG_DIR"
SCALES="${SCALES:-16,32,64,128}"
MASTER_PORT="${MASTER_PORT:-29701}"
POLL_SEC="${POLL_SEC:-20}"
POLL_MAX="${POLL_MAX:-180}"

export AFS_OUT LOG_DIR OPS="${OPS:-all_reduce,all_gather,reduce_scatter,broadcast}"
export SIZES="${SIZES:-1M,16M,64M,256M}"

exec >>"$LOG_DIR/orch.log" 2>&1
echo "==== ORCH $(date '+%Y-%m-%dT%H:%M:%S') SCALES=$SCALES ===="

IFS=',' read -ra ARR <<< "$SCALES"
for world in "${ARR[@]}"; do
  echo "==> fire scale=$world port=$MASTER_PORT"
  cluster_pod_exec "${CLUSTER_POD}" \
    "rm -f $AFS_OUT/scale_${world}.node_* $AFS_OUT/scale_${world}.rank* $AFS_OUT/scale_${world}.jsonl 2>/dev/null; true" || true
  bash "$SCRIPT_DIR/fire_nccl_scale_muxi.sh" "$world" "$MASTER_PORT"
  ok=0
  for ((i=0; i<POLL_MAX; i++)); do
    if bash "$SCRIPT_DIR/poll_nccl_scale_muxi.sh" "$world"; then
      ok=1
      break
    fi
    ec=$?
    if [[ "$ec" -eq 2 ]]; then
      echo "FAIL scale=$world"
      break
    fi
    sleep "$POLL_SEC"
  done
  if [[ "$ok" -eq 1 ]]; then
    echo "OK scale=$world"
  else
    echo "GIVEUP scale=$world"
  fi
  MASTER_PORT=$((MASTER_PORT + 1))
done
echo "ORCH_DONE"
