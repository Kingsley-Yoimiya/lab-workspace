#!/usr/bin/env bash
# Muxi：体质 → 拓扑 → NCCL collective → NCCL P2P（对标 run_constitution_then_comm.sh）
#
# 用法:
#   ./scripts/cluster/run_constitution_then_comm_muxi.sh
#   SKIP_CONSTITUTION=1 SKIP_TOPO=1 ./scripts/cluster/run_constitution_then_comm_muxi.sh  # 只通信
#   SKIP_COMM=1 ./scripts/cluster/run_constitution_then_comm_muxi.sh
#
# 通信步用 fire+poll（pod 内 setsid nohup），不依赖本机长 SSH。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=muxi.env
source "$SCRIPT_DIR/muxi.env"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PIPE_LOG_DIR="${PIPE_LOG_DIR:-/Users/yinjinrun/random-thing/logs/pipeline-muxi-${STAMP}}"
mkdir -p "$PIPE_LOG_DIR"
exec >>"$PIPE_LOG_DIR/pipeline.log" 2>&1
echo "==== PIPELINE $(date '+%Y-%m-%dT%H:%M:%S') ===="

SKIP_CONSTITUTION="${SKIP_CONSTITUTION:-0}"
SKIP_PLOT="${SKIP_PLOT:-0}"
SKIP_TOPO="${SKIP_TOPO:-0}"
SKIP_COMM="${SKIP_COMM:-0}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"

NCCL_SCALES="${NCCL_SCALES:-8,16,32,64,128}"
P2P_SCALES="${P2P_SCALES:-16,128}"
POLL_SEC="${POLL_SEC:-20}"
POLL_MAX="${POLL_MAX:-180}"

run_step() {
  local name="$1"
  shift
  echo ""
  echo "======== STEP: $name ========"
  echo "CMD: $*"
  local rc=0
  "$@" || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "WARN step_failed name=$name rc=$rc" | tee -a "$PIPE_LOG_DIR/failures.txt"
    if [[ "$CONTINUE_ON_FAIL" != "1" ]]; then
      echo "ABORT pipeline"
      exit "$rc"
    fi
  else
    echo "OK step=$name"
  fi
}

wait_poll() {
  local kind="$1" world="$2" # scale|p2p
  local i=0
  while [[ "$i" -lt "$POLL_MAX" ]]; do
    if [[ "$kind" == "p2p" ]]; then
      if bash "$SCRIPT_DIR/poll_nccl_p2p_muxi.sh" "$world"; then return 0; fi
      local ec=$?
    else
      if bash "$SCRIPT_DIR/poll_nccl_scale_muxi.sh" "$world"; then return 0; fi
      local ec=$?
    fi
    if [[ "${ec:-1}" -eq 2 ]]; then return 2; fi
    sleep "$POLL_SEC"
    i=$((i + 1))
  done
  return 1
}

echo "==> PIPE_LOG_DIR=$PIPE_LOG_DIR PROFILE=muxi JOB=$CLUSTER_JOB"
echo "==> list: constitution -> plot -> muxi_topo -> nccl_scale -> nccl_p2p"

# --- 1) 体质 ---
if [[ "$SKIP_CONSTITUTION" != "1" ]]; then
  export LOG_DIR="${LOG_DIR:-/Users/yinjinrun/random-thing/logs/muxi-constitution-${STAMP}}"
  run_step constitution bash "$SCRIPT_DIR/run_card_constitution_muxi.sh"
  CONST_LOG_DIR="$LOG_DIR"
else
  echo "==> skip constitution"
  CONST_LOG_DIR="${CONST_LOG_DIR:-}"
fi

# --- 2) 出图 ---
if [[ "$SKIP_PLOT" != "1" ]]; then
  PLOT_DATA="${CONST_LOG_DIR}/results"
  if [[ -n "${CONST_LOG_DIR:-}" && -d "$PLOT_DATA" ]]; then
    run_step plot_constitution \
      python3 "$OPS_ROOT/reports/plot_card_constitution.py" \
        --data-dir "$PLOT_DATA" \
        --out-dir "$OPS_ROOT/reports/rounds"
  else
    echo "==> skip plot (no results dir)"
  fi
fi

# --- 3) 拓扑（可与 SKIP_COMM 独立）---
if [[ "$SKIP_TOPO" != "1" ]]; then
  unset TOPO_AFS_OUT || true
  export TOPO_LOG_DIR="$PIPE_LOG_DIR/muxi-topo"
  mkdir -p "$TOPO_LOG_DIR"
  run_step muxi_topo bash "$SCRIPT_DIR/probe_muxi_topology.sh"
else
  echo "==> skip topo"
fi

# --- 4/5) NCCL collective + P2P ---
if [[ "$SKIP_COMM" != "1" ]]; then
  export AFS_OUT="${AFS_RESULTS}/nccl-pipe-${STAMP}"
  export LOG_DIR="$PIPE_LOG_DIR/nccl-scale"
  mkdir -p "$LOG_DIR"
  cluster_pod_exec "$CLUSTER_POD" "mkdir -p '$AFS_OUT'"
  PORT="${MASTER_PORT:-29901}"
  IFS=',' read -ra ARR <<< "$NCCL_SCALES"
  for world in "${ARR[@]}"; do
    echo "==> nccl scale=$world port=$PORT"
    bash "$SCRIPT_DIR/fire_nccl_scale_muxi.sh" "$world" "$PORT" || true
    wait_poll scale "$world" || echo "WARN scale=$world poll failed"
    PORT=$((PORT + 1))
  done

  export AFS_OUT="${AFS_RESULTS}/nccl-p2p-pipe-${STAMP}"
  export LOG_DIR="$PIPE_LOG_DIR/nccl-p2p"
  mkdir -p "$LOG_DIR"
  cluster_pod_exec "$CLUSTER_POD" "mkdir -p '$AFS_OUT'"
  export SIZES="${P2P_SIZES:-64K,16M}"
  export STRATEGIES="${STRATEGIES:-ring}"
  IFS=',' read -ra PARR <<< "$P2P_SCALES"
  for world in "${PARR[@]}"; do
    echo "==> nccl p2p=$world port=$PORT"
    bash "$SCRIPT_DIR/fire_nccl_p2p_muxi.sh" "$world" "$PORT" || true
    wait_poll p2p "$world" || echo "WARN p2p=$world poll failed"
    PORT=$((PORT + 1))
  done
else
  echo "==> skip comm"
fi

echo "PIPELINE_MUXI_DONE → $PIPE_LOG_DIR"
[[ -f "$PIPE_LOG_DIR/failures.txt" ]] && cat "$PIPE_LOG_DIR/failures.txt" || true
