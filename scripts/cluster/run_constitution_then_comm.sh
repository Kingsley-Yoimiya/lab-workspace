#!/usr/bin/env bash
# 烤机 → 拓扑 → 通信 串行 list（同一集群窗口接着跑）
#
# 顺序：
#   1) CARD_SCREEN constitution128（含 10 shape BNMK + HBM 多模式 + Stage C）
#   2) 本地出体质分布图（有结果目录时）
#   3) HCCL 拓扑探测（机内 HCCS / hccn / ranktable；通信 list 最前）
#   4) HCCL collective scale（16/32/64/128）
#   5) HCCL P2P（默认先 16，再 128 ring-only）
#
# 用法：
#   ./scripts/cluster/run_constitution_then_comm.sh
#   SKIP_CONSTITUTION=1 ./scripts/cluster/run_constitution_then_comm.sh   # 只跑通信
#   SKIP_COMM=1 ./scripts/cluster/run_constitution_then_comm.sh           # 只烤机
#   SKIP_TOPO=1 ./scripts/cluster/run_constitution_then_comm.sh            # 跳过拓扑
#   CLUSTER_JOB=huawei-8node-copy2 ./scripts/cluster/run_constitution_then_comm.sh
#
# 任一步失败默认中止；设 CONTINUE_ON_FAIL=1 则记 WARN 继续。
# 拓扑步默认 CONTINUE 友好：缺 hccn_tool 不视为失败（脚本自身 best-effort）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PIPE_LOG_DIR="${PIPE_LOG_DIR:-$OPS_ROOT/../../logs/pipeline-constitution-comm-${STAMP}}"
mkdir -p "$PIPE_LOG_DIR"
exec > >(tee -a "$PIPE_LOG_DIR/pipeline.log") 2>&1

SKIP_CONSTITUTION="${SKIP_CONSTITUTION:-0}"
SKIP_PLOT="${SKIP_PLOT:-0}"
SKIP_TOPO="${SKIP_TOPO:-0}"
SKIP_COMM="${SKIP_COMM:-0}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"

# 通信默认：collective 全档；P2P 先 16 再 128（ring 由 bench 自动）
HCCL_SCALES="${HCCL_SCALES:-16,32,64,128}"
P2P_SCALES="${P2P_SCALES:-16,128}"

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
      echo "ABORT pipeline (set CONTINUE_ON_FAIL=1 to keep going)"
      exit "$rc"
    fi
  else
    echo "OK step=$name"
  fi
}

echo "==> PIPE_LOG_DIR=$PIPE_LOG_DIR"
echo "==> CLUSTER_JOB=$CLUSTER_JOB"
echo "==> list: constitution(10-shape) -> plot -> hccl_topo -> hccl_scale -> hccl_p2p"

# --- 1) 烤机 ---
if [[ "$SKIP_CONSTITUTION" != "1" ]]; then
  export CASE_NAME="${CASE_NAME:-constitution128}"
  export CONFIG_NAME="${CONFIG_NAME:-config.constitution128.yaml}"
  export LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/card-constitution-128-${STAMP}-${CASE_NAME}}"
  run_step constitution "$SCRIPT_DIR/run_card_constitution_128.sh"
  CONST_LOG_DIR="$LOG_DIR"
else
  echo "==> skip constitution"
  CONST_LOG_DIR="${CONST_LOG_DIR:-}"
fi

# --- 2) 出图（本机）---
if [[ "$SKIP_PLOT" != "1" ]]; then
  PLOT_DATA="${CONST_LOG_DIR}/results"
  if [[ -n "${CONST_LOG_DIR:-}" && -d "$PLOT_DATA" ]]; then
    run_step plot_constitution \
      python3 "$OPS_ROOT/reports/plot_card_constitution.py" \
        --data-dir "$PLOT_DATA" \
        --out-dir "$OPS_ROOT/reports/rounds"
  else
    echo "==> skip plot (no $PLOT_DATA); pull results later then re-run plot"
  fi
fi

# --- 3) 拓扑探测（通信 list 最前：烤机后、collective 前）---
if [[ "$SKIP_COMM" != "1" && "$SKIP_TOPO" != "1" ]]; then
  export LOG_DIR="$PIPE_LOG_DIR/hccl-topo"
  mkdir -p "$LOG_DIR"
  run_step hccl_topo "$SCRIPT_DIR/probe_hccl_topology.sh"
elif [[ "$SKIP_TOPO" == "1" ]]; then
  echo "==> skip topo"
fi

# --- 4) 通信 collective ---
if [[ "$SKIP_COMM" != "1" ]]; then
  export SCALES="$HCCL_SCALES"
  export LOG_DIR="$PIPE_LOG_DIR/hccl-scale"
  mkdir -p "$LOG_DIR"
  run_step hccl_scale "$SCRIPT_DIR/run_hccl_scale.sh"

  # --- 5) 通信 P2P ---
  export SCALES="$P2P_SCALES"
  export STRATEGIES="${STRATEGIES:-ring}"
  export LOG_DIR="$PIPE_LOG_DIR/hccl-p2p"
  mkdir -p "$LOG_DIR"
  run_step hccl_p2p "$SCRIPT_DIR/run_hccl_p2p_128.sh"
else
  echo "==> skip comm"
fi

echo ""
echo "PIPELINE_DONE → $PIPE_LOG_DIR"
if [[ -f "$PIPE_LOG_DIR/failures.txt" ]]; then
  echo "failures:"
  cat "$PIPE_LOG_DIR/failures.txt"
  exit 1
fi
