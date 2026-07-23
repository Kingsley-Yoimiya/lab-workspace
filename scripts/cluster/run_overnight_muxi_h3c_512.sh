#!/usr/bin/env bash
# Muxi h3c-test 512 卡夜间：smoke → sentinel → constitution；并行通信门禁
# 用法:
#   CLUSTER_JOB_OVERRIDE=yinjinrun-cs512-... ./scripts/cluster/run_overnight_muxi_h3c_512.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=muxi.env
source "$SCRIPT_DIR/muxi.env"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
CAMPAIGN="${CAMPAIGN:-overnight512-${STAMP}}"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$OPS_ROOT/../../logs/muxi-${CAMPAIGN}}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/campaign.log") 2>&1

echo "==> CAMPAIGN=$CAMPAIGN JOB=$CLUSTER_JOB N_WORKERS=$CLUSTER_N_WORKERS"
echo "==> AFS_CS=$AFS_CS AFS_RESULTS=$AFS_RESULTS"
n_pods="$(cluster_pods_running | wc -l | tr -d ' ')"
echo "==> running pods=$n_pods"
[[ "$n_pods" -ge 1 ]] || { echo "ERROR: no running pods"; exit 1; }

# --- V1 smoke ---
echo "==== V1 SMOKE $(date -Iseconds) ===="
CASE_NAME=smoke RUN_ID="${STAMP}-muxi-smoke512" \
  LOG_DIR="$LOG_ROOT/smoke" \
  CLUSTER_FANOUT_PARALLEL="${CLUSTER_FANOUT_PARALLEL:-16}" \
  "$SCRIPT_DIR/run_card_screen_muxi.sh" all \
  || echo "WARN: smoke had failures (see fail_pods)"

# --- V2 sentinel（复用 constitution durable launcher + sentinel config）---
echo "==== V2 SENTINEL $(date -Iseconds) ===="
CASE_NAME=sentinel512 RUN_ID="${STAMP}-muxi-sentinel512" \
  CONFIG_SRC="$SCRIPT_DIR/../../projects/CARD_SCREEN/config.phase1_sentinel.yaml" \
  CONFIG_NAME=config.phase1_sentinel.yaml \
  SDC_ROUNDS=0 GEMM_N=4096 SUSTAINED_S=8 \
  LOG_DIR="$LOG_ROOT/sentinel" \
  "$SCRIPT_DIR/run_card_constitution_muxi.sh" \
  || echo "WARN: sentinel launch/poll issues"

# --- V3 constitution ---
echo "==== V3 CONSTITUTION $(date -Iseconds) ===="
CASE_NAME=constitution512 RUN_ID="${STAMP}-muxi-constitution512" \
  LOG_DIR="$LOG_ROOT/constitution" \
  "$SCRIPT_DIR/run_card_constitution_muxi.sh" \
  || echo "WARN: constitution launch/poll issues"

echo "==== SCREEN PIPELINE DONE $(date -Iseconds) ===="
echo "LOG_ROOT=$LOG_ROOT"
