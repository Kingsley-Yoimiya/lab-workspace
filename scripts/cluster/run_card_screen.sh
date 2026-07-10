#!/usr/bin/env bash
# 在 master pod 上冒烟 CARD_SCREEN（快速参数）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/cluster-run-$STAMP}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/card_screen.log") 2>&1

CS_DIR="${AFS_WORKSPACE}/projects/CARD_SCREEN"
echo "==> LOG_DIR=$LOG_DIR"
echo "==> CS_DIR=$CS_DIR POD=$CLUSTER_POD"

cluster_pod_exec "$CLUSTER_POD" "
set -euo pipefail
cd '$CS_DIR'
pwd
ls -la | head
# 依赖
python -c 'import torch; print(\"torch\", torch.__version__)' 2>&1 || true
python -c 'import torch_npu; print(\"torch_npu_ok\")' 2>&1 || echo 'torch_npu missing (may still detect npu via device)'
python -c 'import yaml; print(\"pyyaml_ok\")' 2>&1 || pip install -q pyyaml

# 快速冒烟：小规模，几分钟内验证流程
export CASE_NAME=smoke
export DEVICE=all
export SDC_ROUNDS=3
export GEMM_N=4096
export SUSTAINED_S=10
OUT_DIR='results/${STAMP}-ops-smoke'
mkdir -p \"\$OUT_DIR\"
echo '==> running screen.py smoke'
python screen.py --device all --sdc-rounds 3 --gemm-n 4096 --sustained-s 10 --out \"\$OUT_DIR/smoke.jsonl\"
echo CARD_SCREEN_SMOKE_OK
ls -la \"\$OUT_DIR\" | head
" | tee "$LOG_DIR/card_screen.remote.log"

echo "==> CARD_SCREEN 冒烟结束 → $LOG_DIR"
