#!/usr/bin/env bash
# Muxi CARD_SCREEN 冒烟 / 128 卡扇出（KUBECONFIG 隔离 + 有界并行）
#
# 用法:
#   ./scripts/cluster/run_card_screen_muxi.sh           # master 8 卡
#   ./scripts/cluster/run_card_screen_muxi.sh all       # 全 pod 扇出
#   ./scripts/cluster/run_card_screen_muxi.sh retry     # 只重跑上次失败的 pod（读 FAIL_PODS 文件）
#   CLUSTER_FANOUT_PARALLEL=4 SDC_ROUNDS=3 ./scripts/cluster/run_card_screen_muxi.sh all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=muxi.env
source "$SCRIPT_DIR/muxi.env"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

MODE="${1:-master}"
STAMP="$(date +%Y%m%d_%H%M%S)"
CASE_NAME="${CASE_NAME:-smoke}"
RUN_ID="${RUN_ID:-${STAMP}-muxi-${CASE_NAME}}"
SDC_ROUNDS="${SDC_ROUNDS:-3}"
GEMM_N="${GEMM_N:-4096}"
SUSTAINED_S="${SUSTAINED_S:-10}"

OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/muxi-card-screen-${RUN_ID}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/fanout.log") 2>&1

AFS_OUT_DIR="${AFS_RESULTS}/card_screen-${RUN_ID}"
# retry 模式可复用已有 OUT
if [[ "$MODE" == "retry" ]]; then
  PREV_RUN="${PREV_RUN:?retry 需要 PREV_RUN=... 指向已有 RUN_ID 目录名}"
  RUN_ID="$PREV_RUN"
  LOG_DIR="${LOG_DIR_OVERRIDE:-$OPS_ROOT/../../logs/muxi-card-screen-${RUN_ID}}"
  AFS_OUT_DIR="${AFS_RESULTS}/card_screen-${RUN_ID}"
  OUT_JSONL="${AFS_OUT_DIR}/${CASE_NAME}.jsonl"
else
  OUT_JSONL="${AFS_OUT_DIR}/${CASE_NAME}.jsonl"
fi

echo "==> PROFILE=muxi KUBECONFIG=$CLUSTER_KUBECONFIG"
echo "==> JOB=$CLUSTER_JOB MODE=$MODE PARALLEL=$CLUSTER_FANOUT_PARALLEL"
echo "==> AFS_CS=$AFS_CS"
echo "==> AFS_OUT_DIR=$AFS_OUT_DIR"
echo "==> LOG_DIR=$LOG_DIR"
echo "==> SDC_ROUNDS=$SDC_ROUNDS GEMM_N=$GEMM_N SUSTAINED_S=$SUSTAINED_S"

PODS=()
if [[ "$MODE" == "retry" ]]; then
  FAIL_FILE="${FAIL_FILE:-$LOG_DIR/fail_pods.txt}"
  if [[ ! -f "$FAIL_FILE" ]]; then
    echo "ERROR: missing $FAIL_FILE"
    exit 1
  fi
  while IFS= read -r _pod; do
    [[ -n "$_pod" ]] && PODS+=("$_pod")
  done < "$FAIL_FILE"
else
  while IFS= read -r _pod; do
    [[ -n "$_pod" ]] && PODS+=("$_pod")
  done < <(cluster_pods_running)
fi

if [[ ${#PODS[@]} -eq 0 ]]; then
  echo "ERROR: no pods"
  ./scripts/cluster/switch_kube_context.sh status || true
  exit 1
fi

if [[ "$MODE" == "master" ]]; then
  PODS=("${CLUSTER_POD}")
fi
echo "==> pods (${#PODS[@]}): ${PODS[*]}"

if [[ "$MODE" != "retry" ]]; then
  cluster_pod_exec "${PODS[0]}" "
set -euo pipefail
mkdir -p '$AFS_OUT_DIR' '$(dirname "$AFS_RESULTS")'
test -f '$AFS_CS/screen.py'
python -c 'import yaml' 2>/dev/null || pip install -q pyyaml
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'cuda not available'
print('torch', torch.__version__, 'ndev', torch.cuda.device_count(), torch.cuda.get_device_name(0))
PY
which mx-smi && mx-smi 2>/dev/null | head -8 || true
"
fi

run_one() {
  local pod="$1"
  local logf="$LOG_DIR/${pod}.log"
  echo "==> start $pod"
  local prefix
  prefix="$(_cluster_vcctl_prefix)"
  if ssh -o BatchMode=yes -o ConnectTimeout=30 "$CLUSTER_SSH_HOST" \
    "${prefix} pod exec ${pod} -- bash -lc $(printf '%q' "
set -euo pipefail
cd '$AFS_CS'
export PYTHONUNBUFFERED=1
python screen.py \
  --device all \
  --sdc-rounds $SDC_ROUNDS \
  --gemm-n $GEMM_N \
  --sustained-s $SUSTAINED_S \
  --out '$OUT_JSONL' \
  --no-plot
echo SCREEN_DONE_${pod}
")" >"$logf" 2>&1; then
    echo "==> done $pod"
    return 0
  fi
  echo "FAIL $pod (see $logf)"
  return 1
}

# 有界并行（bash 3.2 友好）
FAIL_PODS=()
ACTIVE=0
PIDS=()
POD_FOR_PID=()

wait_one_slot() {
  local i pid rc
  while [[ "$ACTIVE" -ge "$CLUSTER_FANOUT_PARALLEL" ]]; do
    for i in "${!PIDS[@]}"; do
      pid="${PIDS[$i]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        rc=0
        wait "$pid" || rc=$?
        if [[ "$rc" -ne 0 ]]; then
          FAIL_PODS+=("${POD_FOR_PID[$i]}")
        fi
        unset "PIDS[$i]"
        unset "POD_FOR_PID[$i]"
        ACTIVE=$((ACTIVE - 1))
      fi
    done
    # re-pack
    local _p=() _n=()
    for i in "${!PIDS[@]}"; do
      _p+=("${PIDS[$i]}")
      _n+=("${POD_FOR_PID[$i]}")
    done
    PIDS=("${_p[@]+"${_p[@]}"}")
    POD_FOR_PID=("${_n[@]+"${_n[@]}"}")
    [[ "$ACTIVE" -ge "$CLUSTER_FANOUT_PARALLEL" ]] && sleep 0.4
  done
}

for pod in "${PODS[@]}"; do
  wait_one_slot
  run_one "$pod" &
  PIDS+=("$!")
  POD_FOR_PID+=("$pod")
  ACTIVE=$((ACTIVE + 1))
done
# drain
for i in "${!PIDS[@]}"; do
  rc=0
  wait "${PIDS[$i]}" || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    FAIL_PODS+=("${POD_FOR_PID[$i]}")
  fi
done

printf '%s\n' "${FAIL_PODS[@]+"${FAIL_PODS[@]}"}" > "$LOG_DIR/fail_pods.txt"
echo "==> fail_pods (${#FAIL_PODS[@]}): ${FAIL_PODS[*]:-none}"

echo "==> aggregate"
cluster_pod_exec "${CLUSTER_POD}" "
set -euo pipefail
cd '$AFS_CS'
python - <<'PY'
from card_screen.cluster.aggregate import aggregate
from pathlib import Path
out = Path('$OUT_JSONL')
summary = aggregate(str(out), slow_frac=0.2)
print('n_cards', summary.get('n_cards'))
print('summary', summary.get('summary'))
print('nodes', len(summary.get('nodes') or []))
print('wrote', str(out).replace('.jsonl', '') + '.cluster.json')
PY
ls -la '$AFS_OUT_DIR' | head -40
"

echo "==> pull results → $LOG_DIR/results"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${CLUSTER_POD} -- bash -c 'tar -C $AFS_OUT_DIR -cf - .' " \
  > "$LOG_DIR/results.tar" || true
mkdir -p "$LOG_DIR/results"
tar -xf "$LOG_DIR/results.tar" -C "$LOG_DIR/results" 2>/dev/null || true

if [[ ${#FAIL_PODS[@]} -ne 0 ]]; then
  echo "部分节点失败: ${FAIL_PODS[*]}"
  echo "重跑: PREV_RUN=$RUN_ID FAIL_FILE=$LOG_DIR/fail_pods.txt $0 retry"
  exit 1
fi
echo "MUXI_CARD_SCREEN_OK → $AFS_OUT_DIR / $LOG_DIR"
