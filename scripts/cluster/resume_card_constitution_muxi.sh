#!/usr/bin/env bash
# 补跑 muxi 体质：只跑 POD_LIST 文件里的 pod，写入已有 AFS_OUT_DIR
# 用法:
#   POD_LIST=.../remain_pods.txt RUN_ID=20260711_140024-muxi-constitution128 \
#     ./scripts/cluster/resume_card_constitution_muxi.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=muxi.env
source "$SCRIPT_DIR/muxi.env"
export AFS_WORKSPACE="${AFS_WORKSPACE_OVERRIDE:-/afs-a3-weight-share/yinjinrun.p/lab-workspace}"
export AFS_CS="${AFS_CS_OVERRIDE:-${AFS_WORKSPACE}/projects/CARD_SCREEN}"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

RUN_ID="${RUN_ID:?need RUN_ID}"
POD_LIST="${POD_LIST:?need POD_LIST}"
CASE_NAME="${CASE_NAME:-constitution128}"
AFS_OUT_DIR="${AFS_RESULTS}/card_screen-${RUN_ID}"
OUT_JSONL="${AFS_OUT_DIR}/${CASE_NAME}.jsonl"
CONFIG_NAME="${CONFIG_NAME:-config.constitution128.yaml}"
SDC_ROUNDS="${SDC_ROUNDS:-5}"
GEMM_N="${GEMM_N:-8192}"
SUSTAINED_S="${SUSTAINED_S:-30}"

OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/muxi-constitution-${RUN_ID}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/resume.log") 2>&1

PODS=()
while IFS= read -r _pod; do
  [[ -n "$_pod" ]] && PODS+=("$_pod")
done < "$POD_LIST"

echo "==> RESUME RUN_ID=$RUN_ID PARALLEL=$CLUSTER_FANOUT_PARALLEL"
echo "==> pods (${#PODS[@]}): ${PODS[*]}"
echo "==> OUT=$OUT_JSONL"

run_one() {
  local pod="$1"
  local logf="$LOG_DIR/${pod}.resume.log"
  echo "==> start $pod"
  local prefix
  prefix="$(_cluster_vcctl_prefix)"
  if ssh -o BatchMode=yes -o ConnectTimeout=30 "$CLUSTER_SSH_HOST" \
    "${prefix} pod exec ${pod} -- bash -lc $(printf '%q' "
set -euo pipefail
cd '$AFS_CS'
export PYTHONUNBUFFERED=1
python -u screen.py \
  --device all \
  --config '$CONFIG_NAME' \
  --sdc-rounds $SDC_ROUNDS \
  --gemm-n $GEMM_N \
  --sustained-s $SUSTAINED_S \
  --out '$OUT_JSONL' \
  --no-plot
echo CONSTITUTION_DONE_${pod}
")" >"$logf" 2>&1; then
    echo "==> done $pod"
    return 0
  fi
  echo "FAIL $pod (see $logf)"
  return 1
}

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
        [[ "$rc" -ne 0 ]] && FAIL_PODS+=("${POD_FOR_PID[$i]}")
        unset "PIDS[$i]"; unset "POD_FOR_PID[$i]"
        ACTIVE=$((ACTIVE - 1))
      fi
    done
    local _p=() _n=()
    for i in "${!PIDS[@]}"; do _p+=("${PIDS[$i]}"); _n+=("${POD_FOR_PID[$i]}"); done
    PIDS=("${_p[@]+"${_p[@]}"}")
    POD_FOR_PID=("${_n[@]+"${_n[@]}"}")
    [[ "$ACTIVE" -ge "$CLUSTER_FANOUT_PARALLEL" ]] && sleep 2
  done
}

for pod in "${PODS[@]}"; do
  wait_one_slot
  run_one "$pod" &
  PIDS+=("$!")
  POD_FOR_PID+=("$pod")
  ACTIVE=$((ACTIVE + 1))
  sleep 1  # stagger ssh
done
for i in "${!PIDS[@]}"; do
  rc=0
  wait "${PIDS[$i]}" || rc=$?
  [[ "$rc" -ne 0 ]] && FAIL_PODS+=("${POD_FOR_PID[$i]}")
done

printf '%s\n' "${FAIL_PODS[@]+"${FAIL_PODS[@]}"}" > "$LOG_DIR/fail_pods.txt"
echo "==> fail_pods (${#FAIL_PODS[@]}): ${FAIL_PODS[*]:-none}"

echo "==> aggregate all"
cluster_pod_exec "${CLUSTER_POD}" "
set -euo pipefail
cd '$AFS_CS'
python - <<'PY'
from card_screen.cluster.aggregate import aggregate
from pathlib import Path
out = Path('$OUT_JSONL')
summary = aggregate(str(out), slow_frac=0.2)
print('n_cards', summary.get('n_cards'))
print('nodes', len(summary.get('nodes') or []))
print('summary', summary.get('summary'))
PY
ls -la '$AFS_OUT_DIR' | head -40
"

ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${CLUSTER_POD} -- bash -c 'tar -C $AFS_OUT_DIR -cf - .' " \
  > "$LOG_DIR/results.tar" || true
mkdir -p "$LOG_DIR/results"
tar -xf "$LOG_DIR/results.tar" -C "$LOG_DIR/results" 2>/dev/null || true

if [[ ${#FAIL_PODS[@]} -ne 0 ]]; then
  echo "RESUME_PARTIAL_FAIL"
  exit 1
fi
echo "RESUME_CONSTITUTION_OK"
