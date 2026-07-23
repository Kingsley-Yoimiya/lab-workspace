#!/usr/bin/env bash
# 在每个 Running pod 上 setsid nohup 启动 screen.py（短连，防本机 nohup 被杀）
# 用法:
#   CASE_NAME=smoke CONFIG_NAME=  GEMM_N=4096 SUSTAINED_S=10 SDC_ROUNDS=3 \
#     ./scripts/cluster/fire_screen_durable_muxi.sh
#   CASE_NAME=sentinel CONFIG_NAME=config.phase1_sentinel.yaml ...
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
CASE_NAME="${CASE_NAME:-smoke}"
RUN_ID="${RUN_ID:-${STAMP}-muxi-${CASE_NAME}512}"
AFS_OUT_DIR="${AFS_RESULTS}/card_screen-${RUN_ID}"
OUT_JSONL="${AFS_OUT_DIR}/${CASE_NAME}.jsonl"
CONFIG_NAME="${CONFIG_NAME:-}"
SDC_ROUNDS="${SDC_ROUNDS:-3}"
GEMM_N="${GEMM_N:-4096}"
SUSTAINED_S="${SUSTAINED_S:-10}"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/muxi-${CASE_NAME}-${RUN_ID}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/fire.log") 2>&1

echo "==> JOB=$CLUSTER_JOB CASE=$CASE_NAME RUN_ID=$RUN_ID"
echo "==> AFS_CS=$AFS_CS OUT=$OUT_JSONL CONFIG=${CONFIG_NAME:-none}"

PODS=()
while IFS= read -r p; do [[ -n "$p" ]] && PODS+=("$p"); done < <(cluster_pods_running)
echo "==> pods ${#PODS[@]}"
[[ ${#PODS[@]} -gt 0 ]] || exit 1

# optional push config
if [[ -n "$CONFIG_NAME" ]]; then
  CONFIG_SRC="${CONFIG_SRC:-$SCRIPT_DIR/../../projects/CARD_SCREEN/$CONFIG_NAME}"
  [[ -f "$CONFIG_SRC" ]] || { echo "missing $CONFIG_SRC"; exit 1; }
  TMP_B64="$(base64 < "$CONFIG_SRC" | tr -d '\n')"
  cluster_pod_exec "${CLUSTER_POD}" "
mkdir -p '$AFS_OUT_DIR' '$AFS_CS'
echo '$TMP_B64' | base64 -d > '$AFS_CS/$CONFIG_NAME'
test -f '$AFS_CS/screen.py'
"
else
  cluster_pod_exec "${CLUSTER_POD}" "mkdir -p '$AFS_OUT_DIR'; test -f '$AFS_CS/screen.py'"
fi

CFG_ARG=""
[[ -n "$CONFIG_NAME" ]] && CFG_ARG="--config $CONFIG_NAME"

run_one() {
  local pod="$1"
  local logf="$LOG_DIR/${pod}.launch.log"
  local tag="$pod"
  if cluster_pod_exec "$pod" "
set -euo pipefail
mkdir -p '$AFS_OUT_DIR'
python3 -c \"from pathlib import Path; p=Path('$AFS_OUT_DIR')/('$tag'+'.run.sh'); p.write_text('#!/bin/bash\\nset -euo pipefail\\ncd $AFS_CS\\nexport PYTHONUNBUFFERED=1\\npython -u screen.py --device all $CFG_ARG --sdc-rounds $SDC_ROUNDS --gemm-n $GEMM_N --sustained-s $SUSTAINED_S --no-require-idle --out $OUT_JSONL --no-plot\\necho SCREEN_DONE\\ntouch $AFS_OUT_DIR/$tag.done\\n'); p.chmod(0o755)\"
old=\$(ps -eo pid,args | awk '\$2 ~ /^python/ && /screen\\.py/ {print \$1}')
if [[ -n \"\${old:-}\" ]]; then kill -9 \$old || true; sleep 1; fi
nohup bash '$AFS_OUT_DIR/${tag}.run.sh' > '$AFS_OUT_DIR/${tag}.run.log' 2>&1 < /dev/null &
echo STARTED_\$!
sleep 4
ps -eo etime,args | awk '\$2 ~ /^python/ && /screen\\.py/ {print; found=1} END{if(!found) exit 1}'
" >"$logf" 2>&1; then
    echo "==> launched $pod"
    return 0
  fi
  echo "FAIL_LAUNCH $pod"
  return 1
}

FAIL=()
ACTIVE=0
PIDS=()
POD_FOR_PID=()
for pod in "${PODS[@]}"; do
  while [[ "$ACTIVE" -ge "$CLUSTER_FANOUT_PARALLEL" ]]; do
    for i in "${!PIDS[@]}"; do
      if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
        wait "${PIDS[$i]}" || FAIL+=("${POD_FOR_PID[$i]}")
        unset "PIDS[$i]"; unset "POD_FOR_PID[$i]"
        ACTIVE=$((ACTIVE - 1))
      fi
    done
    if [[ ${#PIDS[@]} -gt 0 ]]; then PIDS=("${PIDS[@]}"); POD_FOR_PID=("${POD_FOR_PID[@]}"); else PIDS=(); POD_FOR_PID=(); fi
    [[ "$ACTIVE" -ge "$CLUSTER_FANOUT_PARALLEL" ]] && sleep 0.3
  done
  run_one "$pod" &
  PIDS+=("$!"); POD_FOR_PID+=("$pod"); ACTIVE=$((ACTIVE + 1))
done
for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" || FAIL+=("${POD_FOR_PID[$i]}"); done
printf '%s\n' "${FAIL[@]+"${FAIL[@]}"}" > "$LOG_DIR/fail_launch.txt"
echo "==> launched ok=$(( ${#PODS[@]} - ${#FAIL[@]} )) fail=${#FAIL[@]}"
echo "AFS_OUT_DIR=$AFS_OUT_DIR"
echo "POLL: watch for python screen.py gone + jsonl growth; then aggregate"
echo "$AFS_OUT_DIR" > "$LOG_DIR/AFS_OUT_DIR.txt"
echo "$OUT_JSONL" > "$LOG_DIR/OUT_JSONL.txt"
echo "$RUN_ID" > "$LOG_DIR/RUN_ID.txt"
