#!/usr/bin/env bash
# 沐曦体质 durable 点火：逐 pod 短 SSH + 远端 nohup，再轮询至结束
#
# 用法:
#   ./scripts/cluster/fire_constitution_durable_muxi.sh
#   RUN_ID=... CLUSTER_FANOUT_PARALLEL=1 ./scripts/cluster/fire_constitution_durable_muxi.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=muxi.env
source "$SCRIPT_DIR/muxi.env"
export AFS_WORKSPACE="${AFS_WORKSPACE_OVERRIDE:-/afs-a3-weight-share/montyyin/lab-workspace}"
export AFS_CS="${AFS_CS_OVERRIDE:-${AFS_WORKSPACE}/projects/CARD_SCREEN}"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
CASE_NAME="${CASE_NAME:-constitution128}"
RUN_ID="${RUN_ID:-${STAMP}-muxi-${CASE_NAME}}"
AFS_OUT_DIR="${AFS_RESULTS}/card_screen-${RUN_ID}"
OUT_JSONL="${AFS_OUT_DIR}/${CASE_NAME}.jsonl"
CONFIG_SRC="${CONFIG_SRC:-$SCRIPT_DIR/../../projects/CARD_SCREEN/config.constitution128.yaml}"
CONFIG_NAME="${CONFIG_NAME:-config.constitution128.yaml}"
SDC_ROUNDS="${SDC_ROUNDS:-5}"
GEMM_N="${GEMM_N:-8192}"
SUSTAINED_S="${SUSTAINED_S:-30}"
LAUNCH_STAGGER_S="${LAUNCH_STAGGER_S:-3}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-120}"
POLL_MAX_S="${POLL_MAX_S:-14400}"

OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/muxi-constitution-${RUN_ID}}"
mkdir -p "$LOG_DIR"
# 不用 process-substitution tee（nohup/后台易早退）；直接 append
exec >>"$LOG_DIR/fanout.log" 2>&1

echo "==> PROFILE=muxi RUN_ID=$RUN_ID"
echo "==> AFS_CS=$AFS_CS"
echo "==> AFS_OUT_DIR=$AFS_OUT_DIR"
echo "==> LOG_DIR=$LOG_DIR"

test -f "$CONFIG_SRC"
TMP_B64="$(base64 < "$CONFIG_SRC" | tr -d '\n')"
cluster_pod_exec "${CLUSTER_POD}" "
set -euo pipefail
mkdir -p '$AFS_OUT_DIR' '$AFS_CS'
test -f '$AFS_CS/screen.py'
test -f '$AFS_CS/card_screen/probes/stage_c.py'
echo '$TMP_B64' | base64 -d > '$AFS_CS/$CONFIG_NAME'
python - <<'PY'
import torch
assert torch.cuda.is_available()
print('torch', torch.__version__, 'ndev', torch.cuda.device_count(), torch.cuda.get_device_name(0))
PY
"

PODS=()
while IFS= read -r _pod; do
  [[ -n "$_pod" ]] && PODS+=("$_pod")
done < <(cluster_pods_running)
echo "==> pods (${#PODS[@]}): ${PODS[*]}"
[[ ${#PODS[@]} -gt 0 ]]

FAIL_PODS=()
for pod in "${PODS[@]}"; do
  echo "==> start $pod"
  logf="$LOG_DIR/${pod}.launch.log"
  if cluster_pod_exec "$pod" "
set -euo pipefail
mkdir -p '$AFS_OUT_DIR'
python3 -c \"from pathlib import Path; p=Path('$AFS_OUT_DIR') / ('$pod' + '.run.sh'); p.write_text('#!/bin/bash\\nset -euo pipefail\\ncd $AFS_CS\\nexport PYTHONUNBUFFERED=1\\nexec python -u screen.py --device all --config $CONFIG_NAME --sdc-rounds $SDC_ROUNDS --gemm-n $GEMM_N --sustained-s $SUSTAINED_S --no-require-idle --out $OUT_JSONL --no-plot\\n'); p.chmod(0o755); print('wrote', p)\"
old=\$(ps -eo pid,args | awk '\$2 ~ /^python/ && /screen\\.py/ {print \$1}')
if [[ -n \"\${old:-}\" ]]; then kill -9 \$old || true; sleep 1; fi
setsid nohup '$AFS_OUT_DIR/${pod}.run.sh' > '$AFS_OUT_DIR/${pod}.run.log' 2>&1 < /dev/null &
echo STARTED_\$!
sleep 5
ps -eo etime,args | awk '\$2 ~ /^python/ && /screen\\.py/ {print; found=1} END{if(!found) exit 1}'
" >"$logf" 2>&1; then
    echo "==> launched $pod"
  else
    echo "FAIL launch $pod (see $logf)"
    FAIL_PODS+=("$pod")
  fi
  sleep "$LAUNCH_STAGGER_S"
done

printf '%s\n' "${FAIL_PODS[@]+"${FAIL_PODS[@]}"}" > "$LOG_DIR/fail_launch.txt"
echo "==> launch_fail (${#FAIL_PODS[@]}): ${FAIL_PODS[*]:-none}"

echo "==> poll every ${POLL_INTERVAL_S}s"
elapsed=0
while (( elapsed < POLL_MAX_S )); do
  alive=0
  for pod in "${PODS[@]}"; do
    n=$(cluster_pod_exec "$pod" "ps -eo args | awk '\$1 ~ /^python/ && /screen\\.py/ {c++} END{print c+0}'" 2>/dev/null | tail -1 | tr -d ' \r' || echo 0)
    if [[ "${n:-0}" -gt 0 ]]; then
      alive=$((alive + 1))
    fi
  done
  echo "==> t=${elapsed}s alive=$alive / ${#PODS[@]}"
  if [[ "$alive" -eq 0 ]]; then
    echo "==> all screens exited"
    break
  fi
  sleep "$POLL_INTERVAL_S"
  elapsed=$((elapsed + POLL_INTERVAL_S))
done

echo "==> aggregate"
cluster_pod_exec "${CLUSTER_POD}" "
set -euo pipefail
cd '$AFS_CS'
python - <<'PY'
from card_screen.cluster.aggregate import aggregate
from pathlib import Path
out = Path('$OUT_JSONL')
# host-suffixed jsonl siblings
files = list(out.parent.glob(out.stem + '*.jsonl'))
print('jsonl_files', len(files))
summary = aggregate(str(out), slow_frac=0.2)
print('n_cards', summary.get('n_cards'))
print('summary', summary.get('summary'))
PY
ls -la '$AFS_OUT_DIR' | head -50
"

echo "==> pull"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${CLUSTER_POD} -- bash -c 'tar -C $AFS_OUT_DIR -cf - .' " \
  > "$LOG_DIR/results.tar" || true
mkdir -p "$LOG_DIR/results"
tar -xf "$LOG_DIR/results.tar" -C "$LOG_DIR/results" 2>/dev/null || true

if [[ ${#FAIL_PODS[@]} -ne 0 ]]; then
  echo "部分节点启动失败: ${FAIL_PODS[*]}"
  exit 1
fi
echo "MUXI_CONSTITUTION_OK → $AFS_OUT_DIR / $LOG_DIR"
