#!/usr/bin/env bash
# Muxi 128 卡体质（constitution）扇出 — 对标 run_card_constitution_128.sh
#
# 用法:
#   source scripts/cluster/muxi.env   # 或本脚本自行 source
#   ./scripts/cluster/run_card_constitution_muxi.sh
#   CLUSTER_FANOUT_PARALLEL=4 SUSTAINED_S=30 ./scripts/cluster/run_card_constitution_muxi.sh
#
# 依赖: AFS 上已有含 stage_c 的 CARD_SCREEN（见 sync_card_screen_muxi.sh）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=muxi.env
source "$SCRIPT_DIR/muxi.env"
# 体质用自有树（含 stage_c + Metax），不覆盖 yushan 冒烟树
export AFS_WORKSPACE="${AFS_WORKSPACE_OVERRIDE:-/afs-a3-weight-share/montyyin/lab-workspace}"
export AFS_CS="${AFS_CS_OVERRIDE:-${AFS_WORKSPACE}/projects/CARD_SCREEN}"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
CASE_NAME="${CASE_NAME:-constitution128}"
RUN_ID="${RUN_ID:-${STAMP}-muxi-${CASE_NAME}}"
AFS_OUT_DIR="${AFS_RESULTS}/card_screen-${RUN_ID}"
OUT_JSONL="${AFS_OUT_DIR}/${CASE_NAME}.jsonl"

SDC_ROUNDS="${SDC_ROUNDS:-5}"
GEMM_N="${GEMM_N:-8192}"
SUSTAINED_S="${SUSTAINED_S:-30}"
IDLE_MAX_MIB="${IDLE_MAX_MIB:-1024}"
REQUIRE_IDLE="${REQUIRE_IDLE:-0}"
CONFIG_SRC="${CONFIG_SRC:-$SCRIPT_DIR/../../projects/CARD_SCREEN/config.constitution128.yaml}"
CONFIG_NAME="${CONFIG_NAME:-config.constitution128.yaml}"

OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/muxi-constitution-${RUN_ID}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/fanout.log") 2>&1

echo "==> PROFILE=muxi KUBECONFIG=$CLUSTER_KUBECONFIG"
echo "==> RUN_ID=$RUN_ID PARALLEL=$CLUSTER_FANOUT_PARALLEL"
echo "==> AFS_CS=$AFS_CS"
echo "==> AFS_OUT_DIR=$AFS_OUT_DIR"
echo "==> LOG_DIR=$LOG_DIR"
echo "==> SDC_ROUNDS=$SDC_ROUNDS GEMM_N=$GEMM_N SUSTAINED_S=$SUSTAINED_S"

if [[ ! -f "$CONFIG_SRC" ]]; then
  echo "ERROR: missing $CONFIG_SRC" >&2
  exit 1
fi

PODS=()
while IFS= read -r _pod; do
  [[ -n "$_pod" ]] && PODS+=("$_pod")
done < <(cluster_pods_running)
if [[ ${#PODS[@]} -eq 0 ]]; then
  echo "ERROR: no Running pods"
  exit 1
fi
echo "==> pods (${#PODS[@]}): ${PODS[*]}"

# 推送 constitution 配置
TMP_B64="$(base64 < "$CONFIG_SRC" | tr -d '\n')"
cluster_pod_exec "${CLUSTER_POD}" "
set -euo pipefail
mkdir -p '$AFS_OUT_DIR' '$AFS_CS'
test -f '$AFS_CS/screen.py' || { echo 'ERROR: CARD_SCREEN missing on AFS; run sync_card_screen_muxi.sh first'; exit 1; }
test -f '$AFS_CS/card_screen/probes/stage_c.py' || { echo 'ERROR: stage_c.py missing'; exit 1; }
echo '$TMP_B64' | base64 -d > '$AFS_CS/$CONFIG_NAME'
ls -la '$AFS_CS/$CONFIG_NAME'
python -c 'import yaml' 2>/dev/null || pip install -q pyyaml
python - <<'PY'
import torch
assert torch.cuda.is_available()
print('torch', torch.__version__, 'ndev', torch.cuda.device_count(), torch.cuda.get_device_name(0))
PY
"

IDLE_ARGS=()
if [[ "$REQUIRE_IDLE" == "1" ]]; then
  IDLE_ARGS=(--require-idle --idle-max-memory-mib "$IDLE_MAX_MIB")
fi
# bash 3.2: expand for remote
IDLE_CLI=""
if [[ "$REQUIRE_IDLE" == "1" ]]; then
  IDLE_CLI="--require-idle --idle-max-memory-mib $IDLE_MAX_MIB"
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
  --config '$CONFIG_NAME' \
  --sdc-rounds $SDC_ROUNDS \
  --gemm-n $GEMM_N \
  --sustained-s $SUSTAINED_S \
  $IDLE_CLI \
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
        if [[ "$rc" -ne 0 ]]; then
          FAIL_PODS+=("${POD_FOR_PID[$i]}")
        fi
        unset "PIDS[$i]"
        unset "POD_FOR_PID[$i]"
        ACTIVE=$((ACTIVE - 1))
      fi
    done
    local _p=() _n=()
    for i in "${!PIDS[@]}"; do
      _p+=("${PIDS[$i]}")
      _n+=("${POD_FOR_PID[$i]}")
    done
    PIDS=("${_p[@]+"${_p[@]}"}")
    POD_FOR_PID=("${_n[@]+"${_n[@]}"}")
    [[ "$ACTIVE" -ge "$CLUSTER_FANOUT_PARALLEL" ]] && sleep 1
  done
}

for pod in "${PODS[@]}"; do
  wait_one_slot
  run_one "$pod" &
  PIDS+=("$!")
  POD_FOR_PID+=("$pod")
  ACTIVE=$((ACTIVE + 1))
done
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
print('nodes', len(summary.get('nodes') or []))
print('summary', summary.get('summary'))
PY
ls -la '$AFS_OUT_DIR' | head -40
"

echo "==> pull → $LOG_DIR/results"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${CLUSTER_POD} -- bash -c 'tar -C $AFS_OUT_DIR -cf - .' " \
  > "$LOG_DIR/results.tar" || true
mkdir -p "$LOG_DIR/results"
tar -xf "$LOG_DIR/results.tar" -C "$LOG_DIR/results" 2>/dev/null || true

if [[ ${#FAIL_PODS[@]} -ne 0 ]]; then
  echo "部分节点失败: ${FAIL_PODS[*]}"
  exit 1
fi
echo "MUXI_CONSTITUTION_OK → $AFS_OUT_DIR / $LOG_DIR"
