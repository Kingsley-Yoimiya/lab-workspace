#!/usr/bin/env bash
# 128 卡「体质」筛查：Cube + HBM + sustained + Vector/Scalar 吞吐 + launch 延迟
# 用法:
#   ./scripts/cluster/run_card_constitution_128.sh
#   SDC_ROUNDS=5 SUSTAINED_S=30 ./scripts/cluster/run_card_constitution_128.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
CASE_NAME="${CASE_NAME:-constitution128}"
RUN_ID="${RUN_ID:-${STAMP}-${CASE_NAME}}"
AFS_CS="${AFS_WORKSPACE}/projects/CARD_SCREEN"
AFS_OUT_ROOT="${AFS_RESULTS:-/afs-a3-weight-share/yinjinrun.p-huawei/results}"
AFS_OUT_DIR="${AFS_OUT_ROOT}/card_screen-${RUN_ID}"
OUT_JSONL="${AFS_OUT_DIR}/${CASE_NAME}.jsonl"

SDC_ROUNDS="${SDC_ROUNDS:-5}"
GEMM_N="${GEMM_N:-8192}"
SUSTAINED_S="${SUSTAINED_S:-30}"
IDLE_MAX_MIB="${IDLE_MAX_MIB:-1024}"
CONFIG_SRC="${CONFIG_SRC:-$SCRIPT_DIR/../../projects/CARD_SCREEN/config.constitution128.yaml}"
CONFIG_NAME="${CONFIG_NAME:-config.constitution128.yaml}"

OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/card-constitution-128-${RUN_ID}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/fanout.log") 2>&1

PODS=(
  "${CLUSTER_JOB}-master-0"
  "${CLUSTER_JOB}-worker-0"
  "${CLUSTER_JOB}-worker-1"
  "${CLUSTER_JOB}-worker-2"
  "${CLUSTER_JOB}-worker-3"
  "${CLUSTER_JOB}-worker-4"
  "${CLUSTER_JOB}-worker-5"
  "${CLUSTER_JOB}-worker-6"
)

echo "==> RUN_ID=$RUN_ID CASE_NAME=$CASE_NAME"
echo "==> AFS_OUT_DIR=$AFS_OUT_DIR"
echo "==> LOG_DIR=$LOG_DIR"
echo "==> CONFIG_SRC=$CONFIG_SRC"

# 把体质配置推到 AFS（经 master pod）
if [[ ! -f "$CONFIG_SRC" ]]; then
  echo "ERROR: missing $CONFIG_SRC" >&2
  exit 1
fi

# 经跳板把本地 yaml 写到 AFS
TMP_B64="$(base64 < "$CONFIG_SRC" | tr -d '\n')"
cluster_pod_exec "${PODS[0]}" "
set -euo pipefail
mkdir -p '$AFS_OUT_DIR' '$AFS_CS'
echo '$TMP_B64' | base64 -d > '$AFS_CS/$CONFIG_NAME'
# 同步增强后的 CARD_SCREEN 代码（若 AFS 已有旧树，用本机 sync 更稳；此处至少保证 config 到位）
ls -la '$AFS_CS/$CONFIG_NAME' '$AFS_OUT_DIR'
head -n 5 '$AFS_CS/$CONFIG_NAME'
"

run_one() {
  local pod="$1"
  local logf="$LOG_DIR/${pod}.log"
  echo "==> start $pod"
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER_SSH_HOST" \
    "vcctl pod exec ${pod} -- bash -lc $(printf '%q' "
set -euo pipefail
cd '$AFS_CS'
export PYTHONUNBUFFERED=1
python screen.py \
  --device all \
  --config '$CONFIG_NAME' \
  --sdc-rounds $SDC_ROUNDS \
  --gemm-n $GEMM_N \
  --sustained-s $SUSTAINED_S \
  --require-idle \
  --idle-max-memory-mib $IDLE_MAX_MIB \
  --out '$OUT_JSONL' \
  --no-plot
echo CONSTITUTION_DONE_${pod}
")" >"$logf" 2>&1 || {
    echo "FAIL $pod (see $logf)"
    return 1
  }
  echo "==> done $pod"
}

FAIL=0
pids=()
for pod in "${PODS[@]}"; do
  run_one "$pod" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done

echo "==> aggregate"
cluster_pod_exec "${PODS[0]}" "
set -euo pipefail
cd '$AFS_CS'
python - <<'PY'
from card_screen.cluster.aggregate import aggregate
from pathlib import Path
out = Path('$OUT_JSONL')
summary = aggregate(str(out), slow_frac=0.2)
print('n_cards', summary.get('n_cards'))
print('summary', summary.get('summary'))
print('wrote', str(out) + '.cluster.json')
PY
ls -la '$AFS_OUT_DIR' | head -40
"

echo "==> pull to $LOG_DIR"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "vcctl pod exec -i ${PODS[0]} -- bash -c 'tar -C $AFS_OUT_DIR -cf - .' " \
  > "$LOG_DIR/results.tar" || true
mkdir -p "$LOG_DIR/results"
tar -xf "$LOG_DIR/results.tar" -C "$LOG_DIR/results" 2>/dev/null || true
ls -la "$LOG_DIR/results" | head -30

if [[ "$FAIL" -ne 0 ]]; then
  echo "部分节点失败，检查 $LOG_DIR/*.log"
  exit 1
fi
echo "CARD_CONSTITUTION_128_OK → $AFS_OUT_DIR / $LOG_DIR"
echo "NOTE: 跑前请先 sync 含 stage_c.py 的 CARD_SCREEN 到 AFS（sync_to_afs.sh）"
