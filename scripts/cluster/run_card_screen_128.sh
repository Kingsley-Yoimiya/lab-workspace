#!/usr/bin/env bash
# 经 vcctl 扇出 CARD_SCREEN 到 8 个 pod（128 卡），结果写 AFS 再可 pull
# 用法:
#   ./scripts/cluster/run_card_screen_128.sh
#   SDC_ROUNDS=5 SUSTAINED_S=30 ./scripts/cluster/run_card_screen_128.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
CASE_NAME="${CASE_NAME:-perf128}"
RUN_ID="${RUN_ID:-${STAMP}-${CASE_NAME}}"
AFS_CS="${AFS_WORKSPACE}/projects/CARD_SCREEN"
AFS_OUT_ROOT="${AFS_RESULTS:-/afs-a3-241ceshi-shared/montyyin/results}"
AFS_OUT_DIR="${AFS_OUT_ROOT}/card_screen-${RUN_ID}"
OUT_JSONL="${AFS_OUT_DIR}/${CASE_NAME}.jsonl"

SDC_ROUNDS="${SDC_ROUNDS:-5}"
GEMM_N="${GEMM_N:-8192}"
SUSTAINED_S="${SUSTAINED_S:-30}"
IDLE_MAX_MIB="${IDLE_MAX_MIB:-1024}"
CONFIG_NAME="${CONFIG_NAME:-config.perf128.yaml}"

OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/card-screen-128-${RUN_ID}}"
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

echo "==> RUN_ID=$RUN_ID"
echo "==> AFS_OUT_DIR=$AFS_OUT_DIR"
echo "==> LOG_DIR=$LOG_DIR"
echo "==> SDC_ROUNDS=$SDC_ROUNDS GEMM_N=$GEMM_N SUSTAINED_S=$SUSTAINED_S"

# 轻量配置：开 shape_sweep，缩短单 shape 封顶
cluster_pod_exec "${PODS[0]}" "
set -euo pipefail
mkdir -p '$AFS_OUT_DIR'
cat > '$AFS_CS/$CONFIG_NAME' <<'YAML'
func_perf:
  gemm_n: 8192
  iters: 50
  warmup: 20
  dtype: bf16
  tol: 0.02
hbm:
  mb: 1024
  iters: 50
  warmup: 20
sustained:
  seconds: 30.0
  window: 50
  flatline_min_samples: 20
  flatline_iter_ms_range: 0.001
shape_sweep:
  start: 128
  stop: 16880
  min_seconds: 3.0
  min_windows: 3
  max_seconds: 20.0
  window: 50
  warmup: 10
  dtype: bf16
sdc:
  rounds: 5
health:
  ecc_uncorrected_max: 0
  telemetry_interval_s: 0.5
preflight:
  require_idle: true
  max_memory_used_mib: 1024.0
probes:
  health:
    enabled: true
  func_perf:
    enabled: true
  hbm:
    enabled: true
  sustained:
    enabled: true
  shape_sweep:
    enabled: true
  sdc_cube_gemm:
    enabled: true
  sdc_vector_fma:
    enabled: true
  sdc_sfu_identity:
    enabled: true
  sdc_mem_pattern:
    enabled: true
  sdc_reduce_chain:
    enabled: true
YAML
ls -la '$AFS_CS/$CONFIG_NAME' '$AFS_OUT_DIR'
"

run_one() {
  local pod="$1"
  local logf="$LOG_DIR/${pod}.log"
  echo "==> start $pod"
  # shellcheck disable=SC2029
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
echo SCREEN_DONE_${pod}
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
# aggregate expects the base out path; it globs result.<host>.jsonl siblings
summary = aggregate(str(out), slow_frac=0.2)
print('n_cards', summary.get('n_cards'))
print('summary', summary.get('summary'))
print('wrote', str(out) + '.cluster.json')
PY
ls -la '$AFS_OUT_DIR' | head -40
"

echo "==> pull to $LOG_DIR"
# 经 master 打包再 scp
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
echo "CARD_SCREEN_128_OK → $AFS_OUT_DIR / $LOG_DIR"
