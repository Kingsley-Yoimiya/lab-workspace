#!/usr/bin/env bash
# HCCL P2P 抽样微基准扇出（默认 smoke: 16 卡；可扩到 128）
# 用法:
#   ./scripts/cluster/run_hccl_p2p_128.sh              # SCALES=16
#   SCALES=16,128 ./scripts/cluster/run_hccl_p2p_128.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
AFS_OUT="${AFS_RESULTS:-/afs-a3-241ceshi-shared/montyyin/results}/hccl-p2p-${STAMP}"
AFS_SCRIPTS="${AFS_WORKSPACE}/scripts/cluster"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Track B R0 本地日志约定: logs/hccl-cluster-r0-<ts>/
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/hccl-cluster-r0-${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/hccl_p2p.log") 2>&1

# smoke 默认单节点 16；全量: SCALES=16,32,64,128 或 SCALES=128
SCALES="${SCALES:-16}"
SIZES="${SIZES:-64K,16M}"
# 空=Python 按 world 自动选（>=64 仅 ring，防 star@128 SIGSEGV）；显式: STRATEGIES=ring,star
STRATEGIES="${STRATEGIES-}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"

MASTER_ADDR="${MASTER_ADDR:-huawei-8node-copy-master-0.huawei-8node-copy}"
MASTER_PORT="${MASTER_PORT:-29601}"

echo "==> sync p2p bench script to AFS"
cluster_pod_exec "${CLUSTER_JOB}-master-0" "mkdir -p '$AFS_SCRIPTS' '$AFS_OUT'"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'cat > $AFS_SCRIPTS/hccl_p2p_bench.py'" \
  < "$SCRIPT_DIR/hccl_p2p_bench.py"

pod_for_rank() {
  local r="$1"
  if [[ "$r" -eq 0 ]]; then echo "${CLUSTER_JOB}-master-0"; else echo "${CLUSTER_JOB}-worker-$((r-1))"; fi
}

run_scale() {
  local world_npu="$1"
  local nnodes=$((world_npu / 16))
  local out="$AFS_OUT/scale_${world_npu}.jsonl"
  echo "==> p2p scale=$world_npu nnodes=$nnodes port=$MASTER_PORT"
  local pids=()
  local r=0
  while [[ "$r" -lt "$nnodes" ]]; do
    local pod
    pod="$(pod_for_rank "$r")"
    local logf="$LOG_DIR/scale${world_npu}_rank${r}.log"
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER_SSH_HOST" \
      "vcctl pod exec ${pod} -- bash -lc $(printf '%q' "
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /tmp
cp -f '$AFS_SCRIPTS/hccl_p2p_bench.py' /tmp/hccl_p2p_bench.py
torchrun \
  --nnodes=$nnodes \
  --node_rank=$r \
  --nproc_per_node=16 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  /tmp/hccl_p2p_bench.py \
  --sizes '$SIZES' \
  --strategies '$STRATEGIES' \
  --warmup $WARMUP \
  --iters $ITERS \
  --out '$out'
echo HCCL_P2P_${world_npu}_RANK_${r}_OK
")" >"$logf" 2>&1 &
    pids+=("$!")
    r=$((r + 1))
  done
  local fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "FAIL p2p scale=$world_npu"
    return 1
  fi
  # 合并 per-rank JSONL → scale_N.jsonl（便于下游）
  cluster_pod_exec "${CLUSTER_JOB}-master-0" "
set -euo pipefail
shopt -s nullglob
parts=(\$(ls '$AFS_OUT'/scale_${world_npu}.rank*.jsonl 2>/dev/null | sort -V))
if [[ \${#parts[@]} -gt 0 ]]; then
  cat \"\${parts[@]}\" > '$out'
  echo MERGED_\${#parts[@]}_PARTS_TO_$out
else
  echo WARN_NO_RANK_PARTS_scale_${world_npu}
fi
" || true
  echo "OK p2p scale=$world_npu → $out"
}

IFS=',' read -ra SCALE_ARR <<< "$SCALES"
for s in "${SCALE_ARR[@]}"; do
  run_scale "$s" || true
  # 换 port 避免 TIME_WAIT
  MASTER_PORT=$((MASTER_PORT + 1))
done

echo "==> pull"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'tar -C $AFS_OUT -cf - .' " \
  > "$LOG_DIR/results.tar" || true
mkdir -p "$LOG_DIR/results"
tar -xf "$LOG_DIR/results.tar" -C "$LOG_DIR/results" 2>/dev/null || true
ls -la "$LOG_DIR/results" || true
echo "HCCL_P2P_DONE → $AFS_OUT / $LOG_DIR"
