#!/usr/bin/env bash
# HCCL scale benchmark via torch.distributed (16/32/64/128)
# 用法: ./scripts/cluster/run_hccl_scale.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
AFS_OUT="${AFS_RESULTS:-/afs-a3-241ceshi-shared/montyyin/results}/hccl-${STAMP}"
AFS_SCRIPTS="${AFS_WORKSPACE}/scripts/cluster"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/hccl-${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/hccl.log") 2>&1

SCALES="${SCALES:-16,32,64,128}"
SIZES="${SIZES:-1M,16M,64M,256M}"
OPS="${OPS:-all_reduce,all_gather,reduce_scatter,broadcast}"

MASTER_ADDR="${MASTER_ADDR:-huawei-8node-copy-master-0.huawei-8node-copy}"
MASTER_PORT="${MASTER_PORT:-29501}"

# 上传 bench 脚本到 AFS
echo "==> sync bench script to AFS"
scp -o BatchMode=yes "$SCRIPT_DIR/hccl_torch_bench.py" "${CLUSTER_SSH_HOST}:/tmp/hccl_torch_bench.py"
cluster_pod_exec "${CLUSTER_JOB}-master-0" "
mkdir -p '$AFS_SCRIPTS' '$AFS_OUT'
cat > '$AFS_SCRIPTS/hccl_torch_bench.py' < /dev/null
"
# pipe file via ssh+vcctl
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'cat > $AFS_SCRIPTS/hccl_torch_bench.py'" \
  < "$SCRIPT_DIR/hccl_torch_bench.py"

# pod list by scale
pod_for_rank() {
  local r="$1"
  if [[ "$r" -eq 0 ]]; then echo "${CLUSTER_JOB}-master-0"; else echo "${CLUSTER_JOB}-worker-$((r-1))"; fi
}

run_scale() {
  local world_npu="$1"
  local nnodes=$((world_npu / 16))
  local out="$AFS_OUT/scale_${world_npu}.jsonl"
  echo "==> scale=$world_npu nnodes=$nnodes"
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
# ensure script
cp -f '$AFS_SCRIPTS/hccl_torch_bench.py' /tmp/hccl_torch_bench.py
torchrun \
  --nnodes=$nnodes \
  --node_rank=$r \
  --nproc_per_node=16 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  /tmp/hccl_torch_bench.py \
  --ops '$OPS' \
  --sizes '$SIZES' \
  --out '$out'
echo HCCL_SCALE_${world_npu}_RANK_${r}_OK
")" >"$logf" 2>&1 &
    pids+=("$!")
    r=$((r + 1))
  done
  local fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "FAIL scale=$world_npu"
    return 1
  fi
  echo "OK scale=$world_npu → $out"
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
echo "HCCL_SCALE_DONE → $AFS_OUT / $LOG_DIR"
