#!/usr/bin/env bash
# HCCL op-level microbench 上机推送（scp bench → torchrun on pods）
# 环境变量:
#   SCALES     = "128,256,512,1024"          # world_size 阶梯
#   SIZES      = "1M,16M,64M,256M"           # primitive nbytes
#   OPS        = "all_reduce,barrier,mm_all_reduce,moe_dispatch"
#   ITERS      = 50
#   WARMUP     = 5
#   DTYPE      = "fp16"                       # fp16 / bf16 / fp32
#   MSPROF     = 0                            # last N iter 开 msprof；0=关
#   MASTER_ADDR / MASTER_PORT                 # 默认 huawei-8node-copy-master-0
#   AFS_OUT_ROOT  = /afs-a3-weight-share/yinjinrun.p-huawei/results
#   LOG_DIR       = <auto>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
AFS_OUT_ROOT="${AFS_OUT_ROOT:-/afs-a3-weight-share/yinjinrun.p-huawei/results}"
AFS_OUT="${AFS_OUT_ROOT}/hccl-op-${STAMP}"
AFS_SCRIPTS="${AFS_WORKSPACE}/scripts/cluster"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/hccl-op-${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/hccl.log") 2>&1

SCALES="${SCALES:-16,32,64,128}"
SIZES="${SIZES:-1M,16M,64M,256M}"
OPS="${OPS:-all_reduce,all_gather,reduce_scatter,broadcast}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"
DTYPE="${DTYPE:-fp16}"
MSPROF="${MSPROF:-0}"

MASTER_ADDR="${MASTER_ADDR:-huawei-8node-copy-master-0.huawei-8node-copy}"
MASTER_PORT="${MASTER_PORT:-29501}"

echo "==> SCALES=$SCALES SIZES=$SIZES OPS=$OPS ITERS=$ITERS DTYPE=$DTYPE MSPROF=$MSPROF"
echo "==> AFS_OUT=$AFS_OUT"
echo "==> LOG_DIR=$LOG_DIR"

# 上传 bench 脚本到 AFS（一次性）
echo "==> sync bench script to AFS"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'mkdir -p \"$AFS_SCRIPTS\" \"$AFS_OUT\"'"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'cat > \"$AFS_SCRIPTS/hccl_torch_bench.py\"'" \
  < "$SCRIPT_DIR/hccl_torch_bench.py"

pod_for_rank() {
  local r="$1"
  if [[ "$r" -eq 0 ]]; then echo "${CLUSTER_JOB}-master-0"; else echo "${CLUSTER_JOB}-worker-$((r-1))"; fi
}

run_scale() {
  local world_npu="$1"
  local nnodes=$((world_npu / 16))
  local scale_dir="$AFS_OUT/scale_${world_npu}"
  local out="$scale_dir/scale_${world_npu}.jsonl"
  local msprof_dir="$scale_dir/msprof"

  echo "==> scale=$world_npu nnodes=$nnodes -> $scale_dir"

  # mkdir 一次
  ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'mkdir -p \"$scale_dir\" \"$msprof_dir\"'"

  local pids=()
  local r=0
  while [[ "$r" -lt "$nnodes" ]]; do
    local pod
    pod="$(pod_for_rank "$r")"
    local logf="$LOG_DIR/scale${world_npu}_rank${r}.log"
    local cmd
    cmd="$(cat <<PYCMD
set -euo pipefail
export PYTHONUNBUFFERED=1
cp -f '$AFS_SCRIPTS/hccl_torch_bench.py' /tmp/hccl_torch_bench.py
torchrun \\
  --nnodes=$nnodes \\
  --node_rank=$r \\
  --nproc_per_node=16 \\
  --master_addr=$MASTER_ADDR \\
  --master_port=$MASTER_PORT \\
  /tmp/hccl_torch_bench.py \\
  --ops '$OPS' \\
  --sizes '$SIZES' \\
  --iters $ITERS \\
  --warmup $WARMUP \\
  --dtype $DTYPE \\
  --msprof $MSPROF \\
  --msprof-dir '$msprof_dir' \\
  --out '$out'
echo HCCL_OP_${world_npu}_RANK_${r}_OK
PYCMD
)"
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER_SSH_HOST" \
      "vcctl pod exec ${pod} -- bash -lc $(printf '%q' "$cmd")" >"$logf" 2>&1 &
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

  # 合并 per-rank jsonl → merged 便于本机分析
  ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c '
set -euo pipefail
shopt -s nullglob
periter_parts=( \$(ls \"$scale_dir\"/scale_${world_npu}.rank*.periter.jsonl 2>/dev/null | sort -V) )
summary_parts=( \$(ls \"$scale_dir\"/scale_${world_npu}.rank*.summary.jsonl 2>/dev/null | sort -V) )
if [[ \${#periter_parts[@]} -gt 0 ]]; then
  cat \"\${periter_parts[@]}\" > \"$scale_dir/scale_${world_npu}.periter.jsonl\"
  echo MERGED_PERITER_\${#periter_parts[@]}
fi
if [[ \${#summary_parts[@]} -gt 0 ]]; then
  cat \"\${summary_parts[@]}\" > \"$scale_dir/scale_${world_npu}.summary.jsonl\"
  echo MERGED_SUMMARY_\${#summary_parts[@]}
fi
'" || true
  echo "OK scale=$world_npu → $scale_dir"
}

IFS=',' read -ra SCALE_ARR <<< "$SCALES"
for s in "${SCALE_ARR[@]}"; do
  run_scale "$s" || echo "WARN scale=$s failed, continuing"
  MASTER_PORT=$((MASTER_PORT + 1))
done

echo "==> AFS_OUT=$AFS_OUT"
echo "HCCL_OP_DONE"
