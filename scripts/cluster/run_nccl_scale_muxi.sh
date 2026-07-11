#!/usr/bin/env bash
# Muxi NCCL scale（对标 run_hccl_scale.sh；每节点 8 卡）
#
# 关键：在 pod 内 nohup 启动 torchrun，本机只短连启动 + 轮询 AFS done 标记。
# 这样 Cursor/nohup 父进程被杀时，远端 torchrun 不会跟着死。
#
# 用法:
#   ./scripts/cluster/run_nccl_scale_muxi.sh
#   SCALES=16,32,64,128 AFS_OUT=... LOG_DIR=... ./scripts/cluster/run_nccl_scale_muxi.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=muxi.env
source "$SCRIPT_DIR/muxi.env"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
AFS_OUT="${AFS_OUT:-${AFS_RESULTS}/nccl-${STAMP}}"
AFS_SCRIPTS="/afs-a3-weight-share/montyyin/lab-workspace/scripts/cluster"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/muxi-nccl-${STAMP}}"
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/nccl.log" 2>&1
echo "==== START $(date '+%Y-%m-%dT%H:%M:%S') ===="

SCALES="${SCALES:-8,16,32,64,128}"
SIZES="${SIZES:-1M,16M,64M,256M}"
OPS="${OPS:-all_reduce,all_gather,reduce_scatter,broadcast}"
NPROC="${DEVICES_PER_NODE:-8}"
POLL_SEC="${POLL_SEC:-20}"
POLL_MAX="${POLL_MAX:-180}" # 20s * 180 = 60min / scale

MASTER_ADDR="${MASTER_ADDR:-${CLUSTER_JOB}-master-0.${CLUSTER_JOB}}"
MASTER_PORT="${MASTER_PORT:-29601}"

echo "==> PROFILE=muxi KUBE=$CLUSTER_KUBECONFIG"
echo "==> SCALES=$SCALES NPROC=$NPROC MASTER_ADDR=$MASTER_ADDR"
echo "==> AFS_OUT=$AFS_OUT LOG_DIR=$LOG_DIR"

POD_NODES=("${CLUSTER_JOB}-master-0")
for i in $(seq 0 "$((CLUSTER_N_WORKERS - 1))"); do
  POD_NODES+=("${CLUSTER_JOB}-worker-${i}")
done
echo "==> nodes(${#POD_NODES[@]}): ${POD_NODES[*]}"

echo "==> upload nccl_torch_bench.py"
cluster_pod_exec "${CLUSTER_POD}" "mkdir -p '$AFS_SCRIPTS' '$AFS_OUT'"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${CLUSTER_POD} -- bash -c 'cat > $AFS_SCRIPTS/nccl_torch_bench.py'" \
  < "$SCRIPT_DIR/nccl_torch_bench.py"

fire_node() {
  local world="$1" nnodes="$2" r="$3" out="$4"
  local pod="${POD_NODES[$r]}"
  local donef="$AFS_OUT/scale_${world}.node_${r}.done"
  local failf="$AFS_OUT/scale_${world}.node_${r}.fail"
  local rlog="$AFS_OUT/scale_${world}.node_${r}.log"
  local prefix
  prefix="$(_cluster_vcctl_prefix)"
  # 远端 nohup：SSH 断开不影响 torchrun
  ssh -o BatchMode=yes -o ConnectTimeout=30 "$CLUSTER_SSH_HOST" \
    "${prefix} pod exec ${pod} -- bash -lc $(printf '%q' "
set -euo pipefail
rm -f '$donef' '$failf'
cp -f '$AFS_SCRIPTS/nccl_torch_bench.py' /tmp/nccl_torch_bench.py
nohup bash -lc '
export PYTHONUNBUFFERED=1
if torchrun --nnodes=$nnodes --node_rank=$r --nproc_per_node=$NPROC \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  /tmp/nccl_torch_bench.py --ops \"$OPS\" --sizes \"$SIZES\" --out \"$out\"
then
  echo OK > \"$donef\"
else
  echo FAIL > \"$failf\"
fi
' >'$rlog' 2>&1 &
echo STARTED_\$!
")" >"$LOG_DIR/scale${world}_noderank${r}.fire.log" 2>&1
}

wait_scale() {
  local world="$1" nnodes="$2"
  local i=0
  while [[ "$i" -lt "$POLL_MAX" ]]; do
    local status
    status="$(cluster_pod_exec "${CLUSTER_POD}" "
ok=0; fail=0; miss=0
for r in \$(seq 0 $((nnodes - 1))); do
  if [[ -f $AFS_OUT/scale_${world}.node_\$r.done ]]; then ok=\$((ok+1))
  elif [[ -f $AFS_OUT/scale_${world}.node_\$r.fail ]]; then fail=\$((fail+1))
  else miss=\$((miss+1)); fi
done
echo OK=\$ok FAIL=\$fail MISS=\$miss
" 2>/dev/null | tail -1)"
    echo "  poll[$i] scale=$world $status"
    if [[ "$status" == OK=${nnodes}\ FAIL=0\ MISS=0 ]]; then
      return 0
    fi
    if [[ "$status" == *"FAIL="* ]] && [[ "$status" != *"FAIL=0"* ]]; then
      echo "FAIL scale=$world ($status)"
      return 1
    fi
    # 把远端 log 尾部拉到本地便于盯梢
    cluster_pod_exec "${CLUSTER_POD}" "
for r in 0; do
  f=$AFS_OUT/scale_${world}.node_\$r.log
  [[ -f \$f ]] && tail -3 \$f || true
done
" 2>/dev/null | sed 's/^/    /' || true
    sleep "$POLL_SEC"
    i=$((i + 1))
  done
  echo "TIMEOUT scale=$world"
  return 1
}

run_scale() {
  local world="$1"
  local nnodes=$((world / NPROC))
  if [[ $((world % NPROC)) -ne 0 ]]; then
    echo "ERROR: world=$world not divisible by NPROC=$NPROC"
    return 1
  fi
  if [[ "$nnodes" -gt "${#POD_NODES[@]}" ]]; then
    echo "ERROR: need $nnodes nodes, have ${#POD_NODES[@]}"
    return 1
  fi
  local out="$AFS_OUT/scale_${world}.jsonl"
  echo "==> scale=$world nnodes=$nnodes port=$MASTER_PORT"
  local r=0
  while [[ "$r" -lt "$nnodes" ]]; do
    fire_node "$world" "$nnodes" "$r" "$out"
    r=$((r + 1))
    sleep 1
  done
  wait_scale "$world" "$nnodes" || return 1
  cluster_pod_exec "${CLUSTER_POD}" "
set -euo pipefail
shopt -s nullglob
parts=(\$(ls '$AFS_OUT'/scale_${world}.rank*.jsonl 2>/dev/null | sort -V))
if [[ \${#parts[@]} -gt 0 ]]; then
  cat \"\${parts[@]}\" > '$out'
  echo MERGED_\${#parts[@]}_TO_$out
else
  echo WARN_NO_RANK_PARTS
fi
" || true
  echo "OK scale=$world"
}

IFS=',' read -ra SCALE_ARR <<< "$SCALES"
for s in "${SCALE_ARR[@]}"; do
  run_scale "$s" || true
  MASTER_PORT=$((MASTER_PORT + 1))
done

echo "==> pull"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${CLUSTER_POD} -- bash -c 'tar -C $AFS_OUT -cf - .' " \
  > "$LOG_DIR/results.tar" || true
mkdir -p "$LOG_DIR/results"
tar -xf "$LOG_DIR/results.tar" -C "$LOG_DIR/results" 2>/dev/null || true
ls -la "$LOG_DIR/results" | head -30
echo "NCCL_SCALE_DONE → $AFS_OUT / $LOG_DIR"
