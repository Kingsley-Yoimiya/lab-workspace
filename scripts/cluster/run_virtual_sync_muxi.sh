#!/usr/bin/env bash
# MUXI Phase0 虚拟同步实验编排（机间不通约束版）
# 实验0：单节点 8 卡 real_sync vs independent 校准
# 实验1：16 节点 × 8 卡独立负载，事后子集重构
#
# 用法（本机）:
#   CLUSTER_FORCE_JUMP=1 bash scripts/cluster/run_virtual_sync_muxi.sh
#   PHASE=0 CLUSTER_FORCE_JUMP=1 bash scripts/cluster/run_virtual_sync_muxi.sh
#   PHASE=1 CLUSTER_FORCE_JUMP=1 bash scripts/cluster/run_virtual_sync_muxi.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_FORCE_JUMP="${CLUSTER_FORCE_JUMP:-1}"
export CLUSTER_SSH_HOST_OVERRIDE="${CLUSTER_SSH_HOST_OVERRIDE:-ais-cf3e61a5}"
# shellcheck source=muxi.env
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

PHASE="${PHASE:-all}"  # 0|1|all
ITERS="${ITERS:-1000}"
WARMUP="${WARMUP:-20}"
HIDDEN="${HIDDEN:-4096}"
SEQ="${SEQ:-2048}"
LAYERS="${LAYERS:-8}"
BATCH="${BATCH:-2}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
AFS_BENCH="/afs-a3-weight-share/yinjinrun.p/lab-workspace/scripts/cluster"
AFS_OUT="${AFS_OUT:-/afs-a3-weight-share/yinjinrun.p/results/muxi-vsync-${STAMP}}"
REPO_LOG="${REPO_LOG:-/Users/yinjinrun/random-thing/logs/muxi-vsync-${STAMP}}"
mkdir -p "$REPO_LOG"
exec > >(tee -a "$REPO_LOG/orchestrator.log") 2>&1

JOB="$CLUSTER_JOB"
MASTER="${JOB}-master-0"
pods=("${JOB}-master-0")
for i in $(seq 0 14); do pods+=("${JOB}-worker-$i"); done

echo "==== MUXI virtual-sync PHASE=$PHASE STAMP=$STAMP ===="
echo "JOB=$JOB AFS_OUT=$AFS_OUT ITERS=$ITERS"

safe_kill_pod() {
  local pod="$1"
  # awk 字符类避免匹配本 shell；跳过自身 PID 与 defunct
  cluster_pod_exec "$pod" 'set +e; me=$$; ps -eo pid=,args= | awk -v me="$me" '\''$1==me{next} /defunct/{next} /[t]orchrun|[p]retrain_gpt|[m]fu_train_bench|[v]irtual_sync_bench|[n]ccl_torch_bench/{print $1}'\'' | while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done; echo KILL_OK $(hostname)'
}

echo "==> kill residual trainers on all 16 pods"
# 有界并行，避开 cluster_fanout_run 在 set -u 下空数组问题
running=0
for pod in "${pods[@]}"; do
  while [[ "$running" -ge 4 ]]; do
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  done
  ( safe_kill_pod "$pod" || echo "WARN kill $pod" ) &
  running=$((running + 1))
done
wait || true
echo "KILL_PHASE_DONE"

echo "==> upload bench + mkdir"
cluster_pod_exec "$MASTER" "mkdir -p '$AFS_BENCH' '$AFS_OUT'"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${MASTER} -- bash -c 'cat > ${AFS_BENCH}/virtual_sync_bench.py && wc -c ${AFS_BENCH}/virtual_sync_bench.py'" \
  < "$SCRIPT_DIR/virtual_sync_bench.py" | tee "$REPO_LOG/upload_bench.log"

# 写节点启动脚本到 AFS
cluster_pod_exec "$MASTER" "cat > '$AFS_OUT/env_common.sh' <<'EOF'
export PATH=/opt/conda/bin:\${PATH:-/usr/bin}
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0
export MCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
# 机间不通：强制走机内
export NCCL_IB_DISABLE=\${NCCL_IB_DISABLE:-1}
export MCCL_IB_DISABLE=\${MCCL_IB_DISABLE:-1}
CU_BRIDGE_BIN=/opt/maca/tools/cu-bridge/bin
if [[ -x \"\$CU_BRIDGE_BIN/cucc\" && ! -e \"\$CU_BRIDGE_BIN/nvcc\" ]]; then
  ln -sfn \"\$CU_BRIDGE_BIN/cucc\" \"\$CU_BRIDGE_BIN/nvcc\" || true
fi
export CUDA_HOME=/opt/maca/tools/cu-bridge
EOF"

fire_exp0() {
  local out_real="$AFS_OUT/exp0_real"
  local out_virt="$AFS_OUT/exp0_virtual"
  cluster_pod_exec "$MASTER" "mkdir -p '$out_real' '$out_virt'"

  echo "==> EXP0-A real_sync on master 8 GPUs"
  cluster_pod_exec "$MASTER" "
set -euo pipefail
source '$AFS_OUT/env_common.sh'
cd '$AFS_OUT'
rm -f '$out_real'/done_rank*.txt '$out_real'/step_times_rank*.jsonl
nohup bash -lc '
source $AFS_OUT/env_common.sh
export GPUS_PER_NODE=8 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501
torchrun --standalone --nproc_per_node=8 $AFS_BENCH/virtual_sync_bench.py \
  --mode real_sync --iters $ITERS --warmup $WARMUP \
  --hidden $HIDDEN --seq $SEQ --layers $LAYERS --batch $BATCH \
  --out-dir $out_real --tag exp0_real \
  >$out_real/torchrun.log 2>&1
echo REAL_DONE \$? >>$out_real/torchrun.log
' >/dev/null 2>&1 &
echo FIRED_REAL \$!
"

  # 等 real 完成（预估：1000 step × ~0.3-1s ≈ 5-20min）
  local deadline=$((SECONDS + ${EXP0_TIMEOUT:-1800}))
  while (( SECONDS < deadline )); do
    local n
    n=$(cluster_pod_exec "$MASTER" "ls '$out_real'/done_rank*.txt 2>/dev/null | wc -l" | tr -dc '0-9')
    n=${n:-0}
    echo "  exp0_real done=$n/8"
    [[ "$n" -ge 8 ]] && break
    # 早期失败检测
    if cluster_pod_exec "$MASTER" "test -f '$out_real/torchrun.log' && grep -qE 'Error|Traceback|NCCL error' '$out_real/torchrun.log'" >/dev/null 2>&1; then
      echo "FAIL exp0_real early error"; cluster_pod_exec "$MASTER" "tail -40 '$out_real/torchrun.log'"; return 1
    fi
    sleep 20
  done
  n=$(cluster_pod_exec "$MASTER" "ls '$out_real'/done_rank*.txt 2>/dev/null | wc -l" | tr -dc '0-9')
  [[ "${n:-0}" -ge 8 ]] || { echo "FAIL exp0_real timeout done=$n"; cluster_pod_exec "$MASTER" "tail -60 '$out_real/torchrun.log'"; return 1; }

  echo "==> EXP0-B independent on master 8 GPUs"
  cluster_pod_exec "$MASTER" "
set -euo pipefail
source '$AFS_OUT/env_common.sh'
pkill -9 -f '[v]irtual_sync_bench' 2>/dev/null || true
sleep 2
rm -f '$out_virt'/done_rank*.txt '$out_virt'/step_times_rank*.jsonl
for g in \$(seq 0 7); do
  nohup bash -lc \"
    source $AFS_OUT/env_common.sh
    export CUDA_VISIBLE_DEVICES=\$g
    export LOCAL_RANK=0 NODE_RANK=0 GPUS_PER_NODE=8 GLOBAL_RANK=\$g
    python3 $AFS_BENCH/virtual_sync_bench.py \
      --mode independent --iters $ITERS --warmup $WARMUP \
      --hidden $HIDDEN --seq $SEQ --layers $LAYERS --batch $BATCH \
      --out-dir $out_virt --tag exp0_virtual \
      >$out_virt/gpu\${g}.log 2>&1
  \" >/dev/null 2>&1 &
  echo FIRED_VIRT_GPU\$g \$!
done
"
  deadline=$((SECONDS + ${EXP0_TIMEOUT:-1800}))
  while (( SECONDS < deadline )); do
    n=$(cluster_pod_exec "$MASTER" "ls '$out_virt'/done_rank*.txt 2>/dev/null | wc -l" | tr -dc '0-9')
    n=${n:-0}
    echo "  exp0_virtual done=$n/8"
    [[ "$n" -ge 8 ]] && break
    sleep 20
  done
  n=$(cluster_pod_exec "$MASTER" "ls '$out_virt'/done_rank*.txt 2>/dev/null | wc -l" | tr -dc '0-9')
  [[ "${n:-0}" -ge 8 ]] || { echo "FAIL exp0_virtual timeout done=$n"; return 1; }
  echo "EXP0_DONE"
}

fire_exp1() {
  local out="$AFS_OUT/exp1_independent"
  cluster_pod_exec "$MASTER" "mkdir -p '$out'"
  echo "==> EXP1 independent on all 16 nodes × 8 GPUs"

  local idx=0
  for pod in "${pods[@]}"; do
    local node_rank=$idx
    cluster_pod_exec "$pod" "
set +e
source '$AFS_OUT/env_common.sh'
pkill -9 -f '[v]irtual_sync_bench' 2>/dev/null || true
sleep 1
mkdir -p '$out'
for g in \$(seq 0 7); do
  gr=\$(( $node_rank * 8 + g ))
  nohup bash -lc \"
    source $AFS_OUT/env_common.sh
    export CUDA_VISIBLE_DEVICES=\$g
    export LOCAL_RANK=0 NODE_RANK=$node_rank GPUS_PER_NODE=8 GLOBAL_RANK=\$gr
    python3 $AFS_BENCH/virtual_sync_bench.py \
      --mode independent --iters $ITERS --warmup $WARMUP \
      --hidden $HIDDEN --seq $SEQ --layers $LAYERS --batch $BATCH \
      --out-dir $out --tag exp1 \
      >$out/node${node_rank}_gpu\${g}.log 2>&1
  \" >/dev/null 2>&1 &
done
echo FIRED_NODE $node_rank on \$(hostname)
" | tee -a "$REPO_LOG/exp1_fire.log"
    idx=$((idx + 1))
    sleep 1
  done

  local deadline=$((SECONDS + ${EXP1_TIMEOUT:-3600}))
  while (( SECONDS < deadline )); do
    local n
    n=$(cluster_pod_exec "$MASTER" "ls '$out'/done_rank*.txt 2>/dev/null | wc -l" | tr -dc '0-9')
    n=${n:-0}
    echo "  exp1 done=$n/128"
    [[ "$n" -ge 128 ]] && break
    # 抽样看一个 log
    if (( n == 0 )); then
      cluster_pod_exec "$MASTER" "tail -5 '$out'/node0_gpu0.log 2>/dev/null || true" || true
    fi
    sleep 30
  done
  n=$(cluster_pod_exec "$MASTER" "ls '$out'/done_rank*.txt 2>/dev/null | wc -l" | tr -dc '0-9')
  [[ "${n:-0}" -ge 128 ]] || { echo "FAIL exp1 timeout done=$n"; return 1; }
  echo "EXP1_DONE n=$n"
}

case "$PHASE" in
  0) fire_exp0 ;;
  1) fire_exp1 ;;
  all) fire_exp0; fire_exp1 ;;
  *) echo "bad PHASE=$PHASE"; exit 2 ;;
esac

echo "AFS_OUT=$AFS_OUT"
echo "REPO_LOG=$REPO_LOG"
echo "NEXT: 拉回结果后跑 parse_virtual_sync.py"
echo "ORCH_DONE"
