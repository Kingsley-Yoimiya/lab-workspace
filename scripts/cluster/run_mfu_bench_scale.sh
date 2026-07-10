#!/usr/bin/env bash
# 用 mfu_train_bench.py 在 16/32/64/128 上测 dense/moe MFU（torchrun+HCCL）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

MODE="${MODE:-dense}"
SCALES="${SCALES:-16,32,64,128}"
ITERS="${ITERS:-15}"
STAMP="$(date +%Y%m%d_%H%M%S)"
AFS_OUT="/afs-a3-241ceshi-shared/montyyin/results/mfu-${MODE}-${STAMP}"
AFS_SCRIPTS="/afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/mfu-${MODE}-${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/mfu.log") 2>&1

MASTER_ADDR="huawei-8node-copy-master-0.huawei-8node-copy"
MASTER_PORT="${MASTER_PORT:-29701}"

echo "==> MODE=$MODE SCALES=$SCALES OUT=$AFS_OUT"

ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'mkdir -p $AFS_SCRIPTS $AFS_OUT && cat > $AFS_SCRIPTS/mfu_train_bench.py'" \
  < "$SCRIPT_DIR/mfu_train_bench.py"

pod_for_rank() {
  local r="$1"
  if [[ "$r" -eq 0 ]]; then echo "${CLUSTER_JOB}-master-0"; else echo "${CLUSTER_JOB}-worker-$((r-1))"; fi
}

run_scale() {
  local world="$1"
  local nnodes=$((world / 16))
  local out="$AFS_OUT/scale_${world}.jsonl"
  echo "==> scale=$world"
  local pids=() r=0
  while [[ "$r" -lt "$nnodes" ]]; do
    local pod logf
    pod="$(pod_for_rank "$r")"
    logf="$LOG_DIR/scale${world}_rank${r}.log"
    # 写每 rank 启动脚本，避免 vcctl 参数解析问题
    local launcher="$AFS_OUT/launch_w${world}_r${r}.sh"
    ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
      "vcctl pod exec ${CLUSTER_JOB}-master-0 -- bash -lc $(printf '%q' "cat > $launcher <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export HCCL_IF_BASE_PORT=$((MASTER_PORT + 2000))
torchrun --nnodes=$nnodes --node_rank=$r --nproc_per_node=16 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  $AFS_SCRIPTS/mfu_train_bench.py --mode $MODE --iters $ITERS --out $out
echo MFU_OK_w${world}_r${r}
EOF
chmod +x $launcher")" >/dev/null
    ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
      "vcctl pod exec ${pod} -- bash $launcher" >"$logf" 2>&1 &
    pids+=("$!")
    r=$((r+1))
  done
  local fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  if [[ "$fail" -ne 0 ]]; then echo "FAIL scale=$world"; return 1; fi
  echo "OK scale=$world"
}

IFS=',' read -ra ARR <<< "$SCALES"
for s in "${ARR[@]}"; do
  run_scale "$s" || true
  # 下一档换端口，避免 Bind_IP_Port
  MASTER_PORT=$((MASTER_PORT + 10))
done

ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "vcctl pod exec -i ${CLUSTER_JOB}-master-0 -- bash -c 'tar -C $AFS_OUT -cf - .' " \
  > "$LOG_DIR/results.tar"
mkdir -p "$LOG_DIR/results"
tar -xf "$LOG_DIR/results.tar" -C "$LOG_DIR/results"
ls -la "$LOG_DIR/results"
echo "MFU_BENCH_DONE → $AFS_OUT / $LOG_DIR"
