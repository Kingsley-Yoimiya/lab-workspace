#!/usr/bin/env bash
# 在 ais-jump 上执行：对 montyyin-moe-scale-96 扇出 MoE 训练
# SCALES 支持「波次」：逗号分隔轮次，加号表示同轮并行占卡。
#   例：32+64,96  → 第 1 轮并行 32(2 节点)+64(4 节点)=96 卡；第 2 轮单独 96。
# 由本机 scp/ssh 上传后启动；不依赖本机长时间 SSH 挂住训练进程。
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/root/.kube/config.huawei-a3-241ceshi}"
JOB="${JOB:-montyyin-moe-scale-96}"
MASTER_POD="${JOB}-master-0"
SCALES="${SCALES:-32+64,96}"
TP="${TP:-1}"; PP="${PP:-4}"; EP="${EP:-4}"; ETP="${ETP:-1}"
# 96 卡 TP1/PP4 → DP=24，GBS 必须被 24 整除；2048%24=8 会直接 ChildFailed。
# 1920 同时整除 DP=8/16/24，便于 32/64/96 对照。
GBS="${GBS:-1920}"; MBS="${MBS:-1}"; SEQ="${SEQ_LENGTH:-4096}"
ITERS="${TRAIN_ITERS:-8}"
PROBING="${PROBING:-0}"
SKIP_TB="${SKIP_TB:-1}"
NPUS=16
MASTER_PORT_BASE="${MASTER_PORT:-25200}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/afs-a3-weight-share/yinjinrun.p-huawei/results/mfu_moe_scale/${STAMP}}"
AFS_WRAPPERS="/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/wrappers"
MEGATRON="/afs-a3-241ceshi-shared/geruijun/Megatron-LM-0.12.3"
DATA_ROOT="/afs-a3-241ceshi-shared/geruijun"
PROBING_HOME="/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/projects/Probing_plus"
LOCAL_LOG="${LOCAL_LOG:-/tmp/moe-jumphost-${STAMP}.log}"
exec > >(tee -a "$LOCAL_LOG") 2>&1

pod_for_global() {
  local g="$1"
  if [[ "$g" -eq 0 ]]; then echo "${JOB}-master-0"; else echo "${JOB}-worker-$((g-1))"; fi
}

master_dns_for_global() {
  local g="$1"
  local pod
  pod=$(pod_for_global "$g")
  echo "${pod}.${JOB}"
}

count_done() {
  local scale_dir="$1" nnodes="$2"
  local n
  n=$(vcctl pod exec "$MASTER_POD" -- bash -lc \
    "grep -l 'TRAIN_RANK_.*_DONE' $scale_dir/rank*.log $scale_dir/nohup_rank*.log 2>/dev/null | wc -l" \
    2>/dev/null | grep -oE '[0-9]+' | tail -1)
  echo "${n:-0}"
}

scale_alive_on_pods() {
  # 在指定 global node 列表上是否仍有 torchrun/pretrain
  local pods=("$@")
  local p total=0 n
  for p in "${pods[@]}"; do
    n=$(vcctl pod exec "$p" -- bash -lc \
      "ps -ef | grep -E 'torchrun|pretrain_gpt' | grep -v grep | wc -l" \
      2>/dev/null | grep -oE '[0-9]+' | tail -1)
    total=$((total + ${n:-0}))
  done
  echo "$total"
}

spawn_scale() {
  local world="$1" node_offset="$2" master_port="$3"
  local nnodes=$((world / NPUS))
  local scale_dir="$RUN_ROOT/scale_${world}"
  local master_addr
  master_addr=$(master_dns_for_global "$node_offset")
  local pods=()
  local r g pod
  for ((r=0;r<nnodes;r++)); do
    g=$((node_offset + r))
    pods+=("$(pod_for_global "$g")")
  done

  echo "==> scale=$world nnodes=$nnodes offset=$node_offset MASTER=$master_addr:$master_port pods=${pods[*]}"
  vcctl pod exec "$MASTER_POD" -- bash -lc "mkdir -p '$scale_dir'"

  local done_n alive
  done_n=$(count_done "$scale_dir" "$nnodes")
  alive=$(scale_alive_on_pods "${pods[@]}")
  if [[ "$done_n" -ge "$nnodes" ]]; then
    echo "  skip spawn: already DONE ($done_n/$nnodes)"
    return 0
  fi
  if [[ "$alive" -gt 0 ]]; then
    echo "  skip spawn: trainers already alive=$alive on target pods (reuse in-flight)"
    return 0
  fi

  for pod in "${pods[@]}"; do
    vcctl pod exec "$pod" -- bash -lc \
      "pkill -9 -f pretrain_gpt.py || true; pkill -9 -f 'torchrun.*pretrain' || true" \
      >/dev/null 2>&1 || true
  done
  sleep 2

  for ((r=0;r<nnodes;r++)); do
    g=$((node_offset + r))
    pod=$(pod_for_global "$g")
    launch="$scale_dir/launch_rank${r}.sh"
    vcctl pod exec "$MASTER_POD" -- bash -lc "cat > '$launch' <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
export TP=$TP PP=$PP EP=$EP ETP=$ETP MBS=$MBS GBS=$GBS SEQ_LENGTH=$SEQ
export SKIP_TB=$SKIP_TB SKIP_SAVE=1 SKIP_PROFILE=1 TRAIN_ITERS=$ITERS
export PROBING=$PROBING PROBING_HOME=$PROBING_HOME
export PATH=/root/miniconda3/envs/llm_test/bin:\$PATH
export PYTHONPATH=/MindSpeed-LLM/MindSpeed:\${PYTHONPATH:-}
export WORLD_SIZE=$nnodes NNODES=$nnodes RANK=$r NODE_RANK=$r
export MASTER_ADDR=$master_addr MASTER_PORT=$master_port
export NPUS_PER_NODE=$NPUS GPUS_PER_NODE=$NPUS
export DATA_ROOT=$DATA_ROOT RUN_DIR=$scale_dir LOG_DIR=$scale_dir/
export TENSORBOARD_DIR=$scale_dir/tb CKPT_SAVE_DIR=$scale_dir/ckpt
export HCCL_IF_BASE_PORT=$((master_port+2000))
export HCCL_EXEC_TIMEOUT=\${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_CONNECT_TIMEOUT=\${HCCL_CONNECT_TIMEOUT:-3600}
if [[ \"\$PROBING\" != \"0\" ]]; then
  [[ -f /afs-a3-weight-share/yinjinrun.p-huawei/toolchains/rust-env.sh ]] && source /afs-a3-weight-share/yinjinrun.p-huawei/toolchains/rust-env.sh || true
  export PYTHONPATH=\$PROBING_HOME:\$PROBING_HOME/python:\$PYTHONPATH
fi
mkdir -p \"\$RUN_DIR\" \"\$LOG_DIR\" \"\$TENSORBOARD_DIR\" \"\$CKPT_SAVE_DIR\"
cd $MEGATRON
bash $AFS_WRAPPERS/train_qwen3_30B_A3B_ascend.sh 2>&1 | tee $scale_dir/rank${r}.log
rc=\${PIPESTATUS[0]}
echo TRAIN_RANK_${r}_DONE rc=\$rc | tee -a $scale_dir/rank${r}.log
exit \$rc
EOF
chmod +x '$launch'"
    vcctl pod exec "$pod" -- bash -lc \
      "setsid nohup bash $launch >$scale_dir/nohup_rank${r}.log 2>&1 & echo SPAWNED_\$!"
    sleep 5
  done
}

wait_scale() {
  local world="$1" node_offset="$2"
  local nnodes=$((world / NPUS))
  local scale_dir="$RUN_ROOT/scale_${world}"
  local pods=() r g
  for ((r=0;r<nnodes;r++)); do
    g=$((node_offset + r))
    pods+=("$(pod_for_global "$g")")
  done
  local deadline=$((SECONDS + ${SCALE_TIMEOUT_SEC:-3600}))
  local done_n alive
  while (( SECONDS < deadline )); do
    done_n=$(count_done "$scale_dir" "$nnodes")
    alive=$(scale_alive_on_pods "${pods[@]}")
    echo "  scale=$world done=$done_n/$nnodes alive=$alive $(date +%H:%M:%S)"
    if [[ "$done_n" -ge "$nnodes" ]]; then
      break
    fi
    if [[ "$alive" -eq 0 && "$done_n" -eq 0 && SECONDS -gt 120 ]]; then
      echo "  FAIL early exit scale=$world"
      break
    fi
    sleep 45
  done
  if ! vcctl pod exec "$MASTER_POD" -- true >/dev/null 2>&1; then
    echo "FAIL scale=$world (job/pod gone during run — preempted or deleted)"
    return 1
  fi
  vcctl pod exec "$MASTER_POD" -- bash -lc \
    "grep -hE 'throughput per GPU|elapsed time per iteration|TRAIN_RANK_|OOM|ChildFailed|not divisible' $scale_dir/*.log 2>/dev/null | tail -60" \
    || true
  if vcctl pod exec "$MASTER_POD" -- bash -lc \
      "grep -qiE 'out of memory|NPU out of memory|ChildFailedError|not divisible' $scale_dir/*.log"; then
    echo "FAIL scale=$world"
    return 1
  fi
  # 无吞吐也算未成功（避免空跑 OK）
  if ! vcctl pod exec "$MASTER_POD" -- bash -lc \
      "grep -q 'throughput per GPU' $scale_dir/*.log"; then
    echo "FAIL scale=$world (no throughput lines)"
    return 1
  fi
  echo "OK scale=$world"
  return 0
}

run_wave() {
  local wave="$1"  # e.g. 32+64 or 96
  local port_cursor="$2"
  echo "==> WAVE '$wave' port_base=$port_cursor"
  local parts=()
  IFS='+' read -ra parts <<< "$wave"
  local total_npu=0 w
  for w in "${parts[@]}"; do
    total_npu=$((total_npu + w))
  done
  if (( total_npu > 96 )); then
    echo "FAIL wave '$wave' needs ${total_npu} NPUs > 96"
    return 1
  fi

  local offset=0
  local -a worlds=() offsets=() ports=()
  local i=0
  for w in "${parts[@]}"; do
    worlds+=("$w")
    offsets+=("$offset")
    ports+=("$((port_cursor + i))")
    offset=$((offset + w / NPUS))
    i=$((i + 1))
  done

  for i in "${!worlds[@]}"; do
    spawn_scale "${worlds[$i]}" "${offsets[$i]}" "${ports[$i]}"
  done

  local fail=0
  for i in "${!worlds[@]}"; do
    wait_scale "${worlds[$i]}" "${offsets[$i]}" || fail=1
  done
  return "$fail"
}

echo "==> JUMPHOST MoE scale STAMP=$STAMP SCALES=$SCALES PROBING=$PROBING"
echo "==> RUN_ROOT=$RUN_ROOT (waves: comma=sequential, plus=parallel pack)"

vcctl pod exec "$MASTER_POD" -- bash -lc "mkdir -p '$RUN_ROOT' '$AFS_WRAPPERS'"

PORT="$MASTER_PORT_BASE"
IFS=',' read -ra WAVES <<< "$SCALES"
for wave in "${WAVES[@]}"; do
  wave="${wave// /}"
  [[ -n "$wave" ]] || continue
  run_wave "$wave" "$PORT" || true
  # 每波用一段端口，避免与残留 HCCL 冲突；并行子任务已占 +0..n-1
  nparts=$(awk -F'+' '{print NF}' <<<"$wave")
  PORT=$((PORT + nparts + 2))
done
echo "JUMPHOST_MOE_DONE stamp=$STAMP → $RUN_ROOT"
