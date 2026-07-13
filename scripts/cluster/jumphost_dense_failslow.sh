#!/usr/bin/env bash
# 跳板：Dense FailSlow 实验一（固定 TP4×PP2，只扩 DP；波次占卡）
# SCALES 例：32+64,96,16
# 每卡 step time 由 hooks/failslow_step_timer 落盘（不依赖 probing._core）
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/root/.kube/config.huawei-a3-241ceshi}"
JOB="${JOB:-montyyin-moe96-r2}"
MASTER_POD="${JOB}-master-0"
SCALES="${SCALES:-32+64,96,16}"
TP="${TP:-4}"; PP="${PP:-2}"; EP="${EP:-1}"; ETP="${ETP:-1}"
GBS="${GBS:-1920}"; MBS="${MBS:-1}"; SEQ="${SEQ_LENGTH:-4096}"
# GBS_PROP_DP=1：GBS = DP * MBS * MICROBATCHES_PER_DP（每副本负载恒定，等步长对照）
GBS_PROP_DP="${GBS_PROP_DP:-0}"
MICROBATCHES_PER_DP="${MICROBATCHES_PER_DP:-160}"
ITERS="${TRAIN_ITERS:-220}"
PROBING="${PROBING:-1}"
FAILSLOW_STEP_LOG="${FAILSLOW_STEP_LOG:-1}"
SKIP_TB="${SKIP_TB:-1}"
NPUS=16
MASTER_PORT_BASE="${MASTER_PORT:-25600}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/afs-a3-weight-share/yinjinrun.p-huawei/results/dense_failslow/${STAMP}}"
AFS_WRAPPERS="/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/wrappers"
AFS_HOOKS="/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/hooks"
MEGATRON="/afs-a3-241ceshi-shared/geruijun/Megatron-LM-0.12.3"
DATA_ROOT="/afs-a3-241ceshi-shared/geruijun"
PROBING_HOME="/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/projects/Probing_plus"
WRAPPER_NAME="train_qwen3_8B_ascend.sh"
LOCAL_LOG="${LOCAL_LOG:-/tmp/dense-failslow-${STAMP}.log}"
exec > >(tee -a "$LOCAL_LOG") 2>&1

gbs_for_world() {
  local world="$1"
  if [[ "$GBS_PROP_DP" == "1" ]]; then
    local dp=$((world / (TP * PP)))
    echo $((dp * MBS * MICROBATCHES_PER_DP))
  else
    echo "$GBS"
  fi
}

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
  local scale_gbs
  scale_gbs=$(gbs_for_world "$world")
  local dp=$((world / (TP * PP)))
  local master_addr
  master_addr=$(master_dns_for_global "$node_offset")
  local pods=() r g pod
  for ((r=0;r<nnodes;r++)); do
    g=$((node_offset + r))
    pods+=("$(pod_for_global "$g")")
  done

  echo "==> scale=$world nnodes=$nnodes DP=$dp GBS=$scale_gbs (prop_dp=$GBS_PROP_DP mb/dp=$MICROBATCHES_PER_DP) offset=$node_offset MASTER=$master_addr:$master_port pods=${pods[*]}"
  vcctl pod exec "$MASTER_POD" -- bash -lc "mkdir -p '$scale_dir'"

  local done_n alive
  done_n=$(count_done "$scale_dir" "$nnodes")
  alive=$(scale_alive_on_pods "${pods[@]}")
  if [[ "$done_n" -ge "$nnodes" ]]; then
    echo "  skip spawn: already DONE ($done_n/$nnodes)"
    return 0
  fi
  if [[ "$alive" -gt 0 ]]; then
    echo "  skip spawn: trainers already alive=$alive on target pods"
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
export TP=$TP PP=$PP EP=$EP ETP=$ETP MBS=$MBS GBS=$scale_gbs SEQ_LENGTH=$SEQ
export SKIP_TB=$SKIP_TB SKIP_SAVE=1 SKIP_PROFILE=1 TRAIN_ITERS=$ITERS
export PROBING=$PROBING PROBING_HOME=$PROBING_HOME FAILSLOW_STEP_LOG=$FAILSLOW_STEP_LOG
export PATH=/root/miniconda3/envs/llm_test/bin:\$PATH
export PYTHONPATH=$AFS_HOOKS:/MindSpeed-LLM/MindSpeed:\${PYTHONPATH:-}
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
# 强制挂接每卡 step timer（.pth 才会在 torchrun 子进程生效）
SP=\$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)
if [[ -n \"\$SP\" && -d \"\$SP\" ]]; then
  printf '%s\\nimport failslow_step_timer\\n' \"$AFS_HOOKS\" > \"\$SP/zz_failslow_step.pth\"
fi
cd $MEGATRON
bash $AFS_WRAPPERS/$WRAPPER_NAME 2>&1 | tee $scale_dir/rank${r}.log
rc=\${PIPESTATUS[0]}
echo TRAIN_RANK_${r}_DONE rc=\$rc | tee -a $scale_dir/rank${r}.log
# 汇总本节点 step_times 文件数
ls $scale_dir/step_times_rank*.jsonl 2>/dev/null | wc -l | xargs -I{} echo STEP_FILES_NODE_${r}={}
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
  # 注意：bash SECONDS 是 shell 启动起算，跨波次会累积；这里用本 wait 的墙钟
  local t0 now elapsed
  t0=$(date +%s)
  local timeout_sec=${SCALE_TIMEOUT_SEC:-10800}
  # Dense 首 iter 编译/建链可 >3min；grace 用墙钟，避免误判 early exit
  local grace_sec=${SCALE_GRACE_SEC:-600}
  local done_n alive
  while true; do
    now=$(date +%s)
    elapsed=$((now - t0))
    if (( elapsed >= timeout_sec )); then
      echo "  TIMEOUT scale=$world after ${elapsed}s"
      break
    fi
    done_n=$(count_done "$scale_dir" "$nnodes")
    alive=$(scale_alive_on_pods "${pods[@]}")
    echo "  scale=$world done=$done_n/$nnodes alive=$alive elapsed=${elapsed}s $(date +%H:%M:%S)"
    if [[ "$done_n" -ge "$nnodes" ]]; then
      break
    fi
    if [[ "$alive" -eq 0 && "$done_n" -eq 0 && elapsed -gt grace_sec ]]; then
      echo "  FAIL early exit scale=$world (alive=0 after ${elapsed}s)"
      break
    fi
    sleep 60
  done
  if ! vcctl pod exec "$MASTER_POD" -- true >/dev/null 2>&1; then
    echo "FAIL scale=$world (job/pod gone)"
    return 1
  fi
  vcctl pod exec "$MASTER_POD" -- bash -lc \
    "echo STEP_FILES=\$(ls $scale_dir/step_times_rank*.jsonl 2>/dev/null | wc -l); \
     grep -hE 'throughput per GPU|elapsed time per iteration|TRAIN_RANK_|failslow_step_timer|ChildFailed|not divisible' $scale_dir/*.log 2>/dev/null | tail -40" \
    || true
  if vcctl pod exec "$MASTER_POD" -- bash -lc \
      "grep -qiE 'out of memory|NPU out of memory|ChildFailedError|not divisible' $scale_dir/*.log"; then
    echo "FAIL scale=$world"
    return 1
  fi
  if ! vcctl pod exec "$MASTER_POD" -- bash -lc \
      "grep -q 'throughput per GPU' $scale_dir/*.log"; then
    echo "FAIL scale=$world (no throughput)"
    return 1
  fi
  # 主指标门闩：至少要有 step_times 文件
  local nstep
  nstep=$(vcctl pod exec "$MASTER_POD" -- bash -lc \
    "ls $scale_dir/step_times_rank*.jsonl 2>/dev/null | wc -l" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  nstep=${nstep:-0}
  if [[ "$nstep" -lt 1 ]]; then
    echo "FAIL scale=$world (no step_times_rank*.jsonl — failslow hook missing)"
    return 1
  fi
  echo "OK scale=$world step_files=$nstep"
  return 0
}

run_wave() {
  local wave="$1"
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

echo "==> JUMPHOST Dense FailSlow STAMP=$STAMP SCALES=$SCALES PROBING=$PROBING FAILSLOW_STEP_LOG=$FAILSLOW_STEP_LOG"
echo "==> TP=$TP PP=$PP GBS=$GBS GBS_PROP_DP=$GBS_PROP_DP MICROBATCHES_PER_DP=$MICROBATCHES_PER_DP ITERS=$ITERS RUN_ROOT=$RUN_ROOT"

vcctl pod exec "$MASTER_POD" -- bash -lc "mkdir -p '$RUN_ROOT' '$AFS_WRAPPERS' '$AFS_HOOKS'"

PORT="$MASTER_PORT_BASE"
IFS=',' read -ra WAVES <<< "$SCALES"
for wave in "${WAVES[@]}"; do
  wave="${wave// /}"
  [[ -n "$wave" ]] || continue
  run_wave "$wave" "$PORT" || true
  nparts=$(awk -F'+' '{print NF}' <<<"$wave")
  PORT=$((PORT + nparts + 2))
done
echo "JUMPHOST_DENSE_FAILSLOW_DONE stamp=$STAMP → $RUN_ROOT"
