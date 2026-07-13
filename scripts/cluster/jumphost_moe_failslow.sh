#!/usr/bin/env bash
# MoE FailSlow 短窗：TP1 PP4 EP4，挂 failslow_step_timer
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/root/.kube/config.huawei-a3-241ceshi}"
JOB="${JOB:-montyyin-moe96-r2}"
MASTER_POD="${JOB}-master-0"
SCALES="${SCALES:-32+64}"
TP="${TP:-1}"; PP="${PP:-4}"; EP="${EP:-4}"; ETP="${ETP:-1}"
GBS="${GBS:-1920}"; MBS="${MBS:-1}"; SEQ="${SEQ_LENGTH:-4096}"
ITERS="${TRAIN_ITERS:-40}"
PROBING="${PROBING:-0}"
FAILSLOW_STEP_LOG="${FAILSLOW_STEP_LOG:-1}"
SKIP_TB="${SKIP_TB:-1}"
NPUS=16
MASTER_PORT_BASE="${MASTER_PORT:-26200}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/afs-a3-weight-share/yinjinrun.p-huawei/results/moe_failslow/${STAMP}}"
AFS_WRAPPERS="/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/wrappers"
AFS_HOOKS="/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/hooks"
MEGATRON="/afs-a3-241ceshi-shared/geruijun/Megatron-LM-0.12.3"
DATA_ROOT="/afs-a3-241ceshi-shared/geruijun"
PROBING_HOME="/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/projects/Probing_plus"
LOCAL_LOG="${LOCAL_LOG:-/tmp/moe-failslow-${STAMP}.log}"
exec > >(tee -a "$LOCAL_LOG") 2>&1

pod_for_global() {
  local g="$1"
  if [[ "$g" -eq 0 ]]; then echo "${JOB}-master-0"; else echo "${JOB}-worker-$((g-1))"; fi
}

master_dns_for_global() {
  echo "$(pod_for_global "$1").${JOB}"
}

count_done() {
  local scale_dir="$1" nnodes="$2"
  n=$(vcctl pod exec "$MASTER_POD" -- bash -lc \
    "grep -l 'TRAIN_RANK_.*_DONE' $scale_dir/rank*.log $scale_dir/nohup_rank*.log 2>/dev/null | wc -l" \
    2>/dev/null | grep -oE '[0-9]+' | tail -1)
  echo "${n:-0}"
}

scale_alive_on_pods() {
  local pods=("$@") p total=0 n
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
  local master_addr pods=() r g pod
  master_addr=$(master_dns_for_global "$node_offset")
  for ((r=0;r<nnodes;r++)); do
    g=$((node_offset + r))
    pods+=("$(pod_for_global "$g")")
  done
  echo "==> scale=$world nnodes=$nnodes offset=$node_offset MASTER=$master_addr:$master_port"
  vcctl pod exec "$MASTER_POD" -- bash -lc "mkdir -p '$scale_dir'"
  local done_n alive
  done_n=$(count_done "$scale_dir" "$nnodes")
  alive=$(scale_alive_on_pods "${pods[@]}")
  if [[ "$done_n" -ge "$nnodes" ]]; then echo "  skip: DONE"; return 0; fi
  if [[ "$alive" -gt 0 ]]; then echo "  skip: alive=$alive"; return 0; fi
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
SP=\$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)
if [[ -n \"\$SP\" && -d \"\$SP\" ]]; then
  printf '%s\\nimport failslow_step_timer\\n' \"$AFS_HOOKS\" > \"\$SP/zz_failslow_step.pth\"
fi
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
  local t0 now elapsed
  t0=$(date +%s)
  local timeout_sec=${SCALE_TIMEOUT_SEC:-7200}
  local grace_sec=${SCALE_GRACE_SEC:-900}
  local done_n alive
  while true; do
    now=$(date +%s)
    elapsed=$((now - t0))
    if (( elapsed >= timeout_sec )); then echo "  TIMEOUT scale=$world"; break; fi
    done_n=$(count_done "$scale_dir" "$nnodes")
    alive=$(scale_alive_on_pods "${pods[@]}")
    echo "  scale=$world done=$done_n/$nnodes alive=$alive elapsed=${elapsed}s $(date +%H:%M:%S)"
    [[ "$done_n" -ge "$nnodes" ]] && break
    if [[ "$alive" -eq 0 && "$done_n" -eq 0 && elapsed -gt grace_sec ]]; then
      echo "  FAIL early exit scale=$world"; break
    fi
    sleep 60
  done
  vcctl pod exec "$MASTER_POD" -- bash -lc \
    "grep -hE 'throughput per GPU|TRAIN_RANK_|ChildFailed' $scale_dir/*.log 2>/dev/null | tail -30" || true
  if ! vcctl pod exec "$MASTER_POD" -- bash -lc "grep -q 'throughput per GPU' $scale_dir/*.log"; then
    echo "FAIL scale=$world (no throughput)"; return 1
  fi
  nstep=$(vcctl pod exec "$MASTER_POD" -- bash -lc \
    "ls $scale_dir/step_times_rank*.jsonl 2>/dev/null | wc -l" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  nstep=${nstep:-0}
  if [[ "$nstep" -lt 1 ]]; then echo "FAIL scale=$world (no step_times)"; return 1; fi
  echo "OK scale=$world step_files=$nstep"
  return 0
}

run_wave() {
  local wave="$1" port_cursor="$2"
  echo "==> WAVE '$wave' port_base=$port_cursor"
  local parts=() total_npu=0 w
  IFS='+' read -ra parts <<< "$wave"
  for w in "${parts[@]}"; do total_npu=$((total_npu + w)); done
  (( total_npu <= 96 )) || { echo "FAIL wave >96"; return 1; }
  local offset=0 worlds=() offsets=() ports=() i=0
  for w in "${parts[@]}"; do
    worlds+=("$w"); offsets+=("$offset"); ports+=("$((port_cursor + i))")
    offset=$((offset + w / NPUS)); i=$((i + 1))
  done
  for i in "${!worlds[@]}"; do spawn_scale "${worlds[$i]}" "${offsets[$i]}" "${ports[$i]}"; done
  local fail=0
  for i in "${!worlds[@]}"; do wait_scale "${worlds[$i]}" "${offsets[$i]}" || fail=1; done
  return "$fail"
}

echo "==> JUMPHOST MoE FailSlow STAMP=$STAMP SCALES=$SCALES ITERS=$ITERS RUN_ROOT=$RUN_ROOT"
vcctl pod exec "$MASTER_POD" -- bash -lc "mkdir -p '$RUN_ROOT'"
PORT="$MASTER_PORT_BASE"
IFS=',' read -ra WAVES <<< "$SCALES"
for wave in "${WAVES[@]}"; do
  wave="${wave// /}"; [[ -n "$wave" ]] || continue
  run_wave "$wave" "$PORT" || true
  nparts=$(awk -F'+' '{print NF}' <<<"$wave")
  PORT=$((PORT + nparts + 2))
done
echo "JUMPHOST_MOE_FAILSLOW_DONE stamp=$STAMP → $RUN_ROOT"
