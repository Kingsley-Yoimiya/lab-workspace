#!/usr/bin/env bash
# 加强版 Block D：更大 DELAY_MS + 更长窗；可与其它 16 卡任务并行（指定 POD）
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/root/.kube/config.huawei-a3-241ceshi}"
JOB="${JOB:-montyyin-moe96-r2}"
POD="${POD:-${JOB}-worker-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REMOTE_DIR="${REMOTE_DIR:-/root/montyyin-lab-remote}"
ROOT="/afs-a3-weight-share/yinjinrun.p-huawei/results/dense_pp_inject_strong/${STAMP}"
LOG="/tmp/exp3_strong_${STAMP}.log"
DELAY_MS="${DELAY_MS:-2500}"
TRAIN_ITERS="${TRAIN_ITERS:-48}"
exec > >(tee -a "$LOG") 2>&1
echo "==> EXP3_STRONG STAMP=$STAMP POD=$POD DELAY_MS=$DELAY_MS $(date -Iseconds)"

cat "$REMOTE_DIR/failslow_step_timer.py" | vcctl pod exec -i "${JOB}-master-0" -- bash -lc \
  "cat > /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/hooks/failslow_step_timer.py"
for f in parse_pp_inject_ab.py parse_failslow_gap.py; do
  cat "$REMOTE_DIR/$f" | vcctl pod exec -i "${JOB}-master-0" -- bash -lc \
    "cat > /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/$f"
done

vcctl pod exec "$POD" -- bash -lc \
  'pkill -9 -f pretrain_gpt || true; pkill -9 -f torchrun || true' >/dev/null 2>&1 || true
sleep 2

run_one() {
  local tag="$1" inject="$2" port="$3"
  local scale_dir="$ROOT/$tag"
  echo "==> RUN $tag DELAY_INJECT=$inject"
  vcctl pod exec "${JOB}-master-0" -- bash -lc "mkdir -p $scale_dir"
  vcctl pod exec "${JOB}-master-0" -- bash -lc "cat > $scale_dir/launch_rank0.sh <<EOF
#!/usr/bin/env bash
set -uo pipefail
export TP=4 PP=2 EP=1 ETP=1 MBS=1 GBS=320 SEQ_LENGTH=4096
export SKIP_TB=1 SKIP_SAVE=1 SKIP_PROFILE=1 TRAIN_ITERS=$TRAIN_ITERS
export PROBING=0 FAILSLOW_STEP_LOG=1
export DELAY_INJECT=$inject DELAY_STAGE=1 DELAY_MS=$DELAY_MS DELAY_EVERY=4 DELAY_BURST=2
export PP_SIZE=2 WORLD_SIZE_NPUS=16
export PATH=/root/miniconda3/envs/llm_test/bin:\\\$PATH
export PYTHONPATH=/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/hooks:/MindSpeed-LLM/MindSpeed:\\\${PYTHONPATH:-}
export WORLD_SIZE=1 NNODES=1 RANK=0 NODE_RANK=0
export MASTER_ADDR=${POD}.${JOB} MASTER_PORT=$port
export NPUS_PER_NODE=16 GPUS_PER_NODE=16
export DATA_ROOT=/afs-a3-241ceshi-shared/geruijun
export RUN_DIR=$scale_dir LOG_DIR=$scale_dir/
export TENSORBOARD_DIR=$scale_dir/tb CKPT_SAVE_DIR=$scale_dir/ckpt
export HCCL_IF_BASE_PORT=$((port+2000))
mkdir -p \\\"\\\$RUN_DIR\\\" \\\"\\\$LOG_DIR\\\" \\\"\\\$TENSORBOARD_DIR\\\" \\\"\\\$CKPT_SAVE_DIR\\\"
SP=\\\$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)
if [[ -n \\\"\\\$SP\\\" && -d \\\"\\\$SP\\\" ]]; then
  printf '%s\\\\nimport failslow_step_timer\\\\n' \\\"/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/hooks\\\" > \\\"\\\$SP/zz_failslow_step.pth\\\"
fi
cd /afs-a3-241ceshi-shared/geruijun/Megatron-LM-0.12.3
bash /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/wrappers/train_qwen3_8B_ascend.sh 2>&1 | tee $scale_dir/rank0.log
rc=\\\${PIPESTATUS[0]}
echo TRAIN_RANK_0_DONE rc=\\\$rc | tee -a $scale_dir/rank0.log
exit \\\$rc
EOF
chmod +x $scale_dir/launch_rank0.sh"
  vcctl pod exec "$POD" -- bash -lc \
    "setsid nohup bash $scale_dir/launch_rank0.sh >$scale_dir/nohup_rank0.log 2>&1 & echo SPAWNED_\$!"
  local t0=$(date +%s)
  while true; do
    local el=$(( $(date +%s) - t0 ))
    local done=$(vcctl pod exec "${JOB}-master-0" -- bash -lc "grep -c TRAIN_RANK_0_DONE $scale_dir/rank0.log 2>/dev/null || echo 0" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
    local steps=$(vcctl pod exec "${JOB}-master-0" -- bash -lc "wc -l < $scale_dir/step_times_rank0.jsonl 2>/dev/null || echo 0" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
    echo "  $tag steps=${steps:-0}/$TRAIN_ITERS done=${done:-0} elapsed=${el}s"
    [[ "${done:-0}" -ge 1 ]] && break
    (( el > 3600 )) && { echo TIMEOUT; break; }
    sleep 40
  done
}

run_one baseline 0 27600
vcctl pod exec "$POD" -- bash -lc 'pkill -9 -f pretrain_gpt || true; pkill -9 -f torchrun || true' >/dev/null 2>&1 || true
sleep 3
run_one inject 1 27610

vcctl pod exec "${JOB}-master-0" -- bash -lc "
cd /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster
python3 parse_pp_inject_ab.py --baseline $ROOT/baseline --inject $ROOT/inject --pp 2 --world 16 --drop-first 6 --out $ROOT/pp_inject_ab.json
cat $ROOT/pp_inject_ab.json
"
echo "EXP3_STRONG_DONE stamp=$STAMP → $ROOT"
