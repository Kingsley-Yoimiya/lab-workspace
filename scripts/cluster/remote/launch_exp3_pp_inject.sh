#!/usr/bin/env bash
# 实验三 lite + 实验二 lite：Dense 16 卡 baseline→inject，顺带 npu-smi 采样
# 在 ais-jump 上 nohup 跑；本机可离线
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/root/.kube/config.huawei-a3-241ceshi}"
JOB="${JOB:-montyyin-moe96-r2}"
POD="${POD:-${JOB}-master-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REMOTE_DIR="${REMOTE_DIR:-/root/montyyin-lab-remote}"
ROOT="/afs-a3-241ceshi-shared/montyyin/results/dense_pp_inject/${STAMP}"
LOG="/tmp/exp3_pp_inject_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "==> EXP3 STAMP=$STAMP POD=$POD $(date -Iseconds)"

# sync hooks
for f in failslow_step_timer.py; do
  [[ -f "$REMOTE_DIR/$f" ]] || continue
  cat "$REMOTE_DIR/$f" | vcctl pod exec -i "$POD" -- bash -lc \
    "cat > /afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster/hooks/$f"
done
for f in parse_pp_inject_ab.py parse_failslow_gap.py; do
  [[ -f "$REMOTE_DIR/$f" ]] || continue
  cat "$REMOTE_DIR/$f" | vcctl pod exec -i "$POD" -- bash -lc \
    "cat > /afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster/$f"
done

# kill leftover trainers
for p in ${JOB}-master-0 ${JOB}-worker-0 ${JOB}-worker-1 ${JOB}-worker-2 ${JOB}-worker-3 ${JOB}-worker-4; do
  vcctl pod exec "$p" -- bash -lc \
    'for pid in $(ps -eo pid,cmd | awk "/[p]retrain_gpt.py|[t]orchrun/{print \$1}"); do kill -9 $pid 2>/dev/null; done; true' \
    >/dev/null 2>&1 || true
done
sleep 2

# npu-smi sampler (exp2 lite) — power/temp on card 0..15
start_smi() {
  local tag="$1" out="$2"
  vcctl pod exec "$POD" -- bash -lc "
    mkdir -p $(dirname $out)
    nohup bash -c '
      while true; do
        ts=\$(date +%s)
        for i in \$(seq 0 15); do
          line=\$(npu-smi info -t power -i \$i 2>/dev/null | tr \"\\n\" \" \" )
          temp=\$(npu-smi info -t temp -i \$i 2>/dev/null | tr \"\\n\" \" \" )
          echo \"\$ts card=\$i power=\$line temp=\$temp\"
        done
        sleep 10
      done
    ' > $out 2>&1 &
    echo SMI_PID=\$!
  "
}

stop_smi() {
  vcctl pod exec "$POD" -- bash -lc 'pkill -f "npu-smi info -t power" || true; pkill -f "while true; do" || true' >/dev/null 2>&1 || true
}

run_one() {
  local tag="$1" inject="$2"
  local scale_dir="$ROOT/$tag"
  local port=26300
  [[ "$tag" == "inject" ]] && port=26310
  echo "==> RUN $tag DELAY_INJECT=$inject"
  vcctl pod exec "$POD" -- bash -lc "mkdir -p $scale_dir"

  # write launch（可在任意单节点 pod 上跑 Dense-16）
  vcctl pod exec "$POD" -- bash -lc "cat > $scale_dir/launch_rank0.sh <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
export TP=4 PP=2 EP=1 ETP=1 MBS=1 GBS=320 SEQ_LENGTH=4096
export SKIP_TB=1 SKIP_SAVE=1 SKIP_PROFILE=1 TRAIN_ITERS=36
export PROBING=0 FAILSLOW_STEP_LOG=1
export DELAY_INJECT=$inject DELAY_STAGE=1 DELAY_MS=800 DELAY_EVERY=4 DELAY_BURST=2
export PP_SIZE=2 WORLD_SIZE_NPUS=16
export PATH=/root/miniconda3/envs/llm_test/bin:\$PATH
export PYTHONPATH=/afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster/hooks:/MindSpeed-LLM/MindSpeed:\${PYTHONPATH:-}
export WORLD_SIZE=1 NNODES=1 RANK=0 NODE_RANK=0
export MASTER_ADDR=${POD}.${JOB} MASTER_PORT=$port
export NPUS_PER_NODE=16 GPUS_PER_NODE=16
export DATA_ROOT=/afs-a3-241ceshi-shared/geruijun
export RUN_DIR=$scale_dir LOG_DIR=$scale_dir/
export TENSORBOARD_DIR=$scale_dir/tb CKPT_SAVE_DIR=$scale_dir/ckpt
export HCCL_IF_BASE_PORT=$((port+2000))
mkdir -p \"\$RUN_DIR\" \"\$LOG_DIR\" \"\$TENSORBOARD_DIR\" \"\$CKPT_SAVE_DIR\"
SP=\$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)
if [[ -n \"\$SP\" && -d \"\$SP\" ]]; then
  printf '%s\\nimport failslow_step_timer\\n' \"/afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster/hooks\" > \"\$SP/zz_failslow_step.pth\"
fi
cd /afs-a3-241ceshi-shared/geruijun/Megatron-LM-0.12.3
bash /afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster/wrappers/train_qwen3_8B_ascend.sh 2>&1 | tee $scale_dir/rank0.log
rc=\${PIPESTATUS[0]}
echo TRAIN_RANK_0_DONE rc=\$rc | tee -a $scale_dir/rank0.log
exit \$rc
EOF
chmod +x $scale_dir/launch_rank0.sh"

  stop_smi
  start_smi "$tag" "$scale_dir/npu_smi_sample.log"
  vcctl pod exec "$POD" -- bash -lc \
    "setsid nohup bash $scale_dir/launch_rank0.sh >$scale_dir/nohup_rank0.log 2>&1 & echo SPAWNED_\$!"

  # wait up to 45 min
  local t0=$(date +%s)
  while true; do
    local now=$(date +%s)
    local el=$((now - t0))
    local steps=$(vcctl pod exec "$POD" -- bash -lc "wc -l < $scale_dir/step_times_rank0.jsonl 2>/dev/null || echo 0" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
    steps=${steps:-0}
    local done=$(vcctl pod exec "$POD" -- bash -lc "grep -c TRAIN_RANK_0_DONE $scale_dir/rank0.log 2>/dev/null || echo 0" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
    done=${done:-0}
    echo "  $tag steps=$steps/36 done=$done elapsed=${el}s"
    [[ "$done" -ge 1 ]] && break
    if (( el > 2700 )); then echo "  TIMEOUT $tag"; break; fi
    sleep 45
  done
  stop_smi
}

run_one baseline 0
# clean before inject
vcctl pod exec "$POD" -- bash -lc \
  'for pid in $(ps -eo pid,cmd | awk "/[p]retrain_gpt.py|[t]orchrun/{print \$1}"); do kill -9 $pid 2>/dev/null; done; true' \
  >/dev/null 2>&1 || true
sleep 3
run_one inject 1

# parse
vcctl pod exec "$POD" -- bash -lc "
cd /afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster
python3 parse_pp_inject_ab.py \
  --baseline $ROOT/baseline --inject $ROOT/inject \
  --pp 2 --world 16 --drop-first 5 \
  --out $ROOT/pp_inject_ab.json
python3 parse_failslow_gap.py $ROOT/baseline --drop-first 5 --csv $ROOT/baseline/gap_vs_n.csv || true
python3 parse_failslow_gap.py $ROOT/inject --drop-first 5 --csv $ROOT/inject/gap_vs_n.csv || true
# append note to offline report
mkdir -p /afs-a3-241ceshi-shared/montyyin/results/reports/offline_20260713
{
  echo ''
  echo '## 实验三 lite：PP stage 注入 AB (Dense 16, TP4PP2 GBS=320)'
  echo ''
  echo \"Stamp: $STAMP\"
  echo ''
  echo '\\\`\\\`\\\`json'
  cat $ROOT/pp_inject_ab.json 2>/dev/null || echo '{}'
  echo '\\\`\\\`\\\`'
  echo ''
  echo '## 实验二 lite：npu-smi 采样'
  echo ''
  echo \"- baseline: $ROOT/baseline/npu_smi_sample.log\"
  echo \"- inject: $ROOT/inject/npu_smi_sample.log\"
} >> /afs-a3-241ceshi-shared/montyyin/results/reports/offline_20260713/SUMMARY.md
"

echo "EXP3_DONE stamp=$STAMP → $ROOT"
