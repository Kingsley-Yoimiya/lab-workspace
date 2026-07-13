#!/usr/bin/env bash
# 无限租约连环战役：等当前 BlockC → D强/E并行 → MoE32+64 → Dense 长窗 → 再 BlockC
# 在 ais-jump nohup；本机可离线
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/root/.kube/config.huawei-a3-241ceshi}"
JOB="${JOB:-montyyin-moe96-r2}"
REMOTE_DIR="${REMOTE_DIR:-/root/montyyin-lab-remote}"
CAMPAIGN_STAMP="${CAMPAIGN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG="/tmp/infinite_campaign_${CAMPAIGN_STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> INFINITE_CAMPAIGN $CAMPAIGN_STAMP $(date -Iseconds)"

wait_marker() {
  local file="$1" pat="$2" max_sec="${3:-7200}"
  local t0=$(date +%s)
  while true; do
    if [[ -f "$file" ]] && grep -qE "$pat" "$file" 2>/dev/null; then
      echo "  marker hit: $pat in $file"
      return 0
    fi
    local el=$(( $(date +%s) - t0 ))
    echo "  waiting $pat (${el}s/${max_sec}s) ..."
    (( el > max_sec )) && { echo "  WAIT_TIMEOUT $pat"; return 1; }
    sleep 45
  done
}

kill_trainers() {
  for p in ${JOB}-master-0 ${JOB}-worker-0 ${JOB}-worker-1 ${JOB}-worker-2 ${JOB}-worker-3 ${JOB}-worker-4; do
    vcctl pod exec "$p" -- bash -lc \
      'pkill -9 -f pretrain_gpt || true; pkill -9 -f torchrun || true; pkill -9 -f virtual_sync || true; pkill -9 -f npu_busy || true' \
      >/dev/null 2>&1 || true
  done
  sleep 4
}

# ---------- 0) 等当前 Block C（若在跑）----------
echo "==> PHASE0 wait Block C if running"
if ps -ef | grep -v grep | grep -q launch_blockC_full96; then
  # 找最新 log
  latest=$(ls -t /tmp/blockC_full96_*.log 2>/dev/null | head -1 || true)
  if [[ -n "${latest:-}" ]]; then
    wait_marker "$latest" "BLOCK_C_FULL96_DONE" 7200 || true
  fi
fi
# 也等 jumphost dense 结束
while ps -ef | grep -v grep | grep -q '[j]umphost_dense_failslow'; do
  echo "  jumphost_dense still alive ..."
  sleep 30
done
kill_trainers

# ---------- 1) D强 + E 并行 ----------
echo "==> PHASE1 Exp45 (E+D rematch) + Exp3 strong AB"
STAMP1="${CAMPAIGN_STAMP}_p1"
export STAMP="$STAMP1" REMOTE_DIR
nohup bash "$REMOTE_DIR/launch_exp45_parallel.sh" > /tmp/exp45_${STAMP1}_nohup.out 2>&1 &
PID45=$!
# 等 exp45 清场并占住 master/worker-1 后再开 strong，避免被 pkill
sleep 90
export STAMP="${STAMP1}_strong" POD="${JOB}-worker-2" DELAY_MS=2500 TRAIN_ITERS=48
nohup bash "$REMOTE_DIR/launch_exp3_strong.sh" > /tmp/exp3_strong_${STAMP}_nohup.out 2>&1 &
PID3=$!
echo "  PIDs exp45=$PID45 strong=$PID3"
wait_marker "/tmp/exp45_${STAMP1}.log" "EXP45_DONE" 5400 || true
wait_marker "/tmp/exp3_strong_${STAMP1}_strong.log" "EXP3_STRONG_DONE" 5400 || true
kill_trainers

# ---------- 2) MoE failslow 32+64 并行，再 96 ----------
echo "==> PHASE2 MoE failslow"
cp -f "$REMOTE_DIR/jumphost_moe_failslow.sh" /tmp/jumphost_moe_failslow.sh
chmod +x /tmp/jumphost_moe_failslow.sh
# sync hook
cat "$REMOTE_DIR/failslow_step_timer.py" | vcctl pod exec -i ${JOB}-master-0 -- bash -lc \
  "cat > /afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster/hooks/failslow_step_timer.py" || true

STAMP2="${CAMPAIGN_STAMP}_moe"
export JOB STAMP="$STAMP2" SCALES='32+64' TRAIN_ITERS=40 PROBING=0 FAILSLOW_STEP_LOG=1
export MASTER_PORT=28000 RUN_ROOT="/afs-a3-241ceshi-shared/montyyin/results/moe_failslow/${STAMP2}"
export LOCAL_LOG="/tmp/moe_failslow_${STAMP2}.log"
nohup bash /tmp/jumphost_moe_failslow.sh > /tmp/moe_failslow_${STAMP2}_nohup.out 2>&1 &
wait_marker "/tmp/moe_failslow_${STAMP2}.log" "JUMPHOST_MOE_FAILSLOW_DONE|MOE_FAILSLOW_DONE|DONE stamp" 7200 || true
# parse if possible
vcctl pod exec ${JOB}-master-0 -- bash -lc "
cd /afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster
python3 parse_failslow_gap.py /afs-a3-241ceshi-shared/montyyin/results/moe_failslow/${STAMP2} --drop-first 5 \
  --csv /afs-a3-241ceshi-shared/montyyin/results/moe_failslow/${STAMP2}/gap_vs_n.csv || true
" || true
kill_trainers

# MoE 96
STAMP2b="${CAMPAIGN_STAMP}_moe96"
export STAMP="$STAMP2b" SCALES=96 TRAIN_ITERS=40 MASTER_PORT=28100
export RUN_ROOT="/afs-a3-241ceshi-shared/montyyin/results/moe_failslow/${STAMP2b}"
export LOCAL_LOG="/tmp/moe_failslow_${STAMP2b}.log"
nohup bash /tmp/jumphost_moe_failslow.sh > /tmp/moe_failslow_${STAMP2b}_nohup.out 2>&1 &
wait_marker "/tmp/moe_failslow_${STAMP2b}.log" "JUMPHOST_MOE_FAILSLOW_DONE|MOE_FAILSLOW_DONE|DONE stamp" 7200 || true
vcctl pod exec ${JOB}-master-0 -- bash -lc "
cd /afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster
python3 parse_failslow_gap.py /afs-a3-241ceshi-shared/montyyin/results/moe_failslow/${STAMP2b} --drop-first 5 \
  --csv /afs-a3-241ceshi-shared/montyyin/results/moe_failslow/${STAMP2b}/gap_vs_n.csv || true
" || true
kill_trainers

# ---------- 3) Dense 长窗 GBS∝DP 16+32 然后 64+96 ----------
echo "==> PHASE3 Dense long-window GBS∝DP"
cp -f "$REMOTE_DIR/jumphost_dense_failslow.sh" /tmp/jumphost_dense_failslow.sh
chmod +x /tmp/jumphost_dense_failslow.sh
STAMP3="${CAMPAIGN_STAMP}_dense_long"
export STAMP="${STAMP3}_a" SCALES='16+32' GBS_PROP_DP=1 MICROBATCHES_PER_DP=160
export TRAIN_ITERS=80 PROBING=0 FAILSLOW_STEP_LOG=1 MASTER_PORT=28200
export SCALE_TIMEOUT_SEC=7200 SCALE_GRACE_SEC=900 TP=4 PP=2
export RUN_ROOT="/afs-a3-241ceshi-shared/montyyin/results/dense_failslow_gbsprop_long/${STAMP3}"
export LOCAL_LOG="/tmp/dense_long_${STAMP3}_a.log"
nohup bash /tmp/jumphost_dense_failslow.sh > /tmp/dense_long_${STAMP3}_a_nohup.out 2>&1 &
wait_marker "/tmp/dense_long_${STAMP3}_a.log" "JUMPHOST_DENSE_FAILSLOW_DONE" 9000 || true
kill_trainers

export STAMP="${STAMP3}_b" SCALES='64+96' MASTER_PORT=28300
export LOCAL_LOG="/tmp/dense_long_${STAMP3}_b.log"
# same RUN_ROOT to merge scales
nohup bash /tmp/jumphost_dense_failslow.sh > /tmp/dense_long_${STAMP3}_b_nohup.out 2>&1 &
wait_marker "/tmp/dense_long_${STAMP3}_b.log" "JUMPHOST_DENSE_FAILSLOW_DONE" 12000 || true
vcctl pod exec ${JOB}-master-0 -- bash -lc "
cd /afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster
python3 parse_failslow_gap.py /afs-a3-241ceshi-shared/montyyin/results/dense_failslow_gbsprop_long/${STAMP3} --drop-first 10 \
  --csv /afs-a3-241ceshi-shared/montyyin/results/dense_failslow_gbsprop_long/${STAMP3}/gap_vs_n.csv || true
python3 parse_network_contrib.py \
  --indep-root /afs-a3-241ceshi-shared/montyyin/results/blockA_indep/20260713_110548 \
  --real-csv /afs-a3-241ceshi-shared/montyyin/results/dense_failslow_gbsprop_long/${STAMP3}/gap_vs_n.csv \
  --drop-first 10 \
  --out /afs-a3-241ceshi-shared/montyyin/results/reports/ascend_campaign_20260713/network_contrib_long.csv || true
" || true
kill_trainers

# ---------- 4) 再来一轮 Block C full96 ----------
echo "==> PHASE4 Block C full96 again"
STAMP4="${CAMPAIGN_STAMP}_c2"
export STAMP="$STAMP4"
nohup bash "$REMOTE_DIR/launch_blockC_full96.sh" > /tmp/blockC_full96_${STAMP4}_nohup.out 2>&1 &
wait_marker "/tmp/blockC_full96_${STAMP4}.log" "BLOCK_C_FULL96_DONE" 7200 || true
kill_trainers

# ---------- 5) MoE 再试一轮 32 only 长窗（稳）----------
echo "==> PHASE5 MoE32 long"
STAMP5="${CAMPAIGN_STAMP}_moe32long"
export STAMP="$STAMP5" SCALES=32 TRAIN_ITERS=60 MASTER_PORT=28400
export RUN_ROOT="/afs-a3-241ceshi-shared/montyyin/results/moe_failslow/${STAMP5}"
export LOCAL_LOG="/tmp/moe_failslow_${STAMP5}.log"
nohup bash /tmp/jumphost_moe_failslow.sh > /tmp/moe_failslow_${STAMP5}_nohup.out 2>&1 &
wait_marker "/tmp/moe_failslow_${STAMP5}.log" "JUMPHOST_MOE_FAILSLOW_DONE|MOE_FAILSLOW_DONE|DONE stamp" 7200 || true

# finalize note
vcctl pod exec ${JOB}-master-0 -- bash -lc "
mkdir -p /afs-a3-241ceshi-shared/montyyin/results/reports/ascend_campaign_20260713
{
  echo ''
  echo \"## Infinite campaign $CAMPAIGN_STAMP\"
  echo \"phases: C→D/E→MoE→DenseLong→C2→MoE32long\"
  echo \"finished: \$(date -Iseconds)\"
} >> /afs-a3-241ceshi-shared/montyyin/results/reports/offline_20260713/SUMMARY.md
" || true

echo "INFINITE_CAMPAIGN_DONE stamp=$CAMPAIGN_STAMP $(date -Iseconds)"
