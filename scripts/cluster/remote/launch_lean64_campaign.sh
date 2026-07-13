#!/usr/bin/env bash
# 从 MoE 起恢复：串行 32,64（空出≥32卡），不跑任何 96
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/root/.kube/config.huawei-a3-241ceshi}"
JOB="${JOB:-montyyin-moe96-r2}"
REMOTE_DIR="${REMOTE_DIR:-/root/montyyin-lab-remote}"
CAMPAIGN_STAMP="${CAMPAIGN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG="/tmp/lean64_campaign_${CAMPAIGN_STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> LEAN64_CAMPAIGN $CAMPAIGN_STAMP $(date -Iseconds)"

wait_marker() {
  local file="$1" pat="$2" max_sec="${3:-10800}"
  local t0=$(date +%s)
  while true; do
    if [[ -f "$file" ]] && grep -qE "$pat" "$file" 2>/dev/null; then
      echo "  marker hit: $pat"; return 0
    fi
    local el=$(( $(date +%s) - t0 ))
    echo "  waiting $pat (${el}s) ..."
    (( el > max_sec )) && { echo "  WAIT_TIMEOUT"; return 1; }
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

kill_trainers
cp -f "$REMOTE_DIR/jumphost_moe_failslow.sh" /tmp/jumphost_moe_failslow.sh
cp -f "$REMOTE_DIR/jumphost_dense_failslow.sh" /tmp/jumphost_dense_failslow.sh
chmod +x /tmp/jumphost_moe_failslow.sh /tmp/jumphost_dense_failslow.sh
cat "$REMOTE_DIR/failslow_step_timer.py" | vcctl pod exec -i ${JOB}-master-0 -- bash -lc \
  "cat > /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/hooks/failslow_step_timer.py" || true

echo "==> MoE SCALES=32,64 sequential (32卡时闲64；64卡时闲32)"
STAMP2="${CAMPAIGN_STAMP}_moe"
export JOB STAMP="$STAMP2" SCALES='32,64' TRAIN_ITERS=40 PROBING=0 FAILSLOW_STEP_LOG=1
export MASTER_PORT=28500 RUN_ROOT="/afs-a3-weight-share/yinjinrun.p-huawei/results/moe_failslow/${STAMP2}"
export LOCAL_LOG="/tmp/moe_failslow_${STAMP2}.log"
nohup bash /tmp/jumphost_moe_failslow.sh > /tmp/moe_failslow_${STAMP2}_nohup.out 2>&1 &
wait_marker "/tmp/moe_failslow_${STAMP2}.log" "JUMPHOST_MOE_FAILSLOW_DONE" 10800 || true
vcctl pod exec ${JOB}-master-0 -- bash -lc "
cd /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster
python3 parse_failslow_gap.py \$RUN_ROOT --drop-first 5 --csv \$RUN_ROOT/gap_vs_n.csv || true
" || true
# fix: RUN_ROOT not expanded in remote - use stamp path
vcctl pod exec ${JOB}-master-0 -- bash -lc "
cd /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster
python3 parse_failslow_gap.py /afs-a3-weight-share/yinjinrun.p-huawei/results/moe_failslow/${STAMP2} --drop-first 5 \
  --csv /afs-a3-weight-share/yinjinrun.p-huawei/results/moe_failslow/${STAMP2}/gap_vs_n.csv || true
cat /afs-a3-weight-share/yinjinrun.p-huawei/results/moe_failslow/${STAMP2}/gap_vs_n.csv || true
" || true
kill_trainers

echo "==> Dense long max64: 16+32 then 64"
STAMP3="${CAMPAIGN_STAMP}_dense_long"
export STAMP="${STAMP3}_a" SCALES='16+32' GBS_PROP_DP=1 MICROBATCHES_PER_DP=160
export TRAIN_ITERS=80 PROBING=0 FAILSLOW_STEP_LOG=1 MASTER_PORT=28600
export SCALE_TIMEOUT_SEC=7200 SCALE_GRACE_SEC=900 TP=4 PP=2
export RUN_ROOT="/afs-a3-weight-share/yinjinrun.p-huawei/results/dense_failslow_gbsprop_long/${STAMP3}"
export LOCAL_LOG="/tmp/dense_long_${STAMP3}_a.log"
nohup bash /tmp/jumphost_dense_failslow.sh > /tmp/dense_long_${STAMP3}_a_nohup.out 2>&1 &
wait_marker "/tmp/dense_long_${STAMP3}_a.log" "JUMPHOST_DENSE_FAILSLOW_DONE" 9000 || true
kill_trainers
export STAMP="${STAMP3}_b" SCALES=64 MASTER_PORT=28700
export LOCAL_LOG="/tmp/dense_long_${STAMP3}_b.log"
nohup bash /tmp/jumphost_dense_failslow.sh > /tmp/dense_long_${STAMP3}_b_nohup.out 2>&1 &
wait_marker "/tmp/dense_long_${STAMP3}_b.log" "JUMPHOST_DENSE_FAILSLOW_DONE" 9000 || true
vcctl pod exec ${JOB}-master-0 -- bash -lc "
cd /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster
python3 parse_failslow_gap.py /afs-a3-weight-share/yinjinrun.p-huawei/results/dense_failslow_gbsprop_long/${STAMP3} --drop-first 10 \
  --csv /afs-a3-weight-share/yinjinrun.p-huawei/results/dense_failslow_gbsprop_long/${STAMP3}/gap_vs_n.csv || true
" || true
kill_trainers

echo "==> MoE32 long only"
STAMP5="${CAMPAIGN_STAMP}_moe32long"
export STAMP="$STAMP5" SCALES=32 TRAIN_ITERS=60 MASTER_PORT=28800
export RUN_ROOT="/afs-a3-weight-share/yinjinrun.p-huawei/results/moe_failslow/${STAMP5}"
export LOCAL_LOG="/tmp/moe_failslow_${STAMP5}.log"
nohup bash /tmp/jumphost_moe_failslow.sh > /tmp/moe_failslow_${STAMP5}_nohup.out 2>&1 &
wait_marker "/tmp/moe_failslow_${STAMP5}.log" "JUMPHOST_MOE_FAILSLOW_DONE" 7200 || true

echo "LEAN64_CAMPAIGN_DONE stamp=$CAMPAIGN_STAMP $(date -Iseconds)"
