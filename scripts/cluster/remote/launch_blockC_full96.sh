#!/usr/bin/env bash
# Block C 全负载：Dense-96 GBS∝DP 短窗 + 全节点 npu-smi 热力图采样
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/root/.kube/config.huawei-a3-241ceshi}"
JOB="${JOB:-montyyin-moe96-r2}"
POD0="${JOB}-master-0"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REMOTE_DIR="${REMOTE_DIR:-/root/montyyin-lab-remote}"
ROOT="/afs-a3-241ceshi-shared/montyyin/results/blockC_full96/${STAMP}"
LOG="/tmp/blockC_full96_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> BLOCK_C_FULL96 STAMP=$STAMP $(date -Iseconds)"

# sync jumphost + hook
for f in jumphost_dense_failslow.sh failslow_step_timer.py parse_failslow_gap.py parse_npusmi_heatmap.py; do
  [[ -f "$REMOTE_DIR/$f" ]] || continue
  if [[ "$f" == failslow_step_timer.py ]]; then
    cat "$REMOTE_DIR/$f" | vcctl pod exec -i "$POD0" -- bash -lc \
      "cat > /afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster/hooks/$f"
  elif [[ "$f" == jumphost_dense_failslow.sh ]]; then
    cp "$REMOTE_DIR/$f" /tmp/jumphost_dense_failslow.sh && chmod +x /tmp/jumphost_dense_failslow.sh
  else
    cat "$REMOTE_DIR/$f" | vcctl pod exec -i "$POD0" -- bash -lc \
      "cat > /afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster/$f"
  fi
done

for p in ${JOB}-master-0 ${JOB}-worker-0 ${JOB}-worker-1 ${JOB}-worker-2 ${JOB}-worker-3 ${JOB}-worker-4; do
  vcctl pod exec "$p" -- bash -lc \
    'pkill -9 -f pretrain_gpt || true; pkill -9 -f torchrun || true; pkill -9 -f virtual_sync || true; pkill -9 -f npu_busy || true' \
    >/dev/null 2>&1 || true
done
sleep 3

SMI_OUT="$ROOT/npu_smi"
vcctl pod exec "$POD0" -- bash -lc "mkdir -p $SMI_OUT $ROOT/train"
for p in ${JOB}-master-0 ${JOB}-worker-0 ${JOB}-worker-1 ${JOB}-worker-2 ${JOB}-worker-3 ${JOB}-worker-4; do
  vcctl pod exec "$p" -- bash -lc "
    nohup bash -c '
      for i in \$(seq 1 80); do
        echo TS=\$(date -Iseconds) >> $SMI_OUT/${p}.log
        npu-smi info 2>/dev/null | head -120 >> $SMI_OUT/${p}.log
        echo --- >> $SMI_OUT/${p}.log
        sleep 10
      done
    ' > $SMI_OUT/${p}.nohup 2>&1 &
    echo SMI_${p}_\$!
  "
done

export JOB STAMP="${STAMP}_train" SCALES=96 GBS_PROP_DP=1 MICROBATCHES_PER_DP=160
export TRAIN_ITERS=40 PROBING=0 FAILSLOW_STEP_LOG=1 MASTER_PORT=27500
export SCALE_TIMEOUT_SEC=3600 SCALE_GRACE_SEC=600 TP=4 PP=2
export RUN_ROOT="$ROOT/train"
export LOCAL_LOG="/tmp/blockC_full96_train_${STAMP}.log"
nohup bash /tmp/jumphost_dense_failslow.sh > /tmp/blockC_full96_train_${STAMP}_nohup.out 2>&1 &
TRAIN_PID=$!
echo TRAIN_ORCH=$TRAIN_PID

for i in $(seq 1 80); do
  if grep -q JUMPHOST_DENSE_FAILSLOW_DONE /tmp/blockC_full96_train_${STAMP}.log 2>/dev/null; then break; fi
  n=$(vcctl pod exec "$POD0" -- bash -lc "wc -l < $ROOT/train/scale_96/step_times_rank0.jsonl 2>/dev/null || echo 0" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  echo "  dense96 steps=${n:-0}/40"
  sleep 30
done

# stop smi
for p in ${JOB}-master-0 ${JOB}-worker-0 ${JOB}-worker-1 ${JOB}-worker-2 ${JOB}-worker-3 ${JOB}-worker-4; do
  vcctl pod exec "$p" -- bash -lc 'pkill -f "npu-smi info" || true; pkill -f "seq 1 80" || true' >/dev/null 2>&1 || true
done

vcctl pod exec "$POD0" -- bash -lc "
cd /afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster
python3 parse_failslow_gap.py $ROOT/train --drop-first 8 --csv $ROOT/train/gap_vs_n.csv || true
python3 parse_npusmi_heatmap.py $SMI_OUT --out $ROOT/heatmap_means.csv || true
mkdir -p /afs-a3-241ceshi-shared/montyyin/results/reports/ascend_campaign_20260713
cp -f $ROOT/train/gap_vs_n.csv /afs-a3-241ceshi-shared/montyyin/results/reports/ascend_campaign_20260713/gap_real96_blockC.csv 2>/dev/null || true
cp -f $ROOT/heatmap_means.csv /afs-a3-241ceshi-shared/montyyin/results/reports/ascend_campaign_20260713/ 2>/dev/null || true
{
  echo ''
  echo '## Block C full96'
  echo \"stamp: $STAMP\"
  echo \"train: $ROOT/train\"
  echo \"smi: $SMI_OUT\"
  cat $ROOT/train/gap_vs_n.csv 2>/dev/null || true
} >> /afs-a3-241ceshi-shared/montyyin/results/reports/offline_20260713/SUMMARY.md
"
echo "BLOCK_C_FULL96_DONE stamp=$STAMP → $ROOT"
