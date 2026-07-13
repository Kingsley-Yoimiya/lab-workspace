#!/usr/bin/env bash
# Block A 快速双轨：indep 8/16/32/64 + real 补 8/16（GBS∝DP）
# 在 ais-jump nohup；占 montyyin-moe96-r2 96 卡内扇出
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/root/.kube/config.huawei-a3-241ceshi}"
JOB="${JOB:-montyyin-moe96-r2}"
POD0="${JOB}-master-0"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REMOTE_DIR="${REMOTE_DIR:-/root/montyyin-lab-remote}"
INDEP_ROOT="/afs-a3-weight-share/yinjinrun.p-huawei/results/blockA_indep/${STAMP}"
REAL_ROOT="/afs-a3-weight-share/yinjinrun.p-huawei/results/blockA_real/${STAMP}"
LOG="/tmp/blockA_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> BLOCK_A STAMP=$STAMP $(date -Iseconds)"

# sync scripts
for f in virtual_sync_bench_npu.py parse_network_contrib.py parse_failslow_gap.py jumphost_dense_failslow.sh failslow_step_timer.py; do
  [[ -f "$REMOTE_DIR/$f" ]] || continue
  if [[ "$f" == failslow_step_timer.py ]]; then
    cat "$REMOTE_DIR/$f" | vcctl pod exec -i "$POD0" -- bash -lc \
      "cat > /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/hooks/$f"
  elif [[ "$f" == jumphost_dense_failslow.sh ]]; then
    cp "$REMOTE_DIR/$f" /tmp/jumphost_dense_failslow.sh
    chmod +x /tmp/jumphost_dense_failslow.sh
  else
    cat "$REMOTE_DIR/$f" | vcctl pod exec -i "$POD0" -- bash -lc \
      "cat > /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/$f"
  fi
done

for p in ${JOB}-master-0 ${JOB}-worker-0 ${JOB}-worker-1 ${JOB}-worker-2 ${JOB}-worker-3 ${JOB}-worker-4; do
  vcctl pod exec "$p" -- bash -lc \
    'for pid in $(ps -eo pid,cmd | awk "/[p]retrain_gpt.py|[t]orchrun|[v]irtual_sync/{print \$1}"); do kill -9 $pid 2>/dev/null; done; true' \
    >/dev/null 2>&1 || true
done
sleep 2

pod_for() {
  local g="$1"
  if [[ "$g" -eq 0 ]]; then echo "${JOB}-master-0"; else echo "${JOB}-worker-$((g-1))"; fi
}

# --- indep wave: spawn torchrun independent on contiguous nodes ---
spawn_indep() {
  local world="$1" node_offset="$2" port="$3"
  local nnodes=$(( (world + 15) / 16 ))
  local nproc=16
  # N=8: half node
  if [[ "$world" -eq 8 ]]; then nnodes=1; nproc=8; fi
  local scale_dir="$INDEP_ROOT/scale_${world}"
  echo "==> INDEP scale=$world nnodes=$nnodes nproc=$nproc offset=$node_offset"
  vcctl pod exec "$POD0" -- bash -lc "mkdir -p $scale_dir"
  local r g pod
  for ((r=0;r<nnodes;r++)); do
    g=$((node_offset + r))
    pod=$(pod_for "$g")
    vcctl pod exec "$POD0" -- bash -lc "cat > $scale_dir/launch_node${r}.sh <<EOF
#!/usr/bin/env bash
set -uo pipefail
export PATH=/root/miniconda3/envs/llm_test/bin:\\\$PATH
export PYTHONPATH=/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster:\\\$PYTHONPATH
export NODE_RANK=$r
export NPUS_PER_NODE=$nproc LOCAL_WORLD_SIZE=$nproc
export MASTER_ADDR=$(pod_for $node_offset).${JOB}
export MASTER_PORT=$port
cd /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster
if [[ $nproc -eq 8 ]]; then
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
torchrun --nproc_per_node=$nproc --nnodes=$nnodes --node_rank=$r \\
  --master_addr=\\\$MASTER_ADDR --master_port=\\\$MASTER_PORT \\
  virtual_sync_bench_npu.py --mode independent --iters 80 --warmup 8 \\
  --hidden 4096 --seq 1024 --layers 4 --out-dir $scale_dir --tag indep_$world \\
  2>&1 | tee $scale_dir/node${r}.log
echo INDEP_NODE_${r}_DONE
EOF
chmod +x $scale_dir/launch_node${r}.sh"
    vcctl pod exec "$pod" -- bash -lc \
      "setsid nohup bash $scale_dir/launch_node${r}.sh >$scale_dir/nohup_node${r}.log 2>&1 & echo SPAWNED_\$!"
    sleep 2
  done
}

wait_indep() {
  local world="$1" expect_files="$2"
  local scale_dir="$INDEP_ROOT/scale_${world}"
  local t0=$(date +%s)
  while true; do
    local n=$(vcctl pod exec "$POD0" -- bash -lc "ls $scale_dir/done_rank*.txt 2>/dev/null | wc -l" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
    n=${n:-0}
    local el=$(( $(date +%s) - t0 ))
    echo "  indep_$world done_files=$n/$expect_files elapsed=${el}s"
    [[ "$n" -ge "$expect_files" ]] && break
    if (( el > 900 )); then echo "  TIMEOUT indep_$world"; break; fi
    sleep 20
  done
}

# Wave1: 64+32 = 96
spawn_indep 64 0 27000
spawn_indep 32 4 27010
wait_indep 64 64
wait_indep 32 32

# Wave2: 16+8 on first two node slots (reuse after kill)
for p in ${JOB}-master-0 ${JOB}-worker-0; do
  vcctl pod exec "$p" -- bash -lc \
    'for pid in $(ps -eo pid,cmd | awk "/[v]irtual_sync|[t]orchrun/{print \$1}"); do kill -9 $pid 2>/dev/null; done; true' \
    >/dev/null 2>&1 || true
done
sleep 2
spawn_indep 16 0 27100
spawn_indep 8 1 27110
wait_indep 16 16
wait_indep 8 8

# Wave3: real Dense GBS∝DP for 8+16 (missing points), short iters
echo "==> REAL dense GBS∝DP 8+16"
pkill -f jumphost_dense_failslow || true
export JOB STAMP="${STAMP}_real" SCALES='8+16' GBS_PROP_DP=1 MICROBATCHES_PER_DP=160
export TRAIN_ITERS=40 PROBING=0 FAILSLOW_STEP_LOG=1 MASTER_PORT=27200
export SCALE_TIMEOUT_SEC=3600 SCALE_GRACE_SEC=600 TP=4 PP=2
export RUN_ROOT="$REAL_ROOT"
export LOCAL_LOG="/tmp/blockA_real_${STAMP}.log"
# N=8 needs NPUS=8 — jumphost assumes 16; for 8 use visible devices hack via wrapper env
# Simpler: only run 16 real if 8 unsupported; try 16 alone + skip 8 or use 16 only
# Override: run scales 16 only if 8 fails - for now spawn 16 only via SCALES=16, and 8 as special
export SCALES='16'
nohup bash /tmp/jumphost_dense_failslow.sh > /tmp/blockA_real_${STAMP}_nohup.out 2>&1 &
REAL_PID=$!
echo REAL_ORCH=$REAL_PID

# wait real 16
for i in $(seq 1 60); do
  if grep -q JUMPHOST_DENSE_FAILSLOW_DONE /tmp/blockA_real_${STAMP}.log 2>/dev/null; then break; fi
  n=$(vcctl pod exec "$POD0" -- bash -lc "wc -l < $REAL_ROOT/scale_16/step_times_rank0.jsonl 2>/dev/null || echo 0" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  echo "  real16 steps=${n:-0}/40"
  sleep 30
done

# parse network contrib
vcctl pod exec "$POD0" -- bash -lc "
cd /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster
python3 parse_failslow_gap.py $INDEP_ROOT --drop-first 8 --csv $INDEP_ROOT/gap_vs_n.csv || true
python3 parse_failslow_gap.py $REAL_ROOT --drop-first 8 --csv $REAL_ROOT/gap_vs_n.csv || true
python3 parse_network_contrib.py \
  --indep-root $INDEP_ROOT \
  --real-csv /afs-a3-weight-share/yinjinrun.p-huawei/results/dense_failslow_gbsprop/20260713_071316/gap_vs_n.csv \
  --real-csv $REAL_ROOT/gap_vs_n.csv \
  --real-csv /afs-a3-weight-share/yinjinrun.p-huawei/results/dense_failslow/20260713_001230/gap_vs_n.csv \
  --drop-first 8 \
  --out /afs-a3-weight-share/yinjinrun.p-huawei/results/reports/offline_20260713/network_contrib.csv
mkdir -p /afs-a3-weight-share/yinjinrun.p-huawei/results/reports/offline_20260713
{
  echo ''
  echo '## Block A：network_contrib = gap_real − gap_indep'
  echo ''
  echo \"indep stamp: $STAMP\"
  echo ''
  cat /afs-a3-weight-share/yinjinrun.p-huawei/results/reports/offline_20260713/network_contrib.csv
} >> /afs-a3-weight-share/yinjinrun.p-huawei/results/reports/offline_20260713/SUMMARY.md
"
echo "BLOCK_A_DONE stamp=$STAMP"
