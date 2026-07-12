#!/usr/bin/env bash
# 本机：等 AFS WAVE1.done，再跑离线+B1+B2+C1+报告
set +e
DAY_ROOT=$(cat /tmp/muxi_day_root.txt)
AFS=$(cat /tmp/muxi_day_afs.txt)
STAMP=$(cat /tmp/muxi_day_stamp.txt)
LOG=$DAY_ROOT/pipeline_rest2.log
MASTER=yushan-muxi-card-screen-128-cp-copy-master-0
export CLUSTER_FORCE_JUMP=1 STAMP DAY_ROOT AFS_OUT=$AFS

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
log "wait AFS WAVE1.done"

while true; do
  line=$(ssh -o BatchMode=yes -o ConnectTimeout=30 ais-cf3e61a5 \
    "KUBECONFIG=/root/.kube/config.muxi-mohe vcctl pod exec $MASTER -- bash -lc 'test -f $AFS/WAVE1.done && echo YES || echo NO; tail -2 /tmp/muxi_day_drive.log'" 2>/dev/null)
  log "$line"
  echo "$line" | grep -q YES && break
  sleep 90
done

touch "$DAY_ROOT/WAVE1_CLUSTER.done" "$DAY_ROOT/A1.done" "$DAY_ROOT/A2.done"
log "offline"
bash /Users/yinjinrun/random-thing/project/lab-workspace/scripts/cluster/offline_muxi_day_wave1.sh >>"$DAY_ROOT/offline.log" 2>&1
log "B1"
bash /Users/yinjinrun/random-thing/project/lab-workspace/scripts/cluster/run_muxi_b1_pp_mask.sh >>"$DAY_ROOT/b1_outer.log" 2>&1
mkdir -p "$DAY_ROOT/results/B1/baseline" "$DAY_ROOT/results/B1/inject" "$DAY_ROOT/results/analysis"
for mode in baseline inject; do
  ssh -o BatchMode=yes ais-cf3e61a5 \
    "KUBECONFIG=/root/.kube/config.muxi-mohe vcctl pod exec $MASTER -- bash -lc 'cd $AFS/B1/$mode && tar czf - step_times_rank*.jsonl 2>/dev/null'" \
    >"$DAY_ROOT/results/B1/${mode}.tgz"
  tar xzf "$DAY_ROOT/results/B1/${mode}.tgz" -C "$DAY_ROOT/results/B1/$mode" 2>/dev/null
done
python3 /Users/yinjinrun/random-thing/project/lab-workspace/scripts/cluster/parse_b1_pp_mask.py \
  --baseline "$DAY_ROOT/results/B1/baseline" --inject "$DAY_ROOT/results/B1/inject" \
  --out "$DAY_ROOT/results/analysis/B1_summary.json"
touch "$DAY_ROOT/B1.done"
log "B2"
bash /Users/yinjinrun/random-thing/project/lab-workspace/scripts/cluster/run_muxi_b2_preempt.sh >>"$DAY_ROOT/b2_outer.log" 2>&1
mkdir -p "$DAY_ROOT/results/B2/preempt"
ssh -o BatchMode=yes ais-cf3e61a5 \
  "KUBECONFIG=/root/.kube/config.muxi-mohe vcctl pod exec yushan-muxi-card-screen-128-cp-copy-worker-5 -- bash -lc 'cd $AFS/B2/preempt && tar czf - step_times_rank*.jsonl diagnose.json ps_snapshot.txt preempt*.log 2>/dev/null'" \
  >"$DAY_ROOT/results/B2.tgz"
tar xzf "$DAY_ROOT/results/B2.tgz" -C "$DAY_ROOT/results/B2/preempt" 2>/dev/null
touch "$DAY_ROOT/B2.done"
log "C1"
mkdir -p "$DAY_ROOT/results/C1"
LOCAL_C1="$DAY_ROOT/results/C1" AFS_OUT="$AFS" \
  bash /Users/yinjinrun/random-thing/project/lab-workspace/scripts/cluster/probe_muxi_pair_matrix.sh >>"$DAY_ROOT/c1_outer.log" 2>&1
ssh -o BatchMode=yes ais-cf3e61a5 \
  "KUBECONFIG=/root/.kube/config.muxi-mohe vcctl pod exec $MASTER -- bash -lc 'cat $AFS/C1/pair_summary.json'" \
  >"$DAY_ROOT/results/C1/pair_summary.json" 2>/dev/null
ssh -o BatchMode=yes ais-cf3e61a5 \
  "KUBECONFIG=/root/.kube/config.muxi-mohe vcctl pod exec $MASTER -- bash -lc 'cat $AFS/C1/pair_matrix.json'" \
  >"$DAY_ROOT/results/C1/pair_matrix.json" 2>/dev/null
touch "$DAY_ROOT/C1.done"
touch "$DAY_ROOT/PIPELINE_REST.done"
log ALL_REST_DONE
