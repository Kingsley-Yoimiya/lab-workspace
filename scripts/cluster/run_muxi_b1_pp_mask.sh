#!/usr/bin/env bash
# B1: 单节点 8 卡独立负载 + PP stage 延迟注入 vs 基线
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_FORCE_JUMP="${CLUSTER_FORCE_JUMP:-1}"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="${STAMP:-$(cat /tmp/muxi_day_stamp.txt)}"
AFS_OUT="${AFS_OUT:-$(cat /tmp/muxi_day_afs.txt)}"
DAY_ROOT="${DAY_ROOT:-$(cat /tmp/muxi_day_root.txt)}"
AFS_BENCH="/afs-a3-weight-share/montyyin/lab-workspace/scripts/cluster"
ITERS="${ITERS:-1500}"
POD="${POD:-${CLUSTER_JOB}-master-0}"
LOGF="$DAY_ROOT/b1.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGF"; }

run_mode() {
  local tag="$1" inject="$2"
  local out="$AFS_OUT/B1/$tag"
  cluster_pod_exec "$POD" "mkdir -p '$out'; rm -f '$out'/done_rank*.txt '$out'/step_times_rank*.jsonl"
  cluster_pod_exec "$POD" 'set +e; me=$$; ps -eo pid=,args= | awk -v me="$me" '\''$1==me{next} /[v]irtual_sync_bench/{print $1}'\'' | while read p; do kill -9 $p 2>/dev/null; done; echo K' || true
  sleep 2
  cluster_pod_exec "$POD" "
set +e
for g in \$(seq 0 7); do
  nohup env PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1 CUDA_HOME=/opt/maca/tools/cu-bridge \
    PYTHONPATH=$AFS_BENCH:\$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=\$g LOCAL_RANK=0 NODE_RANK=0 GPUS_PER_NODE=8 GLOBAL_RANK=\$g WORLD_SIZE=8 PP_SIZE=4 \
    DELAY_INJECT=$inject DELAY_STAGE=1 DELAY_MS=80 DELAY_EVERY=20 DELAY_BURST=3 \
    python3 $AFS_BENCH/virtual_sync_bench.py \
      --mode independent --iters $ITERS --warmup 20 \
      --hidden 4096 --seq 2048 --layers 8 --batch 2 \
      --out-dir $out --tag B1_$tag \
      >$out/gpu\${g}.log 2>&1 &
done
echo FIRED_$tag
"
  local start=$SECONDS n
  while (( SECONDS - start < 3600 )); do
    n=$(cluster_pod_exec "$POD" "ls '$out'/done_rank*.txt 2>/dev/null|wc -l" | tr -dc '0-9')
    n=${n:-0}
    log "  $tag done=$n/8"
    [[ "$n" -ge 8 ]] && break
    sleep 20
  done
}

log "B1 baseline"
run_mode baseline 0
log "B1 inject stage1"
run_mode inject 1
cluster_pod_exec "$POD" "echo OK > '$AFS_OUT/B1/DONE'"
echo B1_DONE > "$DAY_ROOT/B1.done"
log "B1_DONE"
