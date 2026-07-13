#!/usr/bin/env bash
# Wave4：加深 B1 注入幅度扫描（DELAY_MS × DELAY_EVERY）
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_FORCE_JUMP="${CLUSTER_FORCE_JUMP:-1}"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"
set +e

STAMP="${STAMP:-$(cat /tmp/muxi_day_stamp.txt)}"
AFS_OUT="${AFS_OUT:-$(cat /tmp/muxi_day_afs.txt)}"
DAY_ROOT="${DAY_ROOT:-$(cat /tmp/muxi_day_root.txt)}"
AFS_BENCH="/afs-a3-weight-share/yinjinrun.p/lab-workspace/scripts/cluster"
ITERS="${ITERS:-600}"
POD="${POD:-${CLUSTER_JOB}-master-0}"
LOGF="$DAY_ROOT/b1_scan.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGF"; }

# 脚本已由战役驱动 base64 上传到 AFS；此处仅确认存在
cluster_pod_exec "$POD" "ls -l $AFS_BENCH/delay_inject.py $AFS_BENCH/virtual_sync_bench.py" || true

run_one() {
  local tag="$1" inject="$2" dms="$3" every="$4" burst="$5"
  local out="$AFS_OUT/B1_scan/$tag"
  cluster_pod_exec "$POD" "mkdir -p '$out'; rm -f '$out'/done_rank*.txt '$out'/step_times_rank*.jsonl"
  cluster_pod_exec "$POD" 'set +e; me=$$; ps -eo pid=,args= | awk -v me="$me" '\''$1==me{next} /[v]irtual_sync_bench/{print $1}'\'' | while read p; do kill -9 $p 2>/dev/null; done; echo K' || true
  sleep 2
  cluster_pod_exec "$POD" "
set +e
for g in \$(seq 0 7); do
  nohup env PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1 CUDA_HOME=/opt/maca/tools/cu-bridge \
    PYTHONPATH=$AFS_BENCH:\$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=\$g LOCAL_RANK=0 NODE_RANK=0 GPUS_PER_NODE=8 GLOBAL_RANK=\$g WORLD_SIZE=8 PP_SIZE=4 \
    DELAY_INJECT=$inject DELAY_STAGE=1 DELAY_MS=$dms DELAY_EVERY=$every DELAY_BURST=$burst \
    python3 $AFS_BENCH/virtual_sync_bench.py \
      --mode independent --iters $ITERS --warmup 10 \
      --hidden 4096 --seq 2048 --layers 8 --batch 2 \
      --out-dir $out --tag B1_$tag \
      >$out/gpu\${g}.log 2>&1 &
done
echo FIRED_$tag
"
  local start=$SECONDS n
  while (( SECONDS - start < 2400 )); do
    n=$(cluster_pod_exec "$POD" "ls '$out'/done_rank*.txt 2>/dev/null|wc -l" | tr -dc '0-9')
    n=${n:-0}
    log "  $tag done=$n/8"
    [[ "$n" -ge 8 ]] && break
    sleep 15
  done
}

mkdir -p "$DAY_ROOT/results/B1_scan" "$DAY_ROOT/results/analysis"
log "B1_scan baseline"
run_one baseline 0 0 20 1
for cfg in "ms80_e20_b3:80:20:3" "ms200_e10_b5:200:10:5" "ms400_e5_b3:400:5:3"; do
  tag=${cfg%%:*}; rest=${cfg#*:}; dms=${rest%%:*}; rest=${rest#*:}; every=${rest%%:*}; burst=${rest#*:}
  log "B1_scan inject $tag DELAY_MS=$dms EVERY=$every BURST=$burst"
  run_one "$tag" 1 "$dms" "$every" "$burst"
done

# 拉回并解析
BASE_LOCAL="$DAY_ROOT/results/B1_scan/baseline"
mkdir -p "$BASE_LOCAL"
ssh -o BatchMode=yes ais-cf3e61a5 \
  "KUBECONFIG=/root/.kube/config.muxi-mohe vcctl pod exec $POD -- bash -lc 'cd $AFS_OUT/B1_scan/baseline && tar czf - step_times_rank*.jsonl'" \
  >"$DAY_ROOT/results/B1_scan/baseline.tgz"
tar xzf "$DAY_ROOT/results/B1_scan/baseline.tgz" -C "$BASE_LOCAL" 2>/dev/null
SCAN_JSON="$DAY_ROOT/results/analysis/B1_scan_summary.json"
echo '[' >"$SCAN_JSON.tmp"
first=1
for tag in ms80_e20_b3 ms200_e10_b5 ms400_e5_b3; do
  loc="$DAY_ROOT/results/B1_scan/$tag"
  mkdir -p "$loc"
  ssh -o BatchMode=yes ais-cf3e61a5 \
    "KUBECONFIG=/root/.kube/config.muxi-mohe vcctl pod exec $POD -- bash -lc 'cd $AFS_OUT/B1_scan/$tag && tar czf - step_times_rank*.jsonl'" \
    >"$DAY_ROOT/results/B1_scan/${tag}.tgz"
  tar xzf "$DAY_ROOT/results/B1_scan/${tag}.tgz" -C "$loc" 2>/dev/null
  outj="$DAY_ROOT/results/analysis/B1_${tag}.json"
  python3 "$SCRIPT_DIR/parse_b1_pp_mask.py" --baseline "$BASE_LOCAL" --inject "$loc" --out "$outj"
  [[ $first -eq 1 ]] || echo ',' >>"$SCAN_JSON.tmp"
  first=0
  python3 -c "import json;print(json.dumps({'tag':'$tag',**json.load(open('$outj'))},ensure_ascii=False))" >>"$SCAN_JSON.tmp"
done
echo ']' >>"$SCAN_JSON.tmp"
mv "$SCAN_JSON.tmp" "$SCAN_JSON"
cluster_pod_exec "$POD" "echo OK > '$AFS_OUT/B1_scan/DONE'"
touch "$DAY_ROOT/B1_SCAN.done" "$DAY_ROOT/WAVE4.done"
log "B1_SCAN_DONE"
cat "$SCAN_JSON"
