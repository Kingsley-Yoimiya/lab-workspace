#!/usr/bin/env bash
# B2: 外部抢占 — 独立负载 + GPU busy 干扰
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_FORCE_JUMP="${CLUSTER_FORCE_JUMP:-1}"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="${STAMP:-$(cat /tmp/muxi_day_stamp.txt)}"
AFS_OUT="${AFS_OUT:-$(cat /tmp/muxi_day_afs.txt)}"
DAY_ROOT="${DAY_ROOT:-$(cat /tmp/muxi_day_root.txt)}"
AFS_BENCH="/afs-a3-weight-share/yinjinrun.p/lab-workspace/scripts/cluster"
ITERS="${ITERS:-800}"
POD="${POD:-${CLUSTER_JOB}-worker-5}"
LOGF="$DAY_ROOT/b2.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGF"; }

out="$AFS_OUT/B2/preempt"
cluster_pod_exec "$POD" "mkdir -p '$out'; rm -f '$out'/*"
cluster_pod_exec "$POD" 'set +e; me=$$; ps -eo pid=,args= | awk -v me="$me" '\''$1==me{next} /[v]irtual_sync_bench|[g]pu_busy/{print $1}'\'' | while read p; do kill -9 $p 2>/dev/null; done; echo K' || true
sleep 1

# 先起负载（8 卡）
cluster_pod_exec "$POD" "
set +e
for g in \$(seq 0 7); do
  nohup env PATH=/opt/conda/bin:\$PATH PYTHONUNBUFFERED=1 CUDA_HOME=/opt/maca/tools/cu-bridge \
    CUDA_VISIBLE_DEVICES=\$g LOCAL_RANK=0 NODE_RANK=0 GPUS_PER_NODE=8 GLOBAL_RANK=\$g \
    python3 $AFS_BENCH/virtual_sync_bench.py \
      --mode independent --iters $ITERS --warmup 10 \
      --hidden 4096 --seq 2048 --layers 8 --batch 2 \
      --out-dir $out --tag B2 \
      >$out/gpu\${g}.log 2>&1 &
done
echo LOAD_FIRED
"
sleep 15
# 在 GPU0/1 上抢占 ~一半时间
cluster_pod_exec "$POD" "
set +e
nohup env PATH=/opt/conda/bin:\$PATH CUDA_VISIBLE_DEVICES=0 \
  python3 $AFS_BENCH/gpu_busy_preempt.py --seconds 90 --size 8192 >$out/preempt0.log 2>&1 &
nohup env PATH=/opt/conda/bin:\$PATH CUDA_VISIBLE_DEVICES=1 \
  python3 $AFS_BENCH/gpu_busy_preempt.py --seconds 90 --size 8192 >$out/preempt1.log 2>&1 &
echo PREEMPT_FIRED
ps -ef | grep -E 'virtual_sync|gpu_busy' | grep -v grep | head -20 > $out/ps_snapshot.txt
"
log "waiting B2 load"
start=$SECONDS
while (( SECONDS - start < 2400 )); do
  n=$(cluster_pod_exec "$POD" "ls '$out'/done_rank*.txt 2>/dev/null|wc -l" | tr -dc '0-9')
  n=${n:-0}
  log "  B2 done=$n/8"
  [[ "$n" -ge 8 ]] && break
  sleep 20
done
# 诊断摘要
cluster_pod_exec "$POD" "
python3 - <<'PY'
import json,statistics
from pathlib import Path
out=Path('$out')
by={}
for p in out.glob('step_times_rank*.jsonl'):
  rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
  if not rows: continue
  r=rows[0]['global_rank']
  ms=[x['ms'] for x in rows[100:]] or [x['ms'] for x in rows]
  by[r]=statistics.median(ms)
med=statistics.median(by.values())
slow=sorted(by.items(), key=lambda x:-x[1])[:3]
open(out/'diagnose.json','w').write(json.dumps({'median_ms':med,'per_rank':by,'slowest':slow},indent=2))
print('DIAG', slow)
PY
cat $out/ps_snapshot.txt | head -15
echo OK > $out/../DONE
"
echo B2_DONE > "$DAY_ROOT/B2.done"
log "B2_DONE"
