#!/usr/bin/env bash
# 并行：实验四 NPU 抢占（master 16卡）+ 实验三 rematch rank 注入（worker-1 16卡）
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/root/.kube/config.huawei-a3-241ceshi}"
JOB="${JOB:-montyyin-moe96-r2}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REMOTE_DIR="${REMOTE_DIR:-/root/montyyin-lab-remote}"
ROOT="/afs-a3-weight-share/yinjinrun.p-huawei/results/exp45_parallel/${STAMP}"
LOG="/tmp/exp45_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> EXP45 STAMP=$STAMP $(date -Iseconds)"

POD0=${JOB}-master-0
POD1=${JOB}-worker-1

# sync files
for f in failslow_step_timer.py npu_busy_preempt.py parse_failslow_gap.py parse_pp_inject_ab.py; do
  [[ -f "$REMOTE_DIR/$f" ]] || continue
  dst_hooks=/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/hooks/$f
  dst=/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/$f
  if [[ "$f" == failslow_step_timer.py ]]; then
    cat "$REMOTE_DIR/$f" | vcctl pod exec -i "$POD0" -- bash -lc "cat > $dst_hooks"
  else
    cat "$REMOTE_DIR/$f" | vcctl pod exec -i "$POD0" -- bash -lc "cat > $dst"
  fi
done
# also copy preempt to AFS scripts
cat "$REMOTE_DIR/npu_busy_preempt.py" | vcctl pod exec -i "$POD0" -- bash -lc \
  "cat > /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/npu_busy_preempt.py"

for p in $POD0 $POD1 ${JOB}-worker-0 ${JOB}-worker-2 ${JOB}-worker-3 ${JOB}-worker-4; do
  vcctl pod exec "$p" -- bash -lc \
    'for pid in $(ps -eo pid,cmd | awk "/[p]retrain_gpt.py|[t]orchrun|[n]pu_busy/{print \$1}"); do kill -9 $pid 2>/dev/null; done; true' \
    >/dev/null 2>&1 || true
done
sleep 2

spawn_dense() {
  local pod="$1" tag="$2" port="$3" inject="$4" ranks="${5:-}" master_dns="$6"
  local scale_dir="$ROOT/$tag"
  vcctl pod exec "$POD0" -- bash -lc "mkdir -p $scale_dir"
  vcctl pod exec "$POD0" -- bash -lc "cat > $scale_dir/launch_rank0.sh <<EOF
#!/usr/bin/env bash
set -uo pipefail
export TP=4 PP=2 EP=1 ETP=1 MBS=1 GBS=320 SEQ_LENGTH=4096
export SKIP_TB=1 SKIP_SAVE=1 SKIP_PROFILE=1 TRAIN_ITERS=30
export PROBING=0 FAILSLOW_STEP_LOG=1
export DELAY_INJECT=$inject DELAY_STAGE=1 DELAY_MS=1500 DELAY_EVERY=3 DELAY_BURST=1
export DELAY_RANKS='$ranks'
export PP_SIZE=2 WORLD_SIZE_NPUS=16
export PATH=/root/miniconda3/envs/llm_test/bin:\\\$PATH
export PYTHONPATH=/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/hooks:/MindSpeed-LLM/MindSpeed:\\ foresPYTHONPATH:-}
export WORLD_SIZE=1 NNODES=1 RANK=0 NODE_RANK=0
export MASTER_ADDR=$master_dns MASTER_PORT=$port
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
  vcctl pod exec "$pod" -- bash -lc \
    "setsid nohup bash $scale_dir/launch_rank0.sh >$scale_dir/nohup_rank0.log 2>&1 & echo SPAWNED_\$!"
}

echo "==> Exp4: dense on $POD0 + NPU preempt cards 14,15 mid-run + npu-smi 双信号"
spawn_dense "$POD0" exp4_preempt 26400 0 "" "${POD0}.${JOB}"
# Block E：全程采 npu-smi（与 step timer 对照）
vcctl pod exec "$POD0" -- bash -lc "
mkdir -p $ROOT/exp4_preempt
nohup bash -c '
  for i in \$(seq 1 90); do
    echo TS=\$(date -Iseconds) >> $ROOT/exp4_preempt/npu_smi_sample.log
    npu-smi info 2>/dev/null | head -120 >> $ROOT/exp4_preempt/npu_smi_sample.log
    echo --- >> $ROOT/exp4_preempt/npu_smi_sample.log
    sleep 8
  done
' > $ROOT/exp4_preempt/npu_smi_sample.nohup 2>&1 &
echo SMI_PID=\$!
"
sleep 90
# start preempt for 120s on device 14 and 15
vcctl pod exec "$POD0" -- bash -lc "
cd /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster
echo PREEMPT_T0=\$(date -Iseconds) | tee $ROOT/exp4_preempt/preempt_meta.txt
nohup python3 npu_busy_preempt.py --device 14 --seconds 120 --size 4096 > $ROOT/exp4_preempt/preempt14.log 2>&1 &
nohup python3 npu_busy_preempt.py --device 15 --seconds 120 --size 4096 > $ROOT/exp4_preempt/preempt15.log 2>&1 &
echo PREEMPT_STARTED
"

echo "==> Exp3 rematch: rank inject 12,13,14,15 on $POD1"
spawn_dense "$POD1" exp3_rank_inject 26410 1 "12,13,14,15" "${POD1}.${JOB}"

# wait both
for i in $(seq 1 50); do
  s4=$(vcctl pod exec "$POD0" -- bash -lc "wc -l < $ROOT/exp4_preempt/step_times_rank0.jsonl 2>/dev/null || echo 0" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  s3=$(vcctl pod exec "$POD0" -- bash -lc "wc -l < $ROOT/exp3_rank_inject/step_times_rank0.jsonl 2>/dev/null || echo 0" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  d4=$(vcctl pod exec "$POD0" -- bash -lc "grep -c TRAIN_RANK_0_DONE $ROOT/exp4_preempt/rank0.log 2>/dev/null || echo 0" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  d3=$(vcctl pod exec "$POD0" -- bash -lc "grep -c TRAIN_RANK_0_DONE $ROOT/exp3_rank_inject/rank0.log 2>/dev/null || echo 0" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  s4=${s4:-0}; s3=${s3:-0}; d4=${d4:-0}; d3=${d3:-0}
  echo "tick=$i exp4=$s4/30 done=$d4 | exp3=$s3/30 done=$d3"
  if [[ "$d4" -ge 1 && "$d3" -ge 1 ]]; then break; fi
  if [[ "$d4" -ge 1 && "$s3" -eq 0 && "$i" -gt 20 ]]; then echo "exp3 may have failed"; break; fi
  sleep 45
done

# parse（修 timer 后：delay 应计入 ms；Exp4 双信号对照）
vcctl pod exec "$POD0" -- bash -lc "
cd /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster
python3 parse_failslow_gap.py $ROOT/exp4_preempt --drop-first 5 --csv $ROOT/exp4_preempt/gap_vs_n.csv || true
python3 parse_failslow_gap.py $ROOT/exp3_rank_inject --drop-first 5 --csv $ROOT/exp3_rank_inject/gap_vs_n.csv || true
python3 - <<'PY'
import json, statistics, re
from pathlib import Path
from collections import defaultdict

def rank_medians(root, drop=5):
    by=defaultdict(list); delayed=defaultdict(int)
    for p in Path(root).glob('step_times_rank*.jsonl'):
        for line in p.read_text().splitlines():
            if not line.strip(): continue
            r=json.loads(line)
            if int(r['iter'])<=drop: continue
            by[int(r['rank'])].append(float(r['ms']))
            if r.get('delayed'): delayed[int(r['rank'])]+=1
    med={k:statistics.median(v) for k,v in by.items() if v}
    return med, delayed

# Exp3 rematch：注入 rank 应明显更慢
med3, del3 = rank_medians('$ROOT/exp3_rank_inject')
inj=[med3[k] for k in range(12,16) if k in med3]
oth=[med3[k] for k in range(0,12) if k in med3]
out3={
  'timer_fix': 'delay_inside_perf_counter',
  'injected_ranks_median_ms': statistics.median(inj) if inj else None,
  'other_ranks_median_ms': statistics.median(oth) if oth else None,
  'delta_ms': (statistics.median(inj)-statistics.median(oth)) if inj and oth else None,
  'delayed_counts': {str(k):del3.get(k,0) for k in range(16)},
  'per_rank_median_ms': {str(k):med3[k] for k in sorted(med3)},
  'verdict': ('PASS' if (inj and oth and statistics.median(inj)>statistics.median(oth)+500) else 'WEAK'),
}
Path('$ROOT/exp3_rank_inject/rank_contrast.json').write_text(json.dumps(out3,indent=2))
print('EXP3', json.dumps(out3,indent=2))

# Exp4：step 最慢卡 vs npu-smi AICore 异常卡
med4, _ = rank_medians('$ROOT/exp4_preempt')
slow=sorted(med4.items(), key=lambda kv: -kv[1])[:4]
# 粗解析 npu-smi：按 Phy-ID 取 AICore 序列均值
aicore=defaultdict(list)
pending=None
text=Path('$ROOT/exp4_preempt/npu_smi_sample.log').read_text(errors='ignore') if Path('$ROOT/exp4_preempt/npu_smi_sample.log').exists() else ''
re_pwr=re.compile(r'\\|\\s*(\\d+)\\s+Ascend\\S+\\s+\\|\\s+\\S+\\s+\\|\\s+([\\d.]+|-)\\s+')
re_chip=re.compile(r'\\|\\s*(\\d+)\\s+(\\d+)\\s+\\|\\s+\\S+\\s+\\|\\s+(\\d+)\\s+')
for line in text.splitlines():
    if re_pwr.search(line):
        continue
    m=re_chip.search(line)
    if m:
        aicore[int(m.group(2))].append(int(m.group(3)))
ac_mean={k: statistics.mean(v) for k,v in aicore.items() if v}
# 抢占卡预期 AICore 更高或 step 更慢
out4={
  'slowest_ranks_by_step': [{'rank':r,'median_ms':m} for r,m in slow],
  'aicore_mean_by_phy': {str(k):ac_mean[k] for k in sorted(ac_mean)},
  'preempt_targets': [14,15],
  'step_hits_preempt': [r for r,_ in slow if r in (14,15)],
  'aicore_top': sorted(ac_mean.items(), key=lambda kv:-kv[1])[:4],
}
combo = bool(set(out4['step_hits_preempt'])) or any(p in dict(out4['aicore_top']) for p in (14,15))
out4['dual_signal_verdict'] = 'PASS' if combo else 'WEAK'
Path('$ROOT/exp4_preempt/dual_signal.json').write_text(json.dumps(out4,indent=2))
print('EXP4', json.dumps(out4,indent=2))
PY
pkill -f npu_smi_sample || true
mkdir -p /afs-a3-weight-share/yinjinrun.p-huawei/results/reports/offline_20260713
{
  echo ''
  echo '## 实验四 rematch（timer 修复后）：NPU busy + npu-smi 双信号'
  echo \"Stamp: $ROOT/exp4_preempt\"
  echo ''
  cat $ROOT/exp4_preempt/dual_signal.json 2>/dev/null || true
  echo ''
  echo '## 实验三 rematch（timer 修复后）：DELAY_RANKS=12-15'
  echo ''
  cat $ROOT/exp3_rank_inject/rank_contrast.json 2>/dev/null || true
} >> /afs-a3-weight-share/yinjinrun.p-huawei/results/reports/offline_20260713/SUMMARY.md
"
echo "EXP45_DONE stamp=$STAMP → $ROOT"
