#!/usr/bin/env bash
# 长窗口：干扰对主进程影响（补实验）
# 用法: PHYS=11 bash run_main_vs_inject.sh
set -euo pipefail
PHYS="${PHYS:?PHYS required (physical card)}"
HOST_CS="${HOST_CS:-$HOME/CARD_SCREEN}"
HOST_SCRIPTS="${HOST_SCRIPTS:-$HOME/lab-workspace/scripts/cluster}"
IMG="${IMG:-quay.io/ascend/vllm-ascend:v0.19.1rc1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)-main-vs-inject-d${PHYS}}"
OUT_HOST="$HOST_CS/results/$RUN_ID"
mkdir -p "$OUT_HOST" "$HOST_CS/logs"

DEVICES=()
for i in $(seq 0 15); do DEVICES+=(--device="/dev/davinci${i}"); done
DEVICES+=(--device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc)

WINDOW_S="${WINDOW_S:-8}"
REPEATS="${REPEATS:-2}"
WORKLOADS="${WORKLOADS:-gemm block}"
TL_SEGMENT_S="${TL_SEGMENT_S:-8}"
TL_REPEATS="${TL_REPEATS:-2}"
DO_TIMELINE="${DO_TIMELINE:-1}"

LOG="$HOST_CS/logs/${RUN_ID}.log"
{
  echo "RUN_ID=$RUN_ID PHYS=$PHYS WINDOW_S=$WINDOW_S REPEATS=$REPEATS"
  date -Is
} | tee "$LOG"

sudo -n docker run --rm --name "main-vs-inject-${RUN_ID}" --network=host --ipc=host \
  "${DEVICES[@]}" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$HOST_CS":/workspace/CARD_SCREEN \
  -v "$HOST_SCRIPTS":/workspace/inject \
  -w /workspace/inject \
  -e "ASCEND_RT_VISIBLE_DEVICES=${PHYS}" \
  -e PYTHONUNBUFFERED=1 \
  "$IMG" bash -lc "
set -euo pipefail
OUT=/workspace/CARD_SCREEN/results/${RUN_ID}
mkdir -p \"\$OUT\"

# 1) ABBA 稳态：不同干扰对主进程吞吐
for wl in ${WORKLOADS}; do
  echo \"=== ABBA workload=\$wl ===\"
  python3 main_vs_inject_bench_npu.py \
    --device 0 \
    --protocol abba \
    --workload \"\$wl\" \
    --factors placebo,cube,vector,hbm_mte \
    --window-s ${WINDOW_S} \
    --repeats ${REPEATS} \
    --duty 1.0 \
    --out \"\$OUT/abba_\${wl}.jsonl\"
done

# 2) Timeline 突发：quiet→burst→quiet
if [ \"${DO_TIMELINE}\" = \"1\" ]; then
  for wl in ${WORKLOADS}; do
    echo \"=== TIMELINE workload=\$wl ===\"
    python3 main_vs_inject_bench_npu.py \
      --device 0 \
      --protocol timeline \
      --workload \"\$wl\" \
      --factors cube,hbm_mte \
      --window-s ${TL_SEGMENT_S} \
      --repeats ${TL_REPEATS} \
      --duty 1.0 \
      --out \"\$OUT/timeline_\${wl}.jsonl\"
  done
fi

python3 - <<'PY'
import json
from pathlib import Path
out = Path('/workspace/CARD_SCREEN/results/${RUN_ID}')
summary = {'abba': {}, 'timeline': {}}
for p in sorted(out.glob('abba_*.jsonl')):
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    summary['abba'][p.stem] = [r for r in rows if r.get('record') == 'factor_summary']
for p in sorted(out.glob('timeline_*.jsonl')):
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    summary['timeline'][p.stem] = [r for r in rows if r.get('record') == 'timeline_summary']
out.joinpath('summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
print('WROTE', out / 'summary.json')
print('## ABBA')
for wl, sums in summary['abba'].items():
    print(' ', wl)
    for s in sums:
        print(f\"    {s['factor']:10s} thru_drop={s['main_slowdown_pct']:.1f}%  iter_p50_up={s.get('iter_ms_p50_increase_pct', float('nan')):.1f}%\")
print('## TIMELINE')
for wl, sums in summary['timeline'].items():
    print(' ', wl)
    for s in sums:
        print(f\"    {s['factor']:10s} rep={s['repeat']} burst_slowdown={s['burst_vs_pre_slowdown_pct']:.1f}% post_residual={s['post_vs_pre_slowdown_pct']:.1f}%\")
PY
echo MAIN_VS_INJECT_DONE
" 2>&1 | tee -a "$LOG"

echo "$OUT_HOST"
