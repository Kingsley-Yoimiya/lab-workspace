#!/usr/bin/env bash
# 轮询 durable screen：数 *.done 标记，齐了再 aggregate
# 用法: LOG_DIR=... EXPECT_PODS=64 ./poll_screen_durable_muxi.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

LOG_DIR="${LOG_DIR:?set LOG_DIR}"
AFS_OUT_DIR="$(cat "$LOG_DIR/AFS_OUT_DIR.txt")"
OUT_JSONL="$(cat "$LOG_DIR/OUT_JSONL.txt")"
EXPECT_PODS="${EXPECT_PODS:-$(cluster_pods_running | wc -l | tr -d ' ')}"
POLL_SEC="${POLL_SEC:-45}"
POLL_MAX="${POLL_MAX:-200}"

for i in $(seq 1 "$POLL_MAX"); do
  st=$(cluster_pod_exec "${CLUSTER_POD}" "
done=\$(ls -1 '$AFS_OUT_DIR'/*.done 2>/dev/null | wc -l | tr -d ' ')
fail=\$(ls -1 '$AFS_OUT_DIR'/*.fail 2>/dev/null | wc -l | tr -d ' ')
lines=0
for f in '$AFS_OUT_DIR'/*.jsonl; do
  [[ -f \$f ]] || continue
  n=\$(wc -l < \$f | tr -d ' ')
  lines=\$((lines+n))
done
echo DONE=\$done FAIL=\$fail LINES=\$lines
" 2>/dev/null | tail -1)
  echo "poll[$i] expect=$EXPECT_PODS $st"
  done_n=$(echo "$st" | sed -n 's/.*DONE=\([0-9]*\).*/\1/p')
  if [[ -n "$done_n" && "$done_n" -ge "$EXPECT_PODS" ]]; then
    echo ALL_DONE
    break
  fi
  sleep "$POLL_SEC"
done

echo "==> aggregate"
cluster_pod_exec "${CLUSTER_POD}" "
set -euo pipefail
cd '$AFS_CS'
python - <<'PY'
from card_screen.cluster.aggregate import aggregate
from pathlib import Path
out = Path('$OUT_JSONL')
parent = Path('$AFS_OUT_DIR')
# screen often writes result.<host>.jsonl next to --out
cands = sorted(parent.glob('*.jsonl'))
cands = [c for c in cands if not c.name.endswith('.cluster.json')]
print('candidates', len(cands))
merged = parent / (out.stem + '.merged.jsonl')
with merged.open('w') as w:
    for c in cands:
        txt = c.read_text()
        w.write(txt)
        if txt and not txt.endswith('\n'):
            w.write('\n')
summary = aggregate(str(merged), slow_frac=0.2)
print('n_cards', summary.get('n_cards'))
print('summary', summary.get('summary'))
print('wrote', str(merged).replace('.jsonl','') + '.cluster.json')
PY
ls -la '$AFS_OUT_DIR' | head -50
"
echo "POLL_OK $AFS_OUT_DIR"
