#!/usr/bin/env bash
# 轮询 MFU fire 的 done 标记
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

WORLD="${1:?usage: poll_mfu_bench_muxi.sh <world>}"
MODE="${MODE:-dense}"
NPROC="${DEVICES_PER_NODE:-8}"
nnodes=$((WORLD / NPROC))
AFS_OUT="${AFS_OUT:?set AFS_OUT}"
prefix="mfu_${MODE}_${WORLD}"

status="$(cluster_pod_exec "${CLUSTER_POD}" "
ok=0; fail=0; miss=0
for r in \$(seq 0 $((nnodes - 1))); do
  if [[ -f $AFS_OUT/${prefix}.node_\$r.done ]]; then ok=\$((ok+1))
  elif [[ -f $AFS_OUT/${prefix}.node_\$r.fail ]]; then fail=\$((fail+1))
  else miss=\$((miss+1)); fi
done
echo OK=\$ok FAIL=\$fail MISS=\$miss
tail -5 $AFS_OUT/${prefix}.node_0.log 2>/dev/null || true
")"
echo "$status"

if echo "$status" | grep -q "OK=${nnodes} FAIL=0 MISS=0"; then
  cluster_pod_exec "${CLUSTER_POD}" "
[[ -f $AFS_OUT/${prefix}.jsonl ]] && wc -l $AFS_OUT/${prefix}.jsonl && cat $AFS_OUT/${prefix}.jsonl
"
  echo COMPLETE mfu mode=$MODE world=$WORLD
  exit 0
fi
if echo "$status" | grep -qE 'FAIL=[1-9]'; then
  echo FAILED mfu mode=$MODE world=$WORLD
  exit 2
fi
echo RUNNING mfu mode=$MODE world=$WORLD
exit 1
