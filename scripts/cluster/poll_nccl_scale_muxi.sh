#!/usr/bin/env bash
# 轮询某 scale 的 pod-nohup done 标记并合并 jsonl
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

WORLD="${1:?usage: poll_nccl_scale_muxi.sh <world>}"
NPROC="${DEVICES_PER_NODE:-8}"
nnodes=$((WORLD / NPROC))
AFS_OUT="${AFS_OUT:?set AFS_OUT}"

status="$(cluster_pod_exec "${CLUSTER_POD}" "
ok=0; fail=0; miss=0
for r in \$(seq 0 $((nnodes - 1))); do
  if [[ -f $AFS_OUT/scale_${WORLD}.node_\$r.done ]]; then ok=\$((ok+1))
  elif [[ -f $AFS_OUT/scale_${WORLD}.node_\$r.fail ]]; then fail=\$((fail+1))
  else miss=\$((miss+1)); fi
done
echo OK=\$ok FAIL=\$fail MISS=\$miss
tail -2 $AFS_OUT/scale_${WORLD}.node_0.log 2>/dev/null || true
ps -eo cmd | grep -E '[t]orchrun|[n]ccl_torch' | head -3 || true
")"
echo "$status"

if echo "$status" | grep -q "OK=${nnodes} FAIL=0 MISS=0"; then
  cluster_pod_exec "${CLUSTER_POD}" "
shopt -s nullglob
parts=(\$(ls '$AFS_OUT'/scale_${WORLD}.rank*.jsonl 2>/dev/null | sort -V))
if [[ \${#parts[@]} -gt 0 ]]; then
  cat \"\${parts[@]}\" > '$AFS_OUT/scale_${WORLD}.jsonl'
  echo MERGED_\${#parts[@]}
  wc -l '$AFS_OUT/scale_${WORLD}.jsonl'
fi
"
  echo COMPLETE scale=$WORLD
  exit 0
fi
if echo "$status" | grep -qE 'FAIL=[1-9]'; then
  echo FAILED scale=$WORLD
  exit 2
fi
echo RUNNING scale=$WORLD
exit 1
