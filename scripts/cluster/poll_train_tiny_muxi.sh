#!/usr/bin/env bash
# 轮询 G9 tiny train done/fail
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

AFS_OUT="${AFS_OUT:?set AFS_OUT}"

status="$(cluster_pod_exec "${CLUSTER_POD}" "
if [[ -f $AFS_OUT/train.done ]]; then echo DONE
elif [[ -f $AFS_OUT/train.fail ]]; then echo FAIL
else echo RUNNING; fi
# 进度线索
tail -8 $AFS_OUT/train.log 2>/dev/null || tail -8 $AFS_OUT/train.outer.log 2>/dev/null || true
")"
echo "$status"

if echo "$status" | grep -q '^DONE'; then
  # 抽 iteration / throughput 行
  cluster_pod_exec "${CLUSTER_POD}" "
grep -E 'iteration|throughput|TFLOP|tokens per|elapsed time per' $AFS_OUT/train.log | tail -30
echo ---
wc -l $AFS_OUT/train.log
"
  echo COMPLETE train-tiny AFS_OUT=$AFS_OUT
  exit 0
fi
if echo "$status" | grep -q '^FAIL'; then
  cluster_pod_exec "${CLUSTER_POD}" "
tail -60 $AFS_OUT/train.log 2>/dev/null || tail -60 $AFS_OUT/train.outer.log
"
  echo FAILED train-tiny AFS_OUT=$AFS_OUT
  exit 2
fi
echo RUNNING train-tiny AFS_OUT=$AFS_OUT
exit 1
