#!/usr/bin/env bash
# 跳板常驻：盯 MoE failslow，完成后 parse + 写 AFS 简报
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/root/.kube/config.huawei-a3-241ceshi}"
MOE_STAMP="${1:-}"
ARCHIVE_STAMP="${2:-20260713_offline}"
JOB="${JOB:-montyyin-moe96-r2}"
POD="${JOB}-master-0"
REMOTE_DIR="${REMOTE_DIR:-/root/montyyin-lab-remote}"
INTERVAL="${INTERVAL:-300}"
LOG="/tmp/remote_monitor_${MOE_STAMP}.log"
exec >>"$LOG" 2>&1

echo "MONITOR_START moe_stamp=$MOE_STAMP interval=${INTERVAL}s $(date -Iseconds)"

finalize_once() {
  local moe_root="/afs-a3-weight-share/yinjinrun.p-huawei/results/moe_failslow/${MOE_STAMP}"
  echo "FINALIZE $(date -Iseconds)"
  # Dense CSV 若尚未在 AFS 结果目录，在 pod 内 parse
  parse_root() {
    local root="$1" drop="$2"
    [[ -d "$root" ]] || return 0
    vcctl pod exec "$POD" -- bash -lc "
      cd /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster
      python3 parse_failslow_gap.py '$root' --drop-first $drop --csv '$root/gap_vs_n.csv' 2>&1 | tail -5
    " || true
  }
  parse_root "/afs-a3-weight-share/yinjinrun.p-huawei/results/dense_failslow/20260713_001230" 20
  parse_root "/afs-a3-weight-share/yinjinrun.p-huawei/results/dense_failslow_gbsprop/20260713_071316" 10
  parse_root "$moe_root" 8

  vcctl pod exec "$POD" -- bash -lc "
    python3 /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/remote_finalize_reports.py
  " || python3 "$REMOTE_DIR/remote_finalize_reports.py" || true

  echo "FINALIZE_DONE archive=/afs-a3-weight-share/yinjinrun.p-huawei/archive/${ARCHIVE_STAMP}"
}

while true; do
  if grep -q "JUMPHOST_MOE_FAILSLOW_DONE stamp=${MOE_STAMP}" "/tmp/moe-failslow-${MOE_STAMP}.log" 2>/dev/null; then
    finalize_once
    echo "MONITOR_EXIT success"
    exit 0
  fi
  if ! pgrep -f "jumphost_moe_failslow.sh" >/dev/null; then
    # orch 已死但未 DONE：仍尝试收口
    if [[ -f "/tmp/moe-failslow-${MOE_STAMP}.log" ]]; then
      finalize_once
      echo "MONITOR_EXIT orch_dead"
      exit 1
    fi
  fi
  vcctl pod exec "$POD" -- bash -lc "
    R=/afs-a3-weight-share/yinjinrun.p-huawei/results/moe_failslow/${MOE_STAMP}
    for s in 32 64; do
      d=\$R/scale_\$s
      [[ -d \$d ]] && echo scale_\$s=\$(wc -l < \$d/step_times_rank0.jsonl 2>/dev/null || echo 0)/40
    done
  " 2>/dev/null || true
  echo "MONITOR_TICK $(date +%H:%M:%S)"
  sleep "$INTERVAL"
done
