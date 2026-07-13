#!/usr/bin/env bash
# 每节点 npu-smi + hccn_tool 链路健康汇总
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
AFS_OUT="${AFS_RESULTS:-/afs-a3-weight-share/yinjinrun.p-huawei/results}/link-health-${STAMP}"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/link-health-${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/link.log") 2>&1

PODS=(
  "${CLUSTER_JOB}-master-0"
  "${CLUSTER_JOB}-worker-0"
  "${CLUSTER_JOB}-worker-1"
  "${CLUSTER_JOB}-worker-2"
  "${CLUSTER_JOB}-worker-3"
  "${CLUSTER_JOB}-worker-4"
  "${CLUSTER_JOB}-worker-5"
  "${CLUSTER_JOB}-worker-6"
)

cluster_pod_exec "${PODS[0]}" "mkdir -p '$AFS_OUT'"

for pod in "${PODS[@]}"; do
  echo "==> $pod"
  ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "vcctl pod exec ${pod} -- bash -lc $(printf '%q' "
set -euo pipefail
OUT='$AFS_OUT/${pod}.txt'
{
  echo HOST=\$HOSTNAME
  echo '=== npu-smi info ==='
  npu-smi info || true
  echo '=== npu-smi info -t health ==='
  for i in \$(seq 0 15); do
    echo \"-- device \$i --\"
    npu-smi info -t health -i \$i 2>/dev/null || true
  done
  HCCN=\$(find /usr/local/Ascend /usr/local -name hccn_tool 2>/dev/null | head -1 || true)
  echo HCCN_TOOL=\$HCCN
  if [[ -n \"\$HCCN\" ]]; then
    for i in \$(seq 0 15); do
      echo \"=== hccn device \$i ===\"
      \"\$HCCN\" -i \$i -link -g 2>/dev/null || true
      \"\$HCCN\" -i \$i -speed -g 2>/dev/null || true
      \"\$HCCN\" -i \$i -stat -g 2>/dev/null || true
    done
  else
    echo 'hccn_tool not found'
  fi
} > \"\$OUT\" 2>&1
ls -la \"\$OUT\"
echo LINK_DONE_${pod}
")" | tee "$LOG_DIR/${pod}.log"
done

echo "==> pull"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "vcctl pod exec -i ${PODS[0]} -- bash -c 'tar -C $AFS_OUT -cf - .' " \
  > "$LOG_DIR/results.tar"
mkdir -p "$LOG_DIR/results"
tar -xf "$LOG_DIR/results.tar" -C "$LOG_DIR/results"
ls -la "$LOG_DIR/results"
echo "LINK_HEALTH_OK → $AFS_OUT / $LOG_DIR"
