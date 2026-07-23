#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"
source "$SCRIPT_DIR/parallel_retry.sh"

NNODES="${NNODES:?set NNODES}"
PREFLIGHT_PARALLELISM="${PREFLIGHT_PARALLELISM:-$NNODES}"
PREFLIGHT_RETRIES="${PREFLIGHT_RETRIES:-3}"
PREFLIGHT_LOG_DIR="${PREFLIGHT_LOG_DIR:?set PREFLIGHT_LOG_DIR}"

POD_NODES=("${CLUSTER_JOB}-master-0")
for i in $(seq 0 "$((CLUSTER_N_WORKERS - 1))"); do
  POD_NODES+=("${CLUSTER_JOB}-worker-${i}")
done
[[ "$NNODES" -le "${#POD_NODES[@]}" ]]

preflight_one() {
  local rank="$1"
  local attempt="$2"
  local log="$3"
  local pod="${POD_NODES[$rank]}"
  local state ec=0
  {
    echo "POD=$pod NODE_RANK=$rank ATTEMPT=$attempt STARTED_AT=$(date -Iseconds)"
    _cluster_kubectl_env
    state="$(kubectl get pod "$pod" \
      -o jsonpath='{.status.phase}/{.status.containerStatuses[0].ready}/{.status.containerStatuses[0].restartCount}')" ||
      ec=$?
    echo "STATE=$state GET_RC=$ec"
    if [[ "$ec" -eq 0 && "$state" != "Running/true/0" ]]; then
      echo "SEMANTIC_STATE_FAILURE expected=Running/true/0 actual=$state"
      ec=41
    fi
    if [[ "$ec" -eq 0 ]]; then
      cluster_pod_exec "$pod" \
        "set -e; \
         ps -eo args | awk 'BEGIN{IGNORECASE=1} /torchrun|nccl_torch_bench|all_reduce_perf|constitution|burn[-_]?in/ && \$0 !~ /awk/ {found=1; print} END{exit found?1:0}'; \
         gpu=\$(mx-smi 2>/dev/null); printf '%s\\n' \"\$gpu\" | grep -q 'no process found'; \
         for h in xscale_0 xscale_1 xscale_2 xscale_3; do test -d /sys/class/infiniband/\$h; done; \
         test -n \"\$(cat /sys/class/infiniband/xscale_0/ports/1/gids/5)\"; \
         echo GPU0_IDLE_ENV_OK" ||
        ec=$?
    fi
    echo "ENDED_AT=$(date -Iseconds) EXIT_CODE=$ec"
  } >"$log" 2>&1
  return "$ec"
}

parallel_retry_run \
  preflight "$NNODES" "$PREFLIGHT_PARALLELISM" "$PREFLIGHT_RETRIES" \
  "$PREFLIGHT_LOG_DIR" preflight_one
