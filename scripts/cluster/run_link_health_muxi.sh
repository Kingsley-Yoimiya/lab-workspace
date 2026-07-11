#!/usr/bin/env bash
# Muxi 链路/设备健康（对标 run_link_health.sh；mx-smi）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
AFS_OUT="${LINK_AFS_OUT:-${AFS_RESULTS}/link-health-muxi-${STAMP}}"
LOG_DIR="${LINK_LOG_DIR:-/Users/yinjinrun/random-thing/logs/link-health-muxi-${STAMP}}"
mkdir -p "$LOG_DIR"
echo "AFS_OUT=$AFS_OUT LOG_DIR=$LOG_DIR"

POD_NODES=("${CLUSTER_JOB}-master-0")
for i in $(seq 0 "$((CLUSTER_N_WORKERS - 1))"); do
  POD_NODES+=("${CLUSTER_JOB}-worker-${i}")
done
N="${LINK_NODES:-${#POD_NODES[@]}}"

cluster_pod_exec "${CLUSTER_POD}" "mkdir -p '$AFS_OUT'"

i=0
while [[ "$i" -lt "$N" ]]; do
  pod="${POD_NODES[$i]}"
  outf="$AFS_OUT/${pod}.txt"
  run="/tmp/run_link_health_muxi.sh"
  body=$(cat <<EOF
#!/usr/bin/env bash
set +e
OUT='$outf'
{
  echo HOST=\$(hostname)
  echo POD='$pod'
  echo TS=\$(date '+%Y-%m-%dT%H:%M:%S')
  echo '=== mx-smi -L ==='
  mx-smi -L 2>&1
  echo '=== mx-smi --show-temperature ==='
  mx-smi --show-temperature 2>&1
  echo '=== mx-smi --show-memory ==='
  mx-smi --show-memory 2>&1
  echo '=== mx-smi --show-ecc-state ==='
  mx-smi --show-ecc-state 2>&1
  echo '=== mx-smi --show-board-power ==='
  mx-smi --show-board-power 2>&1
  echo '=== mx-smi --show-pcie ==='
  mx-smi --show-pcie 2>&1
  echo '=== mx-smi --show-metaxlink-bandwidth ==='
  mx-smi --show-metaxlink-bandwidth 2>&1
  echo '=== mx-smi topo -m ==='
  mx-smi topo -m 2>&1
  echo '=== ibv_devinfo (head) ==='
  ibv_devinfo 2>&1 | head -100
} >"\$OUT" 2>&1
echo OK >'$outf.done'
EOF
)
  printf '%s\n' "$body" | ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec -i ${pod} -- bash -c \"cat > $run && chmod +x $run && wc -c $run\"" \
    >"$LOG_DIR/${pod}.fire.log" 2>&1
  ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec ${pod} -- bash -c \"setsid nohup bash $run </dev/null >/dev/null 2>&1 & echo STARTED\"" \
    >>"$LOG_DIR/${pod}.fire.log" 2>&1
  echo "fired $pod"
  i=$((i + 1))
  sleep 1
done
echo "FIRE_LINK_HEALTH_DONE n=$N → $AFS_OUT"
