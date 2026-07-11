#!/usr/bin/env bash
# Muxi 拓扑探测（对标 probe_hccl_topology.sh；mx-smi topo）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
# 勿复用外层 AFS_OUT（常被 NCCL 跑残留污染）
AFS_OUT="${TOPO_AFS_OUT:-${AFS_RESULTS}/muxi-topo-${STAMP}}"
LOG_DIR="${TOPO_LOG_DIR:-/Users/yinjinrun/random-thing/logs/muxi-topo-${STAMP}}"
mkdir -p "$LOG_DIR"
echo "AFS_OUT=$AFS_OUT LOG_DIR=$LOG_DIR"

POD_NODES=("${CLUSTER_JOB}-master-0")
for i in $(seq 0 "$((CLUSTER_N_WORKERS - 1))"); do
  POD_NODES+=("${CLUSTER_JOB}-worker-${i}")
done
# 默认采全部 16 节点；可 TOPO_NODES=2 冒烟
N="${TOPO_NODES:-${#POD_NODES[@]}}"

cluster_pod_exec "${CLUSTER_POD}" "mkdir -p '$AFS_OUT'"

probe_one() {
  local pod="$1"
  local host_tag="${pod##*-}"
  # 写到 AFS，文件名用 pod 后缀
  local outf="$AFS_OUT/${pod}.raw.txt"
  local run="/tmp/probe_muxi_topo.sh"
  local body
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
  echo '=== mx-smi topo -t ==='
  mx-smi topo -t 2>&1
  echo '=== mx-smi topo -d ==='
  mx-smi topo -d 2>&1
  echo '=== mx-smi topo -m ==='
  mx-smi topo -m 2>&1
  echo '=== mx-smi topo -n ==='
  mx-smi topo -n 2>&1
  echo '=== mx-smi --show-metaxlink-bandwidth ==='
  mx-smi --show-metaxlink-bandwidth 2>&1
  echo '=== mx-smi --show-pcie-bandwidth ==='
  mx-smi --show-pcie-bandwidth 2>&1
  echo '=== net ifaces ==='
  cat /proc/net/dev 2>&1 | head -20
  echo '=== ibv_devinfo (head) ==='
  ibv_devinfo 2>&1 | head -80
} >"\$OUT" 2>&1
echo OK >'${outf}.done'
EOF
)
  printf '%s\n' "$body" | ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec -i ${pod} -- bash -c \"cat > $run && chmod +x $run && wc -c $run\"" \
    >"$LOG_DIR/${pod}.fire.log" 2>&1
  ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec ${pod} -- bash -c \"setsid nohup bash $run </dev/null >/dev/null 2>&1 & echo STARTED\"" \
    >>"$LOG_DIR/${pod}.fire.log" 2>&1
  echo "fired $pod -> $(tr '\n' ' ' < "$LOG_DIR/${pod}.fire.log")"
}

i=0
while [[ "$i" -lt "$N" ]]; do
  probe_one "${POD_NODES[$i]}"
  i=$((i + 1))
  sleep 1
done
echo "FIRE_TOPO_DONE n=$N → poll: ls $AFS_OUT/*.done"
