#!/usr/bin/env bash
# 在各目标 pod 的 /tmp 写 launcher 并 nohup（避开 AFS 跨节点延迟）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

WORLD="${1:?usage: fire_nccl_scale_muxi.sh <world> [master_port]}"
MASTER_PORT="${2:-${MASTER_PORT:-29631}}"
NPROC="${DEVICES_PER_NODE:-8}"
nnodes=$((WORLD / NPROC))
AFS_OUT="${AFS_OUT:?set AFS_OUT}"
AFS_SCRIPTS="/afs-a3-weight-share/yinjinrun.p/lab-workspace/scripts/cluster"
SIZES="${SIZES:-1M,16M,64M,256M}"
OPS="${OPS:-all_reduce,all_gather,reduce_scatter,broadcast}"
MASTER_ADDR="${MASTER_ADDR:-${CLUSTER_JOB}-master-0.${CLUSTER_JOB}}"
LOG_DIR="${LOG_DIR:-/Users/yinjinrun/random-thing/logs/muxi-nccl-fire}"
mkdir -p "$LOG_DIR"

POD_NODES=("${CLUSTER_JOB}-master-0")
for i in $(seq 0 "$((CLUSTER_N_WORKERS - 1))"); do
  POD_NODES+=("${CLUSTER_JOB}-worker-${i}")
done

if [[ $((WORLD % NPROC)) -ne 0 ]] || [[ "$nnodes" -gt "${#POD_NODES[@]}" ]]; then
  echo "bad world=$WORLD"; exit 1
fi

out="$AFS_OUT/scale_${WORLD}.jsonl"
echo "FIRE scale=$WORLD nnodes=$nnodes port=$MASTER_PORT out=$out"

cluster_pod_exec "${CLUSTER_POD}" "mkdir -p '$AFS_SCRIPTS' '$AFS_OUT'"
cluster_pod_exec_i "${CLUSTER_POD}" "cat > $AFS_SCRIPTS/nccl_torch_bench.py" \
  < "$SCRIPT_DIR/nccl_torch_bench.py"

r=0
while [[ "$r" -lt "$nnodes" ]]; do
  pod="${POD_NODES[$r]}"
  donef="$AFS_OUT/scale_${WORLD}.node_${r}.done"
  failf="$AFS_OUT/scale_${WORLD}.node_${r}.fail"
  rlog="$AFS_OUT/scale_${WORLD}.node_${r}.log"
  run_local="/tmp/run_nccl_scale_${WORLD}_node_${r}.sh"

  run_body=$(cat <<EOF
#!/usr/bin/env bash
export PATH="/opt/conda/bin:\${PATH:-/usr/bin}"
export PYTHONUNBUFFERED=1
# 沐曦多机：强制 eth0 做 NCCL/MCCL socket；IB/RoCE 用 xscale（verbs 可见；mlx5 无 ibv）
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export MCCL_SOCKET_IFNAME=${MCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
# 线上 muxi-128node：xscale_0..3 + GID=5 + VSWITCH（历史 GID=4 跨机失败）
export NCCL_IB_HCA=${NCCL_IB_HCA:-xscale_0,xscale_1,xscale_2,xscale_3}
export MCCL_IB_HCA=${MCCL_IB_HCA:-xscale_0,xscale_1,xscale_2,xscale_3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export MCCL_DEBUG=${MCCL_DEBUG:-WARN}
export FORCE_ACTIVE_WAIT=${FORCE_ACTIVE_WAIT:-2}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-5}
export MCCL_IB_GID_INDEX=${MCCL_IB_GID_INDEX:-5}
export MCCL_IB_TC=${MCCL_IB_TC:-128}
export MCCL_ENABLE_VSWITCH=${MCCL_ENABLE_VSWITCH:-1}
export MCCL_PCIE_BUFFER_MODE=${MCCL_PCIE_BUFFER_MODE:-0}
# 将本机调用时的可选 env 固化进脚本（setsid 不会继承操作机 export）
$( [[ -n "${NCCL_IB_DISABLE:-}" ]] && echo "export NCCL_IB_DISABLE=${NCCL_IB_DISABLE}" )
$( [[ -n "${MCCL_IB_DISABLE:-}" ]] && echo "export MCCL_IB_DISABLE=${MCCL_IB_DISABLE}" )
$( [[ -n "${NCCL_NET:-}" ]] && echo "export NCCL_NET=${NCCL_NET}" )
$( [[ -n "${MCCL_NET:-}" ]] && echo "export MCCL_NET=${MCCL_NET}" )
rm -f '$donef' '$failf'
cp -f '$AFS_SCRIPTS/nccl_torch_bench.py' /tmp/nccl_torch_bench.py
/opt/conda/bin/torchrun --nnodes=$nnodes --node_rank=$r --nproc_per_node=$NPROC \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  /tmp/nccl_torch_bench.py --ops '$OPS' --sizes '$SIZES' --out '$out' \
  >'$rlog' 2>&1
ec=\$?
if [[ \$ec -eq 0 ]]; then echo OK >'$donef'; else echo FAIL >'$failf'; fi
exit \$ec
EOF
)

  # 分两步：先 stdin 写 /tmp 脚本，再 setsid 启动（避免复合命令吃掉 stdin）
  printf '%s\n' "$run_body" | cluster_pod_exec_i "$pod" \
    "cat > $run_local && chmod +x $run_local && wc -c $run_local" \
    >"$LOG_DIR/scale${WORLD}_noderank${r}.fire.log" 2>&1
  cluster_pod_exec "$pod" \
    "setsid nohup bash $run_local </dev/null >/dev/null 2>&1 & echo STARTED \$!; sleep 2; pgrep -af torchrun | head -3" \
    >>"$LOG_DIR/scale${WORLD}_noderank${r}.fire.log" 2>&1
  echo "  fired node_rank=$r pod=$pod -> $(tr '\n' ' ' < "$LOG_DIR/scale${WORLD}_noderank${r}.fire.log")"
  r=$((r + 1))
  sleep 2
done
echo "FIRE_DONE scale=$WORLD"
