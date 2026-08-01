#!/usr/bin/env bash
# fire_bimodal_probe.sh
# 双峰检测 launcher —— 扫 5 个消息大小 × 高 iter，验证 HCCL 通信是否有快慢步双峰
#
# 用法（在跳板机跑）：
#   NNODES=32 TAG=bimodal-512-primitive bash fire_bimodal_probe.sh
#
# 环境变量（可选覆盖）：
#   NNODES        节点数（16=256卡，32=512卡，64=1024卡；单节点 = 1）
#   TAG           输出目录标签（不用带时间戳）
#   NPROC         每节点 NPU 数（默认 16）
#   MASTER_POD    主 pod（默认读 /tmp/hccl_cluster.env 或 yjr-1024-0731-master-0）
#   WORKER_PREFIX worker 前缀（同上）
#   SIZES         消息大小列表（默认覆盖梯度典型大小）
#   ITERS         iter 数（默认 100，双峰检测建议 ≥80）
#   OPS           算子（默认 all_reduce,barrier）
#   HCCL_BUFFSIZE 每 rank buffer（默认 1024，千卡上必须小）
#   TIMEOUT_SEC   单节点 timeout（默认 600）
#   BENCH_SRC     bench.py 路径
#   BENCH_ARGS    若已设置则优先生效（v3 loop）；自动补 --out

set -euo pipefail

# 可变 Job/落盘：跳板 sync 后 source；本机可忽略
if [[ -f /tmp/hccl_cluster.env ]]; then
  # shellcheck disable=SC1091
  source /tmp/hccl_cluster.env
fi

K=${K:-/root/.cache/volcano/kubectl/kubectl}
export KUBECONFIG=${KUBECONFIG:-/tmp/config-vc-a3-241ceshi-songyiyang.yaml}

AFS_ROOT=${AFS_ROOT:-/afs-a3-weight-share/yinjinrun.p-huawei}
BENCH_SRC=${BENCH_SRC:-$AFS_ROOT/lab-workspace/scripts/cluster/hccl_torch_bench.py}
RESULTS_ROOT=${RESULTS_ROOT:-$AFS_ROOT/results}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
TAG=${TAG:?need TAG (e.g. bimodal-512-primitive)}
NNODES=${NNODES:?need NNODES}
MASTER_PORT=${MASTER_PORT:-$((29800 + RANDOM % 200))}
HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}
TIMEOUT_SEC=${TIMEOUT_SEC:-600}
NPROC=${NPROC:-16}
OUT_DIR=${OUT_DIR:-$RESULTS_ROOT/$TAG-$STAMP}
LOG_DIR=${LOG_DIR:-/tmp/hccl-fire-$TAG-$STAMP}
FANOUT=${FANOUT:-$NNODES}

MASTER_POD=${MASTER_POD:-yjr-1024-0731-master-0}
WORKER_PREFIX=${WORKER_PREFIX:-yjr-1024-0731-worker-}

# ★ 双峰探测参数
SIZES=${SIZES:-"4K,64K,1M,64M,256M"}   # 4B 会被 padding；起点 4KB
ITERS=${ITERS:-100}
OPS=${OPS:-"all_reduce,barrier,all_gather,reduce_scatter"}
DTYPE=${DTYPE:-fp16}

# 外部 BENCH_ARGS（fire_hccl_v3_loop）优先；否则用 OPS/SIZES 拼
if [[ -z "${BENCH_ARGS:-}" ]]; then
  BENCH_ARGS="--ops $OPS --sizes $SIZES --iters $ITERS --warmup 5 --dtype $DTYPE --out $OUT_DIR/case.jsonl"
elif [[ "$BENCH_ARGS" != *"--out"* ]]; then
  BENCH_ARGS="$BENCH_ARGS --out $OUT_DIR/case.jsonl"
fi

MASTER_IP=$($K get pod "$MASTER_POD" -o jsonpath='{.status.podIP}')
[[ -n "$MASTER_IP" ]] || { echo "FATAL: no MASTER_IP for $MASTER_POD"; exit 1; }

mkdir -p "$LOG_DIR"
echo "TAG=$TAG STAMP=$STAMP NNODES=$NNODES MASTER=$MASTER_POD IP=$MASTER_IP PORT=$MASTER_PORT"
echo "OPS=$OPS SIZES=$SIZES ITERS=$ITERS HCCL_BUFFSIZE=$HCCL_BUFFSIZE"
echo "OUT=$OUT_DIR"

# Prepare OUT on master AFS
$K exec "$MASTER_POD" -- bash -lc "mkdir -p '$OUT_DIR' && cp -f '$BENCH_SRC' /tmp/hccl_torch_bench.py && wc -l /tmp/hccl_torch_bench.py"

run_node() {
  local rank="$1" pod="$2"
  local logfile="$LOG_DIR/rank${rank}.log"
  $K exec "$pod" -- bash -lc "
set -e
cp -f '$BENCH_SRC' /tmp/hccl_torch_bench.py
export HCCL_BUFFSIZE=$HCCL_BUFFSIZE PROBING=0 PYTHONUNBUFFERED=1
cd /tmp
mkdir -p '$OUT_DIR'
timeout $TIMEOUT_SEC torchrun \
  --nnodes=$NNODES --node_rank=$rank --nproc_per_node=$NPROC \
  --master_addr=$MASTER_IP --master_port=$MASTER_PORT \
  /tmp/hccl_torch_bench.py $BENCH_ARGS
echo NODE_DONE rank=$rank exit=\$?
" >"$logfile" 2>&1
  echo $? >"$LOG_DIR/rank${rank}.ec"
}

# Build pod list
declare -a PODS=()
PODS[0]=$MASTER_POD
for ((r=1; r<NNODES; r++)); do
  PODS[$r]="${WORKER_PREFIX}$((r-1))"
done

# Workers first (background)
pids=()
for ((r=1; r<NNODES; r++)); do
  while (( $(jobs -rp | wc -l) >= FANOUT )); do sleep 0.2; done
  run_node "$r" "${PODS[$r]}" &
  pids+=($!)
done
sleep 5
# Master
run_node 0 "${PODS[0]}" &
pids+=($!)

echo "launched ${#pids[@]} nodes; waiting..."
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

PERITER=$($K exec "$MASTER_POD" -- bash -lc "ls '$OUT_DIR'/*.periter.jsonl 2>/dev/null | wc -l" || echo 0)
ERR=$($K exec "$MASTER_POD" -- bash -lc "grep -h hccl_bench_err '$OUT_DIR'/*.periter.jsonl 2>/dev/null | wc -l" || echo 0)
echo "==============================================="
echo "DONE TAG=$TAG-$STAMP"
echo "  periter=$PERITER / expected $((NNODES * NPROC))"
echo "  err=$ERR"
echo "  fail_wait=$fail"
echo "  OUT=$OUT_DIR"
echo "  LOG=$LOG_DIR"
echo "==============================================="
exit $fail
