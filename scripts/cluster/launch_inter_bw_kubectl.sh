#!/usr/bin/env bash
# 无 AFS + 本机 kubectl：机内/机间 HCCL P2P 带宽探针
#
# 用法:
#   export KUBECONFIG=~/.kube/config-vc-a3-241ceshi
#   export CLUSTER_JOB=whj4stu-copy-copy-copy
#   ./scripts/cluster/launch_inter_bw_kubectl.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CLUSTER_JOB="${CLUSTER_JOB:?set CLUSTER_JOB}"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config-vc-a3-241ceshi}"
REMOTE_PY="${REMOTE_PY:-/root/miniconda3/envs/llm_test/bin/python3}"
WS="${REMOTE_WS:-/data/montyyin/lab-workspace}"
SCRIPTS_R="$WS/scripts/cluster"
RES_ROOT="${REMOTE_RESULTS:-/data/montyyin/results}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PIPE_LOG="${PIPE_LOG_DIR:-$OPS_ROOT/../../logs/inter-bw-${STAMP}}"
mkdir -p "$PIPE_LOG"
exec > >(tee -a "$PIPE_LOG/pipeline.log") 2>&1

MASTER_PORT="${MASTER_PORT:-29577}"
HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"
SIZES="${SIZES:-1M,16M,64M,256M}"
MODES="${MODES:-intra,inter}"
LOCAL_SAMPLES="${LOCAL_SAMPLES:-0,5,10,15}"
WARMUP="${WARMUP:-8}"
ITERS="${ITERS:-30}"
INFLIGHT="${INFLIGHT:-4}"
BIDIR_FLAG="${BIDIR:-0}"
PINGPONG_FLAG="${PINGPONG:-0}"
OUT_DIR="$RES_ROOT/inter-bw-${STAMP}"

PODS=(master-0 worker-0 worker-1 worker-2 worker-3 worker-4 worker-5 worker-6)
MASTER_ADDR="${MASTER_ADDR:-${CLUSTER_JOB}-master-0.${CLUSTER_JOB}.default.svc.cluster.local}"

EXTRA_ARG=""
if [[ "$PINGPONG_FLAG" == "1" ]]; then
  EXTRA_ARG="--pingpong"
elif [[ "$BIDIR_FLAG" == "1" ]]; then
  EXTRA_ARG="--bidir"
fi

echo "==> PIPE=$PIPE_LOG JOB=$CLUSTER_JOB OUT=$OUT_DIR"
echo "==> sizes=$SIZES modes=$MODES inflight=$INFLIGHT iters=$ITERS bidir=$BIDIR_FLAG pingpong=$PINGPONG_FLAG"

kexec() {
  local pod="$1"; shift
  kubectl exec -n default "${CLUSTER_JOB}-${pod}" -- /bin/bash --noprofile --norc -c "$*"
}

# --- upload ---
echo "==> upload probe"
PACK="$PIPE_LOG/pack"
rm -rf "$PACK"; mkdir -p "$PACK"
cp "$SCRIPT_DIR/hccl_inter_bw_probe.py" "$PACK/"
TAR="$PIPE_LOG/probe.tar.gz"
COPYFILE_DISABLE=1 tar -C "$PACK" -czf "$TAR" .
upload_one() {
  local p="$1"
  kubectl exec -i -n default "${CLUSTER_JOB}-${p}" -- /bin/bash --noprofile --norc -c \
    "mkdir -p '$SCRIPTS_R' '$RES_ROOT' && tar -xzf - -C '$SCRIPTS_R' && test -f '$SCRIPTS_R/hccl_inter_bw_probe.py' && echo OK_$p" \
    <"$TAR" >"$PIPE_LOG/upload_${p}.log" 2>&1
}
for p in "${PODS[@]}"; do upload_one "$p" & done
wait
grep -h OK_ "$PIPE_LOG"/upload_*.log || { echo "upload failed"; cat "$PIPE_LOG"/upload_*.log; exit 1; }

# --- durable start（变量在本机展开写入远端脚本）---
echo "==> start torchrun on 8 nodes"
for i in "${!PODS[@]}"; do
  p="${PODS[$i]}"
  # 生成本地 run 脚本再上传，避免嵌套 heredoc 引号地狱
  cat >"$PIPE_LOG/run_${p}.sh" <<EOF
#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
export PATH=/root/miniconda3/envs/llm_test/bin:\$PATH
export PYTHONUNBUFFERED=1
export HCCL_BUFFSIZE=$HCCL_BUFFSIZE
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
cd $WS
exec $REMOTE_PY -m torch.distributed.run \\
  --nnodes=8 --node_rank=$i --nproc_per_node=16 \\
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
  $SCRIPTS_R/hccl_inter_bw_probe.py \\
    --sizes $SIZES --modes $MODES --local-samples $LOCAL_SAMPLES \\
    --warmup $WARMUP --iters $ITERS --inflight $INFLIGHT \\
    $EXTRA_ARG \\
    --out $OUT_DIR/probe.jsonl
EOF
  kubectl cp "$PIPE_LOG/run_${p}.sh" "default/${CLUSTER_JOB}-${p}:/data/montyyin/run_inter_bw_${p}.sh"
  kexec "$p" "
set -e
mkdir -p '$OUT_DIR'
pids=\$(ps -eo pid,cmd | awk '/[h]ccl_inter_bw_probe.py/{print \$1}')
if [ -n \"\$pids\" ]; then kill -9 \$pids || true; sleep 1; fi
chmod +x /data/montyyin/run_inter_bw_${p}.sh
setsid nohup /data/montyyin/run_inter_bw_${p}.sh > '$OUT_DIR/${p}.run.log' 2>&1 < /dev/null &
echo STARTED_\$!
" >"$PIPE_LOG/start_${p}.log" 2>&1 &
done
wait

for p in "${PODS[@]}"; do
  echo -n "$p: "
  grep -E 'STARTED|Error|error|Traceback' "$PIPE_LOG/start_${p}.log" | tail -2 | tr '\n' ' '
  echo
done

echo "$OUT_DIR" >"$PIPE_LOG/out_dir.txt"
cat >"$PIPE_LOG/monitor.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
export KUBECONFIG="$KUBECONFIG"
LOG_DIR="$PIPE_LOG"
OUT_DIR="$OUT_DIR"
JOB="$CLUSTER_JOB"
PODS=(master-0 worker-0 worker-1 worker-2 worker-3 worker-4 worker-5 worker-6)
while true; do
  ts=\$(date -Iseconds)
  echo "==== \$ts ====" | tee -a "\$LOG_DIR/monitor.log"
  run=0
  for p in "\${PODS[@]}"; do
    info=\$(kubectl exec -n default \${JOB}-\$p -- /bin/bash --noprofile --norc -c \
      "py=\$(ps -eo etime,pcpu,cmd | awk '/[h]ccl_inter_bw_probe.py/ && !/awk/ {print \\\$1\\\" cpu\\\"\\\$2; exit}'); done=\$(grep -c INTER_BW_PROBE_DONE \$OUT_DIR/\${p}.run.log 2>/dev/null || echo 0); echo py=\${py:-DEAD} done=\$done" 2>/dev/null | tail -1)
    echo "\$p \$info" | tee -a "\$LOG_DIR/monitor.log"
    echo "\$info" | grep -q 'py=DEAD' || run=\$((run+1))
  done
  echo "running=\$run" | tee -a "\$LOG_DIR/monitor.log"
  if [[ \$run -eq 0 ]]; then
    echo ALL_DONE | tee -a "\$LOG_DIR/monitor.log"
    break
  fi
  sleep 60
done
EOS
chmod +x "$PIPE_LOG/monitor.sh"
nohup bash "$PIPE_LOG/monitor.sh" >"$PIPE_LOG/monitor.nohup" 2>&1 &
echo $! >"$PIPE_LOG/monitor.pid"

echo "INTER_BW_LAUNCHED → $PIPE_LOG"
echo "预计 15–40 min（串行边 × 多 size；看 monitor.log / master-0.run.log）"
