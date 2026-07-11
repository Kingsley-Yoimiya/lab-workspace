#!/usr/bin/env bash
# 无 AFS + 本机 kubectl：拓扑 → HCCL collective → P2P
# 避开 weibozhen 默认 kubeconfig 与沐曦互踩。
#
# 用法:
#   export KUBECONFIG=~/.kube/config-vc-a3-241ceshi
#   export CLUSTER_JOB=whj4stu-copy-copy-copy
#   ./scripts/cluster/launch_comm_kubectl.sh
#   SKIP_TOPO=1 HCCL_SCALES=16 ./scripts/cluster/launch_comm_kubectl.sh
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
PIPE_LOG="${PIPE_LOG_DIR:-$OPS_ROOT/../../logs/pipeline-comm-${STAMP}}"
mkdir -p "$PIPE_LOG"
exec > >(tee -a "$PIPE_LOG/pipeline.log") 2>&1

SKIP_TOPO="${SKIP_TOPO:-0}"
SKIP_HCCL="${SKIP_HCCL:-0}"
SKIP_P2P="${SKIP_P2P:-0}"
HCCL_SCALES="${HCCL_SCALES:-16,32,64,128}"
P2P_SCALES="${P2P_SCALES:-16,128}"
MASTER_ADDR="${MASTER_ADDR:-${CLUSTER_JOB}-master-0.${CLUSTER_JOB}.default.svc.cluster.local}"
MASTER_PORT="${MASTER_PORT:-29501}"
P2P_PORT="${P2P_PORT:-29601}"

PODS=(master-0 worker-0 worker-1 worker-2 worker-3 worker-4 worker-5 worker-6)

echo "==> PIPE=$PIPE_LOG JOB=$CLUSTER_JOB MASTER_ADDR=$MASTER_ADDR"

# 不用 login shell：镜像 .bashrc 会 source 缺失的 setenv.bash，在 set -e 下易把流水线打死
kexec() {
  local pod="$1"; shift
  kubectl exec -n default "${CLUSTER_JOB}-${pod}" -- /bin/bash --noprofile --norc -c "$*"
}

kexec_i() {
  local pod="$1"; shift
  kubectl exec -i -n default "${CLUSTER_JOB}-${pod}" -- /bin/bash --noprofile --norc -c "$*"
}

remote_env() {
  cat <<'EOF'
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
export PATH=/root/miniconda3/envs/llm_test/bin:$PATH
export PYTHONUNBUFFERED=1
EOF
}

# --- upload benches to all pods (tar 一次，并行) ---
echo "==> upload benches"
PACK="$PIPE_LOG/bench_pack"
rm -rf "$PACK"; mkdir -p "$PACK"
cp "$SCRIPT_DIR/hccl_torch_bench.py" "$SCRIPT_DIR/hccl_p2p_bench.py" "$PACK/"
TAR="$PIPE_LOG/benches.tar.gz"
COPYFILE_DISABLE=1 tar -C "$PACK" -czf "$TAR" .
upload_one() {
  local p="$1"
  kubectl exec -i -n default "${CLUSTER_JOB}-${p}" -- /bin/bash --noprofile --norc -c \
    "mkdir -p '$SCRIPTS_R' '$RES_ROOT' && tar -xzf - -C '$SCRIPTS_R' && test -f '$SCRIPTS_R/hccl_torch_bench.py' && echo OK_$p" \
    <"$TAR" >"$PIPE_LOG/upload_${p}.log" 2>&1
}
for p in "${PODS[@]}"; do upload_one "$p" & done
wait
grep -h OK_ "$PIPE_LOG"/upload_*.log || { echo "upload failed"; cat "$PIPE_LOG"/upload_*.log; exit 1; }

# --- TOPO ---
if [[ "$SKIP_TOPO" != "1" ]]; then
  TOPO_OUT="$RES_ROOT/hccl-topo-${STAMP}"
  TOPO_LOG="$PIPE_LOG/hccl-topo"
  mkdir -p "$TOPO_LOG/raw"
  echo "==> STEP topo → $TOPO_OUT"
  for p in "${PODS[@]}"; do
    kexec "$p" "
$(remote_env)
mkdir -p '$TOPO_OUT'
OUT='$TOPO_OUT/${p}.raw.txt'
{
  echo HOST=\$(hostname 2>/dev/null || echo $p)
  echo TS=\$(date -Iseconds 2>/dev/null || date)
  which npu-smi; npu-smi info -l 2>&1 | head -40
  echo '=== topo ==='; npu-smi info -t topo 2>&1 | head -80
  echo '=== hccn.conf ==='; cat /etc/hccn.conf 2>&1 | head -80
  HCCN=\$(find /usr/local/Ascend /usr/local/bin -name hccn_tool 2>/dev/null | head -1)
  echo HCCN_TOOL=\$HCCN
  if [ -n \"\$HCCN\" ]; then \"\$HCCN\" -i 0 -link -g 2>&1 | head -40; fi
  echo '=== env HCCL ==='; env | grep -iE 'HCCL|RANK|MASTER' | sort
} > \"\$OUT\" 2>&1
echo TOPO_OK_$p
wc -c \"\$OUT\"
" >"$TOPO_LOG/${p}.log" 2>&1 &
  done
  wait
  for p in "${PODS[@]}"; do
    kubectl exec -n default "${CLUSTER_JOB}-${p}" -- /bin/bash -c \
      "tar -C $TOPO_OUT -cf - ${p}.raw.txt" >"$TOPO_LOG/raw/${p}.tar" 2>/dev/null || true
    tar -xf "$TOPO_LOG/raw/${p}.tar" -C "$TOPO_LOG/raw" 2>/dev/null || true
  done
  echo "OK topo → $TOPO_LOG/raw"
else
  echo "==> skip topo"
fi

pod_for_rank() {
  local r="$1"
  if [[ "$r" -eq 0 ]]; then echo "master-0"; else echo "worker-$((r-1))"; fi
}

run_torch_scale() {
  local kind="$1" # hccl|p2p
  local world_npu="$2"
  local port="$3"
  local bench="$4"
  local out_root="$5"
  local extra_args="$6"
  local nnodes=$((world_npu / 16))
  local out="$out_root/scale_${world_npu}.jsonl"
  echo "==> $kind scale=$world_npu nnodes=$nnodes port=$port"
  local r=0
  local pids=()
  while [[ "$r" -lt "$nnodes" ]]; do
    local p
    p="$(pod_for_rank "$r")"
    local logf="$PIPE_LOG/${kind}_scale${world_npu}_rank${r}.log"
    kexec "$p" "
$(remote_env)
mkdir -p '$out_root'
cd /tmp
cp -f '$SCRIPTS_R/$bench' /tmp/$bench
torchrun \
  --nnodes=$nnodes \
  --node_rank=$r \
  --nproc_per_node=16 \
  --master_addr=$MASTER_ADDR \
  --master_port=$port \
  /tmp/$bench \
  $extra_args \
  --out '$out_root/scale_${world_npu}.rank${r}.jsonl'
echo ${kind}_SCALE_${world_npu}_RANK_${r}_OK
" >"$logf" 2>&1 &
    pids+=("$!")
    r=$((r + 1))
  done
  local fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  # merge on master
  kexec "master-0" "
set -e
shopt -s nullglob
parts=(\$(ls '$out_root'/scale_${world_npu}.rank*.jsonl 2>/dev/null | sort -V))
if [[ \${#parts[@]} -gt 0 ]]; then cat \"\${parts[@]}\" > '$out'; echo MERGED_\${#parts[@]}; else echo WARN_NO_PARTS; fi
ls -la '$out_root' | head
" || true
  if [[ "$fail" -ne 0 ]]; then
    echo "WARN $kind scale=$world_npu failed (see logs)"
    return 1
  fi
  echo "OK $kind scale=$world_npu"
}

# --- HCCL collective ---
if [[ "$SKIP_HCCL" != "1" ]]; then
  HCCL_OUT="$RES_ROOT/hccl-${STAMP}"
  echo "==> STEP hccl → $HCCL_OUT"
  kexec "master-0" "mkdir -p '$HCCL_OUT'"
  port="$MASTER_PORT"
  IFS=',' read -ra ARR <<< "$HCCL_SCALES"
  for s in "${ARR[@]}"; do
    run_torch_scale hccl "$s" "$port" hccl_torch_bench.py "$HCCL_OUT" \
      "--ops all_reduce,all_gather,reduce_scatter,broadcast --sizes 1M,16M,64M,256M" || true
    port=$((port + 1))
  done
  mkdir -p "$PIPE_LOG/hccl-results"
  kubectl exec -n default "${CLUSTER_JOB}-master-0" -- /bin/bash -c \
    "tar -C $HCCL_OUT -cf - ." >"$PIPE_LOG/hccl-results.tar" 2>/dev/null || true
  tar -xf "$PIPE_LOG/hccl-results.tar" -C "$PIPE_LOG/hccl-results" 2>/dev/null || true
else
  echo "==> skip hccl"
fi

# --- P2P ---
if [[ "$SKIP_P2P" != "1" ]]; then
  P2P_OUT="$RES_ROOT/hccl-p2p-${STAMP}"
  echo "==> STEP p2p → $P2P_OUT"
  kexec "master-0" "mkdir -p '$P2P_OUT'"
  port="$P2P_PORT"
  IFS=',' read -ra ARR <<< "$P2P_SCALES"
  for s in "${ARR[@]}"; do
    # >=64 默认 ring-only（bench 内部也会裁）
    run_torch_scale p2p "$s" "$port" hccl_p2p_bench.py "$P2P_OUT" \
      "--sizes 64K,16M --warmup 5 --iters 20" || true
    port=$((port + 1))
  done
  mkdir -p "$PIPE_LOG/p2p-results"
  kubectl exec -n default "${CLUSTER_JOB}-master-0" -- /bin/bash -c \
    "tar -C $P2P_OUT -cf - ." >"$PIPE_LOG/p2p-results.tar" 2>/dev/null || true
  tar -xf "$PIPE_LOG/p2p-results.tar" -C "$PIPE_LOG/p2p-results" 2>/dev/null || true
else
  echo "==> skip p2p"
fi

echo "PIPELINE_COMM_DONE → $PIPE_LOG"
