#!/usr/bin/env bash
# 在各目标 pod 的 /tmp 写 launcher 并 nohup（避开 AFS 跨节点延迟）
#
# 多节点默认有界并行发射（CLUSTER_FANOUT_PARALLEL，muxi.env 默认 16）。
# 禁止改回 for+sleep 串行；跳板 SSH 过载时只降并发，例如：
#   CLUSTER_FANOUT_PARALLEL=8 ./fire_nccl_scale_muxi.sh 64 29801
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"
source "$SCRIPT_DIR/parallel_retry.sh"

WORLD="${1:?usage: fire_nccl_scale_muxi.sh <world> [master_port]}"
MASTER_PORT="${2:-${MASTER_PORT:-29631}}"
NPROC="${NPROC_OVERRIDE:-${DEVICES_PER_NODE:-8}}"
nnodes=$((WORLD / NPROC))
AFS_OUT="${AFS_OUT:?set AFS_OUT}"
AFS_SCRIPTS="${AFS_SCRIPTS:-${AFS_OUT}/code}"
SIZES="${SIZES:-1M,16M,64M,256M}"
OPS="${OPS:-all_reduce,all_gather,reduce_scatter,broadcast}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
MASTER_ADDR="${MASTER_ADDR:-${CLUSTER_JOB}-master-0.${CLUSTER_JOB}}"
LOG_DIR="${LOG_DIR:-/Users/yinjinrun/random-thing/logs/muxi-nccl-fire}"
PARALLEL="${FIRE_PARALLELISM:-${CLUSTER_FANOUT_PARALLEL:-16}}"
DRY_RUN="${FIRE_DRY_RUN:-0}"
REQUIRE_FULL_PARALLEL="${FIRE_REQUIRE_FULL_PARALLEL:-0}"
UPLOAD_ONLY="${FIRE_UPLOAD_ONLY:-0}"
PREPARE_RETRIES="${FIRE_PREPARE_RETRIES:-3}"
NODE_RETRIES="${FIRE_NODE_RETRIES:-3}"
mkdir -p "$LOG_DIR"

POD_NODES=("${CLUSTER_JOB}-master-0")
for i in $(seq 0 "$((CLUSTER_N_WORKERS - 1))"); do
  POD_NODES+=("${CLUSTER_JOB}-worker-${i}")
done

if [[ $((WORLD % NPROC)) -ne 0 ]] || [[ "$nnodes" -gt "${#POD_NODES[@]}" ]]; then
  echo "bad world=$WORLD"; exit 1
fi
if ! [[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "bad FIRE_PARALLELISM=$PARALLEL"; exit 1
fi
if ! [[ "$PREPARE_RETRIES" =~ ^[1-9][0-9]*$ ]]; then
  echo "bad FIRE_PREPARE_RETRIES=$PREPARE_RETRIES"; exit 1
fi
if ! [[ "$NODE_RETRIES" =~ ^[1-9][0-9]*$ ]]; then
  echo "bad FIRE_NODE_RETRIES=$NODE_RETRIES"; exit 1
fi
if [[ "$REQUIRE_FULL_PARALLEL" == "1" && "$PARALLEL" -lt "$nnodes" ]]; then
  echo "full parallel required: parallel=$PARALLEL nnodes=$nnodes"; exit 1
fi

out="$AFS_OUT/scale_${WORLD}.jsonl"
RUN_TOKEN="${RUN_TOKEN_OVERRIDE:-$(basename "$AFS_OUT")}"
echo "FIRE scale=$WORLD nnodes=$nnodes nproc=$NPROC port=$MASTER_PORT warmup=$WARMUP iters=$ITERS parallel=$PARALLEL out=$out"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN stages=write,start nnodes=$nnodes parallel=$PARALLEL"
  for r in $(seq 0 "$((nnodes - 1))"); do
    echo "PLAN node_rank=$r pod=${POD_NODES[$r]} launcher=/tmp/run_nccl_scale_${WORLD}_node_${r}.sh"
  done
  exit 0
fi

prepare_exec() {
  local name="$1"
  local cmd="$2"
  local attempt ec log
  for ((attempt = 1; attempt <= PREPARE_RETRIES; attempt++)); do
    log="$LOG_DIR/prepare_${name}.attempt${attempt}.log"
    {
      echo "NAME=$name ATTEMPT=$attempt STARTED_AT=$(date -Iseconds)"
      printf 'COMMAND=%q\n' "$cmd"
    } >"$log"
    ec=0
    cluster_pod_exec "${CLUSTER_POD}" "$cmd" >>"$log" 2>&1 || ec=$?
    echo "ENDED_AT=$(date -Iseconds) EXIT_CODE=$ec" >>"$log"
    if [[ "$ec" -eq 0 ]]; then
      return 0
    fi
    if ! grep -Eq '(^|[^A-Za-z])EOF([^A-Za-z]|$)' "$log" || [[ "$attempt" -eq "$PREPARE_RETRIES" ]]; then
      cat "$log" >&2
      return "$ec"
    fi
    echo "RETRY_REASON=transient_EOF BACKOFF_SECONDS=$attempt" >>"$log"
    sleep "$attempt"
  done
}

prepare_upload() {
  local name="$1"
  local src="$2"
  local remote="$3"
  local expected_size expected_sha attempt ec log verify actual_size actual_sha
  expected_size="$(wc -c <"$src" | tr -d ' ')"
  expected_sha="$(sha256sum "$src" | awk '{print $1}')"
  for ((attempt = 1; attempt <= PREPARE_RETRIES; attempt++)); do
    log="$LOG_DIR/prepare_${name}.attempt${attempt}.log"
    {
      echo "NAME=$name ATTEMPT=$attempt STARTED_AT=$(date -Iseconds)"
      echo "SOURCE=$src REMOTE=$remote EXPECTED_SIZE=$expected_size EXPECTED_SHA256=$expected_sha"
    } >"$log"
    ec=0
    cluster_pod_exec_i "${CLUSTER_POD}" \
      "tmp='${remote}.upload.tmp'; cat > \"\$tmp\" && mv -f \"\$tmp\" '$remote'" \
      <"$src" >>"$log" 2>&1 || ec=$?
    echo "UPLOAD_EXIT_CODE=$ec" >>"$log"
    if [[ "$ec" -eq 0 ]]; then
      verify="$(cluster_pod_exec "${CLUSTER_POD}" \
        "printf 'SIZE='; wc -c < '$remote'; printf 'SHA256='; sha256sum '$remote' | awk '{print \$1}'" \
        2>>"$log")" || ec=$?
      printf '%s\n' "$verify" >>"$log"
      actual_size="$(awk -F= '/^SIZE=/{print $2}' <<<"$verify")"
      actual_sha="$(awk -F= '/^SHA256=/{print $2}' <<<"$verify")"
      if [[ "$ec" -eq 0 && "$actual_size" == "$expected_size" && "$actual_sha" == "$expected_sha" ]]; then
        echo "ENDED_AT=$(date -Iseconds) EXIT_CODE=0 HASH_VERIFIED=1" >>"$log"
        return 0
      fi
      echo "ENDED_AT=$(date -Iseconds) EXIT_CODE=65 HASH_VERIFIED=0" >>"$log"
      cat "$log" >&2
      return 65
    fi
    echo "ENDED_AT=$(date -Iseconds) EXIT_CODE=$ec HASH_VERIFIED=0" >>"$log"
    if ! grep -Eq '(^|[^A-Za-z])EOF([^A-Za-z]|$)' "$log" || [[ "$attempt" -eq "$PREPARE_RETRIES" ]]; then
      cat "$log" >&2
      return "$ec"
    fi
    echo "RETRY_REASON=transient_EOF BACKOFF_SECONDS=$attempt" >>"$log"
    sleep "$attempt"
  done
}

PREPARE_BEGIN="$(date -Iseconds)"
echo "PREPARE_BEGIN=$PREPARE_BEGIN mode=$CLUSTER_EXEC_MODE retries=$PREPARE_RETRIES"
prepare_exec mkdir "mkdir -p '$AFS_SCRIPTS' '$AFS_OUT'"
prepare_upload bench "$SCRIPT_DIR/nccl_torch_bench.py" "$AFS_SCRIPTS/nccl_torch_bench.py"
prepare_upload metrics "$SCRIPT_DIR/nccl_torch_bench_metrics.py" "$AFS_SCRIPTS/nccl_torch_bench_metrics.py"
PREPARE_END="$(date -Iseconds)"
echo "PREPARE_END=$PREPARE_END"
{
  echo "PREPARE_BEGIN=$PREPARE_BEGIN"
  echo "PREPARE_END=$PREPARE_END"
  echo "PREPARE_MODE=$CLUSTER_EXEC_MODE"
  echo "PREPARE_RETRIES=$PREPARE_RETRIES"
} >"$LOG_DIR/prepare_timing.env"
prepare_upload timing "$LOG_DIR/prepare_timing.env" "$AFS_OUT/prepare_timing.env"

if [[ "$UPLOAD_ONLY" == "1" ]]; then
  echo "UPLOAD_GATE_OK bench_and_metrics_hash_verified=1"
  exit 0
fi

LOCAL_LAUNCHER_DIR="$LOG_DIR/launchers"
mkdir -p "$LOCAL_LAUNCHER_DIR"

render_run_body() {
  local r="$1"
  local donef="$AFS_OUT/scale_${WORLD}.node_${r}.done"
  local failf="$AFS_OUT/scale_${WORLD}.node_${r}.fail"
  local startedf="$AFS_OUT/scale_${WORLD}.node_${r}.started"
  local rlog="$AFS_OUT/scale_${WORLD}.node_${r}.log"
  cat <<EOF
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
RUN_ID='$RUN_TOKEN'
NODE_RANK='$r'
atomic_marker() {
  path="\$1"; shift
  tmp="\${path}.tmp.\$\$"
  printf '%s\n' "\$@" >"\$tmp"
  mv -f "\$tmp" "\$path"
}
# 将本机调用时的可选 env 固化进脚本（setsid 不会继承操作机 export）
$( [[ -n "${NCCL_IB_DISABLE:-}" ]] && echo "export NCCL_IB_DISABLE=${NCCL_IB_DISABLE}" )
$( [[ -n "${MCCL_IB_DISABLE:-}" ]] && echo "export MCCL_IB_DISABLE=${MCCL_IB_DISABLE}" )
$( [[ -n "${NCCL_NET:-}" ]] && echo "export NCCL_NET=${NCCL_NET}" )
$( [[ -n "${MCCL_NET:-}" ]] && echo "export MCCL_NET=${MCCL_NET}" )
rm -f '$startedf' '$donef' '$failf'
atomic_marker '$startedf' \
  "RUN_ID=\$RUN_ID" "NODE_RANK=\$NODE_RANK" "PID=\$\$" "STARTED_AT=\$(date -Iseconds)"
cp -f '$AFS_SCRIPTS/nccl_torch_bench.py' /tmp/nccl_torch_bench.py
cp -f '$AFS_SCRIPTS/nccl_torch_bench_metrics.py' /tmp/nccl_torch_bench_metrics.py
/opt/conda/bin/torchrun --nnodes=$nnodes --node_rank=$r --nproc_per_node=$NPROC \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  /tmp/nccl_torch_bench.py --ops '$OPS' --sizes '$SIZES' \
  --warmup '$WARMUP' --iters '$ITERS' --out '$out' \
  >'$rlog' 2>&1
ec=\$?
if [[ \$ec -eq 0 ]]; then
  atomic_marker '$donef' \
    "RUN_ID=\$RUN_ID" "NODE_RANK=\$NODE_RANK" "RC=0" "ENDED_AT=\$(date -Iseconds)"
else
  atomic_marker '$failf' \
    "RUN_ID=\$RUN_ID" "NODE_RANK=\$NODE_RANK" "RC=\$ec" "ENDED_AT=\$(date -Iseconds)"
fi
exit \$ec
EOF
}

write_one_rank() {
  local r="$1"
  local attempt="$2"
  local flog="$3"
  local pod="${POD_NODES[$r]}"
  local run_local="/tmp/run_nccl_scale_${WORLD}_node_${r}.sh"
  local run_src="$LOCAL_LAUNCHER_DIR/node_${r}.sh"
  local started ended ec=0 expected_size expected_sha verify actual_size actual_sha
  started="$(date -Iseconds)"
  render_run_body "$r" >"$run_src"
  expected_size="$(wc -c <"$run_src" | tr -d ' ')"
  expected_sha="$(sha256sum "$run_src" | awk '{print $1}')"
  cluster_pod_exec_i "$pod" \
    "tmp='${run_local}.upload.tmp'; cat > \"\$tmp\" && chmod +x \"\$tmp\" && mv -f \"\$tmp\" '$run_local'; printf 'SIZE='; wc -c < '$run_local'; printf 'SHA256='; sha256sum '$run_local' | awk '{print \$1}'" \
    <"$run_src" \
    >"$flog" 2>&1 || ec=$?
  if [[ "$ec" -eq 0 ]]; then
    verify="$(cat "$flog")"
    actual_size="$(awk -F= '/^SIZE=/{print $2}' <<<"$verify")"
    actual_sha="$(awk -F= '/^SHA256=/{print $2}' <<<"$verify")"
    if [[ "$actual_size" != "$expected_size" || "$actual_sha" != "$expected_sha" ]]; then
      ec=65
    fi
  fi
  ended="$(date -Iseconds)"
  printf 'STAGE=write NODE_RANK=%s POD=%s ATTEMPT=%s STARTED_AT=%s ENDED_AT=%s EXPECTED_SIZE=%s EXPECTED_SHA256=%s EXIT_CODE=%s\n' \
    "$r" "$pod" "$attempt" "$started" "$ended" "$expected_size" "$expected_sha" "$ec" >>"$flog"
  return "$ec"
}

start_one_rank() {
  local r="$1"
  local attempt="$2"
  local flog="$3"
  local pod="${POD_NODES[$r]}"
  local run_local="/tmp/run_nccl_scale_${WORLD}_node_${r}.sh"
  local state_file="/tmp/run_nccl_scale_${WORLD}_node_${r}.start.pid"
  local donef="$AFS_OUT/scale_${WORLD}.node_${r}.done"
  local failf="$AFS_OUT/scale_${WORLD}.node_${r}.fail"
  local started ended ec=0
  started="$(date -Iseconds)"
  cluster_pod_exec "$pod" \
    "if [[ -f '$donef' ]]; then echo ALREADY_DONE; exit 0; fi; \
     if [[ -f '$failf' ]]; then echo EXISTING_FAIL_MARKER; exit 66; fi; \
     if [[ -s '$state_file' ]] && grep -qx 'RUN_ID=$RUN_TOKEN' '$state_file'; then oldpid=\$(awk -F= '/^PID=/{print \$2}' '$state_file'); if [[ -n \"\$oldpid\" ]] && kill -0 \$oldpid 2>/dev/null; then echo ALREADY_LAUNCHER_RUNNING PID=\$oldpid; exit 0; fi; fi; \
     existing=\$(ps -eo pid=,comm=,args= | awk '\$2 ~ /python/ && /torchrun/ && /--master_port=$MASTER_PORT/ && /$(basename "$AFS_OUT")/ {print \$1; exit}'); \
     if [[ -n \"\$existing\" ]]; then echo ALREADY_RUNNING PID=\$existing; exit 0; fi; \
     started_at=\$(date -Iseconds); setsid nohup bash $run_local </dev/null >/dev/null 2>&1 & pid=\$!; \
     tmp='${state_file}.tmp'; printf 'RUN_ID=%s\\nPID=%s\\nSTARTED_AT=%s\\n' '$RUN_TOKEN' \"\$pid\" \"\$started_at\" > \"\$tmp\"; mv -f \"\$tmp\" '$state_file'; \
     kill -0 \$pid; echo LAUNCHER_SUBMITTED_AT=\$started_at PID=\$pid" \
    >"$flog" 2>&1 || ec=$?
  ended="$(date -Iseconds)"
  printf 'STAGE=start NODE_RANK=%s POD=%s ATTEMPT=%s STARTED_AT=%s ENDED_AT=%s EXIT_CODE=%s\n' \
    "$r" "$pod" "$attempt" "$started" "$ended" "$ec" >>"$flog"
  return "$ec"
}

run_parallel_stage() {
  local stage="$1"
  local fn="$2"
  local stage_started stage_ended stage_key
  stage_key="$(printf '%s' "$stage" | tr '[:lower:]' '[:upper:]')"
  stage_started="$(date -Iseconds)"
  echo "STAGE_BEGIN stage=$stage at=$stage_started nnodes=$nnodes parallel=$PARALLEL"
  local rc=0
  parallel_retry_run "$stage" "$nnodes" "$PARALLEL" "$NODE_RETRIES" "$LOG_DIR" "$fn" || rc=$?
  stage_ended="$(date -Iseconds)"
  echo "STAGE_END stage=$stage at=$stage_ended rc=$rc"
  {
    echo "${stage_key}_STAGE_BEGIN=$stage_started"
    echo "${stage_key}_STAGE_END=$stage_ended"
    echo "${stage_key}_RC=$rc"
  } >>"$LOG_DIR/fire_timing.env"
  return "$rc"
}

cleanup_one_rank() {
  local r="$1"
  local attempt="$2"
  local flog="$3"
  local pod="${POD_NODES[$r]}"
  cluster_pod_exec "$pod" \
    "state=/tmp/run_nccl_scale_${WORLD}_node_${r}.start.pid; \
     if [[ -s \$state ]] && grep -qx 'RUN_ID=$RUN_TOKEN' \$state; then pid=\$(awk -F= '/^PID=/{print \$2}' \$state); [[ -n \"\$pid\" ]] && kill \$pid 2>/dev/null || true; fi; \
     ids=\$(ps -eo pid=,comm=,args= | awk '\$2 ~ /python/ && /$MASTER_PORT/ && /$(basename "$AFS_OUT")/ {print \$1}'); for p in \$ids; do kill \$p 2>/dev/null || true; done; \
     if grep -q \"RUN_ID='$RUN_TOKEN'\" /tmp/run_nccl_scale_${WORLD}_node_${r}.sh 2>/dev/null; then rm -f /tmp/run_nccl_scale_${WORLD}_node_${r}.sh; fi; \
     rm -f \$state" \
    >"$flog" 2>&1
  printf 'STAGE=cleanup NODE_RANK=%s POD=%s ATTEMPT=%s EXIT_CODE=0\n' \
    "$r" "$pod" "$attempt" >>"$flog"
}

: >"$LOG_DIR/fire_timing.env"
echo "FIRE_PARALLELISM=$PARALLEL" >>"$LOG_DIR/fire_timing.env"
echo "FIRE_NNODES=$nnodes" >>"$LOG_DIR/fire_timing.env"
echo "FIRE_BEGIN=$(date -Iseconds)" >>"$LOG_DIR/fire_timing.env"

if ! run_parallel_stage write write_one_rank; then
  echo "FIRE_FAIL stage=write"
  exit 1
fi
if ! run_parallel_stage start start_one_rank; then
  echo "FIRE_FAIL stage=start; cleaning all target nodes"
  run_parallel_stage cleanup cleanup_one_rank || true
  exit 1
fi
echo "FIRE_END=$(date -Iseconds)" >>"$LOG_DIR/fire_timing.env"
cluster_pod_exec_i "${CLUSTER_POD}" "cat > '$AFS_OUT/fire_timing.env'" <"$LOG_DIR/fire_timing.env"
echo "FIRE_DONE scale=$WORLD nnodes=$nnodes parallel=$PARALLEL"
