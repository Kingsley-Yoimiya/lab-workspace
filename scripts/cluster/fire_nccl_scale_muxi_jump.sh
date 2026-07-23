#!/usr/bin/env bash
# 沐曦 matched-world 跳板直连编排。
# 所有目标 pod 的 vcctl exec 均在 ais-cf3e61a5 内发起；Mac 只上传 bundle 和回拉结果。
set -euo pipefail

RUN_ID="${RUN_ID:?set RUN_ID}"
MASTER_PORT="${MASTER_PORT:?set MASTER_PORT}"
BUNDLE_DIR="${BUNDLE_DIR:?set BUNDLE_DIR}"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config-vc-c550-h3c-test.yaml}"
VCCTL="${VCCTL:-/usr/local/bin/vcctl}"
JOB="${JOB:-yinjinrun-cs512-20260716-221823}"
AFS_OUT="${AFS_OUT:-/afs-a3-weight-share/yinjinrun.p/results/muxi-fabric-w1/$RUN_ID}"
WORK="/tmp/muxi-jump-$RUN_ID"
MASTER="$JOB-master-0"
WORLD="${WORLD:-64}"
NPROC="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-$((WORLD / NPROC))}"
OPS="${OPS:-all_reduce}"
SIZES="${SIZES:-256M}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
NCCL_IB_HCA_CFG="${NCCL_IB_HCA:-xscale_0,xscale_1,xscale_2,xscale_3}"
MCCL_IB_HCA_CFG="${MCCL_IB_HCA:-xscale_0,xscale_1,xscale_2,xscale_3}"
NCCL_DEBUG_LEVEL="${NCCL_DEBUG:-WARN}"
MCCL_DEBUG_LEVEL="${MCCL_DEBUG:-WARN}"
MCCL_ALGO_CFG="${MCCL_ALGO:-}"
MCCL_PROTO_CFG="${MCCL_PROTO:-}"
MCCL_MIN_NCHANNELS_CFG="${MCCL_MIN_NCHANNELS:-}"
MCCL_MAX_NCHANNELS_CFG="${MCCL_MAX_NCHANNELS:-}"
CUDA_VISIBLE_DEVICES_CFG="${CUDA_VISIBLE_DEVICES:-}"
POD_ORDER="${POD_ORDER:-}"
POD_ORDER_DRY_RUN="${POD_ORDER_DRY_RUN:-0}"
[[ $((WORLD % NPROC)) -eq 0 && "$NNODES" -eq $((WORLD / NPROC)) ]]
[[ "$OPS" =~ ^[a-z_,]+$ && "$SIZES" =~ ^[0-9KMGkmg,]+$ ]]
[[ "$WARMUP" =~ ^[0-9]+$ && "$ITERS" =~ ^[1-9][0-9]*$ ]]
[[ "$NCCL_IB_HCA_CFG" =~ ^xscale_[0-3](,xscale_[0-3])*$ ]]
[[ "$MCCL_IB_HCA_CFG" == "$NCCL_IB_HCA_CFG" ]]
[[ "$NCCL_DEBUG_LEVEL" =~ ^(WARN|INFO)$ && "$MCCL_DEBUG_LEVEL" =~ ^(WARN|INFO)$ ]]
[[ "$MCCL_ALGO_CFG" =~ ^[A-Za-z0-9_,-]*$ && "$MCCL_PROTO_CFG" =~ ^[A-Za-z0-9_,-]*$ ]]
[[ "$MCCL_MIN_NCHANNELS_CFG" =~ ^[0-9]*$ && "$MCCL_MAX_NCHANNELS_CFG" =~ ^[0-9]*$ ]]
[[ "$CUDA_VISIBLE_DEVICES_CFG" =~ ^[0-9,]*$ ]]
OP_COUNT="$(awk -F, '{print NF}' <<<"$OPS")"
SIZE_COUNT="$(awk -F, '{print NF}' <<<"$SIZES")"
CASE_COUNT=$((OP_COUNT * SIZE_COUNT))

mkdir -p "$WORK/fire" "$WORK/preflight" "$WORK/cleanup" "$WORK/launchers"
exec > >(tee -a "$WORK/driver.log") 2>&1
export KUBECONFIG

# 物理节点集合固定；POD_ORDER 只决定 node_rank 使用哪个物理 pod。
physical_pods=("$MASTER")
for i in $(seq 0 "$((NNODES - 2))"); do physical_pods+=("$JOB-worker-$i"); done
if [[ -n "$POD_ORDER" ]]; then
  IFS=',' read -r -a pod_order_indices <<<"$POD_ORDER"
else
  pod_order_indices=()
  for ((i=0; i<NNODES; i++)); do pod_order_indices+=("$i"); done
fi
if [[ "${#pod_order_indices[@]}" -ne "$NNODES" ]]; then
  echo "POD_ORDER_COUNT_INVALID expected=$NNODES actual=${#pod_order_indices[@]}" >&2
  exit 64
fi
declare -a seen_indices=() pods=()
for i in "${pod_order_indices[@]}"; do
  if [[ ! "$i" =~ ^[0-9]+$ || "$i" -ge "$NNODES" ]]; then
    echo "POD_ORDER_INDEX_INVALID index=$i nnodes=$NNODES" >&2
    exit 65
  fi
  if [[ -n "${seen_indices[$i]+x}" ]]; then
    echo "POD_ORDER_DUPLICATE index=$i" >&2
    exit 66
  fi
  seen_indices[$i]=1
  pods+=("${physical_pods[$i]}")
done
POD_ORDER_CSV="$(IFS=,; echo "${pod_order_indices[*]}")"
POD_HOST_ORDER_CSV="$(IFS=,; echo "${pods[*]}")"
RENDEZVOUS_POD="${pods[0]}"

if [[ -n "$CUDA_VISIBLE_DEVICES_CFG" ]]; then
  IFS=',' read -r -a visible_gpu_indices <<<"$CUDA_VISIBLE_DEVICES_CFG"
  [[ "${#visible_gpu_indices[@]}" -eq "$NPROC" ]]
  [[ "$(printf '%s\n' "${visible_gpu_indices[@]}" | sort -u | wc -l)" -eq "$NPROC" ]]
else
  visible_gpu_indices=()
  for ((i=0; i<NPROC; i++)); do visible_gpu_indices+=("$i"); done
fi

{
  echo "global_rank,node_rank,pod,host,local_rank,physical_gpu,cuda_visible_devices"
  for ((i=0; i<NNODES; i++)); do
    for ((local_rank=0; local_rank<NPROC; local_rank++)); do
      global_rank=$((i*NPROC+local_rank))
      echo "$global_rank,$i,${pods[$i]},${pods[$i]},$local_rank,${visible_gpu_indices[$local_rank]},${CUDA_VISIBLE_DEVICES_CFG:-unset}"
    done
  done
} >"$WORK/rank_mapping.csv"

if [[ "$POD_ORDER_DRY_RUN" == "1" ]]; then
  echo "POD_ORDER_DRY_RUN nnodes=$NNODES nproc=$NPROC order=$POD_ORDER_CSV rendezvous=$RENDEZVOUS_POD"
  while IFS= read -r line; do echo "MAPPING $line"; done <"$WORK/rank_mapping.csv"
  exit 0
fi

vexec() {
  local pod="$1"; shift
  "$VCCTL" pod exec "$pod" -- bash -lc "$1"
}

vexec_i() {
  local pod="$1"; shift
  "$VCCTL" pod exec -i "$pod" -- bash -c "$1"
}

is_auth_error() {
  grep -Eiq 'Forbidden|Unauthorized|permission denied|certificate|credentials|authentication' "$1"
}

# 首次 pending 为全部节点；后续只重试失败节点，成功节点不重复。
run_stage() {
  local stage="$1" fn="$2" count="$3"
  mkdir -p "$WORK/$stage"
  local -a pending=() permanent=()
  local rank attempt i rc
  for ((rank=0; rank<count; rank++)); do pending+=("$rank"); done
  echo "STAGE_BEGIN stage=$stage count=$count parallel=$count at=$(date -Iseconds)"
  for attempt in 1 2 3; do
    [[ ${#pending[@]} -gt 0 ]] || break
    echo "STAGE_ATTEMPT stage=$stage attempt=$attempt pending=${#pending[@]} ranks=${pending[*]}"
    local -a pids=() ranks=() retry=()
    for rank in "${pending[@]}"; do
      "$fn" "$rank" "$attempt" &
      pids+=("$!"); ranks+=("$rank")
    done
    for i in "${!pids[@]}"; do
      rank="${ranks[$i]}"; rc=0
      wait "${pids[$i]}" || rc=$?
      if [[ "$rc" -eq 0 ]]; then
        continue
      fi
      log="$WORK/$stage/node_${rank}.attempt${attempt}.log"
      if [[ "$rc" -eq 40 || "$rc" -eq 41 ]] || is_auth_error "$log"; then
        permanent+=("$rank:$rc")
      elif [[ "$attempt" -lt 3 ]]; then
        retry+=("$rank")
      else
        permanent+=("$rank:$rc")
      fi
    done
    pending=("${retry[@]}")
    [[ ${#pending[@]} -eq 0 ]] || sleep "$attempt"
  done
  echo "STAGE_END stage=$stage permanent=${permanent[*]:-none} at=$(date -Iseconds)"
  [[ ${#permanent[@]} -eq 0 ]]
}

preflight_one() {
  local rank="$1" attempt="$2"
  local pod="${pods[$rank]}"
  local log="$WORK/preflight/node_${rank}.attempt${attempt}.log" rc=0
  {
    echo "POD=$pod ATTEMPT=$attempt START=$(date -Iseconds)"
    vexec "$pod" \
      "set -e; ps -eo args | awk 'BEGIN{IGNORECASE=1} /torchrun|nccl_torch_bench|all_reduce_perf|constitution|burn[-_]?in/ && \$0 !~ /awk/ {found=1; print} END{exit found?1:0}'; gpu=\$(mx-smi 2>/dev/null); printf '%s\n' \"\$gpu\" | grep -q 'no process found'; echo IDLE"
  } >"$log" 2>&1 || rc=$?
  echo "RC=$rc END=$(date -Iseconds)" >>"$log"
  return "$rc"
}

cleanup_one() {
  local rank="$1" attempt="$2"
  local pod="${pods[$rank]}"
  local log="$WORK/cleanup/node_${rank}.attempt${attempt}.log" rc=0
  {
    vexec "$pod" \
      "ids=\$(ps -eo pid=,comm=,args= | awk '\$2 ~ /python/ && /$MASTER_PORT/ && /$RUN_ID/ {print \$1}'); for p in \$ids; do kill \$p 2>/dev/null || true; done; state=/tmp/${RUN_ID}.rank${rank}.pid; if [[ -s \$state ]]; then pid=\$(cat \$state); kill -TERM -- -\$pid 2>/dev/null || true; for _ in 1 2 3 4 5; do kill -0 -- -\$pid 2>/dev/null || break; sleep 1; done; kill -KILL -- -\$pid 2>/dev/null || true; fi; rm -f /tmp/${RUN_ID}.rank${rank}.sh \$state /tmp/${RUN_ID}.gate; echo CLEAN"
  } >"$log" 2>&1 || rc=$?
  echo "RC=$rc" >>"$log"
  return "$rc"
}

cleanup_all() {
  mkdir -p "$WORK/cleanup"
  run_stage cleanup cleanup_one "$NNODES" || true
}

on_error() {
  echo "DRIVER_ERROR at=$(date -Iseconds)"
  cleanup_all
}
trap on_error ERR

echo "DRIVER_BEGIN run_id=$RUN_ID port=$MASTER_PORT at=$(date -Iseconds)"
echo "KUBECONFIG_PATH=$KUBECONFIG MODE=$(stat -c %a "$KUBECONFIG")"

# 单次列表验证固定 64 pod Running。
"$VCCTL" pod get --job "$JOB" >"$WORK/pods.log"
python3 - "$WORK/pods.log" <<'PY'
import re,sys
lines=open(sys.argv[1]).read().splitlines()
pods=[x for x in lines if "yinjinrun-cs512-20260716-221823-" in x]
running=[x for x in pods if re.search(r"\bRunning\b", x)]
assert len(pods)==64 and len(running)==64, (len(pods),len(running))
print("POD_LIST_OK total=64 running=64")
PY

# 2 节点无通信/小文件门禁。
gate_one() {
  local rank="$1" attempt="$2"
  local pod="${pods[$rank]}"
  local log="$WORK/preflight/gate_${rank}.attempt${attempt}.log" rc=0
  vexec "$pod" \
    "printf '$RUN_ID:%s\n' '$rank' > /tmp/${RUN_ID}.gate; grep -qx '$RUN_ID:$rank' /tmp/${RUN_ID}.gate; rm -f /tmp/${RUN_ID}.gate; echo GATE_OK" \
    >"$log" 2>&1 || rc=$?
  return "$rc"
}
run_stage gate gate_one 2

# 目标 pod 空闲门禁。
run_stage preflight preflight_one "$NNODES"

# 共享代码只经 master 写入 AFS，并逐文件 hash 校验。
vexec "$MASTER" "mkdir -p '$AFS_OUT/code' '$AFS_OUT/raw' '$AFS_OUT/fire'"
vexec_i "$MASTER" "cat > '$AFS_OUT/rank_mapping.csv'" <"$WORK/rank_mapping.csv"
upload_file() {
  local src="$1" remote="$2" name="$3"
  local size sha verify
  size="$(wc -c <"$src" | tr -d ' ')"
  sha="$(sha256sum "$src" | awk '{print $1}')"
  vexec_i "$MASTER" "tmp='${remote}.tmp'; cat > \"\$tmp\" && mv -f \"\$tmp\" '$remote'" <"$src"
  verify="$(vexec "$MASTER" "printf 'SIZE='; wc -c < '$remote'; printf 'SHA='; sha256sum '$remote' | awk '{print \$1}'")"
  [[ "$verify" == *"SIZE=$size"* && "$verify" == *"SHA=$sha"* ]]
  echo "UPLOAD_OK name=$name size=$size sha=$sha"
}
upload_file "$BUNDLE_DIR/nccl_torch_bench.py" "$AFS_OUT/code/nccl_torch_bench.py" bench
upload_file "$BUNDLE_DIR/nccl_torch_bench_metrics.py" "$AFS_OUT/code/nccl_torch_bench_metrics.py" metrics

# 为每个 node_rank 生成独立 launcher，再一次性经 master 写入 AFS。
for rank in $(seq 0 "$((NNODES - 1))"); do
  cat >"$WORK/launchers/rank_${rank}.sh" <<EOF
#!/usr/bin/env bash
export PATH=/opt/conda/bin:\${PATH:-/usr/bin}
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0 MCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA='${NCCL_IB_HCA_CFG}'
export MCCL_IB_HCA='${MCCL_IB_HCA_CFG}'
export NCCL_IB_GID_INDEX=5 MCCL_IB_GID_INDEX=5 MCCL_IB_TC=128
export MCCL_ENABLE_VSWITCH=1 MCCL_PCIE_BUFFER_MODE=0 FORCE_ACTIVE_WAIT=2
export NCCL_DEBUG='${NCCL_DEBUG_LEVEL}' MCCL_DEBUG='${MCCL_DEBUG_LEVEL}'
if [[ -n '${MCCL_ALGO_CFG}' ]]; then export MCCL_ALGO='${MCCL_ALGO_CFG}'; fi
if [[ -n '${MCCL_PROTO_CFG}' ]]; then export MCCL_PROTO='${MCCL_PROTO_CFG}'; fi
if [[ -n '${MCCL_MIN_NCHANNELS_CFG}' ]]; then export MCCL_MIN_NCHANNELS='${MCCL_MIN_NCHANNELS_CFG}'; fi
if [[ -n '${MCCL_MAX_NCHANNELS_CFG}' ]]; then export MCCL_MAX_NCHANNELS='${MCCL_MAX_NCHANNELS_CFG}'; fi
if [[ -n '${CUDA_VISIBLE_DEVICES_CFG}' ]]; then export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES_CFG}'; fi
rm -f '$AFS_OUT/scale_${WORLD}.node_${rank}.done' '$AFS_OUT/scale_${WORLD}.node_${rank}.fail'
cp -f '$AFS_OUT/code/nccl_torch_bench.py' /tmp/nccl_torch_bench.py
cp -f '$AFS_OUT/code/nccl_torch_bench_metrics.py' /tmp/nccl_torch_bench_metrics.py
/opt/conda/bin/torchrun --nnodes=${NNODES} --node_rank=${rank} --nproc_per_node=${NPROC} \
  --master_addr=${RENDEZVOUS_POD}.${JOB} --master_port=${MASTER_PORT} \
  /tmp/nccl_torch_bench.py --ops '${OPS}' --sizes '${SIZES}' --warmup ${WARMUP} --iters ${ITERS} \
  --out '$AFS_OUT/scale_${WORLD}.jsonl' >'$AFS_OUT/scale_${WORLD}.node_${rank}.log' 2>&1
rc=\$?
tmp='$AFS_OUT/scale_${WORLD}.node_${rank}.marker.tmp.'\$\$
if [[ \$rc -eq 0 ]]; then printf 'RUN_ID=%s\nRC=0\n' '$RUN_ID' >"\$tmp"; mv -f "\$tmp" '$AFS_OUT/scale_${WORLD}.node_${rank}.done'; else printf 'RUN_ID=%s\nRC=%s\n' '$RUN_ID' "\$rc" >"\$tmp"; mv -f "\$tmp" '$AFS_OUT/scale_${WORLD}.node_${rank}.fail'; fi
exit \$rc
EOF
  chmod +x "$WORK/launchers/rank_${rank}.sh"
done
(cd "$WORK" && sha256sum launchers/rank_*.sh >launchers/SHA256SUMS)
tar -C "$WORK" -cf - launchers | vexec_i "$MASTER" "tar -C '$AFS_OUT/code' -xf -"
vexec "$MASTER" "cd '$AFS_OUT/code' && sha256sum -c launchers/SHA256SUMS >/dev/null && [[ \$(find launchers -name 'rank_*.sh' | wc -l) -eq $NNODES ]] && echo LAUNCHERS_OK"

start_one() {
  local rank="$1" attempt="$2"
  local pod="${pods[$rank]}"
  local log="$WORK/fire/node_${rank}.attempt${attempt}.log" rc=0
  {
    echo "POD=$pod RANK=$rank ATTEMPT=$attempt START=$(date -Iseconds)"
    vexec "$pod" \
      "cp -f '$AFS_OUT/code/launchers/rank_${rank}.sh' /tmp/${RUN_ID}.rank${rank}.sh; chmod +x /tmp/${RUN_ID}.rank${rank}.sh; setsid nohup bash /tmp/${RUN_ID}.rank${rank}.sh </dev/null >/dev/null 2>&1 & pid=\$!; echo \$pid > /tmp/${RUN_ID}.rank${rank}.pid; echo SUBMITTED PID=\$pid AT=\$(date -Iseconds)"
  } >"$log" 2>&1 || rc=$?
  echo "RC=$rc END=$(date -Iseconds)" >>"$log"
  return "$rc"
}
fire_begin="$(date -Iseconds)"; fire_t0="$(date +%s)"
run_stage fire start_one "$NNODES"
fire_end="$(date -Iseconds)"; fire_s=$(( $(date +%s)-fire_t0 ))
echo "FIRE_TIMING begin=$fire_begin end=$fire_end elapsed_s=$fire_s parallel=$NNODES"

# 单路轮询 master：多 case 在结束前不会写 rank 文件，因此同时跟踪
# master 日志中的 case 输出、字节数和 mtime。任一信号增长都刷新进度。
last_rank=-1; last_case=-1; last_log_bytes=-1; last_log_mtime=-1
last_progress=$(date +%s); poll=0
while true; do
  state="$(vexec "$MASTER" "r=\$(find '$AFS_OUT' -maxdepth 1 -name 'scale_${WORLD}.rank*.jsonl' -type f | wc -l); f=\$(find '$AFS_OUT' -maxdepth 1 -name 'scale_${WORLD}.node_*.fail' -type f | wc -l); d=\$(find '$AFS_OUT' -maxdepth 1 -name 'scale_${WORLD}.node_*.done' -type f | wc -l); log='$AFS_OUT/scale_${WORLD}.node_0.log'; c=0; b=0; m=0; if [[ -f \$log ]]; then c=\$(grep -c '^op=' \$log 2>/dev/null || true); b=\$(wc -c < \$log); m=\$(stat -c %Y \$log); fi; e=0; for x in '$AFS_OUT'/scale_${WORLD}.node_*.log; do [[ -f \$x ]] || continue; if grep -Eiq 'CUDA out of memory|OutOfMemory|SIGSEGV|Segmentation fault|MCCL (ERROR|FATAL)' \$x; then e=\$((e+1)); fi; done; echo RANKS=\$r FAIL=\$f DONE=\$d CASES=\$c LOG_BYTES=\$b LOG_MTIME=\$m ERRORS=\$e")"
  ranks="$(sed -E 's/.*RANKS=([0-9]+).*/\1/' <<<"$state")"
  fails="$(sed -E 's/.*FAIL=([0-9]+).*/\1/' <<<"$state")"
  done_count="$(sed -E 's/.*DONE=([0-9]+).*/\1/' <<<"$state")"
  cases_done="$(sed -E 's/.*CASES=([0-9]+).*/\1/' <<<"$state")"
  log_bytes="$(sed -E 's/.*LOG_BYTES=([0-9]+).*/\1/' <<<"$state")"
  log_mtime="$(sed -E 's/.*LOG_MTIME=([0-9]+).*/\1/' <<<"$state")"
  errors="$(sed -E 's/.*ERRORS=([0-9]+).*/\1/' <<<"$state")"
  now=$(date +%s)
  if [[ "$ranks" -gt "$last_rank" || "$cases_done" -gt "$last_case" ||
        "$log_bytes" -gt "$last_log_bytes" || "$log_mtime" -gt "$last_log_mtime" ]]; then
    last_progress="$now"
  fi
  echo "POLL=$poll $state LAST_PROGRESS=$last_progress AT=$(date -Iseconds)"
  [[ "$fails" -eq 0 && "$errors" -eq 0 ]]
  if [[ "$ranks" -eq "$WORLD" && "$done_count" -eq "$NNODES" &&
        "$cases_done" -eq "$CASE_COUNT" ]]; then break; fi
  [[ $((now-last_progress)) -lt 180 ]]
  last_rank="$ranks"; last_case="$cases_done"; last_log_bytes="$log_bytes"; last_log_mtime="$log_mtime"
  poll=$((poll+1)); sleep 5
done

# INFO 级别 rail 实验必须同时证明环境变量被 MCCL 接受，且 NET/IB 实际
# 选择集合与请求一致。证据仅从 master rank 日志提取，避免额外 16 路查询。
if [[ "$MCCL_DEBUG_LEVEL" == "INFO" ]]; then
  vexec "$MASTER" "set -e; log='$AFS_OUT/scale_${WORLD}.node_0.log'; evidence='$AFS_OUT/hca_evidence.txt'; grep -F 'MCCL_IB_HCA set to $MCCL_IB_HCA_CFG' \$log > \$evidence; awk '/NET\\/IB : Using/{left=5} left>0{print; left--}' \$log >> \$evidence; grep -Fq 'MCCL INFO Using network IB' \$log; for h in \$(printf '%s' '$MCCL_IB_HCA_CFG' | tr ',' ' '); do grep -Fq \"\$h\" \$evidence; done; for h in xscale_0 xscale_1 xscale_2 xscale_3; do case ','$MCCL_IB_HCA_CFG',' in *\",\$h,\"*) ;; *) if grep -Fq \"\$h\" \$evidence; then echo UNEXPECTED_HCA=\$h >&2; exit 42; fi ;; esac; done; echo HCA_EFFECTIVE=$MCCL_IB_HCA_CFG" | tee "$WORK/hca_confirmation.log"
  vexec "$MASTER" "log='$AFS_OUT/scale_${WORLD}.node_0.log'; awk '/MCCL_(ALGO|PROTO|MIN_NCHANNELS|MAX_NCHANNELS).*set by environment|[0-9]+ coll channels|threadThresholds|Trees \\[|Channel 00\\//{print}' \$log" >"$WORK/software_control_evidence.log"
  vexec "$MASTER" "log='$AFS_OUT/scale_${WORLD}.node_0.log'; awk '/GPU_MAPPING rank=|comm .* rank [0-9]+ nranks .* macaDev [0-9]+ busId/{print}' \$log" >"$WORK/gpu_mapping_evidence.log"
fi

# 在 master 内按 (op, size) 校验全部 case、每 rank 记录数和 20 iter schema。
cat >"$WORK/validate.py" <<'PY'
import glob,json,math,os,re,statistics
root=os.environ["AFS_OUT"]
world=int(os.environ["EXPECTED_WORLD"])
nnodes=int(os.environ["EXPECTED_NNODES"])
nproc=int(os.environ["EXPECTED_NPROC"])
host_order=os.environ["EXPECTED_HOST_ORDER"].split(",")
visible_devices=[int(x) for x in os.environ["EXPECTED_VISIBLE_DEVICES"].split(",") if x]
ops=[x for x in os.environ["EXPECTED_OPS"].split(",") if x]
def parse_size(x):
    x=x.strip().upper()
    if x.endswith("K"): return int(float(x[:-1])*1024)
    if x.endswith("M"): return int(float(x[:-1])*1024**2)
    if x.endswith("G"): return int(float(x[:-1])*1024**3)
    return int(x)
sizes=[parse_size(x) for x in os.environ["EXPECTED_SIZES"].split(",") if x]
iters=int(os.environ["EXPECTED_ITERS"])
expected_cases={(op,size) for size in sizes for op in ops}
files=glob.glob(root+f"/scale_{world}.rank*.jsonl")
def rank_of(p): return int(re.search(r"rank(\d+)\.jsonl$",p).group(1))
files=sorted(files,key=rank_of)
errors=[]; recs=[]
if len(files)!=world: errors.append(f"files={len(files)}")
for p in files:
    lines=[x for x in open(p) if x.strip()]
    if len(lines)!=len(expected_cases):
        errors.append(f"{os.path.basename(p)} lines={len(lines)}")
    rank_recs=[]
    for line in lines:
        r=json.loads(line); recs.append(r); rank_recs.append(r)
        case=(r.get("op"),r.get("nbytes"))
        if (r.get("world_size"),r.get("timing_version"),r.get("bw_basis"),r.get("n_iters")) != (world,"w0.1","global_max",iters):
            errors.append(f"fields rank={r.get('rank')} case={case}")
        if len(r.get("iters_s_global_max",[]))!=iters or len(r.get("iters_s_local",[]))!=iters:
            errors.append(f"iters rank={r.get('rank')} case={case}")
    if { (r.get("op"),r.get("nbytes")) for r in rank_recs } != expected_cases:
        errors.append(f"case_set rank_file={os.path.basename(p)}")
for case in sorted(expected_cases):
    rows=[r for r in recs if (r.get("op"),r.get("nbytes"))==case]
    if sorted(r.get("rank",-1) for r in rows)!=list(range(world)):
        errors.append(f"rank_set case={case}")
    if rows and any(r["iters_s_global_max"]!=rows[0]["iters_s_global_max"] for r in rows[1:]):
        errors.append(f"global_vectors case={case}")
    for r in rows:
        rank=r.get("rank",-1)
        node_rank=rank//nproc if rank >= 0 else -1
        if node_rank < 0 or node_rank >= len(host_order) or r.get("host") != host_order[node_rank]:
            errors.append(f"host_mapping rank={rank} host={r.get('host')}")
        local_rank=r.get("local_rank",-1)
        if local_rank < 0 or local_rank >= len(visible_devices) or r.get("physical_gpu") != visible_devices[local_rank]:
            errors.append(f"physical_gpu_mapping rank={rank}")
hosts={}
for r in recs:
    hosts.setdefault(r.get("host"),set()).add(r.get("local_rank"))
if len(hosts)!=nnodes: errors.append(f"hosts={len(hosts)}")
if any(v!=set(range(nproc)) for v in hosts.values()): errors.append("local_rank_mapping")
with open(root+f"/raw/scale_{world}.jsonl","w") as out:
    for r in sorted(recs,key=lambda x:(x["nbytes"],x["op"],x["rank"])):
        out.write(json.dumps(r)+"\n")
cases=[]
for op,size in sorted(expected_cases,key=lambda x:(x[1],x[0])):
    rows=[r for r in recs if (r.get("op"),r.get("nbytes"))==(op,size)]
    base=rows[0] if rows else {}
    vals=base.get("iters_s_global_max",[])
    q=lambda p: sorted(vals)[math.ceil(p*len(vals))-1]*1000 if vals else None
    cases.append({"op":op,"nbytes":size,"records":len(rows),
                  "avg_ms":base.get("avg_s_global_max",0)*1000,
                  "alg_bw_GBps":base.get("alg_bw_GBps_global_max"),
                  "bus_bw_GBps":base.get("bus_bw_GBps_global_max"),
                  "p50_ms":statistics.median(vals)*1000 if vals else None,
                  "p95_ms":q(.95),"p99_ms":q(.99)})
summary={"valid":not errors,"errors":errors,"rank_files":len(files),"records":len(recs),
         "expected_case_count":len(expected_cases),"iters":iters,"cases":cases}
if len(cases)==1:
    only=cases[0]
    rows=[r for r in recs if (r.get("op"),r.get("nbytes"))==(only["op"],only["nbytes"])]
    summary.update({"avg_ms":only["avg_ms"],"alg_bw_GBps":only["alg_bw_GBps"],
                    "bus_bw_GBps":only["bus_bw_GBps"],
                    "iters_s_global_max":rows[0]["iters_s_global_max"] if rows else []})
json.dump(summary,open(root+"/validation_summary.json","w"),indent=2)
if errors: raise SystemExit(1)
PY
AFS_OUT="$AFS_OUT" vexec_i "$MASTER" "AFS_OUT='$AFS_OUT' EXPECTED_WORLD='$WORLD' EXPECTED_NNODES='$NNODES' EXPECTED_NPROC='$NPROC' EXPECTED_HOST_ORDER='$POD_HOST_ORDER_CSV' EXPECTED_VISIBLE_DEVICES='$(IFS=,; echo "${visible_gpu_indices[*]}")' EXPECTED_OPS='$OPS' EXPECTED_SIZES='$SIZES' EXPECTED_ITERS='$ITERS' python3 -" <"$WORK/validate.py"
vexec "$MASTER" "cat '$AFS_OUT/validation_summary.json'" >"$WORK/validation_summary.json"

# 成功后仍只清理本 run 进程/脚本。
cleanup_all
trap - ERR

python3 - "$WORK/validation_summary.json" "$WORK/SUMMARY.md" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
rows=["| op | bytes | avg global-max(ms) | bus_bw(GB/s) | p50/p95/p99(ms) |",
      "|---|---:|---:|---:|---:|"]
for c in s["cases"]:
    rows.append(f"| {c['op']} | {c['nbytes']} | {c['avg_ms']:.6f} | {c['bus_bw_GBps']:.6f} | {c['p50_ms']:.3f}/{c['p95_ms']:.3f}/{c['p99_ms']:.3f} |")
text=f"""# matched-world jump-driver VALID

- {s['rank_files']} rank files, {s['records']} records, {s['expected_case_count']} cases.
- 每 case {s['iters']} iterations，w0.1 / global-max schema valid.

{chr(10).join(rows)}
"""
open(sys.argv[2],"w").write(text)
PY

cat >"$WORK/manifest.yaml" <<EOF
run_id: $RUN_ID
status: VALID
driver: jump_vcctl
kubeconfig_path: $KUBECONFIG
kubeconfig_mode: 600
world_size: $WORLD
nnodes: $NNODES
nproc_per_node: $NPROC
master_port: $MASTER_PORT
fire_parallelism: $NNODES
fire_begin: $fire_begin
fire_end: $fire_end
fire_elapsed_s: $fire_s
local_rank_gpu_range: 0..$((NPROC - 1))
ops: $OPS
sizes: $SIZES
warmup: $WARMUP
iters: $ITERS
expected_cases: $CASE_COUNT
nccl_ib_hca: $NCCL_IB_HCA_CFG
mccl_ib_hca: $MCCL_IB_HCA_CFG
nccl_debug: $NCCL_DEBUG_LEVEL
mccl_debug: $MCCL_DEBUG_LEVEL
mccl_algo: ${MCCL_ALGO_CFG:-default}
mccl_proto: ${MCCL_PROTO_CFG:-default}
mccl_min_nchannels: ${MCCL_MIN_NCHANNELS_CFG:-default}
mccl_max_nchannels: ${MCCL_MAX_NCHANNELS_CFG:-default}
cuda_visible_devices: ${CUDA_VISIBLE_DEVICES_CFG:-unset}
pod_order_indices: $POD_ORDER_CSV
rendezvous_pod: $RENDEZVOUS_POD
rank_mapping_file: rank_mapping.csv
EOF

# 保存 driver/fire/manifest/summary 到 AFS。
vexec_i "$MASTER" "cat > '$AFS_OUT/driver.log'" <"$WORK/driver.log"
vexec_i "$MASTER" "cat > '$AFS_OUT/run.log'" <"$WORK/driver.log"
vexec_i "$MASTER" "cat > '$AFS_OUT/manifest.yaml'" <"$WORK/manifest.yaml"
vexec_i "$MASTER" "cat > '$AFS_OUT/SUMMARY.md'" <"$WORK/SUMMARY.md"
tar -C "$WORK" -cf - fire preflight | vexec_i "$MASTER" "tar -C '$AFS_OUT' -xf -"
if [[ -f "$WORK/hca_confirmation.log" ]]; then
  vexec_i "$MASTER" "cat > '$AFS_OUT/hca_confirmation.log'" <"$WORK/hca_confirmation.log"
fi
if [[ -f "$WORK/software_control_evidence.log" ]]; then
  vexec_i "$MASTER" "cat > '$AFS_OUT/software_control_evidence.log'" <"$WORK/software_control_evidence.log"
fi
if [[ -f "$WORK/gpu_mapping_evidence.log" ]]; then
  vexec_i "$MASTER" "cat > '$AFS_OUT/gpu_mapping_evidence.log'" <"$WORK/gpu_mapping_evidence.log"
fi
echo "DRIVER_VALID run_id=$RUN_ID fire_elapsed_s=$fire_s at=$(date -Iseconds)"
