#!/usr/bin/env bash
# fire_qp_ecmp_muxi.sh — QP 扫描排查沐曦 ECMP 极化（无 AFS，muxi-eval-copy）
#
# 背景: reports/rounds/muxi_fabric_final_20260719.md §4 点名的下一步实验——
# 转 MCCL_IB_QPS_PER_CONNECTION 旋钮，看 AllReduce@256M 带宽是否随 QP 上升。
# 若上升即 ECMP 哈希极化的强端侧证据（多子流=多 UDP src port=打散上行），
# 绕开此前卡死的交换机 counter 死结。
#
# 目标作业 muxi-eval-copy 没挂 AFS: 分发=内联灌 pod /tmp; 完成=pod 内 marker
# + 跳板 fanout 轮询; 收集=逐 pod tar 回跳板→本机; 聚合/校验在本机。
#
# 纪律（AGENTS.md）:
#   - 集群访问 yinjinrun.p，经跳板 ais-cf3e61a5 + vcctl；禁止 Mac 直发多路 exec
#   - 落盘一律 yinjinrun.p；本作业无 AFS → 回拉本机 results/muxi-h3c/<run_id>/
#   - 跳过所有 BUSY 节点并记 skipped_nodes.txt
#   - 偶发 SIGSEGV: 数据齐即 VALID（512 常见 post-destroy 崩）
#
# 用法:
#   NNODES=2 QP_ARMS="default,1,2,4,8,16" MODE=all_reduce \
#     bash fire_qp_ecmp_muxi.sh
#   NNODES=32 REPEATS=3 QP_ARMS="default,1,2,4,8,16,32" \
#     SIZES=256M MODE=all_reduce bash fire_qp_ecmp_muxi.sh
#   NNODES=4 MODE=incast QP_ARMS="1,16" INCAST_SIZES=16M bash fire_qp_ecmp_muxi.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# ---- 执行模式 ----
# ON_JUMP=1: driver 已在跳板上跑，pod exec 走本地 vcctl（零 Mac↔跳板 ssh churn）。
#            这是大规模(32/64节点)唯一稳的方式；Mac 侧启动器 fire_qp_ecmp_on_jump.sh 负责
#            scp 代码上跳板 + nohup 起本脚本 + 回拉结果。
# 否则(Mac 直驱): 走 ssh+ControlMaster 多路复用（仅适合小规模 pilot）。
ON_JUMP="${ON_JUMP:-0}"
export CLUSTER_JOB_OVERRIDE="${CLUSTER_JOB_OVERRIDE:-muxi-eval-copy}"
export CLUSTER_KUBECONFIG_OVERRIDE="${CLUSTER_KUBECONFIG_OVERRIDE:-/root/.kube/config-vc-c550-h3c-test.yaml}"
if [[ "$ON_JUMP" == "1" ]]; then
  export CLUSTER_EXEC_MODE=vcctl_local
  SHA_CMD="sha256sum"
else
  export CLUSTER_EXEC_MODE=vcctl
  export CLUSTER_SSH_CONTROL_PATH="${CLUSTER_SSH_CONTROL_PATH:-/tmp/qp-ecmp-cm-%r-%h.sock}"
  SHA_CMD="shasum -a 256"
fi
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"
source "$SCRIPT_DIR/jump_stage_lib.sh"
# job_helpers.sh 带 set -e；本驱动要让单个 arm 失败非致命，复位关掉 errexit。
set +e

# ---- 实验参数 ----
JOB="${CLUSTER_JOB}"                       # muxi-eval-copy
NPROC="${DEVICES_PER_NODE:-8}"
NNODES="${NNODES:?set NNODES (节点数; world=NNODES*8)}"
WORLD=$((NNODES * NPROC))
MODE="${MODE:-all_reduce}"                 # all_reduce | incast
QP_ARMS="${QP_ARMS:-default,1,2,4,8,16}"   # 逗号分隔; default=不 export
REPEATS="${REPEATS:-1}"
SIZES="${SIZES:-256M}"                      # all_reduce 主口径
INCAST_SIZES="${INCAST_SIZES:-16M}"        # incast per-sender（root 需 N×size）
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
MCCL_DEBUG_LEVEL="${MCCL_DEBUG:-INFO}"     # pilot 用 INFO 发现 QP 打印文本
BASE_PORT="${MASTER_PORT:-29701}"
# 复用单条 ssh 后并发可稍高；默认 8（过高仍可能挤爆多路复用通道）
STAGE_FANOUT_PARALLEL="${STAGE_FANOUT_PARALLEL:-8}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-qp-ecmp-w${WORLD}-${STAMP}}"
if [[ "$ON_JUMP" == "1" ]]; then
  # 跳板本地结果目录；Mac 侧启动器跑完 rsync 回 myportal/results/muxi-h3c/
  LOCAL_ROOT="${LOCAL_ROOT:-/tmp/qp-ecmp-results/$RUN_ID}"
else
  LOCAL_ROOT="${LOCAL_ROOT:-$REPO_ROOT/results/muxi-h3c/$RUN_ID}"
fi
JUMP_WORK="/tmp/$RUN_ID"                    # 跳板侧 bundle 暂存
POD_RUN_BASE="/tmp/$RUN_ID"                 # pod 内工作目录

mkdir -p "$LOCAL_ROOT"/{scan,raw,logs,qp_evidence}
exec > >(tee -a "$LOCAL_ROOT/driver.log") 2>&1
echo "=== QP-ECMP DRIVER run_id=$RUN_ID world=$WORLD nnodes=$NNODES mode=$MODE ==="
echo "QP_ARMS=$QP_ARMS repeats=$REPEATS sizes=$SIZES incast_sizes=$INCAST_SIZES debug=$MCCL_DEBUG_LEVEL"
echo "job=$JOB parallel=$STAGE_FANOUT_PARALLEL local_root=$LOCAL_ROOT"

# 建立执行通道 + 退出清理。ON_JUMP 走本地 vcctl（无 ssh 主连接）；否则建 ssh 复用。
if [[ "$ON_JUMP" == "1" ]]; then
  cleanup_driver() { rm -rf "$JUMP_WORK" 2>/dev/null || true; }
  trap cleanup_driver EXIT
  echo "执行模式: ON_JUMP (本地 vcctl，零 ssh churn)"
else
  cluster_ssh_mux_start || { echo "FATAL: 无法建立跳板 ssh 主连接"; exit 5; }
  cleanup_driver() {
    cluster_ssh "rm -rf '$JUMP_WORK'" 2>/dev/null || true
    cluster_ssh_mux_stop
  }
  trap cleanup_driver EXIT
  echo "执行模式: Mac 直驱 + ssh 多路复用 $CLUSTER_SSH_CONTROL_PATH"
fi

# =====================================================================
# 1. 忙闲全扫（回应"跳过在跑的卡"）：只 select 不 gate
# =====================================================================
scan_one() {
  local pod="$1" _attempt="$2" stage_dir="$3"
  local san; san="$(sanitize_pod "$pod")"
  local state
  # ps 抓训练进程 + mx-smi 查 GPU 忙。BUSY/IDLE 都是有效分类（return 0）；
  # 只有探测本身无输出（ssh/exec 失败）才 return 1 触发重试。
  state="$(cluster_pod_exec "$pod" '
    hot=$(ps -eo comm,pcpu --sort=-pcpu 2>/dev/null | awk "NR>1 && \$2>50" | grep -Ei "python|torchrun|megatron|pretrain" | head -1 | awk "{print \$1}")
    util=$(mx-smi 2>/dev/null | grep -oE "[0-9]+%" | tr -d "%" | sort -rn | head -1)
    if [[ -n "$hot" || "${util:-0}" -ge 20 ]]; then echo "BUSY:proc=${hot:-none},util=${util:-0}"; else echo "IDLE:util=${util:-0}"; fi
  ' 2>/dev/null | tr -d "\r" | tail -1)"
  if [[ -z "$state" ]]; then
    printf '%s\t%s\n' "$pod" "UNKNOWN" >"$stage_dir/$san.state"
    return 1   # 探测失败 → 触发 staged_fanout 重试
  fi
  printf '%s\t%s\n' "$pod" "$state" >"$stage_dir/$san.state"
  return 0
}

echo "--- [1/5] 忙闲全扫 ---"
ALL_PODS=()
while IFS= read -r _p; do [[ -n "$_p" ]] && ALL_PODS+=("$_p"); done < <(cluster_pods_running)
if [[ "${#ALL_PODS[@]}" -eq 0 ]]; then echo "FATAL: 没列到任何 Running pod（检查跳板/kubeconfig/job 名）"; exit 2; fi
echo "Running pod 总数=${#ALL_PODS[@]}"
SCAN_DIR="$LOCAL_ROOT/scan"
# 扫描给 2 次尝试：mux 下仍偶发单点超时，重试一次再归类
staged_fanout scan scan_one 2 "$SCAN_DIR" "${ALL_PODS[@]}" || true

IDLE_PODS=(); BUSY_PODS=()
: >"$LOCAL_ROOT/scan/idle_list.txt"; : >"$LOCAL_ROOT/scan/skipped_nodes.txt"
for pod in "${ALL_PODS[@]}"; do
  san="$(sanitize_pod "$pod")"
  st="$(cut -f2 "$SCAN_DIR/$san.state" 2>/dev/null || echo UNKNOWN)"
  if [[ "$st" == IDLE* ]]; then
    IDLE_PODS+=("$pod"); echo "$pod" >>"$LOCAL_ROOT/scan/idle_list.txt"
  else
    BUSY_PODS+=("$pod"); printf '%s\t%s\n' "$pod" "$st" >>"$LOCAL_ROOT/scan/skipped_nodes.txt"
  fi
done
echo "IDLE=${#IDLE_PODS[@]} BUSY/skip=${#BUSY_PODS[@]}"
if [[ "${#IDLE_PODS[@]}" -lt "$NNODES" ]]; then
  echo "FATAL: 空闲节点 ${#IDLE_PODS[@]} < 需要 $NNODES；跳过清单见 scan/skipped_nodes.txt"; exit 3
fi

# 选点：优先 master-0 做 rendezvous（FQDN 最稳），否则用 idle[0]。
SELECTED=("${IDLE_PODS[@]:0:$NNODES}")
RENDEZVOUS_POD=""
for p in "${SELECTED[@]}"; do
  if [[ "$p" == "${JOB}-master-0" ]]; then RENDEZVOUS_POD="$p"; break; fi
done
[[ -z "$RENDEZVOUS_POD" ]] && RENDEZVOUS_POD="${SELECTED[0]}"
MASTER_ADDR="${RENDEZVOUS_POD}.${JOB}"     # pod FQDN（torchrun rendezvous）
echo "选中 $NNODES 台，rendezvous=$RENDEZVOUS_POD master_addr=$MASTER_ADDR"
printf '%s\n' "${SELECTED[@]}" >"$LOCAL_ROOT/scan/selected_pods.txt"
# 记录多出来未选的空闲节点（也算被"跳过"）
for p in "${IDLE_PODS[@]:$NNODES}"; do printf '%s\tIDLE:unselected\n' "$p" >>"$LOCAL_ROOT/scan/skipped_nodes.txt"; done

# =====================================================================
# 2. 打包通用 bundle（rank 无关；字节对所有 pod 一致）
# =====================================================================
BUNDLE="$JUMP_WORK/bundle.tar"
build_bundle_local() {
  local tmpd; tmpd="$(mktemp -d)"
  cp "$SCRIPT_DIR/nccl_torch_bench.py" "$SCRIPT_DIR/nccl_torch_bench_metrics.py" \
     "$SCRIPT_DIR/nccl_p2p_bench.py" "$tmpd/"
  ( cd "$tmpd" && tar -cf "$LOCAL_ROOT/bundle.tar" . )
  rm -rf "$tmpd"
}
echo "--- [2/5] 打包 bundle + 分发到 pod ---"
build_bundle_local
BUNDLE_SHA="$($SHA_CMD "$LOCAL_ROOT/bundle.tar" | awk '{print $1}')"
echo "bundle sha256=$BUNDLE_SHA"
if [[ "$ON_JUMP" == "1" ]]; then
  mkdir -p "$JUMP_WORK"; cp -f "$LOCAL_ROOT/bundle.tar" "$BUNDLE"
else
  cluster_ssh "mkdir -p '$JUMP_WORK'"
  cat "$LOCAL_ROOT/bundle.tar" | cluster_ssh "cat > '$BUNDLE'"
fi

# 每个选中 pod：建 code 目录并解包 bundle（幂等，QP 各臂共用同一份代码）。
# 两模式统一走 cluster_pod_exec_i(stdin=bundle):
#   ON_JUMP → vcctl_local 纯本地管道，零 ssh；
#   Mac 直驱 → cluster_pod_exec_i 经 ssh 复用（bundle 已在跳板，cat 也在跳板本地）。
upload_one() {
  local pod="$1" _attempt="$2" stage_dir="$3"
  local san; san="$(sanitize_pod "$pod")"
  local remote_cmd="mkdir -p '$POD_RUN_BASE/code' && tar -C '$POD_RUN_BASE/code' -xf - && cd '$POD_RUN_BASE/code' && ls nccl_torch_bench.py nccl_p2p_bench.py >/dev/null && echo UPLOAD_OK"
  if [[ "$ON_JUMP" == "1" ]]; then
    cat "$BUNDLE" | cluster_pod_exec_i "$pod" "$remote_cmd" \
      >"$stage_dir/$san.attempt${_attempt}.log" 2>&1
  else
    # Mac 直驱: 整条 cat|vcctl exec 在跳板本地跑，Mac 只发一条 ssh（bundle 不回 Mac）。
    cluster_ssh "cat '$BUNDLE' | $(_cluster_vcctl_prefix) pod exec -i ${pod} -- bash -c $(printf '%q' "$remote_cmd")" \
      >"$stage_dir/$san.attempt${_attempt}.log" 2>&1
  fi
}
echo "--- 分发 bundle 到 $NNODES 台 ---"
staged_fanout upload upload_one 3 "$LOCAL_ROOT/logs/upload" "${SELECTED[@]}"
if [[ "${#STAGED_FANOUT_FAIL_PODS[@]}" -gt 0 ]]; then
  echo "FATAL: bundle 上传失败: ${STAGED_FANOUT_FAIL_PODS[*]}"; exit 4
fi

# =====================================================================
# 3. 单个 (QP,rep,mode) run：fire → poll → collect → cleanup
# =====================================================================
# 全局(每 run 重设): RUN_SELECTED 顺序即 node_rank；node_rank_of 查下标
declare -a RUN_SELECTED
run_one_arm() {
  local qp="$1" rep="$2" port="$3"
  local tag="qp${qp}/rep${rep}"
  local pod_out="$POD_RUN_BASE/$tag"          # pod 内本 run 输出
  local local_raw="$LOCAL_ROOT/raw/$tag"
  local local_log="$LOCAL_ROOT/logs/$tag"
  mkdir -p "$local_raw" "$local_log"
  echo "  >>> ARM $tag mode=$MODE port=$port qp=$qp <<<"

  local qp_export=""
  if [[ "$qp" != "default" ]]; then
    qp_export="export MCCL_IB_QPS_PER_CONNECTION=$qp NCCL_IB_QPS_PER_CONNECTION=$qp"
  fi

  # bench 命令行随 mode 切换
  local bench_cmd
  if [[ "$MODE" == "incast" ]]; then
    bench_cmd="/tmp/nccl_p2p_bench.py --mode incast --incast-root 0 --sizes '$INCAST_SIZES' --warmup $WARMUP --iters $ITERS --out '$pod_out/ranks/p2p.jsonl'"
  else
    bench_cmd="/tmp/nccl_torch_bench.py --ops '$MODE' --sizes '$SIZES' --warmup $WARMUP --iters $ITERS --out '$pod_out/ranks/scale.jsonl'"
  fi

  # 3a. fire：每 pod 按其 node_rank 起 torchrun，写 marker
  fire_one() {
    local pod="$1" _attempt="$2" stage_dir="$3"
    local san; san="$(sanitize_pod "$pod")"
    local nr; nr="$(node_rank_of "$pod")"
    local launch="$pod_out/launch.sh"
    # rank 无关脚本；node_rank/QP/addr 用变量注入
    local body
    body=$(cat <<LAUNCH
#!/usr/bin/env bash
export PATH="/opt/conda/bin:\${PATH:-/usr/bin}"
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0 MCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA='${NCCL_IB_HCA}' MCCL_IB_HCA='${MCCL_IB_HCA}'
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} MCCL_IB_GID_INDEX=${MCCL_IB_GID_INDEX} MCCL_IB_TC=${MCCL_IB_TC}
export MCCL_ENABLE_VSWITCH=${MCCL_ENABLE_VSWITCH} MCCL_PCIE_BUFFER_MODE=${MCCL_PCIE_BUFFER_MODE} FORCE_ACTIVE_WAIT=2
export NCCL_DEBUG=${MCCL_DEBUG_LEVEL} MCCL_DEBUG=${MCCL_DEBUG_LEVEL}
${qp_export}
mkdir -p '$pod_out/ranks'
rm -f '$pod_out/node_${nr}.done' '$pod_out/node_${nr}.fail'
cp -f '$POD_RUN_BASE/code/nccl_torch_bench.py' /tmp/nccl_torch_bench.py
cp -f '$POD_RUN_BASE/code/nccl_torch_bench_metrics.py' /tmp/nccl_torch_bench_metrics.py
cp -f '$POD_RUN_BASE/code/nccl_p2p_bench.py' /tmp/nccl_p2p_bench.py
/opt/conda/bin/torchrun --nnodes=$NNODES --node_rank=${nr} --nproc_per_node=$NPROC \\
  --master_addr='$MASTER_ADDR' --master_port=$port \\
  $bench_cmd >'$pod_out/node_${nr}.log' 2>&1
rc=\$?
# 512 常见 post-destroy SIGSEGV：数据齐即 VALID → marker 同时记 DATA 状态
data=MISSING
if grep -q 'wrote .*rank' '$pod_out/node_${nr}.log' 2>/dev/null; then data=OK; fi
if [[ \$rc -eq 0 && \$data == OK ]]; then
  printf 'RC=%s DATA=%s\n' "\$rc" "\$data" >'$pod_out/node_${nr}.done'
else
  printf 'RC=%s DATA=%s\n' "\$rc" "\$data" >'$pod_out/node_${nr}.fail'
fi
exit \$rc
LAUNCH
)
    # 一条 channel 搞定：写 launch.sh + chmod（同步）→ 再 setsid nohup 后台启动。
    # 注意分组: cat/chmod 必须同步完成，只 background torchrun 那步。
    printf '%s\n' "$body" | cluster_pod_exec_i "$pod" \
      "mkdir -p '$pod_out' && cat > '$launch' && chmod +x '$launch' && { setsid nohup bash '$launch' </dev/null >/dev/null 2>&1 & } && echo STARTED \$!" \
      >"$stage_dir/$san.attempt${_attempt}.log" 2>&1
  }
  staged_fanout "fire_$tag" fire_one 3 "$local_log/fire" "${RUN_SELECTED[@]}"
  if [[ "${#STAGED_FANOUT_FAIL_PODS[@]}" -gt 0 ]]; then
    echo "  ARM $tag FIRE_FAIL: ${STAGED_FANOUT_FAIL_PODS[*]}"; return 1
  fi

  # 3b. poll：完成判据 = pod 内已写出的 rank jsonl 总数达 WORLD（数据齐即完成），
  # 而非"全部节点写 done marker"。原因: bench 先落盘再 barrier/destroy_process_group，
  # 大规模常在 destroy 阶段 hang/SIGSEGV（07-19 报告的 post-destroy 现象），
  # 那些节点数据已齐但走不到写 marker → 靠 marker 判完成会误判 STALL。
  local per_pod_jsonl=$NPROC
  local deadline=$(( $(date +%s) + ${ARM_TIMEOUT:-1200} ))
  local last_data=-1 last_progress; last_progress="$(date +%s)"
  while true; do
    poll_one() {
      local pod="$1" _a="$2" sd="$3"; local san; san="$(sanitize_pod "$pod")"; local nr; nr="$(node_rank_of "$pod")"
      # 报: 本 pod 已写 rank jsonl 数 | marker 状态(DONE/FAIL/WAIT)
      cluster_pod_exec "$pod" "
        j=\$(ls '$pod_out/ranks/'*.rank*.jsonl 2>/dev/null | wc -l | tr -d ' ')
        if [[ -f '$pod_out/node_${nr}.done' ]]; then m=DONE; elif [[ -f '$pod_out/node_${nr}.fail' ]]; then m=FAIL; else m=WAIT; fi
        echo \"\$j \$m\"" 2>/dev/null | tr -d "\r" | tail -1 >"$sd/$san.st"
      return 0
    }
    staged_fanout "poll_$tag" poll_one 1 "$local_log/poll" "${RUN_SELECTED[@]}" >/dev/null 2>&1 || true
    local data_n=0 done_n=0 fail_n=0
    for pod in "${RUN_SELECTED[@]}"; do
      san="$(sanitize_pod "$pod")"; st="$(cat "$local_log/poll/$san.st" 2>/dev/null || echo '0 WAIT')"
      local j m; j="$(awk '{print $1}' <<<"$st")"; m="$(awk '{print $2}' <<<"$st")"
      [[ "$j" =~ ^[0-9]+$ ]] || j=0
      data_n=$((data_n + j))
      [[ "$m" == DONE ]] && done_n=$((done_n+1)); [[ "$m" == FAIL ]] && fail_n=$((fail_n+1))
    done
    local now; now="$(date +%s)"
    [[ "$data_n" -gt "$last_data" ]] && last_progress="$now"
    echo "    poll $tag jsonl=$data_n/$WORLD markers(done=$done_n fail=$fail_n)/$NNODES"
    # 完成: 数据齐（首选）或 全部 marker terminal
    if [[ "$data_n" -ge "$WORLD" ]]; then echo "    ARM $tag DATA_COMPLETE ($data_n/$WORLD)"; break; fi
    if [[ $((done_n + fail_n)) -eq "$NNODES" ]]; then echo "    ARM $tag ALL_MARKERS_TERMINAL"; break; fi
    if [[ "$now" -ge "$deadline" ]]; then echo "    ARM $tag TIMEOUT(hard)"; break; fi
    if [[ $((now - last_progress)) -ge "${ARM_STALL:-300}" ]]; then echo "    ARM $tag STALL(${ARM_STALL:-300}s 无进展)"; break; fi
    last_data="$data_n"; sleep 10
  done

  # 3c. collect：逐 pod tar 回本机（ranks + 日志 + marker）
  collect_one() {
    local pod="$1" _a="$2" sd="$3"; local san; san="$(sanitize_pod "$pod")"
    cluster_pod_exec_i "$pod" "cd '$pod_out' && tar -cf - ranks node_*.log node_*.done node_*.fail 2>/dev/null || true" \
      > "$sd/$san.tar" 2>>"$sd/$san.err"
    [[ -s "$sd/$san.tar" ]] && tar -C "$local_raw" -xf "$sd/$san.tar" 2>/dev/null || true
    return 0
  }
  mkdir -p "$local_log/collect"
  staged_fanout "collect_$tag" collect_one 2 "$local_log/collect" "${RUN_SELECTED[@]}" >/dev/null 2>&1 || true
  # tar 内容: ranks/rankN.jsonl（rank 结果）+ 顶层 node_*.log/.done/.fail。
  # 摊平 rank jsonl 到 raw/$tag；node 日志留在 raw/$tag 顶层。
  find "$local_raw/ranks" -name '*.rank*.jsonl' -exec mv -f {} "$local_raw/" \; 2>/dev/null || true
  local nfiles; nfiles="$(find "$local_raw" -name '*.rank*.jsonl' 2>/dev/null | wc -l | tr -d ' ')"
  echo "  ARM $tag collected rank_files=$nfiles (expect $WORLD)"

  # QP 生效证据：node 日志在 raw/$tag 顶层（node_0.log = rendezvous rank 日志）
  grep -hE 'QPS_PER_CONNECTION set by environment|MCCL_IB_HCA set to|NET/IB : Using' \
    "$local_raw"/node_*.log 2>/dev/null \
    | sort -u | head -60 >"$LOCAL_ROOT/qp_evidence/${qp}_rep${rep}.txt" || true

  # 3d. cleanup：kill 本 run 进程（含 hang 在 destroy 的 torchrun/python）+ 删本 run 目录。
  # 大规模 post-destroy hang 会留下不退出的 torchrun/python，占 GPU/端口，必须清干净
  # 否则下一臂 rendezvous 冲突。保留 code/ 供下一臂复用。
  cleanup_one() {
    local pod="$1" _a="$2" sd="$3"
    cluster_pod_exec "$pod" "
      pkill -9 -f '$pod_out/launch.sh' 2>/dev/null
      pkill -9 -f 'master_port=$port' 2>/dev/null
      pkill -9 -f 'nccl_torch_bench.py' 2>/dev/null
      pkill -9 -f 'nccl_p2p_bench.py' 2>/dev/null
      pkill -9 -f 'torchrun' 2>/dev/null
      rm -rf '$pod_out' 2>/dev/null
      echo CLEAN" >/dev/null 2>&1 || true
    return 0
  }
  mkdir -p "$local_log/cleanup"
  staged_fanout "cleanup_$tag" cleanup_one 1 "$local_log/cleanup" "${RUN_SELECTED[@]}" >/dev/null 2>&1 || true
  sleep 5   # 给 GPU/端口释放留窗口，再进下一臂
  return 0
}

# =====================================================================
# 4. 外层 QP × rep 编排
# =====================================================================
echo "--- [3/5] QP 扫描编排 ---"
RUN_SELECTED=("${SELECTED[@]}")
# bash 3.2 无关联数组：node_rank = pod 在 RUN_SELECTED 里的下标
node_rank_of() {
  local target="$1" idx=0 p
  for p in "${RUN_SELECTED[@]}"; do
    [[ "$p" == "$target" ]] && { printf '%s' "$idx"; return 0; }
    idx=$((idx+1))
  done
  printf '%s' -1; return 1
}

IFS=',' read -r -a ARMS <<<"$QP_ARMS"
port=$BASE_PORT
for qp in "${ARMS[@]}"; do
  for rep in $(seq 1 "$REPEATS"); do
    run_one_arm "$qp" "$rep" "$port" || echo "  (arm qp=$qp rep=$rep 非致命失败，继续)"
    port=$((port + 1))
  done
done

# =====================================================================
# 5. 本机聚合 + 出 SUMMARY
# =====================================================================
echo "--- [4/5] 本机聚合 ---"
python3 "$SCRIPT_DIR/aggregate_qp_ecmp.py" \
  --root "$LOCAL_ROOT" --world "$WORLD" --mode "$MODE" \
  --arms "$QP_ARMS" --repeats "$REPEATS" \
  || echo "(聚合脚本非零退出，raw 数据已在 $LOCAL_ROOT/raw)"

echo "--- [5/5] 完成（跳板工作目录 + ssh 主连接由 EXIT trap 清理）---"
echo "=== DONE run_id=$RUN_ID → $LOCAL_ROOT ==="
echo "SUMMARY: $LOCAL_ROOT/SUMMARY.md ; 跳过节点: $LOCAL_ROOT/scan/skipped_nodes.txt"
