#!/usr/bin/env bash
# run_case_pipeline_v4.sh — 单 case pipeline(参数化规模 + 多组端口隔离 + freq 时序分支)
#
# 相对 v3 的改造:
#   - 参数化 NNODES/NPROC(支持 16/64/128 rank), 不再硬编码 NNODES=2
#   - POD_EXEC 抽象: raw 特权 pod 用 kubectl(vcjob 被 RBAC 封), 兼容 vcctl
#   - 多组端口隔离: BASE_PORT 由 GROUP_ID 决定不相交端口块
#   - 默认结果写每台 pod 本地 /workspace/probe-bundle/out，完成后立即回拉
#   - freq case 独立时序: fire 前设低档 → 跑 → 恢复(绝不 kernel 运行中改档)
#   - master_addr 动态解析 raw pod IP(无 vcjob 命名)
#
# 用法(env 驱动):
#   CASE=3a INJECT_KIND=cube INJECT_ARGS="duty=0.3" GROUP_ID=0 \
#   PODS="pod0,pod1" NNODES=2 NPROC=8 ROUNDS=3 MODE=gpu_bound \
#   LOCAL_CODE=/workspace/probe-bundle LOCAL_OUT=/workspace/probe-bundle/out \
#   KUBECONFIG=~/.kube/config-vc-c550-h3c-test.yaml \
#   bash run_case_pipeline_v4.sh
set -uo pipefail

# ===== 参数 =====
CASE="${CASE:?need CASE}"
INJECT_KIND="${INJECT_KIND:?need INJECT_KIND (cube|hbm|1b|1c|2ext|freq|stress_cpu|stress_vm|stress_io|none)}"
INJECT_ARGS="${INJECT_ARGS:-}"
GROUP_ID="${GROUP_ID:-0}"
IFS=',' read -r -a PODS <<< "${PODS:?need PODS csv}"
NNODES="${NNODES:-2}"
NPROC="${NPROC:-8}"
ROUNDS="${ROUNDS:-3}"
ITERS="${ITERS:-500}"
WARMUP="${WARMUP:-50}"
SEED="${SEED:-42}"
MODE="${MODE:-gpu_bound}"
MODEL="${MODEL:-gpt2}"
SEQ="${SEQ:-1024}"
BATCH="${BATCH:-8}"
FREQ_LEVEL="${FREQ_LEVEL:-4}"          # freq case: xcore 档 (0-9), 4≈1066MHz
LOCAL_CODE="${LOCAL_CODE:-/workspace/probe-bundle}"
LOCAL_OUT="${LOCAL_OUT:-/workspace/probe-bundle/out}"
# RUN_DIR 是结果根目录。兼容旧 AFS_RUN_DIR，但未设置时绝不要求 AFS。
RUN_DIR="${RUN_DIR:-${AFS_RUN_DIR:-$LOCAL_OUT}}"
CODE_DIR="${CODE_DIR:-${LOCAL_CODE}}"
RUN_ID="${RUN_ID:-$(basename "$RUN_DIR")}"
LOCAL_FS="${LOCAL_FS:-1}"
NS="${NS:-default}"
PROBING_SPEC="${PROBING_SPEC:-}"       # Line B: 传给 C2 的 PROBING_TORCH_PROFILING spec
SIDECAR_LOCAL_RANK="${SIDECAR_LOCAL_RANK:-$((NPROC-1))}"
SIDECAR_WARMUP="${SIDECAR_WARMUP:-8}"
# measure step 100/300 对应 WARMUP=50 时的全局 step 150--350。
INJECT_START_MEASURE_STEP="${INJECT_START_MEASURE_STEP:-100}"
INJECT_STOP_MEASURE_STEP="${INJECT_STOP_MEASURE_STEP:-300}"
IO_STRESS_DIR="${IO_STRESS_DIR:-/workspace/probe-bundle/io_stress}"
CKPT_DIR="${CKPT_DIR:-/workspace/probe-bundle/ckpt}"

# POD_EXEC 抽象: 默认 kubectl(raw pod); 设 USE_VCCTL=1 切 vcctl(vcjob pod)
if [ "${USE_VCCTL:-0}" = "1" ]; then
  VCCTL="${VCCTL:-/usr/local/bin/vcctl}"
  pexec()   { "$VCCTL" pod exec    "$1" -- bash -c "$2"; }
  pexec_i() { "$VCCTL" pod exec -i "$1" -- bash -c "$2"; }
  pod_ip()  { "$VCCTL" pod view "$1" 2>/dev/null | grep -oE 'IP[: ]+[0-9.]+' | grep -oE '[0-9.]+' | head -1; }
else
  KC="${KUBECONFIG:?need KUBECONFIG for kubectl}"
  pexec()   { kubectl --kubeconfig="$KC" -n "$NS" exec    "$1" -- bash -c "$2"; }
  pexec_i() { kubectl --kubeconfig="$KC" -n "$NS" exec -i "$1" -- bash -c "$2"; }
  pod_ip()  { kubectl --kubeconfig="$KC" -n "$NS" get pod "$1" -o jsonpath='{.status.podIP}' 2>/dev/null; }
fi

DUTY=$(echo "$INJECT_ARGS" | grep -oE 'duty=[0-9.]+' | cut -d= -f2 || echo "0.3")
SIZE=$(echo "$INJECT_ARGS" | grep -oE 'size=[0-9]+'  | cut -d= -f2 || echo "4096")
FRAC=$(echo "$INJECT_ARGS" | grep -oE 'frac=[0-9.]+' | cut -d= -f2 || echo "0.7")
CPU_LOAD=$(echo "$INJECT_ARGS" | grep -oE 'cpu_load=[0-9.]+' | cut -d= -f2 || echo "90")
CPU_N=$(echo "$INJECT_ARGS" | grep -oE 'cpu_n=[0-9]+' | cut -d= -f2 || echo "")
# cpu_frac=0.5 → nproc/2（Quiet）
CPU_FRAC=$(echo "$INJECT_ARGS" | grep -oE 'cpu_frac=[0-9.]+' | cut -d= -f2 || echo "")

MASTER="${PODS[0]}"
MASTER_IP="$(pod_ip "$MASTER")"
[ -z "$MASTER_IP" ] && { echo "FATAL: cannot resolve master IP for $MASTER"; exit 2; }
# 多组端口隔离: 每组 100 端口块
BASE_PORT=$(( 30000 + GROUP_ID * 100 ))
OUT_BASE="$RUN_DIR/$CASE"

echo "╔══════════════════════════════════════════════╗"
echo "║ v4 case=$CASE grp=$GROUP_ID inject=$INJECT_KIND mode=$MODE"
echo "║ pods=${PODS[*]} NNODES=$NNODES NPROC=$NPROC world=$((NNODES*NPROC))"
echo "║ master=$MASTER($MASTER_IP) base_port=$BASE_PORT rounds=$ROUNDS"
echo "║ code=$CODE_DIR out=$OUT_BASE local_fs=$LOCAL_FS run_id=$RUN_ID"
echo "╚══════════════════════════════════════════════╝"

# ===== helpers =====
clean_group() {
  # kubectl exec + pkill 常返回 137；在 set -e 下必须吞掉，否则战役会在 fire 前静默退出
  # 同时释放本组端口块，避免 EADDRINUSE（上轮残留 store）
  local p0=$BASE_PORT p1=$((BASE_PORT+1)) p2=$((BASE_PORT+2))
  for ((n=0; n<NNODES; n++)); do
    pexec "${PODS[$n]}" "pkill -9 -f train_bench_probe 2>/dev/null || true; pkill -9 -f /tmp/tbp.py 2>/dev/null || true; pkill -9 -f torchrun 2>/dev/null || true; pkill -9 -f sidecar_inject 2>/dev/null || true; pkill -9 -f stress-ng 2>/dev/null || true; pkill -9 -f 'fio.*io_stress' 2>/dev/null || true; pkill -9 -f ib_write_bw 2>/dev/null || true; fuser -k ${p0}/tcp ${p1}/tcp ${p2}/tcp 2>/dev/null || true; sleep 1; exit 0" 2>/dev/null &
  done
  wait || true
  sleep 5
  return 0
}

# freq 分支: fire 前设档(privileged pod 需已 remount,rw /sys)
freq_set() {   # $1=level
  pexec "$MASTER" "mount -o remount,rw /sys 2>/dev/null; for i in \$(seq 0 $((NPROC-1))); do mx-smi -i \$i --set-dpm-max xcore,$1 >/dev/null 2>&1; done; echo FREQ_SET_$1" 2>/dev/null
}
freq_restore() {
  pexec "$MASTER" "for i in \$(seq 0 $((NPROC-1))); do mx-smi -i \$i --set-dpm-max xcore,9 >/dev/null 2>&1; done; echo FREQ_RESTORED" 2>/dev/null
}

fire_training() {   # $1=port $2=out_dir $3=detect_env $4=round
  local port="$1" out="$2" denv="$3" rnd="$4"
  for ((n=0; n<NNODES; n++)); do
    local launcher
    launcher=$(cat <<LAUNCHER
#!/usr/bin/env bash
export PATH=/opt/conda/bin:\${PATH:-/usr/bin}
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0 MCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=xscale_0,xscale_1,xscale_2,xscale_3
export MCCL_IB_HCA=xscale_0,xscale_1,xscale_2,xscale_3
export NCCL_IB_GID_INDEX=5 MCCL_IB_GID_INDEX=5 MCCL_IB_TC=128
export MCCL_ENABLE_VSWITCH=1
export NCCL_DEBUG=WARN MCCL_DEBUG=WARN
export PYTHONPATH=$CODE_DIR/pydeps:\${PYTHONPATH:-}
export CKPT_DIR=$CKPT_DIR
${denv}
rm -f '$out/node_${n}.done' '$out/node_${n}.fail'
# 清掉旧 ranks/marker，避免 warmup_done / step_*.marker 残留导致假就绪
rm -rf '$out/ranks'
mkdir -p '$out/ranks'
cp -f '$CODE_DIR/train_bench_probe.py' /tmp/tbp.py
/opt/conda/bin/torchrun --nnodes=$NNODES --nproc_per_node=$NPROC --node_rank=$n \\
  --master_addr=$MASTER_IP --master_port=$port \\
  /tmp/tbp.py --iters=$ITERS --warmup=$WARMUP --seed=$SEED --mode=$MODE --model=$MODEL --seq=$SEQ --batch=$BATCH \\
  --run-id=$RUN_ID --group=$GROUP_ID --config='$(basename "$out")' --round=$rnd \\
  --out-dir='$out/ranks' > '$out/node_${n}.log' 2>&1
rc=\$?
if [ \$rc -eq 0 ]; then touch '$out/node_${n}.done'; else echo \$rc > '$out/node_${n}.fail'; fi
LAUNCHER
)
    printf '%s' "$launcher" | pexec_i "${PODS[$n]}" "cat > /tmp/run_${GROUP_ID}.sh && chmod +x /tmp/run_${GROUP_ID}.sh" 2>/dev/null
  done
  # 全并行发射（kubectl/pkill 类退出码在 set -e 下必须吞掉，否则只打出第一个 ok 就整案退出）
  for ((n=0; n<NNODES; n++)); do
    pexec "${PODS[$n]}" "setsid nohup bash /tmp/run_${GROUP_ID}.sh </dev/null >/dev/null 2>&1 & echo ok; exit 0" 2>/dev/null &
  done
  wait || true
  return 0
}

wait_warmup() {   # $1=out_dir；rank0 位于 master，故 marker 在 master pod
  local out="$1" e=0
  while [ $e -lt 180 ]; do
    if pexec "$MASTER" "test -f '$out/ranks/warmup_done'" 2>/dev/null; then echo "  warmup ok(${e}s)"; return 0; fi
    sleep 5; e=$((e+5))
  done
  echo "  warmup timeout"; return 0
}

wait_measure_step() {  # $1=out_dir $2=measure step marker
  local out="$1" target="$2" e=0
  while [ "$e" -lt 1800 ]; do
    if pexec "$MASTER" "test -f '$out/ranks/step_${target}.marker'" 2>/dev/null; then
      echo "  measure step $target reached (${e}s)"
      return 0
    fi
    sleep 5; e=$((e+5))
  done
  echo "  measure step $target timeout"
  return 1
}

start_sidecar() {   # 在 victim(node0)起注入; freq / 内联 8a 不走这里
  local v="${PODS[0]}"
  case "$INJECT_KIND" in
    cube|hbm)
      pexec "$v" "CUDA_VISIBLE_DEVICES=$SIDECAR_LOCAL_RANK MACA_VISIBLE_DEVICES=$SIDECAR_LOCAL_RANK nohup /opt/conda/bin/python3.12 '$CODE_DIR/sidecar_inject.py' --kind '$INJECT_KIND' --duty '$DUTY' --warmup-seconds '$SIDECAR_WARMUP' --seconds 1800 --size '$SIZE' >'$out/injection.log' 2>&1 & echo SC=\$!" 2>/dev/null ;;
    1b|1c|2b|2c|3c|5b|8b|8c)
      pexec "$v" "CUDA_VISIBLE_DEVICES=$((NPROC-1)) nohup /opt/conda/bin/python3.12 $CODE_DIR/sidecar_inject_v2.py --case $INJECT_KIND --seconds 600 --frac $FRAC >/tmp/sc_${GROUP_ID}.log 2>&1 & echo SC=\$!" 2>/dev/null ;;
    stress_cpu)
      # Loud 默认全核 90%；Quiet/Masked 经 INJECT_ARGS: cpu_n / cpu_load / cpu_frac
      local ncpu="${CPU_N}" cl="${CPU_LOAD:-90}"
      if [ -z "$ncpu" ] && [ -n "$CPU_FRAC" ]; then
        local np
        np=$(pexec "$v" "nproc" 2>/dev/null | tr -d '[:space:]')
        ncpu=$(awk -v f="$CPU_FRAC" -v n="${np:-16}" 'BEGIN{v=int(n*f+0.5); if(v<1)v=1; print v}')
      fi
      if [ -z "$ncpu" ]; then
        pexec "$v" "nohup stress-ng --cpu \$(nproc) --cpu-load $cl --timeout 600s >'$out/injection.log' 2>&1 & echo SC=\$!" 2>/dev/null
      else
        pexec "$v" "nohup stress-ng --cpu $ncpu --cpu-load $cl --timeout 600s >'$out/injection.log' 2>&1 & echo SC=\$!" 2>/dev/null
      fi
      ;;
    stress_vm)
      pexec "$v" "nohup stress-ng --vm 4 --vm-bytes 2G --timeout 600s >'$out/injection.log' 2>&1 & echo SC=\$!" 2>/dev/null ;;
    stress_io)
      # Loud：fio 与训练/ckpt 同盘（pod 本地）
      pexec "$v" "mkdir -p '$IO_STRESS_DIR'; nohup fio --name=io_stress --rw=randrw --bs=4k --size=2G --numjobs=4 --time_based --runtime=600 --directory='$IO_STRESS_DIR' >'$out/injection.log' 2>&1 & echo SC=\$!" 2>/dev/null ;;
    8a|inline_8a|none) : ;;  # 8a 走训练进程 INLINE_INJECT
    *) echo "  WARN: unknown INJECT_KIND=$INJECT_KIND" ;;
  esac
}

# GPU sidecar 需在训练 measure 前预热(MetaX 时间片隔离; pilot 实测: 预热后 +214% vs 未预热 +3%)
is_gpu_sidecar() {
  case "$INJECT_KIND" in
    cube|hbm|1b|1c|2b|2c|3c|5b) return 0 ;;
    *) return 1 ;;
  esac
}
stop_sidecar() {
  pexec "${PODS[0]}" 'pkill -f sidecar_inject || true; pkill -f stress-ng || true; pkill -f "fio.*io_stress" || true; pkill -f ib_write_bw || true; exit 0' 2>/dev/null || true
  return 0
}

wait_done() {   # $1=out_dir $2=是否按 stop marker 停 sidecar
  # LOCAL_FS=1: 第 n 个 pod 只写 node_n.done → 必须到 PODS[n] 上查 node_n.done
  local out="$1" stop_on_marker="${2:-0}" stopped=0 e=0
  while [ $e -lt 900 ]; do
    local d=0 f=0
    if [ "$stop_on_marker" = "1" ] && [ "$stopped" = "0" ] &&
      pexec "$MASTER" "test -f '$out/ranks/step_${INJECT_STOP_MEASURE_STEP}.marker'" 2>/dev/null; then
      stop_sidecar
      stopped=1
      echo "  injection stopped at measure step $INJECT_STOP_MEASURE_STEP"
    fi
    if [ "${LOCAL_FS:-0}" = "1" ]; then
      n=0
      while [ "$n" -lt "$NNODES" ]; do
        if pexec "${PODS[$n]}" "test -f '$out/node_${n}.done'" >/dev/null 2>&1; then
          d=$((d + 1))
        elif pexec "${PODS[$n]}" "test -f '$out/node_${n}.fail'" >/dev/null 2>&1; then
          f=$((f + 1))
        fi
        n=$((n + 1))
      done
    else
      d=$(pexec "$MASTER" "ls '$out'/node_*.done 2>/dev/null | wc -l" 2>/dev/null | tr -d ' \n')
      f=$(pexec "$MASTER" "ls '$out'/node_*.fail 2>/dev/null | wc -l" 2>/dev/null | tr -d ' \n')
    fi
    if [ "${d:-0}" -ge "$NNODES" ] 2>/dev/null; then
      echo "  done markers $d/$NNODES (${e}s)"
      return 0
    fi
    if [ "${f:-0}" != "0" ] && [ "${f:-0}" != "" ]; then
      echo "  FAIL marker seen (f=$f)"
      return 1
    fi
    # 每 30s 打一行进度，避免静默卡住
    if [ $((e % 30)) -eq 0 ]; then
      echo "  waiting done… d=${d:-0}/$NNODES t=${e}s"
    fi
    sleep 5; e=$((e+5))
  done
  echo "  TIMEOUT(${e}s) d=${d:-0}/$NNODES"; return 1
}

# ===== configs (保留 C0-C4) — 用函数替代关联数组(兼容 bash 3.2) =====
CONFIGS=("C0_baseline" "C1_inject_none" "C2_probing" "C3_greyhound" "C4_xputimer")
config_denv() {   # $1=cfg → echo detect_env
  case "$1" in
    C0_baseline|C1_inject_none) echo "" ;;
    C2_probing)
      # D4：开 torch_trace + 强制 GPU 采样（PyPI 无 gpu feature 时表仍可能缺失，由 dump 如实记录）
      local torch_spec="${PROBING_SPEC:-on}"
      echo "export PROBING=2; export PROBING_TORCH_PROFILING='$torch_spec'; export PROBING_GPU=on; export PROBING_GPU_SAMPLE_MS=1000;"
      ;;
    C3_greyhound) echo "export LD_PRELOAD=$CODE_DIR/greyhound/libmcclprobe.so;" ;;
    C4_xputimer)  echo "export LD_PRELOAD=$CODE_DIR/xputimer/libxpu_timer_metax.so;" ;;
    *) echo "" ;;
  esac
}
config_has_inject() {   # $1=cfg → echo yes|no
  case "$1" in
    C0_baseline) echo "no" ;;
    *) echo "yes" ;;
  esac
}

# 允许只跑部分 config(Line B / 调试): CONFIGS_ONLY="C0_baseline,C2_probing"
if [ -n "${CONFIGS_ONLY:-}" ]; then IFS=',' read -r -a CONFIGS <<< "$CONFIGS_ONLY"; fi

# ===== main loop =====
IS_FREQ=0; [ "$INJECT_KIND" = "freq" ] && IS_FREQ=1
port=$BASE_PORT
pipe_rc=0
for r in $(seq 1 "$ROUNDS"); do
  echo ""; echo "══ Round $r/$ROUNDS ══"
  for cfg in "${CONFIGS[@]}"; do
    port=$((port+1))
    out="$OUT_BASE/round_${r}/${cfg}"
    echo "── [$cfg] r=$r port=$port ──"
    clean_group
    denv="$(config_denv "$cfg")"; inj="$(config_has_inject "$cfg")"
    # P3-SW-A：进程内联 8a（外挂 GC 无效）
    if [ "$inj" = "yes" ] && { [ "$INJECT_KIND" = "8a" ] || [ "$INJECT_KIND" = "inline_8a" ]; }; then
      denv="${denv}
export INLINE_INJECT=8a;
export INLINE_VICTIM_LOCAL_RANK=$SIDECAR_LOCAL_RANK;
export INLINE_INJECT_START=$INJECT_START_MEASURE_STEP;
export INLINE_INJECT_STOP=$INJECT_STOP_MEASURE_STEP;"
    fi

    if [ "$IS_FREQ" = "1" ] && [ "$inj" = "yes" ]; then
      # freq 分支: 先设低档再起 kernel(铁律: 绝不 kernel 运行中改档)
      freq_set "$FREQ_LEVEL"
      fire_training "$port" "$out" "$denv" "$r"; echo "  fired(freq=$FREQ_LEVEL)"
      wait_warmup "$out"
    else
      # 统一从训练中触发。WARMUP=50 时 measure 100/300 即总 step 150--350；
      # cube/hbm 自身再预热 >=5 秒，避免 MetaX 时间片隔离导致的伪阴性。
      fire_training "$port" "$out" "$denv" "$r"; echo "  fired"
      wait_warmup "$out"
      if [ "$inj" = "yes" ] && [ "$INJECT_KIND" != "none" ] && [ "$INJECT_KIND" != "8a" ] && [ "$INJECT_KIND" != "inline_8a" ]; then
        if wait_measure_step "$out" "$INJECT_START_MEASURE_STEP"; then
          start_sidecar; echo "  sidecar($INJECT_KIND) up on local_rank=$SIDECAR_LOCAL_RANK"
        else
          echo "  injection skipped: start marker unavailable"
        fi
      elif [ "$inj" = "yes" ] && { [ "$INJECT_KIND" = "8a" ] || [ "$INJECT_KIND" = "inline_8a" ]; }; then
        echo "  inline_8a armed (victim local_rank=$SIDECAR_LOCAL_RANK)"
      fi
    fi

    # C2：注入窗中段拉 Probing SQL（进程必须存活）
    if [ "$cfg" = "C2_probing" ] && [ "${DUMP_PROBING_SQL:-1}" = "1" ]; then
      dump_mid=$(( (INJECT_START_MEASURE_STEP + INJECT_STOP_MEASURE_STEP) / 2 ))
      HERE_PIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
      if wait_measure_step "$out" "$dump_mid"; then
        echo "  dumping Probing SQL at measure step $dump_mid …"
        if [ -f "$HERE_PIPE/dump_probing_sql.sh" ]; then
          pexec_i "$MASTER" "cat > '$CODE_DIR/dump_probing_sql.sh' && chmod +x '$CODE_DIR/dump_probing_sql.sh'" \
            < "$HERE_PIPE/dump_probing_sql.sh" 2>/dev/null || true
        fi
        pexec "$MASTER" \
          "OUT_DIR='$out' CASE='$CASE' CODE_DIR='$CODE_DIR' VICTIM_LOCAL_RANK='$SIDECAR_LOCAL_RANK' \
           bash '$CODE_DIR/dump_probing_sql.sh' >'$out/probing_dump.log' 2>&1; exit 0" 2>/dev/null || true
        echo "  SQL dump attempted → $out/probing/"
      else
        echo "  SQL dump skipped: mid-window marker unavailable"
      fi
    fi

    # wait_done 第二参是 0/1（是否按 inject-stop marker 停 sidecar），不是 yes/no
    # 内联 8a 无需外部 stop
    if [ "$inj" = "yes" ] && [ "$INJECT_KIND" != "8a" ] && [ "$INJECT_KIND" != "inline_8a" ]; then
      stop_flag=1
    else
      stop_flag=0
    fi
    if wait_done "$out" "$stop_flag"; then
      echo "  COMPLETE"
    else
      echo "  FAILED"
      pipe_rc=1
    fi
    stop_sidecar
    if [ "$IS_FREQ" = "1" ]; then freq_restore; fi
  done
done
echo ""; echo "╚═ v4 DONE case=$CASE grp=$GROUP_ID rc=$pipe_rc ═╝"
exit "$pipe_rc"
