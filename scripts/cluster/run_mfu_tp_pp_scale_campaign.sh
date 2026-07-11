#!/usr/bin/env bash
# 固定 TP×PP（+EP）只扩 DP 的 MFU 弱扩展战役
# 读 queue.jsonl → run_train_mfu_scale.sh → 解析 MFU → 写 ledger
# 不 recycle 到 TP1PP1；队列空则退出。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$OPS_ROOT/../.." && pwd)"

# shellcheck source=huawei.env
source "$SCRIPT_DIR/huawei.env"
# 本战役专用 job（clone 自 huawei-8node-copy）
export CLUSTER_JOB="${CLUSTER_JOB_OVERRIDE:-montyyin-mfu-scale}"
export CLUSTER_POD="${CLUSTER_JOB}-master-0"
export MASTER_ADDR="${MASTER_ADDR:-${CLUSTER_JOB}-master-0.${CLUSTER_JOB}}"

STATE_DIR="${STATE_DIR:-$OPS_ROOT/reports/rounds/mfu_tp_pp_scale_state}"
LEDGER="${LEDGER:-$OPS_ROOT/reports/rounds/mfu_tp_pp_scale_ledger.md}"
QUEUE="${QUEUE:-$STATE_DIR/queue.jsonl}"
PEAK="${PEAK:-292.79}"
PARSE="$SCRIPT_DIR/parse_train_mfu_log.py"
TARGET_GBS="${TARGET_GBS:-2048}"
ITERS_DEFAULT="${ITERS_DEFAULT:-5}"

mkdir -p "$STATE_DIR" "$REPO_ROOT/logs" "$OPS_ROOT/reports/rounds"

if [[ ! -f "$LEDGER" ]]; then
  cat > "$LEDGER" <<'L'
# 固定 TP×PP 扩 DP · MFU 弱扩展账本

> peak = 292.79 TFLOPS/卡 · 目标 GBS=2048 · SEQ=4096 · MBS=1  
> 设计见 `mfu_tp_pp_scale_plan_20260711.md`  
> Job：`montyyin-mfu-scale`

| round | id | mode | scale | TP/PP/EP | DP | GBS | steady TFLOP/s/GPU | MFU% | status | log | note |
|------:|----|------|------:|----------|---:|----:|-------------------:|-----:|--------|-----|------|
L
fi

if [[ ! -s "$QUEUE" ]]; then
  echo "FAIL: empty queue $QUEUE" >&2
  exit 2
fi

align_gbs() {
  local world="$1" tp="$2" pp="$3" cp="$4" mbs="$5" gbs="$6"
  local model_par=$((tp * pp * cp))
  if (( world % model_par != 0 )); then
    echo "INVALID" >&2
    return 1
  fi
  local dp=$((world / model_par))
  local unit=$((mbs * dp))
  if (( gbs % unit != 0 )); then
    gbs=$(( ((gbs + unit - 1) / unit) * unit ))
  fi
  echo "$gbs"
}

wait_pods_ready() {
  local deadline=$((SECONDS + ${WAIT_PODS_SEC:-7200}))
  echo "==> waiting for $CLUSTER_JOB pods Running (deadline ${WAIT_PODS_SEC:-7200}s)"
  while (( SECONDS < deadline )); do
    local n
    n="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER_SSH_HOST" \
      "KUBECONFIG=$CLUSTER_KUBECONFIG vcctl pod get --job $CLUSTER_JOB 2>/dev/null" \
      | awk 'NR>1 && $3=="Running" {c++} END{print c+0}')"
    echo "  running_pods=$n / 8  $(date -Iseconds)"
    if [[ "$n" -ge 8 ]]; then
      echo "==> pods ready"
      return 0
    fi
    sleep 60
  done
  echo "FAIL: pods not ready in time" >&2
  return 1
}

ROUND_FILE="$STATE_DIR/round_counter"
ROUND="$(cat "$ROUND_FILE" 2>/dev/null || echo 0)"

# 可选：等资源
if [[ "${WAIT_PODS:-1}" == "1" ]]; then
  wait_pods_ready
fi

while [[ -s "$QUEUE" ]]; do
  head -n 1 "$QUEUE" > "$STATE_DIR/current_job.json"
  tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"

  eval "$(python3 -c '
import json, shlex, sys
j = json.load(open(sys.argv[1]))
defaults = {"ep":1,"etp":1,"cp":1,"iters":5,"scales":"16","mode":"dense","mbs":1,"gbs":2048,"seq":4096,"note":""}
for k in ("id","mode","scales","tp","pp","mbs","gbs","seq","iters","note","ep","etp","cp"):
    v = j.get(k, defaults.get(k, ""))
    print(f"export JOB_{k.upper()}={shlex.quote(str(v))}")
' "$STATE_DIR/current_job.json")"

  ROUND=$((ROUND + 1))
  echo "$ROUND" > "$ROUND_FILE"
  STAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_DIR="${LOG_DIR_OVERRIDE:-$REPO_ROOT/logs/mfu-tp-pp-scale-r${ROUND}-${JOB_ID}-${STAMP}}"
  mkdir -p "$LOG_DIR"
  cp "$STATE_DIR/current_job.json" "$LOG_DIR/job.json"
  echo "round=$ROUND stamp=$STAMP job=$CLUSTER_JOB" | tee "$LOG_DIR/meta.txt"

  export MODE="$JOB_MODE"
  export TP="$JOB_TP" PP="$JOB_PP" CP="${JOB_CP:-1}"
  export EP="${JOB_EP:-1}" ETP="${JOB_ETP:-1}"
  export MBS="$JOB_MBS" SEQ_LENGTH="$JOB_SEQ"
  export TRAIN_ITERS="${JOB_ITERS:-$ITERS_DEFAULT}"
  export SKIP_TB=1 SKIP_SAVE=1 SKIP_PROFILE=1
  export LOG_DIR MASTER_ADDR
  export RUN_ROOT="/afs-a3-241ceshi-shared/montyyin/logs/train-tp-pp-r${ROUND}-${JOB_ID}-${STAMP}"

  echo "==> ROUND $ROUND JOB=$JOB_ID MODE=$MODE TP=$TP PP=$PP EP=$EP scales=$JOB_SCALES"
  IFS=',' read -ra SCALE_ARR <<< "$JOB_SCALES"
  for scale in "${SCALE_ARR[@]}"; do
    scale="$(echo "$scale" | tr -d ' ')"
    model_par=$((TP * PP * CP))
    if (( scale % model_par != 0 )); then
      echo "| $ROUND | $JOB_ID | $MODE | $scale | ${TP}/${PP}/${EP} | - | $JOB_GBS | - | - | SKIP_PAR | $LOG_DIR | world%!TP*PP |" >> "$LEDGER"
      continue
    fi
    DP=$((scale / model_par))
    GBS_USE="$(align_gbs "$scale" "$TP" "$PP" "$CP" "$MBS" "$JOB_GBS")" || {
      echo "| $ROUND | $JOB_ID | $MODE | $scale | ${TP}/${PP}/${EP} | $DP | $JOB_GBS | - | - | INVALID | $LOG_DIR | $JOB_NOTE |" >> "$LEDGER"
      continue
    }
    export GBS="$GBS_USE" SCALES="$scale"
    export MASTER_PORT=$((25000 + ROUND * 20 + scale / 16))
    echo "==> run scale=$scale DP=$DP GBS=$GBS port=$MASTER_PORT"
    set +e
    "$SCRIPT_DIR/run_train_mfu_scale.sh" 2>&1 | tee "$LOG_DIR/scale${scale}_driver.log"
    rc=${PIPESTATUS[0]}
    set -e

    # 解析
    PARSE_OUT="$LOG_DIR/scale${scale}_parse.json"
    set +e
    python3 "$PARSE" --peak "$PEAK" --json \
      "$LOG_DIR/scale${scale}_afs" "$LOG_DIR/scale${scale}_metrics.txt" \
      "$LOG_DIR/scale${scale}_rank"*.log \
      > "$PARSE_OUT" 2>"$LOG_DIR/scale${scale}_parse.err"
    prc=$?
    set -e
    if [[ "$prc" -eq 0 ]] && [[ -s "$PARSE_OUT" ]]; then
      eval "$(python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
s=d.get("steady") or {}
print(f"export P_TFLOP={s.get(\"tflops_median\", \"-\")}")
print(f"export P_MFU={d.get(\"mfu_pct\", \"-\")}")
print(f"export P_STATUS=ok")
' "$PARSE_OUT" 2>/dev/null || echo 'export P_TFLOP=-; export P_MFU=-; export P_STATUS=parse_fail')"
    else
      P_TFLOP="-"; P_MFU="-"; P_STATUS="fail_rc${rc}"
    fi
    # 粗判 OOM
    if grep -qiE 'out of memory|NPU out of memory|ChildFailedError' \
        "$LOG_DIR/scale${scale}_"* 2>/dev/null; then
      P_STATUS="OOM"
    fi
    echo "| $ROUND | $JOB_ID | $MODE | $scale | ${TP}/${PP}/${EP} | $DP | $GBS | $P_TFLOP | $P_MFU | $P_STATUS | $LOG_DIR | $JOB_NOTE |" \
      | tee -a "$LEDGER"
  done

  echo "AGENT_LOOP_TICK_mfu_tp_pp {\"event\":\"round_done\",\"round\":$ROUND,\"id\":\"$JOB_ID\",\"ts\":\"$(date -Iseconds)\"}"
  # 轮间短歇，避免端口/进程残留
  sleep 30
done

echo "==> campaign queue drained → $LEDGER"
