#!/usr/bin/env bash
# 永不停止的 training MFU 优化环（单轮执行器）
# 读 state/next_job.json → 跑 scale → 解析 MFU → 写 ledger → 推进队列
# 由 loop_mfu_forever.sh 或 Cursor /loop 反复调用。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$OPS_ROOT/../.." && pwd)"
STATE_DIR="${STATE_DIR:-$OPS_ROOT/reports/rounds/mfu_loop_state}"
LEDGER="${LEDGER:-$OPS_ROOT/reports/rounds/mfu_loop_ledger.md}"
QUEUE="${QUEUE:-$STATE_DIR/queue.jsonl}"
NEXT_JOB="${NEXT_JOB:-$STATE_DIR/next_job.json}"
PEAK="${PEAK:-292.79}"
PARSE="$SCRIPT_DIR/parse_train_mfu_log.py"

mkdir -p "$STATE_DIR" "$REPO_ROOT/logs" "$OPS_ROOT/reports/rounds"

# 暂停：有 PAUSE 且无 next_job 时不跑训练、不打唤醒 tick
if [[ -f "$STATE_DIR/PAUSE" ]] && [[ ! -s "$NEXT_JOB" ]]; then
  echo "==> PAUSED ($(cat "$STATE_DIR/PAUSE")) — no tick"
  exit 0
fi

# ---------- 初始化默认队列（仅首次）----------
if [[ ! -f "$QUEUE" ]]; then
  cat > "$QUEUE" <<'Q'
{"id":"r1_mbs2","mode":"dense","scales":"16","tp":1,"pp":1,"mbs":2,"gbs":128,"seq":4096,"iters":5,"note":"R1: TP1PP1 上抬 MBS=2"}
{"id":"r1b_scale","mode":"dense","scales":"32,64","tp":1,"pp":1,"mbs":1,"gbs":128,"seq":4096,"iters":5,"note":"R1b: 最佳并行扩 32/64"}
{"id":"r2_gbs256","mode":"dense","scales":"16,32","tp":1,"pp":1,"mbs":1,"gbs":256,"seq":4096,"iters":5,"note":"R2: 加大 GBS 看通信摊销"}
{"id":"r3_tp2pp1","mode":"dense","scales":"16,32","tp":2,"pp":1,"mbs":1,"gbs":128,"seq":4096,"iters":5,"note":"R3: TP=2 PP=1"}
{"id":"r4_tp4pp1","mode":"dense","scales":"16,32","tp":4,"pp":1,"mbs":1,"gbs":128,"seq":4096,"iters":5,"note":"R4: TP=4 PP=1"}
{"id":"r5_tp1pp2","mode":"dense","scales":"16,32","tp":1,"pp":2,"mbs":1,"gbs":128,"seq":4096,"iters":5,"note":"R5: TP=1 PP=2"}
{"id":"r6_tp2pp2","mode":"dense","scales":"16,32,64","tp":2,"pp":2,"mbs":1,"gbs":128,"seq":4096,"iters":5,"note":"R6: 基线 TP2PP2 扩 scale"}
{"id":"r7_seq2k","mode":"dense","scales":"16,32","tp":1,"pp":1,"mbs":1,"gbs":128,"seq":2048,"iters":5,"note":"R7: 短序列 SEQ=2048"}
{"id":"r8_seq8k","mode":"dense","scales":"16","tp":1,"pp":1,"mbs":1,"gbs":64,"seq":8192,"iters":5,"note":"R8: 长序列 SEQ=8192 GBS=64"}
{"id":"r9_best128","mode":"dense","scales":"128","tp":1,"pp":1,"mbs":1,"gbs":128,"seq":4096,"iters":5,"note":"R9: 当前最佳并行打满 128"}
{"id":"r10_moe16","mode":"moe","scales":"16","tp":1,"pp":1,"ep":8,"etp":1,"mbs":1,"gbs":128,"seq":4096,"iters":5,"note":"R10: MoE 30B-A3B 冒烟"}
Q
fi

if [[ ! -f "$LEDGER" ]]; then
  cat > "$LEDGER" <<'L'
# MFU 永不停止优化环 · 账本

> peak = 292.79 TFLOPS/卡（card_screen medians.func_tflops）  
> 基线 R0：TP2PP2 ≈47.7% · TP1PP1 ≈58.3%（16 卡）

| round | id | scale | TP/PP/MBS/GBS/SEQ | steady TFLOP/s/GPU | MFU% | status | log | note |
|------:|----|------:|-------------------|-------------------:|-----:|--------|-----|------|
L
fi

# ---------- 取下一任务 ----------
pick_job() {
  if [[ -f "$NEXT_JOB" ]] && [[ -s "$NEXT_JOB" ]]; then
    cat "$NEXT_JOB"
    return
  fi
  # 从队列取第一行未 done 的；若全做完则循环追加变体
  if [[ ! -s "$QUEUE" ]]; then
    local n
    n="$(date +%s)"
    echo "{\"id\":\"auto_${n}\",\"mode\":\"dense\",\"scales\":\"16,32,64,128\",\"tp\":1,\"pp\":1,\"mbs\":1,\"gbs\":128,\"seq\":4096,\"iters\":5,\"note\":\"auto recycle best-so-far\"}" > "$NEXT_JOB"
    cat "$NEXT_JOB"
    return
  fi
  head -n 1 "$QUEUE" > "$NEXT_JOB"
  tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
  cat "$NEXT_JOB"
}

JOB_JSON="$(pick_job)"
echo "==> JOB: $JOB_JSON"
printf '%s\n' "$JOB_JSON" > "$STATE_DIR/current_job.json"

# 用 python 解析 JSON（避免依赖 jq）
eval "$(python3 -c '
import json, shlex, sys
j = json.load(open(sys.argv[1]))
defaults = {"ep":1,"etp":1,"cp":1,"iters":5,"scales":"16","mode":"dense","note":""}
for k in ("id","mode","scales","tp","pp","mbs","gbs","seq","iters","note","ep","etp","cp"):
    v = j.get(k, defaults.get(k, ""))
    print(f"export JOB_{k.upper()}={shlex.quote(str(v))}")
' "$STATE_DIR/current_job.json")"

ROUND_FILE="$STATE_DIR/round_counter"
ROUND="$(cat "$ROUND_FILE" 2>/dev/null || echo 0)"
ROUND=$((ROUND + 1))
echo "$ROUND" > "$ROUND_FILE"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR_OVERRIDE:-$REPO_ROOT/logs/mfu-loop-r${ROUND}-${JOB_ID}-${STAMP}}"
mkdir -p "$LOG_DIR"
echo "$JOB_JSON" > "$LOG_DIR/job.json"
echo "round=$ROUND stamp=$STAMP" | tee "$LOG_DIR/meta.txt"

# GBS 对齐：必须整除 MBS * DP，DP = world/(TP*PP*CP)
align_gbs() {
  local world="$1" tp="$2" pp="$3" cp="$4" mbs="$5" gbs="$6"
  local model_par=$((tp * pp * cp))
  if (( world % model_par != 0 )); then
    echo "INVALID world=$world not divisible by TP*PP*CP=$model_par" >&2
    return 1
  fi
  local dp=$((world / model_par))
  local unit=$((mbs * dp))
  if (( gbs % unit != 0 )); then
    local new=$(( ((gbs + unit - 1) / unit) * unit ))
    echo "WARN align GBS $gbs → $new (unit=MBS*DP=$unit)" >&2
    gbs=$new
  fi
  echo "$gbs"
}

export MODE="$JOB_MODE"
export TP="$JOB_TP"
export PP="$JOB_PP"
export CP="${JOB_CP:-1}"
export EP="${JOB_EP:-1}"
export ETP="${JOB_ETP:-1}"
export MBS="$JOB_MBS"
export SEQ_LENGTH="$JOB_SEQ"
export TRAIN_ITERS="$JOB_ITERS"
export SKIP_TB=1
export SKIP_SAVE=1
export SKIP_PROFILE=1
export LOG_DIR
export RUN_ROOT="/afs-a3-241ceshi-shared/montyyin/logs/train-loop-r${ROUND}-${JOB_ID}-${STAMP}"

IFS=',' read -ra SCALE_ARR <<< "$JOB_SCALES"
STATUS_ALL=ok
declare -a RESULTS=()

for scale in "${SCALE_ARR[@]}"; do
  scale="$(echo "$scale" | tr -d ' ')"
  GBS_USE="$(align_gbs "$scale" "$TP" "$PP" "$CP" "$MBS" "$JOB_GBS")" || {
    echo "| $ROUND | $JOB_ID | $scale | ${TP}/${PP}/${MBS}/${JOB_GBS}/${SEQ_LENGTH} | - | - | INVALID | $LOG_DIR | $JOB_NOTE |" >> "$LEDGER"
    STATUS_ALL=fail
    continue
  }
  export GBS="$GBS_USE"
  export SCALES="$scale"
  export MASTER_PORT=$((24670 + ROUND * 10 + scale / 16))
  echo "==> RUN round=$ROUND id=$JOB_ID scale=$scale TP=$TP PP=$PP MBS=$MBS GBS=$GBS SEQ=$SEQ_LENGTH"
  set +e
  "$SCRIPT_DIR/run_train_mfu_scale.sh" 2>&1 | tee "$LOG_DIR/scale${scale}_driver.log"
  rc=${PIPESTATUS[0]}
  set -e

  # 解析本地日志 + 尝试 AFS 拉回的 results
  METRICS_JSON="$LOG_DIR/scale${scale}_metrics.json"
  set +e
  # 只解析本 scale 的日志，避免串用上一 scale 的 TFLOP
  python3 "$PARSE" --peak "$PEAK" --json \
    "$LOG_DIR/scale${scale}_afs" \
    "$LOG_DIR/scale${scale}_rank0.log" \
    "$LOG_DIR/scale${scale}_metrics.txt" \
    > "$METRICS_JSON" 2>"$LOG_DIR/scale${scale}_parse.err"
  parse_rc=$?
  set -e

  STEADY="-"
  MFU="-"
  ST="fail"
  if [[ $rc -ne 0 ]]; then
    ST=fail
    STATUS_ALL=fail
  elif [[ $parse_rc -eq 0 ]]; then
    STEADY="$(python3 -c "import json;d=json.load(open('$METRICS_JSON'));print(f\"{d['steady']['tflops_mean']:.2f}\")")"
    MFU="$(python3 -c "import json;d=json.load(open('$METRICS_JSON'));print(f\"{d['mfu_pct']:.2f}\")")"
    ST=ok
  else
    ST=nometrics
    STATUS_ALL=fail
  fi
  RESULTS+=("scale=$scale steady=$STEADY mfu=$MFU status=$ST")
  echo "| $ROUND | \`$JOB_ID\` | $scale | ${TP}/${PP}/${MBS}/${GBS}/${SEQ_LENGTH} | $STEADY | $MFU | $ST | \`$LOG_DIR\` | $JOB_NOTE |" >> "$LEDGER"

  # 单 scale 报告片段
  {
    echo "# MFU loop R${ROUND} · ${JOB_ID} · scale=${scale}"
    echo
    echo "- 配置: mode=$MODE TP=$TP PP=$PP CP=$CP EP=$EP MBS=$MBS GBS=$GBS SEQ=$SEQ_LENGTH iters=$TRAIN_ITERS"
    echo "- 稳态 TFLOP/s/GPU: $STEADY · MFU: ${MFU}% · status=$ST"
    echo "- 假设/备注: $JOB_NOTE"
    echo "- 日志: \`$LOG_DIR\`"
    echo "- AFS: \`$RUN_ROOT\`"
  } > "$OPS_ROOT/reports/rounds/mfu_loop_r${ROUND}_${JOB_ID}_s${scale}.md"

  # scale 之间歇一下，降低跳板 ssh/vcctl 压力
  sleep 15
done

# 清空 next_job，让下一轮从队列取；agent 可写入 next_job.json 插队
rm -f "$NEXT_JOB"

# 若队列空：只补少量高价值项，避免反复重跑同一矩阵
if [[ ! -s "$QUEUE" ]]; then
  BEST_TP="${TP:-1}"
  BEST_PP="${PP:-1}"
  # 若刚跑过 GBS=2048 全矩阵，歇着等人工/下一假设，勿立刻再塞同款
  if [[ "${JOB_ID:-}" == *gbs2k_matrix* ]] || [[ "${JOB_NOTE:-}" == *GBS=2048*矩阵* ]]; then
    echo "==> skip auto-recycle after matrix job; waiting for next_job.json"
  else
    cat >> "$QUEUE" <<EOF
{"id":"recycle_${STAMP}_gbs4k_128","mode":"dense","scales":"128","tp":${BEST_TP},"pp":${BEST_PP},"mbs":1,"gbs":4096,"seq":4096,"iters":5,"note":"recycle: GBS=4096 @128 单点"}
EOF
  fi
fi

SUMMARY="round=$ROUND id=$JOB_ID status=$STATUS_ALL ${RESULTS[*]}"
echo "$SUMMARY" | tee "$LOG_DIR/summary.txt"
ROUND="$ROUND" JOB_ID="$JOB_ID" STATUS_ALL="$STATUS_ALL" LOG_DIR="$LOG_DIR" LEDGER="$LEDGER" RESULTS_STR="${RESULTS[*]}" \
python3 -c '
import json, os
payload = {
  "prompt": "分析本轮 MFU 结果并写入 reports/rounds/mfu_opt_rN.md；根据证据更新 queue 或 next_job.json；继续优化",
  "round": int(os.environ["ROUND"]),
  "id": os.environ["JOB_ID"],
  "status": os.environ["STATUS_ALL"],
  "log": os.environ["LOG_DIR"],
  "results": os.environ.get("RESULTS_STR", ""),
  "ledger": os.environ["LEDGER"],
}
print("AGENT_LOOP_TICK_mfu_opt " + json.dumps(payload, ensure_ascii=False))
'

exit 0
