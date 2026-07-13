#!/usr/bin/env bash
# 在跳板机 weibozhen 上跑：不依赖笔记本在线。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export KUBECONFIG="${KUBECONFIG:-/root/.kube/config.huawei-a3-241ceshi}"
JOB="${CLUSTER_JOB:-montyyin-mfu-scale}"
PEAK=292.79
AFS_WS=/afs-a3-weight-share/yinjinrun.p-huawei
STATE="$ROOT/state"
LEDGER="$ROOT/ledger.md"
QUEUE="$STATE/queue.jsonl"
LOGROOT="$ROOT/logs"
mkdir -p "$STATE" "$LOGROOT"

pexec() {
  local pod="$1"; shift
  vcctl pod exec "$pod" -- bash -lc "$*"
}
pod_rank() {
  local r="$1"
  if [[ "$r" -eq 0 ]]; then echo "${JOB}-master-0"; else echo "${JOB}-worker-$((r-1))"; fi
}

align_gbs() {
  local world="$1" tp="$2" pp="$3" cp="$4" mbs="$5" gbs="$6"
  local mp=$((tp*pp*cp)) unit
  if (( world % mp != 0 )); then echo INVALID; return 1; fi
  unit=$((mbs * world / mp))
  if (( gbs % unit != 0 )); then gbs=$(( ((gbs+unit-1)/unit)*unit )); fi
  echo "$gbs"
}

if [[ ! -f "$LEDGER" ]]; then
  cat > "$LEDGER" <<'L'
# jumphost MFU TP×PP scale ledger
| round | id | mode | scale | TP/PP/EP | DP | GBS | TFLOP | MFU% | status | note |
|------:|----|------|------:|----------|---:|----:|------:|-----:|--------|------|
L
fi

# 注意：勿用 pgrep -f pretrain_gpt.py —— bash -lc 命令行自身会被匹配，永远 ≥1
count_pretrain() {
  pexec "${JOB}-master-0" 'ps -eo args | grep -E "^(/root/miniconda3|python).*(pretrain_gpt\\.py|torchrun.*pretrain)" | grep -v grep | wc -l' \
    2>/dev/null | tr -dc '0-9'
  true
}

echo "==> wait existing pretrain to finish $(date -Iseconds)"
for i in $(seq 1 60); do
  n=$(count_pretrain); n=${n:-0}
  echo "  t=$i pretrain_procs=$n $(date -Iseconds)"
  if [[ "$n" -eq 0 ]]; then break; fi
  sleep 20
done

# 捞一把已有 scale16 指标
OLD16=/afs-a3-weight-share/yinjinrun.p-huawei/logs/train-tp-pp-r1-d_tp8pp1-20260711_202651/scale_16
pexec "${JOB}-master-0" "grep -E 'throughput per GPU' $OLD16/train_mcore_qwen3_8b_rank0.log 2>/dev/null | tail -10" \
  > "$LOGROOT/leftover_scale16.txt" 2>/dev/null || true
if [[ -s "$LOGROOT/leftover_scale16.txt" ]]; then
  eval "$(python3 - <<PY
import re,statistics
text=open("$LOGROOT/leftover_scale16.txt").read()
vals=[float(x) for x in re.findall(r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)", text)]
use=vals[1:] if len(vals)>=2 else vals
if use:
    med=statistics.median(use)
    print(f"export L_TFLOP={med:.2f}")
    print(f"export L_MFU={med/$PEAK*100:.2f}")
else:
    print("export L_TFLOP=-; export L_MFU=-")
PY
)"
  echo "| 0 | d_tp8pp1 | dense | 16 | 8/1/1 | 2 | 2048 | ${L_TFLOP} | ${L_MFU} | leftover | 本机拉起后 jumphost 接管 |" | tee -a "$LEDGER"
fi

ROUND=$(cat "$STATE/round_counter" 2>/dev/null || echo 1)
WRAP_DIR=/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/wrappers
MEG=/afs-a3-241ceshi-shared/geruijun/Megatron-LM-0.12.3
DATA=/afs-a3-241ceshi-shared/geruijun
MASTER_ADDR="${JOB}-master-0.${JOB}"

while [[ -s "$QUEUE" ]]; do
  head -n1 "$QUEUE" > "$STATE/current.json"
  tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
  eval "$(python3 -c '
import json,shlex,sys
j=json.load(open(sys.argv[1]))
d={"ep":1,"etp":1,"cp":1,"iters":5,"mbs":1,"gbs":2048,"seq":4096,"note":"","mode":"dense"}
for k in ("id","mode","scales","tp","pp","mbs","gbs","seq","iters","note","ep","etp","cp"):
    print("export J_%s=%s"%(k.upper(), shlex.quote(str(j.get(k,d.get(k,""))))))
' "$STATE/current.json")"

  ROUND=$((ROUND+1)); echo "$ROUND" > "$STATE/round_counter"
  STAMP=$(date +%Y%m%d_%H%M%S)
  RUN=/afs-a3-weight-share/yinjinrun.p-huawei/logs/jumphost-tp-pp-r${ROUND}-${J_ID}-${STAMP}
  echo "==> ROUND $ROUND $J_ID mode=$J_MODE TP=$J_TP PP=$J_PP EP=$J_EP scales=$J_SCALES"

  IFS=',' read -ra SC <<< "$J_SCALES"
  PORT0=$((26000 + ROUND * 20))
  si=0
  for scale in "${SC[@]}"; do
    scale=$(echo "$scale" | tr -d ' ')
    si=$((si+1))
    GBS=$(align_gbs "$scale" "$J_TP" "$J_PP" "${J_CP:-1}" "$J_MBS" "$J_GBS") || {
      echo "| $ROUND | $J_ID | $J_MODE | $scale | ${J_TP}/${J_PP}/${J_EP} | - | $J_GBS | - | - | INVALID | |" >> "$LEDGER"
      continue
    }
    DP=$((scale / (J_TP * J_PP * ${J_CP:-1})))
    NN=$((scale / 16))
    PORT=$((PORT0 + si))
    SD="$RUN/scale_${scale}"
    echo "==> scale=$scale DP=$DP GBS=$GBS nnodes=$NN port=$PORT"
    pexec "${JOB}-master-0" "mkdir -p '$SD'"

    WRAP_NAME=train_qwen3_8B_ascend.sh
    [[ "$J_MODE" == moe ]] && WRAP_NAME=train_qwen3_30B_A3B_ascend.sh

    r=0
    while [[ $r -lt $NN ]]; do
      # 在跳板机本地生成 launcher，再灌进 AFS
      cat > "$LOGROOT/launch_r${ROUND}_s${scale}_rank${r}.sh" <<EOS
#!/usr/bin/env bash
set -uo pipefail
export PATH=/root/miniconda3/envs/llm_test/bin:\${PATH:-}
export PYTHONPATH=/MindSpeed-LLM/MindSpeed:\${PYTHONPATH:-}
export WORLD_SIZE=$NN NNODES=$NN RANK=$r NODE_RANK=$r
export MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$PORT
export NPUS_PER_NODE=16 GPUS_PER_NODE=16
export DATA_ROOT=$DATA RUN_DIR=$SD LOG_DIR=$SD/
export TENSORBOARD_DIR=$SD/tb CKPT_SAVE_DIR=$SD/ckpt
export TRAIN_ITERS=$J_ITERS SKIP_SAVE=1 SKIP_PROFILE=1 SKIP_TB=1
export TP=$J_TP PP=$J_PP CP=${J_CP:-1} EP=${J_EP:-1} ETP=${J_ETP:-1}
export MBS=$J_MBS GBS=$GBS SEQ_LENGTH=$J_SEQ
export HCCL_IF_BASE_PORT=\$(($PORT+2000))
mkdir -p "\$LOG_DIR" "\$TENSORBOARD_DIR" "\$CKPT_SAVE_DIR"
cd $MEG
bash -lc "bash $WRAP_DIR/$WRAP_NAME" > "$SD/rank${r}.log" 2>&1
echo TRAIN_RANK_${r}_DONE rc=\$?
EOS
      vcctl pod exec -i "${JOB}-master-0" -- bash -c "cat > $SD/launch_rank${r}.sh && chmod +x $SD/launch_rank${r}.sh" \
        < "$LOGROOT/launch_r${ROUND}_s${scale}_rank${r}.sh"
      r=$((r+1))
    done

    r=0
    while [[ $r -lt $NN ]]; do
      pod=$(pod_rank "$r")
      vcctl pod exec "$pod" -- bash -lc "nohup bash $SD/launch_rank${r}.sh >$SD/nohup_rank${r}.log 2>&1 & echo STARTED_\$!"
      sleep 6
      r=$((r+1))
    done

    echo "==> waiting scale=$scale"
    for t in $(seq 1 240); do
      # DONE 写在 nohup_rank*.log（redirect 之外的 echo）
      done_n=$(pexec "${JOB}-master-0" "grep -l 'TRAIN_RANK_.*_DONE' $SD/nohup_rank*.log $SD/rank*.log 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9')
      done_n=${done_n:-0}
      alive=$(count_pretrain); alive=${alive:-0}
      has_mfu=$(pexec "${JOB}-master-0" "grep -c 'throughput per GPU' $SD/train_mcore*.log $SD/rank0.log 2>/dev/null | awk -F: '{s+=\$NF} END{print s+0}'" 2>/dev/null | tr -dc '0-9')
      has_mfu=${has_mfu:-0}
      echo "  wait t=$t done=$done_n/$NN alive=$alive mfu_lines=$has_mfu $(date -Iseconds)"
      if [[ "$done_n" -ge "$NN" ]]; then break; fi
      # 进程已死且已有吞吐日志 → 视为结束（兼容 DONE 漏写）
      if [[ "$t" -gt 3 && "$alive" -eq 0 && "$has_mfu" -ge 1 ]]; then
        sleep 10
        alive2=$(count_pretrain); alive2=${alive2:-0}
        [[ "$alive2" -eq 0 ]] && break
      fi
      if [[ "$t" -gt 5 && "$alive" -eq 0 && "$done_n" -ge 1 ]]; then
        sleep 10
        break
      fi
      sleep 30
    done

    pexec "${JOB}-master-0" "grep -hE 'throughput per GPU' $SD/rank0.log $SD/train_mcore*.log $SD/nohup_rank0.log 2>/dev/null | tail -20" \
      > "$LOGROOT/r${ROUND}_s${scale}_metrics.txt" 2>/dev/null || true
    eval "$(python3 - <<PY
import re,statistics
text=open("$LOGROOT/r${ROUND}_s${scale}_metrics.txt").read()
vals=[float(x) for x in re.findall(r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)", text)]
use=vals[1:] if len(vals)>=2 else vals
if use:
    med=statistics.median(use)
    print(f"export P_TFLOP={med:.2f}")
    print(f"export P_MFU={med/$PEAK*100:.2f}")
    print("export P_STATUS=ok")
else:
    print("export P_TFLOP=-")
    print("export P_MFU=-")
    print("export P_STATUS=fail")
PY
)"
    echo "| $ROUND | $J_ID | $J_MODE | $scale | ${J_TP}/${J_PP}/${J_EP} | $DP | $GBS | $P_TFLOP | $P_MFU | $P_STATUS | $J_NOTE |" \
      | tee -a "$LEDGER"
    # 同步 ledger 到 AFS
    vcctl pod exec -i "${JOB}-master-0" -- bash -c "mkdir -p $AFS_WS/results/mfu_tp_pp_scale && cat >> $AFS_WS/results/mfu_tp_pp_scale/ledger.md" \
      <<< "| $ROUND | $J_ID | $J_MODE | $scale | ${J_TP}/${J_PP}/${J_EP} | $DP | $GBS | $P_TFLOP | $P_MFU | $P_STATUS | $J_NOTE |" || true
    sleep 20
  done
done

echo "==> jumphost campaign drained $(date -Iseconds)"
echo DONE > "$ROOT/FINISHED"
