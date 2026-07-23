#!/usr/bin/env bash
# 跳板执行：前2轮perfect matching，每轮32个node-disjoint pair并行。
set -euo pipefail

RUN_ID="${RUN_ID:?set RUN_ID}"
BUNDLE_DIR="${BUNDLE_DIR:?set BUNDLE_DIR}"
SCHEDULE_JSONL="${SCHEDULE_JSONL:-$BUNDLE_DIR/schedule.jsonl}"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config-vc-c550-h3c-test.yaml}"
VCCTL="${VCCTL:-/usr/local/bin/vcctl}"
JOB="${JOB:-yinjinrun-cs512-20260716-221823}"
AFS_OUT="${AFS_OUT:-/afs-a3-weight-share/yinjinrun.p/results/muxi-fabric-w2/$RUN_ID}"
BASE_PORT="${BASE_PORT:-30400}"
ROUNDS="${ROUNDS:-2}"
PAIR_LIMIT="${PAIR_LIMIT:-32}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
NBYTES="${NBYTES:-16777216}"
ROUND_ORDER="${ROUND_ORDER:-}"
PRECOMPLETED_ROUNDS="${PRECOMPLETED_ROUNDS:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-8}"
SESSION="${SESSION:-main}"
WORK="/tmp/muxi-pair-$RUN_ID-$SESSION"
MASTER="$JOB-master-0"
[[ "$PAIR_LIMIT" =~ ^[0-9]+$ && "$PAIR_LIMIT" -ge 1 && "$PAIR_LIMIT" -le 32 ]]

mkdir -p "$WORK/fire" "$WORK/preflight" "$WORK/postcheck" "$WORK/cleanup" "$WORK/launchers"
exec > >(tee -a "$WORK/run.log") 2>&1
export KUBECONFIG

pods=("$MASTER")
for i in $(seq 0 62); do pods+=("$JOB-worker-$i"); done

vexec() {
  local pod="$1"; shift
  "$VCCTL" pod exec "$pod" -- bash -lc "$1"
}

vexec_i() {
  local pod="$1"; shift
  "$VCCTL" pod exec -i "$pod" -- bash -c "$1"
}

preflight_one() {
  local rank="$1" attempt="$2"
  local pod="${pods[$rank]}"
  local log="$WORK/preflight/node_${rank}.attempt${attempt}.log" rc=0
  vexec "$pod" \
    "set -e; left=\$(ps -eo args= | awk '/torchrun|nccl_p2p|muxi_pair_bench|constitution|burn[-_]?in/ && \$0 !~ /awk/ {print}'); [[ -z \"\$left\" ]]; mx-smi 2>/dev/null | grep -q 'no process found'; echo IDLE" \
    >"$log" 2>&1 || rc=$?
  return "$rc"
}

postcheck_one() {
  local rank="$1" attempt="$2"
  local pod="${pods[$rank]}"
  local log="$WORK/postcheck/node_${rank}.attempt${attempt}.log" rc=0
  vexec "$pod" \
    "left=\$(ps -eo args= | awk '/$RUN_ID|muxi_pair_bench/ && \$0 !~ /awk/ {print}'); [[ -z \"\$left\" ]]; echo CLEAN" \
    >"$log" 2>&1 || rc=$?
  return "$rc"
}

run_stage() {
  local stage="$1" callback="$2" count="$3"
  local -a pending=() permanent=()
  local rank attempt i rc log
  for ((rank=0; rank<count; rank++)); do pending+=("$rank"); done
  echo "STAGE_BEGIN stage=$stage count=$count parallel=$count at=$(date -Iseconds)"
  for attempt in 1 2 3; do
    [[ "${#pending[@]}" -gt 0 ]] || break
    local -a pids=() ranks=() retry=()
    echo "STAGE_ATTEMPT stage=$stage attempt=$attempt pending=${#pending[@]} ranks=${pending[*]}"
    for rank in "${pending[@]}"; do
      "$callback" "$rank" "$attempt" &
      pids+=("$!"); ranks+=("$rank")
    done
    for i in "${!pids[@]}"; do
      rank="${ranks[$i]}"; rc=0
      wait "${pids[$i]}" || rc=$?
      [[ "$rc" -eq 0 ]] && continue
      log="$WORK/$stage/node_${rank}.attempt${attempt}.log"
      if grep -Eiq 'Forbidden|Unauthorized|permission denied|certificate|credentials' "$log"; then
        permanent+=("$rank:$rc")
      elif [[ "$attempt" -lt 3 ]]; then
        retry+=("$rank")
      else
        permanent+=("$rank:$rc")
      fi
    done
    pending=("${retry[@]}")
    [[ "${#pending[@]}" -eq 0 ]] || sleep "$attempt"
  done
  echo "STAGE_END stage=$stage permanent=${permanent[*]:-none} at=$(date -Iseconds)"
  [[ "${#permanent[@]}" -eq 0 ]]
}

echo "PAIR_BEGIN run_id=$RUN_ID rounds=$ROUNDS at=$(date -Iseconds)"
"$VCCTL" pod get --job "$JOB" >"$WORK/pods.log"
python3 - "$WORK/pods.log" <<'PY'
import re,sys
lines=open(sys.argv[1]).read().splitlines()
pods=[x for x in lines if "yinjinrun-cs512-20260716-221823-" in x]
running=[x for x in pods if re.search(r"\bRunning\b",x)]
assert len(pods)==64 and len(running)==64,(len(pods),len(running))
print("POD_LIST_OK total=64 running=64")
PY

vexec "$MASTER" "mkdir -p '$AFS_OUT/code' '$AFS_OUT/raw' '$AFS_OUT/results'"
vexec_i "$MASTER" "cat > '$AFS_OUT/code/muxi_pair_bench.py'" <"$BUNDLE_DIR/muxi_pair_bench.py"
vexec_i "$MASTER" "cat > '$AFS_OUT/schedule.jsonl'" <"$SCHEDULE_JSONL"
vexec_i "$MASTER" "cat > '$AFS_OUT/schedule.csv'" <"$BUNDLE_DIR/schedule.csv"

current_round=0
declare -a round_pods=() round_scripts=()

prepare_round() {
  local round_id="$1"
  local round_dir="$WORK/launchers/round_$round_id"
  mkdir -p "$round_dir" "$WORK/fire/round_$round_id" "$WORK/cleanup/round_$round_id"
  python3 - "$SCHEDULE_JSONL" "$round_id" "$PAIR_LIMIT" >"$WORK/round_${round_id}.tsv" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    r=json.loads(line)
    if r["round"]==int(sys.argv[2]) and r["slot"]<int(sys.argv[3]):
        print("\t".join(str(r[k]) for k in ("slot","edge_id","src_index","dst_index","src_pod","dst_pod")))
PY
  round_pods=(); round_scripts=()
  while IFS=$'\t' read -r slot edge src_index dst_index src_pod dst_pod; do
    port=$((BASE_PORT + round_id*32 + slot))
    for node_rank in 0 1; do
      idx=$((slot*2+node_rank))
      [[ "$node_rank" -eq 0 ]] && pod="$src_pod" || pod="$dst_pod"
      script="$round_dir/node_${idx}.sh"
      log="$AFS_OUT/raw/round_${round_id}.pair_${slot}.node_${node_rank}.log"
      donef="$AFS_OUT/results/round_${round_id}/pair_${slot}.node_${node_rank}.done"
      failf="$AFS_OUT/results/round_${round_id}/pair_${slot}.node_${node_rank}.fail"
      cat >"$script" <<EOF
#!/usr/bin/env bash
export PATH=/opt/conda/bin:\${PATH:-/usr/bin}
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0 MCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=xscale_0 MCCL_IB_HCA=xscale_0
export NCCL_IB_GID_INDEX=5 MCCL_IB_GID_INDEX=5 MCCL_IB_TC=128
export MCCL_ENABLE_VSWITCH=1 MCCL_PCIE_BUFFER_MODE=0 FORCE_ACTIVE_WAIT=2
export NCCL_DEBUG=INFO MCCL_DEBUG=INFO
mkdir -p '$AFS_OUT/results/round_${round_id}'
rm -f '$donef' '$failf'
/opt/conda/bin/torchrun --nnodes=2 --node_rank=$node_rank --nproc_per_node=1 \
  --master_addr=${src_pod}.${JOB} --master_port=\${PAIR_MASTER_PORT:-$port} \
  '$AFS_OUT/code/muxi_pair_bench.py' --nbytes $NBYTES --warmup $WARMUP --iters $ITERS \
  --round $round_id --slot $slot --edge-id $edge --src-index $src_index --dst-index $dst_index \
  --src-pod '$src_pod' --dst-pod '$dst_pod' --hca xscale_0 \
  --out '$AFS_OUT/results/round_${round_id}/pair_${slot}.json' >'$log' 2>&1
rc=\$?
if [[ \$rc -eq 0 ]]; then echo OK >'$donef'; else echo FAIL:\$rc >'$failf'; fi
exit \$rc
EOF
      chmod +x "$script"
      round_pods+=("$pod")
      round_scripts+=("$script")
    done
  done <"$WORK/round_${round_id}.tsv"
  [[ "${#round_pods[@]}" -eq $((PAIR_LIMIT*2)) &&
     "$(printf '%s\n' "${round_pods[@]}" | sort -u | wc -l)" -eq $((PAIR_LIMIT*2)) ]]
  tar -C "$round_dir" -cf - . | vexec_i "$MASTER" "mkdir -p '$AFS_OUT/code/round_$round_id'; tar -C '$AFS_OUT/code/round_$round_id' -xf -"
}

start_round_one() {
  local idx="$1" attempt="$2"
  local pod="${round_pods[$idx]}"
  local log="$WORK/fire/round_${current_round}/node_${idx}.attempt${attempt}.log" rc=0
  {
    echo "POD=$pod NODE_INDEX=$idx ROUND=$current_round ATTEMPT=$attempt"
    # pair任务很短：由跳板并发保持64个vcctl exec前台生命周期，避免pod内detach
    # 后控制面exec结束造成torchrun停在spawn前。timeout覆盖init与短迭代。
    vexec "$pod" \
      "timeout 150 bash '$AFS_OUT/code/round_${current_round}/node_${idx}.sh'"
  } >"$log" 2>&1 || rc=$?
  return "$rc"
}

declare -a ROUND_SUCCESS_SLOTS=() ROUND_FAILED_SLOTS=()

run_pair_one() {
  local slot="$1" attempt="$2"
  local src_idx=$((slot*2)) dst_idx=$((slot*2+1))
  local src_pod="${round_pods[$src_idx]}" dst_pod="${round_pods[$dst_idx]}"
  local port=$((BASE_PORT + current_round*32 + slot + (attempt-1)*4096))
  local dir="$WORK/fire/round_${current_round}" r0=0 r1=0
  if [[ "$attempt" -gt 1 ]]; then
    vexec "$MASTER" \
      "rm -f '$AFS_OUT/results/round_${current_round}/pair_${slot}.json' '$AFS_OUT/results/round_${current_round}/pair_${slot}.node_0.done' '$AFS_OUT/results/round_${current_round}/pair_${slot}.node_1.done' '$AFS_OUT/results/round_${current_round}/pair_${slot}.node_0.fail' '$AFS_OUT/results/round_${current_round}/pair_${slot}.node_1.fail'"
  fi
  {
    echo "POD=$src_pod NODE_RANK=0 ROUND=$current_round SLOT=$slot ATTEMPT=$attempt PORT=$port"
    vexec "$src_pod" \
      "PAIR_MASTER_PORT=$port timeout 150 bash '$AFS_OUT/code/round_${current_round}/node_${src_idx}.sh'"
  } >"$dir/pair_${slot}.attempt${attempt}.node_0.log" 2>&1 & p0=$!
  {
    echo "POD=$dst_pod NODE_RANK=1 ROUND=$current_round SLOT=$slot ATTEMPT=$attempt PORT=$port"
    vexec "$dst_pod" \
      "PAIR_MASTER_PORT=$port timeout 150 bash '$AFS_OUT/code/round_${current_round}/node_${dst_idx}.sh'"
  } >"$dir/pair_${slot}.attempt${attempt}.node_1.log" 2>&1 & p1=$!
  wait "$p0" || r0=$?
  wait "$p1" || r1=$?
  if [[ "$r0" -ne 0 || "$r1" -ne 0 ]]; then
    for pod in "$src_pod" "$dst_pod"; do
      vexec "$pod" \
        "pgs=\$(ps -eo pgid=,args= | awk '/--master_port=$port/ && /$RUN_ID/ && \$0 !~ /awk/ {print \$1}' | sort -u); for g in \$pgs; do kill -TERM -- -\$g 2>/dev/null || true; done; sleep 1; for g in \$pgs; do kill -KILL -- -\$g 2>/dev/null || true; done"
    done
    return 1
  fi
  return 0
}

run_pair_stage() {
  local -a pending=() retry=()
  local slot attempt i rc
  ROUND_SUCCESS_SLOTS=(); ROUND_FAILED_SLOTS=()
  for ((slot=0; slot<PAIR_LIMIT; slot++)); do pending+=("$slot"); done
  echo "PAIR_STAGE_BEGIN round=$current_round pairs=$PAIR_LIMIT parallel=$PAIR_LIMIT"
  for attempt in 1 2; do
    [[ "${#pending[@]}" -gt 0 ]] || break
    local -a pids=() slots=()
    retry=()
    echo "PAIR_STAGE_ATTEMPT round=$current_round attempt=$attempt pending=${pending[*]}"
    for slot in "${pending[@]}"; do
      run_pair_one "$slot" "$attempt" &
      pids+=("$!"); slots+=("$slot")
    done
    for i in "${!pids[@]}"; do
      slot="${slots[$i]}"; rc=0
      wait "${pids[$i]}" || rc=$?
      if [[ "$rc" -eq 0 ]]; then
        ROUND_SUCCESS_SLOTS+=("$slot")
      elif grep -Eiq 'Forbidden|Unauthorized|permission denied|certificate|credentials' \
        "$WORK/fire/round_${current_round}/pair_${slot}.attempt${attempt}.node_"*.log; then
        echo "PAIR_AUTH_FAILURE round=$current_round slot=$slot" >&2
        return 40
      elif [[ "$attempt" -lt 2 ]]; then
        retry+=("$slot")
      else
        ROUND_FAILED_SLOTS+=("$slot")
      fi
    done
    pending=("${retry[@]}")
  done
  echo "PAIR_STAGE_END round=$current_round success=${#ROUND_SUCCESS_SLOTS[@]} failed=${#ROUND_FAILED_SLOTS[@]} failed_slots=${ROUND_FAILED_SLOTS[*]:-none}"
  [[ "${#ROUND_FAILED_SLOTS[@]}" -le 8 ]]
}

cleanup_round_one() {
  local idx="$1" attempt="$2"
  local pod="${round_pods[$idx]}"
  local log="$WORK/cleanup/round_${current_round}/node_${idx}.attempt${attempt}.log" rc=0
  vexec "$pod" \
    "state=/tmp/${RUN_ID}.r${current_round}.n${idx}.pid; if [[ -s \$state ]]; then p=\$(cat \$state); kill -TERM -- -\$p 2>/dev/null || true; sleep 1; kill -KILL -- -\$p 2>/dev/null || true; fi; rm -f /tmp/${RUN_ID}.r${current_round}.n${idx}.sh \$state; echo CLEAN" \
    >"$log" 2>&1 || rc=$?
  return "$rc"
}

on_round_error() {
  echo "PAIR_DRIVER_ERROR round=$current_round at=$(date -Iseconds)"
  if [[ "${#round_pods[@]}" -gt 0 ]]; then
    run_stage "cleanup/round_$current_round" cleanup_round_one "${#round_pods[@]}" || true
  fi
}
trap on_round_error ERR

validate_round() {
  local completed_rounds="$1"
  vexec_i "$MASTER" \
    "python3 - --schedule '$AFS_OUT/schedule.jsonl' --results-dir '$AFS_OUT/results' --round-ids '$completed_rounds' --pair-limit '$PAIR_LIMIT' --expected-iters '$ITERS' --jsonl '$AFS_OUT/pairs.jsonl' --csv '$AFS_OUT/pairs.csv' --summary-json '$AFS_OUT/validation_summary.json' --summary-md '$AFS_OUT/SUMMARY.md'" \
    <"$BUNDLE_DIR/parse_pair_rounds.py"
}

printf 'round,start,end,fire_elapsed_s\n' >"$WORK/round_timing.csv"
printf 'round,slot,status\n' >"$WORK/failed_pairs.csv"
if [[ -n "$ROUND_ORDER" ]]; then
  IFS=',' read -r -a rounds_to_run <<<"$ROUND_ORDER"
else
  rounds_to_run=()
  for ((i=0; i<ROUNDS; i++)); do rounds_to_run+=("$i"); done
fi
completed_csv="$PRECOMPLETED_ROUNDS"
completed_new=0
for current_round in "${rounds_to_run[@]}"; do
  run_stage preflight preflight_one 64
  prepare_round "$current_round"
  begin="$(date -Iseconds)"; t0="$(date +%s)"
  run_pair_stage
  state="$(vexec "$MASTER" "d=\$(find '$AFS_OUT/results/round_${current_round}' -name '*.done' -type f 2>/dev/null | wc -l); f=\$(find '$AFS_OUT/results/round_${current_round}' -name '*.fail' -type f 2>/dev/null | wc -l); p=\$(find '$AFS_OUT/results/round_${current_round}' -name 'pair_*.json' -type f 2>/dev/null | wc -l); echo DONE=\$d FAIL=\$f PAIRS=\$p")"
  echo "ROUND_RESULT round=$current_round $state"
  if [[ "${#ROUND_FAILED_SLOTS[@]}" -eq 0 ]]; then
    vexec "$MASTER" "set -e; evidence='$AFS_OUT/hca_round_${current_round}.txt'; : > \$evidence; for slot in \$(seq 0 $((PAIR_LIMIT-1))); do log='$AFS_OUT/raw/round_${current_round}.pair_'\${slot}'.node_0.log'; grep -Fq 'MCCL_IB_HCA set to xscale_0' \$log; awk '/NET\\/IB : Using/{left=3} left>0{print; left--}' \$log >> \$evidence; done; grep -Fq xscale_0 \$evidence; for h in xscale_1 xscale_2 xscale_3; do ! grep -Fq \$h \$evidence; done; echo HCA_ROUND_EFFECTIVE=$current_round"
    completed_csv="${completed_csv:+$completed_csv,}$current_round"
    validate_round "$completed_csv"
  else
    for slot in "${ROUND_FAILED_SLOTS[@]}"; do
      printf '%s,%s,persistent_failure\n' "$current_round" "$slot" >>"$WORK/failed_pairs.csv"
    done
  fi
  run_stage "cleanup/round_$current_round" cleanup_round_one "${#round_pods[@]}"
  end="$(date -Iseconds)"; elapsed=$(( $(date +%s)-t0 ))
  printf '%s,%s,%s,%s\n' "$current_round" "$begin" "$end" "$elapsed" >>"$WORK/round_timing.csv"
  completed_new=$((completed_new+1))
  echo "ROUND_CHECKPOINT round=$current_round elapsed_s=$elapsed completed_new=$completed_new failed_pairs=${#ROUND_FAILED_SLOTS[@]}"
  if [[ $((completed_new % CHECKPOINT_EVERY)) -eq 0 ]]; then
    vexec_i "$MASTER" "cat > '$AFS_OUT/run.log'" <"$WORK/run.log"
    vexec_i "$MASTER" "cat > '$AFS_OUT/round_timing.csv'" <"$WORK/round_timing.csv"
    vexec_i "$MASTER" "cat > '$AFS_OUT/failed_pairs.csv'" <"$WORK/failed_pairs.csv"
    echo "CHECKPOINT_READY completed_new=$completed_new round=$current_round at=$(date -Iseconds)"
  fi
done

run_stage postcheck postcheck_one 64
trap - ERR
cat >"$WORK/manifest.yaml" <<EOF
run_id: $RUN_ID
session: $SESSION
status: VALID
probe: muxi_pair_round_robin
schedule_algorithm: circle-v1-direction-sha256-v1
rounds_total: 63
rounds_executed: ${#rounds_to_run[@]}
round_order: ${ROUND_ORDER:-sequential_0_to_$((ROUNDS-1))}
precompleted_rounds: ${PRECOMPLETED_ROUNDS:-none}
pairs_per_round: 32
pairs_executed_per_round: $PAIR_LIMIT
nbytes: $NBYTES
warmup: $WARMUP
iters: $ITERS
hca: xscale_0
gid_index: 5
tc: 128
vswitch: 1
parallel_pairs: 32
node_participation_per_round: 1
read_only_inventory: false
EOF
vexec_i "$MASTER" "cat > '$AFS_OUT/run.log'" <"$WORK/run.log"
vexec_i "$MASTER" "cat > '$AFS_OUT/manifest.yaml'" <"$WORK/manifest.yaml"
vexec_i "$MASTER" "cat > '$AFS_OUT/round_timing.csv'" <"$WORK/round_timing.csv"
vexec_i "$MASTER" "cat > '$AFS_OUT/failed_pairs.csv'" <"$WORK/failed_pairs.csv"
tar -C "$WORK" -cf - fire preflight postcheck cleanup | vexec_i "$MASTER" "tar -C '$AFS_OUT' -xf -"
echo "PAIR_VALID run_id=$RUN_ID rounds=$ROUNDS at=$(date -Iseconds)"
