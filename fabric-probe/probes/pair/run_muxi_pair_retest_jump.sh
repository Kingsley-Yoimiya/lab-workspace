#!/usr/bin/env bash
# 跳板执行：预注册 retest schedule（node-disjoint batches），固定 xscale_0/16MiB。
# 单次异常保留并继续；系统性失败/身份权限/清理失败才停。
set -euo pipefail

RUN_ID="${RUN_ID:?set RUN_ID}"
BUNDLE_DIR="${BUNDLE_DIR:?set BUNDLE_DIR}"
SCHEDULE_JSONL="${SCHEDULE_JSONL:-$BUNDLE_DIR/retest_schedule.jsonl}"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config-vc-c550-h3c-test.yaml}"
VCCTL="${VCCTL:-/usr/local/bin/vcctl}"
JOB="${JOB:-yinjinrun-cs512-20260716-221823}"
AFS_OUT="${AFS_OUT:-/afs-a3-weight-share/yinjinrun.p/results/muxi-fabric-w2/$RUN_ID}"
BASE_PORT="${BASE_PORT:-32000}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
NBYTES="${NBYTES:-16777216}"
SESSION="${SESSION:-retest}"
SKIP_BATCHES="${SKIP_BATCHES:-}"
WORK="/tmp/muxi-pair-retest-$RUN_ID-$SESSION"
MASTER="$JOB-master-0"

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

echo "RETEST_BEGIN run_id=$RUN_ID at=$(date -Iseconds)"
"$VCCTL" pod get --job "$JOB" >"$WORK/pods.log"
python3 - "$WORK/pods.log" <<'PY'
import re,sys
lines=open(sys.argv[1]).read().splitlines()
pods=[x for x in lines if "yinjinrun-cs512-20260716-221823-" in x]
running=[x for x in pods if re.search(r"\bRunning\b",x)]
assert len(pods)==64 and len(running)==64,(len(pods),len(running))
print("POD_LIST_OK total=64 running=64")
PY

# identity check
id_out="$(vexec "$MASTER" "cat /var/run/secrets/kubernetes.io/serviceaccount/namespace 2>/dev/null; id; echo AFS_PREFIX_OK=\$(test -d /afs-a3-weight-share/yinjinrun.p && echo 1 || echo 0)")"
echo "$id_out"
echo "$id_out" | grep -q 'AFS_PREFIX_OK=1'

vexec "$MASTER" "mkdir -p '$AFS_OUT/code' '$AFS_OUT/raw' '$AFS_OUT/results' '$AFS_OUT/retest'"
vexec_i "$MASTER" "cat > '$AFS_OUT/code/muxi_pair_bench.py'" <"$BUNDLE_DIR/muxi_pair_bench.py"
vexec_i "$MASTER" "cat > '$AFS_OUT/retest_schedule.jsonl'" <"$SCHEDULE_JSONL"
if [[ -f "$BUNDLE_DIR/retest_manifest.json" ]]; then
  vexec_i "$MASTER" "cat > '$AFS_OUT/retest_manifest.json'" <"$BUNDLE_DIR/retest_manifest.json"
fi

mapfile -t BATCH_IDS < <(python3 - "$SCHEDULE_JSONL" "$SKIP_BATCHES" <<'PY'
import json,sys
skip={int(x) for x in sys.argv[2].split(",") if x.strip()!=""}
rounds=sorted({json.loads(l)["round"] for l in open(sys.argv[1]) if l.strip()} - skip)
print("\n".join(str(r) for r in rounds))
PY
)
echo "BATCHES=${BATCH_IDS[*]} skip=${SKIP_BATCHES:-none}"

current_round=0
declare -a round_pods=() ROUND_SUCCESS_SLOTS=() ROUND_FAILED_SLOTS=()
PAIR_LIMIT=0

prepare_round() {
  local round_id="$1"
  local round_dir="$WORK/launchers/round_$round_id"
  mkdir -p "$round_dir" "$WORK/fire/round_$round_id" "$WORK/cleanup/round_$round_id"
  python3 - "$SCHEDULE_JSONL" "$round_id" >"$WORK/round_${round_id}.tsv" <<'PY'
import json,sys
rid=int(sys.argv[2])
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip() and json.loads(l)["round"]==rid]
rows=sorted(rows, key=lambda r:r["slot"])
for r in rows:
    print("\t".join(str(r[k]) for k in ("slot","edge_id","src_index","dst_index","src_pod","dst_pod")))
print(f"COUNT\t{len(rows)}", file=sys.stderr)
PY
  round_pods=()
  PAIR_LIMIT=0
  while IFS=$'\t' read -r slot edge src_index dst_index src_pod dst_pod; do
    port=$((BASE_PORT + round_id*32 + slot))
    for node_rank in 0 1; do
      idx=$((slot*2+node_rank))
      if [[ "$node_rank" -eq 0 ]]; then pod="$src_pod"; else pod="$dst_pod"; fi
      script="$round_dir/node_${idx}.sh"
      log="$AFS_OUT/raw/batch_${round_id}.pair_${slot}.node_${node_rank}.log"
      donef="$AFS_OUT/results/batch_${round_id}/pair_${slot}.node_${node_rank}.done"
      failf="$AFS_OUT/results/batch_${round_id}/pair_${slot}.node_${node_rank}.fail"
      cat >"$script" <<EOF
#!/usr/bin/env bash
export PATH=/opt/conda/bin:\${PATH:-/usr/bin}
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0 MCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=xscale_0 MCCL_IB_HCA=xscale_0
export NCCL_IB_GID_INDEX=5 MCCL_IB_GID_INDEX=5 MCCL_IB_TC=128
export MCCL_ENABLE_VSWITCH=1 MCCL_PCIE_BUFFER_MODE=0 FORCE_ACTIVE_WAIT=2
export NCCL_DEBUG=INFO MCCL_DEBUG=INFO
mkdir -p '$AFS_OUT/results/batch_${round_id}'
rm -f '$donef' '$failf'
/opt/conda/bin/torchrun --nnodes=2 --node_rank=$node_rank --nproc_per_node=1 \
  --master_addr=${src_pod}.${JOB} --master_port=\${PAIR_MASTER_PORT:-$port} \
  '$AFS_OUT/code/muxi_pair_bench.py' --nbytes $NBYTES --warmup $WARMUP --iters $ITERS \
  --round $round_id --slot $slot --edge-id $edge --src-index $src_index --dst-index $dst_index \
  --src-pod '$src_pod' --dst-pod '$dst_pod' --hca xscale_0 \
  --out '$AFS_OUT/results/batch_${round_id}/pair_${slot}.json' >'$log' 2>&1
rc=\$?
if [[ \$rc -eq 0 ]]; then echo OK >'$donef'; else echo FAIL:\$rc >'$failf'; fi
exit \$rc
EOF
      chmod +x "$script"
      round_pods+=("$pod")
    done
    PAIR_LIMIT=$((PAIR_LIMIT+1))
  done <"$WORK/round_${round_id}.tsv"
  local uniq
  uniq="$(printf '%s\n' "${round_pods[@]}" | sort -u | wc -l)"
  [[ "${#round_pods[@]}" -eq $((PAIR_LIMIT*2)) && "$uniq" -eq $((PAIR_LIMIT*2)) ]]
  tar -C "$round_dir" -cf - . | vexec_i "$MASTER" "mkdir -p '$AFS_OUT/code/batch_$round_id'; tar -C '$AFS_OUT/code/batch_$round_id' -xf -"
  echo "BATCH_PREPARED round=$round_id pairs=$PAIR_LIMIT nodes=${#round_pods[@]}"
}

run_pair_one() {
  local slot="$1" attempt="$2"
  local src_idx=$((slot*2)) dst_idx=$((slot*2+1))
  local src_pod="${round_pods[$src_idx]}" dst_pod="${round_pods[$dst_idx]}"
  local port=$((BASE_PORT + current_round*32 + slot + (attempt-1)*4096))
  local dir="$WORK/fire/round_${current_round}" r0=0 r1=0
  if [[ "$attempt" -gt 1 ]]; then
    vexec "$MASTER" \
      "rm -f '$AFS_OUT/results/batch_${current_round}/pair_${slot}.json' '$AFS_OUT/results/batch_${current_round}/pair_${slot}.node_0.done' '$AFS_OUT/results/batch_${current_round}/pair_${slot}.node_1.done' '$AFS_OUT/results/batch_${current_round}/pair_${slot}.node_0.fail' '$AFS_OUT/results/batch_${current_round}/pair_${slot}.node_1.fail'"
  fi
  {
    echo "POD=$src_pod NODE_RANK=0 ROUND=$current_round SLOT=$slot ATTEMPT=$attempt PORT=$port"
    vexec "$src_pod" \
      "PAIR_MASTER_PORT=$port timeout 150 bash '$AFS_OUT/code/batch_${current_round}/node_${src_idx}.sh'"
  } >"$dir/pair_${slot}.attempt${attempt}.node_0.log" 2>&1 & p0=$!
  {
    echo "POD=$dst_pod NODE_RANK=1 ROUND=$current_round SLOT=$slot ATTEMPT=$attempt PORT=$port"
    vexec "$dst_pod" \
      "PAIR_MASTER_PORT=$port timeout 150 bash '$AFS_OUT/code/batch_${current_round}/node_${dst_idx}.sh'"
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
  # 单次异常保留继续：不因少量失败停止
  return 0
}

cleanup_round_one() {
  local idx="$1" attempt="$2"
  local pod="${round_pods[$idx]}"
  local log="$WORK/cleanup/round_${current_round}/node_${idx}.attempt${attempt}.log" rc=0
  # pkill 无匹配时非0；必须整体 || true，避免假清理失败中止计划
  vexec "$pod" \
    "bash -lc 'pkill -f muxi_pair_bench.py >/dev/null 2>&1 || true; pkill -f torchrun >/dev/null 2>&1 || true; echo CLEAN'" \
    >"$log" 2>&1 || rc=$?
  # vcctl瞬时EOF：只记日志，由run_stage重试；三次仍失败也不要把已写完结果的作业清掉
  return "$rc"
}

on_round_error() {
  echo "RETEST_DRIVER_ERROR round=$current_round at=$(date -Iseconds)"
  if [[ "${#round_pods[@]}" -gt 0 ]]; then
    run_stage "cleanup/round_$current_round" cleanup_round_one "${#round_pods[@]}" || true
  fi
}
trap on_round_error ERR

printf 'batch,start,end,fire_elapsed_s,success,failed\n' >"$WORK/batch_timing.csv"
printf 'batch,slot,status\n' >"$WORK/failed_pairs.csv"
total_fail=0
for current_round in "${BATCH_IDS[@]}"; do
  run_stage preflight preflight_one 64
  prepare_round "$current_round"
  begin="$(date -Iseconds)"; t0="$(date +%s)"
  run_pair_stage
  state="$(vexec "$MASTER" "d=\$(find '$AFS_OUT/results/batch_${current_round}' -name '*.done' -type f 2>/dev/null | wc -l); f=\$(find '$AFS_OUT/results/batch_${current_round}' -name '*.fail' -type f 2>/dev/null | wc -l); p=\$(find '$AFS_OUT/results/batch_${current_round}' -name 'pair_*.json' -type f 2>/dev/null | wc -l); echo DONE=\$d FAIL=\$f PAIRS=\$p")"
  echo "BATCH_RESULT round=$current_round $state"
  for slot in "${ROUND_FAILED_SLOTS[@]:-}"; do
    [[ -n "${slot:-}" ]] || continue
    printf '%s,%s,persistent_failure\n' "$current_round" "$slot" >>"$WORK/failed_pairs.csv"
    total_fail=$((total_fail+1))
  done
  if [[ "${#ROUND_FAILED_SLOTS[@]}" -eq 0 ]]; then
    vexec "$MASTER" "evidence='$AFS_OUT/hca_batch_${current_round}.txt'; : > \$evidence; for slot in \$(seq 0 $((PAIR_LIMIT-1))); do log='$AFS_OUT/raw/batch_${current_round}.pair_'\${slot}'.node_0.log'; grep -Fq 'MCCL_IB_HCA set to xscale_0' \$log || true; awk '/NET\\/IB : Using/{left=3} left>0{print; left--}' \$log >> \$evidence; done; echo HCA_BATCH=${current_round}"
  fi
  # 清理失败不中止已成功写盘的批次；仅记录并继续
  if ! run_stage "cleanup/round_$current_round" cleanup_round_one "${#round_pods[@]}"; then
    echo "CLEANUP_WARN round=$current_round continuing_plan=1 at=$(date -Iseconds)"
  fi
  end="$(date -Iseconds)"; elapsed=$(( $(date +%s)-t0 ))
  printf '%s,%s,%s,%s,%s,%s\n' "$current_round" "$begin" "$end" "$elapsed" "${#ROUND_SUCCESS_SLOTS[@]}" "${#ROUND_FAILED_SLOTS[@]}" >>"$WORK/batch_timing.csv"
  echo "BATCH_CHECKPOINT round=$current_round elapsed_s=$elapsed at=$(date -Iseconds)"
  vexec_i "$MASTER" "cat > '$AFS_OUT/run.log'" <"$WORK/run.log"
  vexec_i "$MASTER" "cat > '$AFS_OUT/batch_timing.csv'" <"$WORK/batch_timing.csv"
  vexec_i "$MASTER" "cat > '$AFS_OUT/failed_pairs.csv'" <"$WORK/failed_pairs.csv"
done

run_stage postcheck postcheck_one 64
trap - ERR
cat >"$WORK/manifest.yaml" <<EOF
run_id: $RUN_ID
session: $SESSION
status: RETEST_DONE
probe: muxi_pair_retest
batches: ${#BATCH_IDS[@]}
nbytes: $NBYTES
warmup: $WARMUP
iters: $ITERS
hca: xscale_0
total_persistent_fail_slots: $total_fail
EOF
vexec_i "$MASTER" "cat > '$AFS_OUT/run.log'" <"$WORK/run.log"
vexec_i "$MASTER" "cat > '$AFS_OUT/manifest.yaml'" <"$WORK/manifest.yaml"
vexec_i "$MASTER" "cat > '$AFS_OUT/batch_timing.csv'" <"$WORK/batch_timing.csv"
vexec_i "$MASTER" "cat > '$AFS_OUT/failed_pairs.csv'" <"$WORK/failed_pairs.csv"
tar -C "$WORK" -cf - fire preflight postcheck cleanup | vexec_i "$MASTER" "tar -C '$AFS_OUT' -xf -"
echo "RETEST_VALID run_id=$RUN_ID batches=${#BATCH_IDS[@]} fail_slots=$total_fail at=$(date -Iseconds)"
