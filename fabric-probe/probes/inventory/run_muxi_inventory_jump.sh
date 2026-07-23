#!/usr/bin/env bash
# 在 ais-cf3e61a5 上执行：64 pod 全并发、只读采集、失败节点最多重试2次。
set -euo pipefail

RUN_ID="${RUN_ID:?set RUN_ID}"
BUNDLE_DIR="${BUNDLE_DIR:?set BUNDLE_DIR}"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config-vc-c550-h3c-test.yaml}"
VCCTL="${VCCTL:-/usr/local/bin/vcctl}"
JOB="${JOB:-yinjinrun-cs512-20260716-221823}"
AFS_OUT="${AFS_OUT:-/afs-a3-weight-share/yinjinrun.p/results/muxi-fabric-w2/$RUN_ID}"
WORK="/tmp/muxi-inventory-$RUN_ID"
MASTER="$JOB-master-0"

mkdir -p "$WORK/attempts" "$WORK/raw" "$WORK/postcheck"
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

is_auth_error() {
  grep -Eiq 'Forbidden|Unauthorized|permission denied|certificate|credentials|authentication' "$1"
}

collect_one() {
  local rank="$1" attempt="$2" pod="${pods[$rank]}"
  local log="$WORK/attempts/node_${rank}.attempt${attempt}.log" rc=0
  "$VCCTL" pod exec -i "$pod" -- bash -s -- "$pod" \
    <"$BUNDLE_DIR/muxi_inventory_probe.sh" >"$log" 2>&1 || rc=$?
  printf '%s\n' "$rc" >"$WORK/attempts/node_${rank}.attempt${attempt}.rc"
  if [[ "$rc" -eq 0 ]]; then
    cp "$log" "$WORK/raw/${pod}.raw.log"
    printf '0\n' >"$WORK/raw/${pod}.rc"
  fi
  return "$rc"
}

postcheck_one() {
  local rank="$1" attempt="$2" pod="${pods[$rank]}"
  local log="$WORK/postcheck/node_${rank}.attempt${attempt}.log" rc=0
  vexec "$pod" \
    "left=\$(ps -eo args= | awk '/[m]uxi_inventory_probe\\.sh/ {print}'); [[ -z \"\$left\" ]]; echo CLEAN" \
    >"$log" 2>&1 || rc=$?
  return "$rc"
}

run_stage() {
  local stage="$1" callback="$2"
  local -a pending=() permanent=()
  local rank attempt i rc log
  for ((rank=0; rank<64; rank++)); do pending+=("$rank"); done
  echo "STAGE_BEGIN stage=$stage count=64 parallel=64 at=$(date -Iseconds)"
  for attempt in 1 2 3; do
    [[ "${#pending[@]}" -gt 0 ]] || break
    echo "STAGE_ATTEMPT stage=$stage attempt=$attempt pending=${#pending[@]} ranks=${pending[*]}"
    local -a pids=() ranks=() retry=()
    for rank in "${pending[@]}"; do
      "$callback" "$rank" "$attempt" &
      pids+=("$!"); ranks+=("$rank")
    done
    for i in "${!pids[@]}"; do
      rank="${ranks[$i]}"; rc=0
      wait "${pids[$i]}" || rc=$?
      [[ "$rc" -eq 0 ]] && continue
      log="$WORK/$stage/node_${rank}.attempt${attempt}.log"
      if is_auth_error "$log"; then
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

archive_work() {
  vexec "$MASTER" "mkdir -p '$AFS_OUT'"
  tar -C "$WORK" -cf - . | vexec_i "$MASTER" "tar -C '$AFS_OUT' -xf -"
}

echo "INVENTORY_BEGIN run_id=$RUN_ID at=$(date -Iseconds)"
echo "KUBECONFIG_PATH=$KUBECONFIG MODE=$(stat -c %a "$KUBECONFIG")"
"$VCCTL" pod get --job "$JOB" >"$WORK/pods.log"
python3 - "$WORK/pods.log" <<'PY'
import re,sys
lines=open(sys.argv[1]).read().splitlines()
pods=[x for x in lines if "yinjinrun-cs512-20260716-221823-" in x]
running=[x for x in pods if re.search(r"\bRunning\b",x)]
assert len(pods)==64 and len(running)==64,(len(pods),len(running))
print("POD_LIST_OK total=64 running=64")
PY

if ! run_stage attempts collect_one; then
  echo "INVENTORY_FAILED stage=collect"
  archive_work
  exit 21
fi

python3 "$BUNDLE_DIR/parse_muxi_inventory.py" \
  --raw-dir "$WORK/raw" \
  --expected-nodes 64 \
  --jsonl "$WORK/inventory.jsonl" \
  --csv "$WORK/rails.csv" \
  --summary-json "$WORK/validation_summary.json" \
  --summary-md "$WORK/SUMMARY.md"

if ! run_stage postcheck postcheck_one; then
  echo "INVENTORY_FAILED stage=postcheck"
  archive_work
  exit 22
fi

cat >"$WORK/manifest.yaml" <<EOF
run_id: $RUN_ID
status: VALID
probe: muxi_inventory
schema_version: muxi.inventory.v1
driver: jump_vcctl
cluster: vc-c550-h3c-test
job: $JOB
identity: yinjinrun.p
kubeconfig_path: $KUBECONFIG
kubeconfig_mode: 600
expected_pods: 64
parallelism: 64
max_attempts: 3
read_only: true
afs_out: $AFS_OUT
unavailable_tools_install: false
EOF

archive_work
echo "INVENTORY_VALID run_id=$RUN_ID at=$(date -Iseconds)"
