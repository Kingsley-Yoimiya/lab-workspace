#!/usr/bin/env bash
# 跳板执行：配置与拓扑事实优先的低风险只读探测（A/B/C + 报告）。
# 身份：yinjinrun.p；KUBECONFIG=~/.kube/config-vc-c550-h3c-test.yaml
set -euo pipefail

RUN_ID="${RUN_ID:?set RUN_ID}"
BUNDLE_DIR="${BUNDLE_DIR:?set BUNDLE_DIR}"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config-vc-c550-h3c-test.yaml}"
VCCTL="${VCCTL:-/usr/local/bin/vcctl}"
KUBECTL="${KUBECTL:-/root/.cache/volcano/kubectl/kubectl}"
JOB="${JOB:-yinjinrun-cs512-20260716-221823}"
AFS_OUT="${AFS_OUT:-/afs-a3-weight-share/yinjinrun.p/results/muxi-topology-facts/$RUN_ID}"
LLDP_DURATION="${LLDP_DURATION:-70}"
WORK="/tmp/muxi-topology-$RUN_ID"
MASTER="$JOB-master-0"

mkdir -p "$WORK"/{control_plane,logical/raw,lldp/pods,report,attempts,postcheck}
exec > >(tee -a "$WORK/run.log") 2>&1
export KUBECONFIG

echo "TOPOLOGY_FACTS_BEGIN run_id=$RUN_ID at=$(date -Iseconds)"
echo "KUBECONFIG_PATH=$KUBECONFIG MODE=$(stat -c %a "$KUBECONFIG")"
echo "VCCTL=$VCCTL KUBECTL=$KUBECTL JOB=$JOB"

is_auth_error() {
  grep -Eiq 'Forbidden|Unauthorized|permission denied|certificate|credentials|authentication|IDENTITY_MISMATCH|AUTH_ERROR' "$1"
}

vexec() {
  local pod="$1"; shift
  "$VCCTL" pod exec "$pod" -- bash -lc "$1"
}

vexec_i() {
  local pod="$1"; shift
  "$VCCTL" pod exec -i "$pod" -- bash -c "$1"
}

pods=("$MASTER")
for i in $(seq 0 62); do pods+=("$JOB-worker-$i"); done

"$VCCTL" pod get --job "$JOB" >"$WORK/pods.log"
python3 - "$WORK/pods.log" <<'PY'
import re,sys
lines=open(sys.argv[1]).read().splitlines()
pods=[x for x in lines if "yinjinrun-cs512-20260716-221823-" in x]
running=[x for x in pods if re.search(r"\bRunning\b",x)]
assert len(pods)==64 and len(running)==64,(len(pods),len(running))
print("POD_LIST_OK total=64 running=64")
PY

# ---------- A. 控制面 ----------
echo "STAGE_A_CONTROL_PLANE begin=$(date -Iseconds)"
python3 "$BUNDLE_DIR/collect_control_plane.py" \
  --job "$JOB" \
  --kubectl "$KUBECTL" \
  --kubeconfig "$KUBECONFIG" \
  --out-dir "$WORK/control_plane" \
  | tee "$WORK/control_plane/collect.stdout"
echo "STAGE_A_CONTROL_PLANE end=$(date -Iseconds)"

# ---------- B. 64×4 逻辑 rail 并发 ----------
echo "STAGE_B_LOGICAL begin=$(date -Iseconds)"
run_logical_rank() {
  local rank="$1" attempt="$2" pod="${pods[$rank]}"
  local log="$WORK/attempts/logical_${rank}.a${attempt}.log" rc=0
  local out_remote="/tmp/logical_${RUN_ID}.json"
  "$VCCTL" pod exec -i "$pod" -- bash -lc "
    set -euo pipefail
    cat > /tmp/muxi_logical_rail_probe.py
    python3 /tmp/muxi_logical_rail_probe.py --pod '$pod' --out '$out_remote'
    cat '$out_remote'
  " <"$BUNDLE_DIR/muxi_logical_rail_probe.py" >"$log" 2>&1 || rc=$?
  printf '%s\n' "$rc" >"$WORK/attempts/logical_${rank}.a${attempt}.rc"
  if [[ "$rc" -eq 0 ]]; then
    # 日志末尾是 summary；完整 JSON 在远程，再拉一次
    "$VCCTL" pod exec "$pod" -- bash -lc "cat '$out_remote'" \
      >"$WORK/logical/raw/${pod}.json" 2>"$WORK/attempts/logical_${rank}.a${attempt}.pull.err" || rc=$?
  fi
  if is_auth_error "$log"; then
    echo "AUTH_ERROR logical rank=$rank" >&2
    return 13
  fi
  return "$rc"
}

pending=()
for ((rank=0; rank<64; rank++)); do pending+=("$rank"); done
for attempt in 1 2 3; do
  [[ "${#pending[@]}" -gt 0 ]] || break
  echo "LOGICAL_ATTEMPT attempt=$attempt pending=${#pending[@]}"
  pids=(); ranks=(); retry=()
  for rank in "${pending[@]}"; do
    run_logical_rank "$rank" "$attempt" &
    pids+=("$!"); ranks+=("$rank")
  done
  auth_fail=0
  for i in "${!pids[@]}"; do
    rc=0
    wait "${pids[$i]}" || rc=$?
    if [[ "$rc" -eq 13 ]]; then auth_fail=1; fi
    if [[ "$rc" -ne 0 ]]; then
      if [[ "$attempt" -lt 3 && "$rc" -ne 13 ]]; then
        retry+=("${ranks[$i]}")
      else
        echo "LOGICAL_PERMANENT rank=${ranks[$i]} rc=$rc"
      fi
    fi
  done
  if [[ "$auth_fail" -eq 1 ]]; then
    echo "STOP_AUTH_ERROR stage=logical"
    exit 13
  fi
  pending=("${retry[@]}")
  [[ "${#pending[@]}" -eq 0 ]] || sleep "$attempt"
done
raw_count=$(ls -1 "$WORK/logical/raw"/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "LOGICAL_RAW_COUNT=$raw_count"
[[ "$raw_count" -eq 64 ]] || { echo "LOGICAL_INCOMPLETE"; exit 21; }

python3 "$BUNDLE_DIR/parse_muxi_logical_rail.py" \
  --raw-dir "$WORK/logical/raw" \
  --out-dir "$WORK/logical" | tee "$WORK/logical/parse.stdout"
echo "STAGE_B_LOGICAL end=$(date -Iseconds)"

# ---------- C. LLDP：先 master，必要时扩 64 ----------
echo "STAGE_C_LLDP_MASTER begin=$(date -Iseconds)"
master_lldp_dir="$WORK/lldp/pods/$MASTER"
mkdir -p "$master_lldp_dir"
"$VCCTL" pod exec -i "$MASTER" -- bash -lc "
  set -euo pipefail
  cat > /tmp/muxi_lldp_listen.py
  rm -rf /tmp/lldp_${RUN_ID}
  mkdir -p /tmp/lldp_${RUN_ID}
  python3 /tmp/muxi_lldp_listen.py \
    --pod '$MASTER' \
    --ifaces eth0,net1,net2,net3,net4 \
    --duration $LLDP_DURATION \
    --out-dir /tmp/lldp_${RUN_ID} \
    >/tmp/lldp_${RUN_ID}/listener.stdout 2>/tmp/lldp_${RUN_ID}/listener.stderr
  # 仅二进制 tar 走 stdout，避免污染归档
  tar -C /tmp/lldp_${RUN_ID} -cf - .
" <"$BUNDLE_DIR/muxi_lldp_listen.py" >"$WORK/lldp/master_bundle.tar" 2>"$WORK/lldp/master_exec.err" || {
  rc=$?
  if is_auth_error "$WORK/lldp/master_exec.err"; then
    echo "STOP_AUTH_ERROR stage=lldp_master"
    exit 13
  fi
  echo "LLDP_MASTER_WARN rc=$rc — continue with empty master result"
}
if [[ -s "$WORK/lldp/master_bundle.tar" ]] && tar -tf "$WORK/lldp/master_bundle.tar" >/dev/null 2>&1; then
  tar -C "$master_lldp_dir" -xf "$WORK/lldp/master_bundle.tar"
else
  echo "LLDP_MASTER_TAR_INVALID size=$(wc -c <"$WORK/lldp/master_bundle.tar" 2>/dev/null || echo 0)"
  # 回退：直接 cat summary.json
  "$VCCTL" pod exec "$MASTER" -- bash -lc "test -f /tmp/lldp_${RUN_ID}/summary.json && cat /tmp/lldp_${RUN_ID}/summary.json" \
    >"$master_lldp_dir/summary.json" 2>>"$WORK/lldp/master_exec.err" || true
fi
python3 - "$master_lldp_dir/summary.json" <<'PY' | tee "$WORK/lldp/master_decision.txt"
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
    print("MASTER_LLDP any=False reason=no_summary")
    raise SystemExit(0)
s=json.loads(p.read_text())
print("MASTER_LLDP any=%s total=%s" % (s.get("any_lldp"), s.get("total_lldp_frames")))
for iface in s.get("ifaces",[]):
    print("IFACE", iface.get("iface"), "lldp", iface.get("lldp_frames"), "raw", iface.get("raw_frames_seen"), "dur", iface.get("duration_s"), "status", iface.get("status"))
PY

expand=0
if [[ -f "$master_lldp_dir/summary.json" ]]; then
  expand=$(python3 -c "import json;print(1 if json.load(open('$master_lldp_dir/summary.json')).get('any_lldp') else 0)")
fi

if [[ "$expand" -eq 1 ]]; then
  echo "STAGE_C_LLDP_EXPAND64 begin=$(date -Iseconds)"
  run_lldp_rank() {
    local rank="$1" pod="${pods[$rank]}"
    local log="$WORK/attempts/lldp_${rank}.log" rc=0
    local dest="$WORK/lldp/pods/$pod"
    mkdir -p "$dest"
    "$VCCTL" pod exec -i "$pod" -- bash -lc "
      set -euo pipefail
      cat > /tmp/muxi_lldp_listen.py
      rm -rf /tmp/lldp_${RUN_ID}
      mkdir -p /tmp/lldp_${RUN_ID}
      python3 /tmp/muxi_lldp_listen.py \
        --pod '$pod' \
        --ifaces eth0,net1,net2,net3,net4 \
        --duration $LLDP_DURATION \
        --out-dir /tmp/lldp_${RUN_ID} \
        >/tmp/lldp_${RUN_ID}/listener.stdout 2>/tmp/lldp_${RUN_ID}/listener.stderr
      tar -C /tmp/lldp_${RUN_ID} -cf - .
    " <"$BUNDLE_DIR/muxi_lldp_listen.py" >"$dest/bundle.tar" 2>"$log" || rc=$?
    if [[ "$rc" -eq 0 && -s "$dest/bundle.tar" ]] && tar -tf "$dest/bundle.tar" >/dev/null 2>&1; then
      tar -C "$dest" -xf "$dest/bundle.tar"
    fi
    if is_auth_error "$log"; then return 13; fi
    return "$rc"
  }
  pids=()
  for rank in $(seq 0 63); do
    # master 已完成，仍重跑以统一产物；可接受
    run_lldp_rank "$rank" &
    pids+=("$!")
  done
  auth_fail=0
  for pid in "${pids[@]}"; do
    rc=0
    wait "$pid" || rc=$?
    [[ "$rc" -eq 13 ]] && auth_fail=1
  done
  [[ "$auth_fail" -eq 0 ]] || { echo "STOP_AUTH_ERROR stage=lldp_expand"; exit 13; }
  echo "STAGE_C_LLDP_EXPAND64 end=$(date -Iseconds)"
else
  echo "STAGE_C_LLDP_NO_EXPAND reason=master_no_lldp"
fi

python3 "$BUNDLE_DIR/parse_muxi_lldp.py" \
  --lldp-root "$WORK/lldp/pods" \
  --out-dir "$WORK/lldp" | tee "$WORK/lldp/parse.stdout"
echo "STAGE_C_LLDP end=$(date -Iseconds)"

# ---------- E. 报告 ----------
python3 "$BUNDLE_DIR/aggregate_topology_report.py" \
  --work-dir "$WORK" \
  --out-dir "$WORK/report" | tee "$WORK/report/aggregate.stdout"

cat >"$WORK/manifest.yaml" <<EOF
run_id: $RUN_ID
status: COMPLETE
probe: muxi_topology_facts
goal: config_and_topology_facts_first
driver: jump_vcctl_kubectl
cluster: vc-c550-h3c-test
job: $JOB
identity: yinjinrun.p
kubeconfig_path: $KUBECONFIG
kubectl_path: $KUBECTL
expected_pods: 64
parallelism: 64
lldp_duration_s: $LLDP_DURATION
lldp_expanded_64: $expand
read_only: true
no_frame_tx: true
afs_out: $AFS_OUT
EOF

# 归档到 AFS
vexec "$MASTER" "mkdir -p '$AFS_OUT'"
tar -C "$WORK" -cf - . | vexec_i "$MASTER" "tar -C '$AFS_OUT' -xf -"
echo "ARCHIVED_AFS=$AFS_OUT"
echo "TOPOLOGY_FACTS_END run_id=$RUN_ID at=$(date -Iseconds)"
