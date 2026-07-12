#!/usr/bin/env bash
# Muxi Dense TP×PP 扩 DP 战役（对标 run_mfu_tp_pp_scale_campaign.sh）
# 用法:
#   QUEUE=... LEDGER=... ./run_mfu_tp_pp_scale_campaign_muxi.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$OPS_ROOT/../.." && pwd)"

export CLUSTER_SSH_HOST_OVERRIDE="${CLUSTER_SSH_HOST_OVERRIDE:-ais-cf3e61a5}"
export CLUSTER_FORCE_JUMP="${CLUSTER_FORCE_JUMP:-1}"
# shellcheck source=muxi.env
source "$SCRIPT_DIR/muxi.env"

STATE_DIR="${STATE_DIR:-$OPS_ROOT/reports/rounds/mfu_tp_pp_scale_muxi_state}"
LEDGER="${LEDGER:-$OPS_ROOT/reports/rounds/mfu_tp_pp_scale_muxi_ledger.md}"
QUEUE="${QUEUE:-$STATE_DIR/queue_dense.jsonl}"
PEAK="${PEAK:-279.9}"
PARSE="$SCRIPT_DIR/parse_train_mfu_log.py"
TARGET_GBS="${TARGET_GBS:-2048}"
ITERS_DEFAULT="${ITERS_DEFAULT:-5}"
NCCL_IB_HCA="${NCCL_IB_HCA:-xscale}"
MCCL_IB_HCA="${MCCL_IB_HCA:-xscale}"

mkdir -p "$STATE_DIR" "$REPO_ROOT/logs" "$OPS_ROOT/reports/rounds"

if [[ ! -f "$LEDGER" ]]; then
  cat > "$LEDGER" <<'L'
# Muxi Dense 固定 TP×PP 扩 DP · MFU 弱扩展账本

> peak = 279.9 TFLOPS/卡 · 目标 GBS=2048 · SEQ=4096 · MBS=1 · IB_HCA=xscale
> Job：`yushan-muxi-card-screen-128-cp-copy` · 跳板 `ais-cf3e61a5`

| round | id | mode | scale | TP/PP/EP | DP | GBS | hostset | steady TFLOP/s/GPU | MFU% | status | log | note |
|------:|----|------|------:|----------|---:|----:|---------|-------------------:|-----:|--------|-----|------|
L
fi

if [[ ! -s "$QUEUE" ]]; then
  echo "FAIL: empty queue $QUEUE" >&2
  exit 2
fi

align_gbs() {
  local world="$1" tp="$2" pp="$3" cp="$4" mbs="$5" gbs="$6"
  local model_par=$((tp * pp * cp))
  if (( world % model_par != 0 )); then echo "INVALID"; return 1; fi
  local dp=$((world / model_par))
  local unit=$((mbs * dp))
  if (( gbs % unit != 0 )); then
    gbs=$(( ((gbs + unit - 1) / unit) * unit ))
  fi
  echo "$gbs"
}

ROUND_FILE="$STATE_DIR/round_counter"
[[ -f "$ROUND_FILE" ]] || echo 0 > "$ROUND_FILE"

while true; do
  line="$(head -n 1 "$QUEUE" || true)"
  if [[ -z "${line:-}" ]]; then
    echo "QUEUE_EMPTY"
    break
  fi
  # queue json fields: id,mode,world,tp,pp,ep,hostset,gbs,iters,note
  id="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["id"])' "$line")"
  mode="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("mode","dense"))' "$line")"
  world="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["world"])' "$line")"
  tp="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["tp"])' "$line")"
  pp="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["pp"])' "$line")"
  ep="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("ep",1))' "$line")"
  hostset="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("hostset","clean"))' "$line")"
  gbs="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("gbs",0) or 0)' "$line")"
  iters="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("iters",0) or 0)' "$line")"
  note="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("note",""))' "$line")"
  [[ "$iters" -gt 0 ]] || iters="$ITERS_DEFAULT"
  [[ "$gbs" -gt 0 ]] || gbs="$TARGET_GBS"
  gbs="$(align_gbs "$world" "$tp" "$pp" 1 1 "$gbs")"
  if [[ "$gbs" == "INVALID" ]]; then
    echo "SKIP invalid world=$world tp=$tp pp=$pp"
    tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
    continue
  fi
  dp=$((world / (tp * pp)))
  round=$(( $(cat "$ROUND_FILE") + 1 ))
  echo "$round" > "$ROUND_FILE"
  stamp="$(date +%Y%m%d_%H%M%S)"
  afs_out="${AFS_RESULTS}/mfu-dense-muxi-${stamp}-${id}-w${world}"
  echo "==> ROUND $round id=$id world=$world TP/PP/EP=$tp/$pp/$ep DP=$dp GBS=$gbs hostset=$hostset"
  echo "{\"round\":$round,\"id\":\"$id\",\"afs\":\"$afs_out\",\"ts\":\"$(date -Iseconds)\"}" > "$STATE_DIR/current.json"

  set +e
  AFS_OUT="$afs_out" WORLD="$world" TP="$tp" PP="$pp" GBS="$gbs" TRAIN_ITERS="$iters" \
    HOSTSET="$hostset" NCCL_IB_HCA="$NCCL_IB_HCA" MCCL_IB_HCA="$MCCL_IB_HCA" \
    bash "$SCRIPT_DIR/fire_train_dense_muxi.sh"
  fire_ec=$?
  # poll until done/fail (max ~2h)
  ok=0
  for _ in $(seq 1 240); do
    sleep 30
    status="$(ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
      "KUBECONFIG=$CLUSTER_KUBECONFIG vcctl pod exec ${CLUSTER_JOB}-master-0 -- bash -lc '
        ok=0; fail=0; miss=0; n=$((world/8))
        for r in \$(seq 0 \$((n-1))); do
          if [[ -f $afs_out/node_\$r.done ]]; then ok=\$((ok+1))
          elif [[ -f $afs_out/node_\$r.fail ]]; then fail=\$((fail+1))
          else miss=\$((miss+1)); fi
        done
        echo OK=\$ok FAIL=\$fail MISS=\$miss
      '" 2>/dev/null || echo OK=0 FAIL=0 MISS=9)"
    echo "  poll $status"
    if echo "$status" | grep -q "FAIL=[1-9]"; then ok=0; break; fi
    nnodes=$((world / 8))
    if echo "$status" | grep -q "OK=${nnodes} FAIL=0 MISS=0"; then ok=1; break; fi
  done
  set -e

  # fetch log and parse
  local_log="$REPO_ROOT/logs/mfu-dense-muxi-${stamp}-${id}/train.log"
  mkdir -p "$(dirname "$local_log")"
  ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "KUBECONFIG=$CLUSTER_KUBECONFIG vcctl pod exec ${CLUSTER_JOB}-master-0 -- bash -lc 'cat $afs_out/train.log 2>/dev/null || cat $afs_out/node_0.outer.log 2>/dev/null'" \
    > "$local_log" 2>/dev/null || true

  tflop="-"
  mfu="-"
  status_s="fail"
  if [[ "$ok" -eq 1 && -s "$local_log" ]]; then
    parsed="$(python3 "$PARSE" --peak "$PEAK" --drop-first 1 "$local_log" 2>/dev/null || true)"
    tflop="$(echo "$parsed" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("median_tflop","-"))' 2>/dev/null || echo -)"
    mfu="$(echo "$parsed" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("mfu_pct","-"))' 2>/dev/null || echo -)"
    status_s="ok"
  fi
  echo "| $round | $id | $mode | $world | ${tp}/${pp}/${ep} | $dp | $gbs | $hostset | $tflop | $mfu | $status_s | $local_log | $note |" >> "$LEDGER"
  echo "{\"ts\":\"$(date -Iseconds)\",\"phase\":\"P2a\",\"id\":\"$id\",\"status\":\"$status_s\",\"mfu\":\"$mfu\"}" \
    >> "$REPO_ROOT/logs/muxi-watchdog-20260712/check.jsonl"

  tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
done

echo "CAMPAIGN_DENSE_MUXI_DONE ledger=$LEDGER"
