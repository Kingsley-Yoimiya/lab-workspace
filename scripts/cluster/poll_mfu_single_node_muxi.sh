#!/usr/bin/env bash
# 轮询单机矩阵 running 表，写回 ledger
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_SSH_HOST_OVERRIDE="${CLUSTER_SSH_HOST_OVERRIDE:-ais-cf3e61a5}"
export CLUSTER_FORCE_JUMP="${CLUSTER_FORCE_JUMP:-1}"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

LEDGER="${LEDGER:-$SCRIPT_DIR/../../reports/rounds/mfu_single_node_muxi_ledger.md}"
RUNNING="${LEDGER}.running"
PARSE="$SCRIPT_DIR/parse_train_mfu_log.py"
PEAK="${PEAK:-279.9}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LOCAL_LOG_ROOT="$REPO_ROOT/logs/muxi-mfu-sn"
WATCH="$REPO_ROOT/logs/muxi-watchdog-20260712/check.jsonl"
mkdir -p "$LOCAL_LOG_ROOT"

[[ -f "$RUNNING" ]] || { echo "no running file"; exit 0; }

tmp="$(mktemp)"
# 用文件描述符 3 读表，避免 ssh 吞掉 while-read 的 stdin
exec 3<"$RUNNING"
while IFS='|' read -r round id mode pod_logic tpe gbs seq afs_out note <&3; do
  [[ -z "${id:-}" ]] && continue
  pod="${CLUSTER_JOB}-${pod_logic}"
  status=$(ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "KUBECONFIG=$CLUSTER_KUBECONFIG vcctl pod exec ${pod} -- bash -lc '
      if [[ -f $afs_out/node_0.done ]]; then echo OK
      elif [[ -f $afs_out/node_0.fail ]]; then echo FAIL
      else echo RUN; fi
    '" </dev/null 2>/dev/null || echo RUN)
  if [[ "$status" == "RUN" ]]; then
    echo "$round|$id|$mode|$pod_logic|$tpe|$gbs|$seq|$afs_out|$note" >> "$tmp"
    echo "RUN $id"
    continue
  fi
  local_log="$LOCAL_LOG_ROOT/${id}.train.log"
  ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "KUBECONFIG=$CLUSTER_KUBECONFIG vcctl pod exec ${pod} -- bash -lc 'cat $afs_out/train.log 2>/dev/null || cat $afs_out/node_0.outer.log'" \
    </dev/null > "$local_log" 2>/dev/null || true
  tflop="-"; mfu="-"; st="$status"
  if [[ "$status" == "OK" && -s "$local_log" ]]; then
    js=$(python3 "$PARSE" --peak "$PEAK" --drop-first 1 --json "$local_log" 2>/dev/null || true)
    tflop=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("steady",{}).get("tflops_median","-"))' "$js" 2>/dev/null || echo -)
    mfu=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); v=d.get("mfu_pct"); print(f"{v:.2f}" if v is not None else "-")' "$js" 2>/dev/null || echo -)
    st="ok"
  else
    st="fail"
  fi
  echo "| $round | $id | $mode | $pod_logic | $tpe | $gbs | $seq | $tflop | $mfu | $st | $local_log | $note |" >> "$LEDGER"
  echo "{\"ts\":\"$(date -Iseconds)\",\"phase\":\"P2sn\",\"id\":\"$id\",\"status\":\"$st\",\"mfu\":\"$mfu\"}" >> "$WATCH"
  echo "DONE $id status=$st mfu=$mfu"
done
exec 3<&-
mv "$tmp" "$RUNNING"
wc -l "$RUNNING" || true
