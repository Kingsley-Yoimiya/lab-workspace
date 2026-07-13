#!/usr/bin/env bash
# 单机 Dense/MoE MFU：严格顺序（一条跑完再下一条），默认全打到同一 clean pod
# queue JSONL: id,mode,pod,tp,pp,ep,gbs,seq,iters,experts,topk,note
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_SSH_HOST_OVERRIDE="${CLUSTER_SSH_HOST_OVERRIDE:-ais-cf3e61a5}"
export CLUSTER_FORCE_JUMP="${CLUSTER_FORCE_JUMP:-1}"
# shellcheck source=muxi.env
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

QUEUE="${1:?queue.jsonl}"
FORCE_POD="${FORCE_POD:-worker-1}"   # 顺序跑固定一台，避免并行残留
PEAK="${PEAK:-279.9}"
POLL_SEC="${POLL_SEC:-30}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-3600}"
LEDGER="${LEDGER:-$SCRIPT_DIR/../../reports/rounds/mfu_single_node_muxi_ledger.md}"
PARSE="$SCRIPT_DIR/parse_train_mfu_log.py"
AFS_WRAPPERS="/afs-a3-weight-share/yinjinrun.p/lab-workspace/scripts/cluster/wrappers"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
WATCH="$REPO_ROOT/logs/muxi-watchdog-20260712/check.jsonl"
LOCAL_LOG_ROOT="$REPO_ROOT/logs/muxi-mfu-sn-seq"
mkdir -p "$(dirname "$LEDGER")" "$(dirname "$WATCH")" "$LOCAL_LOG_ROOT"
: > "${LEDGER}.running"

if [[ ! -f "$LEDGER" ]] || ! grep -q '^| round' "$LEDGER"; then
  cat > "$LEDGER" <<'L'
# Muxi 单机 8 卡 Dense/MoE MFU 账本（跨节点 IB 门禁未过 · 顺序执行）

> peak=279.9 · mock-data · local/unfused · 跳板 ais-cf3e61a5 · FORCE_POD 顺序

| round | id | mode | pod | TP/PP/EP | GBS | SEQ | TFLOP med | MFU% | status | log | note |
|------:|----|------|-----|----------|----:|----:|----------:|-----:|--------|-----|------|
L
fi

# 同步 wrapper（含 MoE --disable-bias-linear）
cluster_pod_exec "${CLUSTER_POD}" "mkdir -p '$AFS_WRAPPERS'"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${CLUSTER_POD} -- bash -c 'cat > ${AFS_WRAPPERS}/train_gpt_dense_muxi.sh && chmod +x ${AFS_WRAPPERS}/train_gpt_dense_muxi.sh'" \
  < "$SCRIPT_DIR/wrappers/train_gpt_dense_muxi.sh" >/dev/null
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${CLUSTER_POD} -- bash -c 'cat > ${AFS_WRAPPERS}/train_gpt_moe_muxi.sh && chmod +x ${AFS_WRAPPERS}/train_gpt_moe_muxi.sh'" \
  < "$SCRIPT_DIR/wrappers/train_gpt_moe_muxi.sh" >/dev/null

# 清目标 pod 残留
pod_full="${CLUSTER_JOB}-${FORCE_POD}"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec ${pod_full} -- bash -lc '
    for p in \$(ps -eo pid,args | grep -E \"pretrain_gpt|torchrun\" | grep -v grep | awk \"{print \\\$1}\"); do
      kill -9 \$p 2>/dev/null || true
    done
    sleep 1
    echo CLEAR_POD
  '" </dev/null

round=0
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  round=$((round + 1))
  id=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["id"])' "$line")
  mode=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("mode","dense"))' "$line")
  tp=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("tp",4))' "$line")
  pp=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("pp",2))' "$line")
  ep=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("ep",1))' "$line")
  gbs=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("gbs",64))' "$line")
  seq=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("seq",2048))' "$line")
  iters=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("iters",5))' "$line")
  experts=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("experts",8))' "$line")
  topk=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("topk",2))' "$line")
  note=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("note",""))' "$line")
  pod_logic="$FORCE_POD"
  pod="${CLUSTER_JOB}-${pod_logic}"
  stamp=$(date +%Y%m%d_%H%M%S)
  afs_out="${AFS_RESULTS}/mfu-seq-${stamp}-${id}"
  port=$((33000 + round))
  wrapper="train_gpt_dense_muxi.sh"
  done_token="TRAIN_DENSE_MUXI_DONE"
  [[ "$mode" == "moe" ]] && wrapper="train_gpt_moe_muxi.sh" && done_token="TRAIN_MOE_MUXI_DONE"

  echo "==> SEQ $round/$id mode=$mode pod=$pod_logic TP/PP/EP=$tp/$pp/$ep GBS=$gbs SEQ=$seq"

  run_body=$(cat <<EOF
#!/usr/bin/env bash
set -uo pipefail
export PATH="/opt/conda/bin:\${PATH:-/usr/bin}" PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0 MCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1 MCCL_IB_DISABLE=1
CU_BRIDGE_BIN=/opt/maca/tools/cu-bridge/bin
[[ -x "\$CU_BRIDGE_BIN/cucc" && ! -e "\$CU_BRIDGE_BIN/nvcc" ]] && ln -sfn "\$CU_BRIDGE_BIN/cucc" "\$CU_BRIDGE_BIN/nvcc" || true
export CUDA_HOME=/opt/maca/tools/cu-bridge
mkdir -p '$afs_out'
rm -f '$afs_out/node_0.done' '$afs_out/node_0.fail'
export RUN_DIR='$afs_out' NNODES=1 NODE_RANK=0
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$port
export GPUS_PER_NODE=8 TRAIN_ITERS=$iters
export TP=$tp PP=$pp EP=$ep GBS=$gbs SEQ_LENGTH=$seq MBS=1 RECOMPUTE=1
export NUM_EXPERTS=$experts MOE_TOPK=$topk
bash '$AFS_WRAPPERS/$wrapper' >'$afs_out/node_0.outer.log' 2>&1
ec=\$?
if [[ \$ec -eq 0 ]] && grep -q $done_token '$afs_out/train.log' 2>/dev/null; then
  echo OK >'$afs_out/node_0.done'
else
  echo FAIL >'$afs_out/node_0.fail'
  [[ -f '$afs_out/train.log' ]] || cp -f '$afs_out/node_0.outer.log' '$afs_out/train.log' 2>/dev/null || true
fi
exit \$ec
EOF
)
  printf '%s\n' "$run_body" | ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec -i ${pod} -- bash -c \"cat > /tmp/run_mfu_seq_${id}.sh && chmod +x /tmp/run_mfu_seq_${id}.sh\"" >/dev/null
  ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec ${pod} -- bash -lc \"setsid nohup bash /tmp/run_mfu_seq_${id}.sh </dev/null >/dev/null 2>&1 & echo STARTED\"" </dev/null >/dev/null
  echo "$round|$id|$mode|$pod_logic|$tp/$pp/$ep|$gbs|$seq|$afs_out|$note" > "${LEDGER}.running"
  echo "{\"ts\":\"$(date -Iseconds)\",\"phase\":\"P2seq\",\"id\":\"$id\",\"status\":\"fired\",\"pod\":\"$pod_logic\"}" >> "$WATCH"

  # 阻塞等到 done/fail
  waited=0
  status="RUN"
  while (( waited < MAX_WAIT_SEC )); do
    status=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=10 \
      -o ServerAliveCountMax=2 "$CLUSTER_SSH_HOST" \
      "$(_cluster_vcctl_prefix) pod exec ${pod} -- bash -lc '
        if [[ -f $afs_out/node_0.done ]]; then echo OK
        elif [[ -f $afs_out/node_0.fail ]]; then echo FAIL
        else echo RUN; fi
      '" </dev/null 2>/dev/null || echo RUN)
    if [[ "$status" != "RUN" ]]; then
      break
    fi
    echo "  ... waiting $id ${waited}s"
    sleep "$POLL_SEC"
    waited=$((waited + POLL_SEC))
  done

  local_log="$LOCAL_LOG_ROOT/${id}.train.log"
  ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec ${pod} -- bash -lc 'cat $afs_out/train.log 2>/dev/null || cat $afs_out/node_0.outer.log'" \
    </dev/null > "$local_log" 2>/dev/null || true

  tflop="-"; mfu="-"; st="fail"
  if [[ "$status" == "OK" && -s "$local_log" ]]; then
    js=$(python3 "$PARSE" --peak "$PEAK" --drop-first 1 --json "$local_log" 2>/dev/null || true)
    tflop=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("steady",{}).get("tflops_median","-"))' "$js" 2>/dev/null || echo -)
    mfu=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); v=d.get("mfu_pct"); print(f"{v:.2f}" if v is not None else "-")' "$js" 2>/dev/null || echo -)
    st="ok"
  elif [[ "$status" == "RUN" ]]; then
    st="timeout"
  fi
  echo "| $round | $id | $mode | $pod_logic | $tp/$pp/$ep | $gbs | $seq | $tflop | $mfu | $st | $local_log | $note |" >> "$LEDGER"
  echo "{\"ts\":\"$(date -Iseconds)\",\"phase\":\"P2seq\",\"id\":\"$id\",\"status\":\"$st\",\"mfu\":\"$mfu\",\"tflop\":\"$tflop\"}" >> "$WATCH"
  echo "==> DONE $id status=$st tflop=$tflop mfu=$mfu"
  : > "${LEDGER}.running"

  # 条间清进程，避免下一条抢 GPU
  ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec ${pod} -- bash -lc '
      for p in \$(ps -eo pid,args | grep -E \"pretrain_gpt|torchrun\" | grep -v grep | awk \"{print \\\$1}\"); do
        kill -9 \$p 2>/dev/null || true
      done
    '" </dev/null >/dev/null || true
  sleep 3
done < "$QUEUE"

echo "SEQ_ALL_DONE"
