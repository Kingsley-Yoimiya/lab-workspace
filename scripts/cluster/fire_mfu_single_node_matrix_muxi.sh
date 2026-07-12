#!/usr/bin/env bash
# 在多个空闲 worker 上并行打单机 8 卡 Dense/MoE 矩阵（跨节点 IB 未通时的主战役）
# queue JSONL: id,mode,pod_logic,tp,pp,ep,gbs,seq,iters,experts,topk,note
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_SSH_HOST_OVERRIDE="${CLUSTER_SSH_HOST_OVERRIDE:-ais-cf3e61a5}"
export CLUSTER_FORCE_JUMP="${CLUSTER_FORCE_JUMP:-1}"
# shellcheck source=muxi.env
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

QUEUE="${1:?queue.jsonl}"
PEAK="${PEAK:-279.9}"
LEDGER="${LEDGER:-$SCRIPT_DIR/../../reports/rounds/mfu_single_node_muxi_ledger.md}"
PARSE="$SCRIPT_DIR/parse_train_mfu_log.py"
AFS_WRAPPERS="/afs-a3-weight-share/montyyin/lab-workspace/scripts/cluster/wrappers"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
WATCH="$REPO_ROOT/logs/muxi-watchdog-20260712/check.jsonl"
mkdir -p "$(dirname "$LEDGER")" "$(dirname "$WATCH")"

if [[ ! -f "$LEDGER" ]]; then
  cat > "$LEDGER" <<'L'
# Muxi 单机 8 卡 Dense/MoE MFU 账本（跨节点 IB 门禁未过）

> peak=279.9 · mock-data · local/unfused · 跳板 ais-cf3e61a5

| round | id | mode | pod | TP/PP/EP | GBS | SEQ | TFLOP med | MFU% | status | log | note |
|------:|----|------|-----|----------|----:|----:|----------:|-----:|--------|-----|------|
L
fi

# upload wrappers once
cluster_pod_exec "${CLUSTER_POD}" "mkdir -p '$AFS_WRAPPERS'"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${CLUSTER_POD} -- bash -c 'cat > ${AFS_WRAPPERS}/train_gpt_dense_muxi.sh && chmod +x ${AFS_WRAPPERS}/train_gpt_dense_muxi.sh'" \
  < "$SCRIPT_DIR/wrappers/train_gpt_dense_muxi.sh" >/dev/null
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${CLUSTER_POD} -- bash -c 'cat > ${AFS_WRAPPERS}/train_gpt_moe_muxi.sh && chmod +x ${AFS_WRAPPERS}/train_gpt_moe_muxi.sh'" \
  < "$SCRIPT_DIR/wrappers/train_gpt_moe_muxi.sh" >/dev/null

round=0
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  round=$((round + 1))
  id=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["id"])' "$line")
  mode=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("mode","dense"))' "$line")
  pod_logic=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["pod"])' "$line")
  tp=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("tp",4))' "$line")
  pp=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("pp",2))' "$line")
  ep=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("ep",1))' "$line")
  gbs=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("gbs",64))' "$line")
  seq=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("seq",2048))' "$line")
  iters=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("iters",5))' "$line")
  experts=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("experts",8))' "$line")
  topk=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("topk",2))' "$line")
  note=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("note",""))' "$line")
  pod="${CLUSTER_JOB}-${pod_logic}"
  stamp=$(date +%Y%m%d_%H%M%S)
  afs_out="${AFS_RESULTS}/mfu-sn-${stamp}-${id}"
  port=$((31000 + round))
  wrapper="train_gpt_dense_muxi.sh"
  done_token="TRAIN_DENSE_MUXI_DONE"
  [[ "$mode" == "moe" ]] && wrapper="train_gpt_moe_muxi.sh" && done_token="TRAIN_MOE_MUXI_DONE"

  echo "==> FIRE $id mode=$mode pod=$pod_logic TP/PP/EP=$tp/$pp/$ep GBS=$gbs SEQ=$seq"

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
    "$(_cluster_vcctl_prefix) pod exec -i ${pod} -- bash -c \"cat > /tmp/run_mfu_sn_${id}.sh && chmod +x /tmp/run_mfu_sn_${id}.sh\"" >/dev/null
  ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "$(_cluster_vcctl_prefix) pod exec ${pod} -- bash -lc \"setsid nohup bash /tmp/run_mfu_sn_${id}.sh </dev/null >/dev/null 2>&1 & echo STARTED\"" >/dev/null
  echo "$round|$id|$mode|$pod_logic|$tp/$pp/$ep|$gbs|$seq|$afs_out|$note" >> "${LEDGER}.running"
  echo "{\"ts\":\"$(date -Iseconds)\",\"phase\":\"P2sn\",\"id\":\"$id\",\"status\":\"fired\",\"pod\":\"$pod_logic\"}" >> "$WATCH"
  sleep 2
done

echo "ALL_FIRED; poll with poll_mfu_single_node_muxi.sh"
