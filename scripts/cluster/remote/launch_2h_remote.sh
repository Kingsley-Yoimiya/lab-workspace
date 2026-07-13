#!/usr/bin/env bash
# 在 ais-jump 上执行：归档 + MoE gap 短跑 + 远程监控（本机可离线）
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/root/.kube/config.huawei-a3-241ceshi}"
JOB="${JOB:-montyyin-moe96-r2}"
POD="${JOB}-master-0"
REMOTE_DIR="${REMOTE_DIR:-/root/montyyin-lab-remote}"
STAMP_MOE="${STAMP_MOE:-$(date +%Y%m%d_%H%M%S)}"
ARCHIVE_STAMP="20260713_offline"
LOG="/tmp/launch_2h_remote_${STAMP_MOE}.log"
exec > >(tee -a "$LOG") 2>&1

echo "==> launch_2h_remote STAMP_MOE=$STAMP_MOE $(date -Iseconds)"

# --- 1) 归档关键 AFS 到同盘（本机离线后仍可读）---
ARCHIVE_ROOT="/afs-a3-weight-share/yinjinrun.p-huawei/archive/${ARCHIVE_STAMP}"
vcctl pod exec "$POD" -- bash -lc "
set -e
ARCH='$ARCHIVE_ROOT'
mkdir -p \"\$ARCH\"
pack() {
  local src=\"\$1\" dst=\"\$2\"
  [[ -d \"\$src\" ]] || return 0
  echo PACK \"\$src\"
  tar czf \"\$dst\" -C \"\$(dirname \"\$src\")\" \"\$(basename \"\$src\")\"
  ls -lh \"\$dst\"
}
pack /afs-a3-weight-share/yinjinrun.p-huawei/results/dense_failslow/20260713_001230 \
     \"\$ARCH/dense_failslow_20260713_001230.tgz\"
pack /afs-a3-weight-share/yinjinrun.p-huawei/results/dense_failslow_gbsprop/20260713_071316 \
     \"\$ARCH/dense_failslow_gbsprop_20260713_071316.tgz\"
pack /afs-a3-weight-share/yinjinrun.p-huawei/results/mfu_moe_scale/20260712_181247 \
     \"\$ARCH/mfu_moe_scale_181247.tgz\"
pack /afs-a3-weight-share/yinjinrun.p-huawei/results/mfu_moe_scale/20260712_221912 \
     \"\$ARCH/mfu_moe_scale_221912.tgz\"
echo ARCHIVE_DONE \"$ARCHIVE_ROOT\"
"

# --- 2) 上传解析脚本到 AFS（若本地 bundle 在 jump）---
if [[ -f "$REMOTE_DIR/parse_failslow_gap.py" ]]; then
  for f in parse_failslow_gap.py parse_train_mfu_log.py remote_finalize_reports.py; do
    [[ -f "$REMOTE_DIR/$f" ]] || continue
    base64 -w0 "$REMOTE_DIR/$f" | vcctl pod exec -i "$POD" -- bash -lc \
      "base64 -d | tee /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/$f >/dev/null && wc -c /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/$f"
  done
fi

# --- 3) 清训练进程 ---
for p in ${JOB}-master-0 ${JOB}-worker-0 ${JOB}-worker-1 ${JOB}-worker-2 ${JOB}-worker-3 ${JOB}-worker-4; do
  vcctl pod exec "$p" -- bash -lc \
    'for pid in $(ps -eo pid,cmd | awk "/[p]retrain_gpt.py|[t]orchrun/{print \$1}"); do kill -9 $pid 2>/dev/null; done; true' \
    >/dev/null 2>&1 || true
done
sleep 3

# --- 4) 启动 MoE FailSlow 32+64 并行 ---
pkill -f jumphost_moe_failslow.sh || true
sleep 1
export JOB STAMP="$STAMP_MOE" SCALES='32+64' TRAIN_ITERS=40 PROBING=0 FAILSLOW_STEP_LOG=1
export MASTER_PORT=26200 SCALE_TIMEOUT_SEC=5400 SCALE_GRACE_SEC=900
export RUN_ROOT="/afs-a3-weight-share/yinjinrun.p-huawei/results/moe_failslow/${STAMP_MOE}"
export LOCAL_LOG="/tmp/moe-failslow-${STAMP_MOE}.log"
nohup bash "$REMOTE_DIR/jumphost_moe_failslow.sh" > "/tmp/moe-failslow-${STAMP_MOE}_nohup.out" 2>&1 &
echo MOE_ORCH_PID=$! STAMP=$STAMP_MOE

# --- 5) 远程监控 + 收口（写 AFS 简报）---
pkill -f remote_monitor_finalize.sh || true
sleep 1
nohup bash "$REMOTE_DIR/remote_monitor_finalize.sh" \
  "$STAMP_MOE" "$ARCHIVE_STAMP" > "/tmp/remote_monitor_${STAMP_MOE}.log" 2>&1 &
echo MONITOR_PID=$!

echo "LAUNCH_2H_REMOTE_DONE log=$LOG moe_stamp=$STAMP_MOE"
