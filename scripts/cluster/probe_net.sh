#!/usr/bin/env bash
# 对比：weibozhen 直连 GitHub / 本机→pod 上传 / pod 写 AFS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/cluster-net-$STAMP}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/probe.log") 2>&1

echo "==> LOG_DIR=$LOG_DIR"

echo ""
echo "======== 1) weibozhen 直连 GitHub HTTPS ========"
cluster_ssh 'bash -s' <<'EOF' || true
set +e
echo "curl github.com:"
curl -sI --connect-timeout 8 --max-time 20 https://github.com | head -8
echo "git ls-remote (max 25s):"
timeout 25 git ls-remote https://github.com/Kingsley-Yoimiya/lab-workspace.git HEAD 2>&1 | head -5
echo "exit=$?"
EOF

echo ""
echo "======== 2) 本机 → pod stdin 吞吐（64MB） ========"
START=$(date +%s)
dd if=/dev/urandom bs=1M count=64 2>/dev/null \
| ssh -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER_SSH_HOST" \
  "vcctl pod exec -i ${CLUSTER_POD} -- bash -c 'dd of=/tmp/upload_probe.bin bs=1M status=none; ls -la /tmp/upload_probe.bin; rm -f /tmp/upload_probe.bin; echo UPLOAD_STDIN_OK'" \
  | tee "$LOG_DIR/upload-throughput.log"
END=$(date +%s)
echo "upload_wall_seconds=$((END-START)) for_64MiB" | tee -a "$LOG_DIR/upload-throughput.log"

echo ""
echo "======== 3) pod 内写 AFS 吞吐（256MB） ========"
cluster_pod_exec "$CLUSTER_POD" "
set -e
DEST='${AFS_WORKSPACE%/*}/.net-probe'
mkdir -p \"\$DEST\"
TIMEFORMAT='afs_write_real=%R'
time dd if=/dev/zero of=\"\$DEST/probe.bin\" bs=1M count=256 conv=fsync status=none
ls -la \"\$DEST/probe.bin\"
rm -f \"\$DEST/probe.bin\"
echo AFS_WRITE_OK
" | tee "$LOG_DIR/afs-write.log"

echo ""
echo "==> 结论: 节点勿直连 GitHub；用本机 pull + sync_to_afs.sh"
echo "==> probe 完成 → $LOG_DIR"
