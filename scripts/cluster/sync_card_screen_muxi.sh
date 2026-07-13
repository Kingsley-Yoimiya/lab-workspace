#!/usr/bin/env bash
# 把本机 CARD_SCREEN（含 stage_c + Metax）同步到沐曦 AFS 个人树（yinjinrun.p）
#
# 用法:
#   ./scripts/cluster/sync_card_screen_muxi.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=muxi.env
source "$SCRIPT_DIR/muxi.env"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

LOCAL_CS="${LOCAL_CS:-$SCRIPT_DIR/../../projects/CARD_SCREEN}"
AFS_DEST="${AFS_DEST:-${AFS_WORKSPACE}/projects/CARD_SCREEN}"
afs_assert_under_home "$AFS_DEST"
STAMP="$(date +%Y%m%d_%H%M%S)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/muxi-sync-cs-${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/sync.log") 2>&1

echo "==> LOCAL_CS=$LOCAL_CS"
echo "==> AFS_DEST=$AFS_DEST via $CLUSTER_POD"
echo "==> KUBECONFIG=$CLUSTER_KUBECONFIG"

test -f "$LOCAL_CS/screen.py"
test -f "$LOCAL_CS/card_screen/probes/stage_c.py"
test -f "$LOCAL_CS/card_screen/backend.py"

# 清空目标后解压（保留父目录；目标必须已过 afs_assert_under_home）
REMOTE_CMD=$(cat <<REMOTE
set -euo pipefail
mkdir -p '$(dirname "$AFS_DEST")'
rm -rf '$AFS_DEST'
mkdir -p '$AFS_DEST'
tar -xpf - -C '$AFS_DEST'
test -f '$AFS_DEST/screen.py'
test -f '$AFS_DEST/card_screen/probes/stage_c.py'
echo AFS_CS_SYNC_OK
du -sh '$AFS_DEST'
ls '$AFS_DEST/card_screen/probes'
REMOTE
)

# macOS: 避免 xattr / AppleDouble
COPYFILE_DISABLE=1 tar -cf - \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='results' \
  --exclude='*.pyc' \
  -C "$LOCAL_CS" . \
  | cluster_pod_exec_i "$CLUSTER_POD" "$REMOTE_CMD"

echo "SYNC_OK → $AFS_DEST / $LOG_DIR"
