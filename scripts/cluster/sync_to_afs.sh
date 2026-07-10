#!/usr/bin/env bash
# 本机 main 工作区 → 经 weibozhen vcctl → 真 AFS
# 用法: ./scripts/cluster/sync_to_afs.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_DIR="$(cd "$OPS_ROOT/.." && pwd)"
MAIN_TREE="${MAIN_TREE:-$PROJECT_DIR/lab-workspace-main}"
AFS_DEST="${AFS_WORKSPACE}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/cluster-sync-$STAMP}"
# 若从 project/lab-workspace 跑，logs 落在 random-thing/logs
if [[ ! -d "$(dirname "$LOG_DIR")" ]]; then
  LOG_DIR="$OPS_ROOT/logs/cluster-sync-$STAMP"
fi
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/sync.log") 2>&1

echo "==> LOG_DIR=$LOG_DIR"
echo "==> OPS_ROOT=$OPS_ROOT"
echo "==> MAIN_TREE=$MAIN_TREE"
echo "==> AFS_DEST=$AFS_DEST"
echo "==> POD=$CLUSTER_POD via $CLUSTER_SSH_HOST"

# 1) 准备 main worktree（与 ops/cluster 分离）
if [[ ! -d "$MAIN_TREE/.git" ]]; then
  echo "==> 创建 main worktree: $MAIN_TREE"
  git -C "$OPS_ROOT" fetch origin main
  git -C "$OPS_ROOT" worktree add -B main "$MAIN_TREE" origin/main
else
  echo "==> 更新已有 main worktree"
  git -C "$MAIN_TREE" fetch origin main
  git -C "$MAIN_TREE" checkout main
  git -C "$MAIN_TREE" pull --ff-only origin main
fi

echo "==> submodule update"
git -C "$MAIN_TREE" submodule update --init --recursive

# Probing_plus 跟踪 kingsley/ascend-lab
if [[ -d "$MAIN_TREE/projects/Probing_plus/.git" ]] || [[ -f "$MAIN_TREE/projects/Probing_plus/.git" ]]; then
  git -C "$MAIN_TREE/projects/Probing_plus" fetch origin kingsley/ascend-lab 2>/dev/null || true
  git -C "$MAIN_TREE/projects/Probing_plus" checkout kingsley/ascend-lab 2>/dev/null \
    || git -C "$MAIN_TREE/projects/Probing_plus" checkout -B kingsley/ascend-lab origin/kingsley/ascend-lab
fi

# CARD_SCREEN 跟踪 montyyin_develop
if [[ -d "$MAIN_TREE/projects/CARD_SCREEN/.git" ]] || [[ -f "$MAIN_TREE/projects/CARD_SCREEN/.git" ]]; then
  git -C "$MAIN_TREE/projects/CARD_SCREEN" fetch origin montyyin_develop 2>/dev/null || true
  git -C "$MAIN_TREE/projects/CARD_SCREEN" checkout montyyin_develop 2>/dev/null \
    || git -C "$MAIN_TREE/projects/CARD_SCREEN" checkout -B montyyin_develop origin/montyyin_develop
fi

echo "==> 本地内容摘要"
du -sh "$MAIN_TREE" "$MAIN_TREE/projects"/* 2>/dev/null | tee "$LOG_DIR/local-size.txt"
git -C "$MAIN_TREE" rev-parse HEAD | tee "$LOG_DIR/main-sha.txt"
git -C "$MAIN_TREE" submodule status | tee "$LOG_DIR/submodule-status.txt"

# 2) tar 管道：本机 → ssh weibozhen → vcctl pod exec → AFS
echo "==> 上传到 AFS（tar 管道）"
START=$(date +%s)
# 远端：清空目标后解压（保留父目录）
REMOTE_CMD=$(cat <<REMOTE
set -euo pipefail
mkdir -p '$(dirname "$AFS_DEST")'
rm -rf '$AFS_DEST'
mkdir -p '$AFS_DEST'
tar -xpf - -C '$AFS_DEST'
echo AFS_EXTRACT_OK
du -sh '$AFS_DEST'
REMOTE
)

# 注意：vcctl 无 TTY；用 bash -c 读 stdin
# macOS 避免 AppleDouble / xattr 污染 AFS
export COPYFILE_DISABLE=1
tar -C "$MAIN_TREE" -cf - \
  --exclude='.DS_Store' \
  --exclude='._*' \
  --exclude='**/._*' \
  --exclude='**/__pycache__' \
  --exclude='**/.venv' \
  --exclude='**/node_modules' \
  --exclude='**/target/debug' \
  --exclude='**/results' \
  . \
| cluster_pod_exec_i "$CLUSTER_POD" "$REMOTE_CMD" \
  | tee "$LOG_DIR/remote-extract.log"

END=$(date +%s)
echo "==> 上传耗时 $((END-START))s" | tee "$LOG_DIR/timing.txt"

# 3) 校验
echo "==> 远端校验"
cluster_pod_exec "$CLUSTER_POD" "ls -la '$AFS_DEST' | head; test -f '$AFS_DEST/README.md' && echo README_OK; test -d '$AFS_DEST/projects/CARD_SCREEN' && echo CARD_SCREEN_OK; test -d '$AFS_DEST/projects/Probing_plus' && echo PROBING_OK; du -sh '$AFS_DEST'" \
  | tee "$LOG_DIR/verify.log"

echo "==> sync 完成"
