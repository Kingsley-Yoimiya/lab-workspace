#!/usr/bin/env bash
# 将 weight-share 上误放在 montyyin / 他人前缀下的自有内容迁到 yinjinrun.p。
# 必须在已 source muxi.env 的本机执行；经 cluster_pod_exec 操作真盘。
#
# 用法:
#   source scripts/cluster/muxi.env
#   ./scripts/cluster/migrate_weight_share_home.sh           # dry-run 盘点
#   ./scripts/cluster/migrate_weight_share_home.sh --apply   # 执行 mv
#
# 禁止：删改 yushan 原有 CARD_SCREEN；动华为盘；在跳板假挂载上操作。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$ROOT/job_helpers.sh"
# shellcheck source=afs_guard.sh
source "$ROOT/afs_guard.sh"

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

AFS_ROOT="${AFS_ROOT:-/afs-a3-weight-share}"
AFS_USER="${AFS_USER:-yinjinrun.p}"
AFS_HOME="${AFS_HOME:-${AFS_ROOT}/${AFS_USER}}"
OLD_MONTY="${AFS_ROOT}/montyyin"
STAMP="$(date +%Y%m%d_%H%M%S)"
# 日志落在仓库上两级 logs/
REPO_ROOT="$(cd "$ROOT/../../.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/cluster-afs-migrate-$STAMP}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/migrate.log"

afs_assert_under_home "$AFS_HOME" || {
  echo "AFS_HOME 异常: $AFS_HOME" >&2
  exit 1
}

if [[ "$AFS_ROOT" != "/afs-a3-weight-share" ]]; then
  echo "仅支持 weight-share 迁移（当前 AFS_ROOT=$AFS_ROOT）" >&2
  exit 1
fi

log() {
  echo "$*" | tee -a "$LOG"
}

remote() {
  cluster_pod_exec "$1"
}

log "==> stamp=$STAMP apply=$APPLY home=$AFS_HOME"
log "==> log_dir=$LOG_DIR"

# --- 盘点 ---
log "==> 盘点 weight-share 顶层"
remote "ls -la ${AFS_ROOT} 2>/dev/null | head -80" | tee -a "$LOG" || true

log "==> 盘点旧前缀 ${OLD_MONTY}"
remote "if [[ -d ${OLD_MONTY} ]]; then du -sh ${OLD_MONTY}/* 2>/dev/null | head -40; ls -la ${OLD_MONTY}; else echo 'NO_OLD_MONTY'; fi" \
  | tee -a "$LOG" || true

log "==> 盘点 yushan 顶层（只列，不迁 CARD_SCREEN）"
remote "if [[ -d ${AFS_ROOT}/yushan ]]; then ls -la ${AFS_ROOT}/yushan; else echo 'NO_YUSHAN'; fi" \
  | tee -a "$LOG" || true

if [[ "$APPLY" -ne 1 ]]; then
  log "==> dry-run 结束。确认后加 --apply 执行迁移。"
  exit 0
fi

# --- 建目标 ---
log "==> mkdir $AFS_HOME/{lab-workspace,results}"
remote "mkdir -p '${AFS_HOME}/lab-workspace' '${AFS_HOME}/results' && touch '${AFS_HOME}/.write_test' && rm -f '${AFS_HOME}/.write_test' && echo MKDIR_OK"

# --- 迁移 montyyin 整树 ---
# lab-workspace → home/lab-workspace；results → home/results；其它 → home/ 下同名
migrate_monty_child() {
  local name="$1"
  local src="${OLD_MONTY}/${name}"
  local dest="${AFS_HOME}/${name}"
  remote "
set -euo pipefail
SRC='${src}'
DEST='${dest}'
STAMP='${STAMP}'
if [[ ! -e \"\$SRC\" ]]; then
  echo SKIP_MISSING \$SRC
  exit 0
fi
if [[ -e \"\$DEST\" ]]; then
  DEST=\"\${DEST}_migrated_\${STAMP}\"
  echo CONFLICT_RENAME \$DEST
fi
mv \"\$SRC\" \"\$DEST\"
echo MOVED \$SRC '->' \$DEST
"
}

log "==> 迁移 montyyin 子树"
# 常见子目录
for child in lab-workspace results; do
  migrate_monty_child "$child" | tee -a "$LOG"
done
# 其余顶层条目
remote "
set -euo pipefail
OLD='${OLD_MONTY}'
HOME='${AFS_HOME}'
STAMP='${STAMP}'
if [[ ! -d \"\$OLD\" ]]; then
  echo NO_OLD_DIR
  exit 0
fi
shopt -s nullglob
for p in \"\$OLD\"/* \"\$OLD\"/.[!.]* \"\$OLD\"/..?*; do
  [[ -e \"\$p\" ]] || continue
  base=\$(basename \"\$p\")
  case \"\$base\" in
    lab-workspace|results) continue ;;
  esac
  dest=\"\$HOME/\$base\"
  if [[ -e \"\$dest\" ]]; then
    dest=\"\${dest}_migrated_\${STAMP}\"
  fi
  mv \"\$p\" \"\$dest\"
  echo MOVED \$p '->' \$dest
done
# 若空则删旧目录
if [[ -d \"\$OLD\" ]] && [[ -z \"\$(ls -A \"\$OLD\" 2>/dev/null)\" ]]; then
  rmdir \"\$OLD\" && echo REMOVED_EMPTY \$OLD
else
  echo OLD_REMAINS
  ls -la \"\$OLD\" 2>/dev/null || true
fi
" | tee -a "$LOG"

# --- yushan：仅迁明确属于我们的额外目录名（保守白名单）---
# 不迁 CARD_SCREEN / 对方原有资产。若存在我们误建的常见名则迁走。
log "==> 检查 yushan 下可能的自有尾巴（白名单）"
remote "
set -euo pipefail
Y='${AFS_ROOT}/yushan'
HOME='${AFS_HOME}'
STAMP='${STAMP}'
if [[ ! -d \"\$Y\" ]]; then
  echo NO_YUSHAN
  exit 0
fi
# 白名单：我们可能误建的名字（不含 CARD_SCREEN）
for name in lab-workspace results montyyin-results scripts; do
  src=\"\$Y/\$name\"
  [[ -e \"\$src\" ]] || continue
  dest=\"\$HOME/\$name\"
  if [[ -e \"\$dest\" ]]; then
    dest=\"\${dest}_migrated_from_yushan_\${STAMP}\"
  fi
  mv \"\$src\" \"\$dest\"
  echo MOVED_FROM_YUSHAN \$src '->' \$dest
done
echo YUSHAN_DONE
ls -la \"\$Y\" | head -40
" | tee -a "$LOG"

log "==> 最终 home"
remote "ls -la '${AFS_HOME}' && du -sh '${AFS_HOME}'/* 2>/dev/null | head -20" | tee -a "$LOG"

log "==> 迁移完成。清单: $LOG"
