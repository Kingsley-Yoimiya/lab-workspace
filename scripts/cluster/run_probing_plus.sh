#!/usr/bin/env bash
# 在 master pod 上冒烟 Probing_plus（安装/help；完整 make develop 可能较久）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/cluster-run-$STAMP}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/probing_plus.log") 2>&1

PP_DIR="${AFS_WORKSPACE}/projects/Probing_plus"
MODE="${MODE:-smoke}"   # smoke | develop
echo "==> LOG_DIR=$LOG_DIR"
echo "==> PP_DIR=$PP_DIR MODE=$MODE POD=$CLUSTER_POD"

cluster_pod_exec "$CLUSTER_POD" "
set -euo pipefail
cd '$PP_DIR'
pwd
ls -la | head
git rev-parse --abbrev-ref HEAD 2>/dev/null || true
git log -1 --oneline 2>/dev/null || true

echo '==> python/rust toolchain'
python -V
which rustc cargo maturin 2>/dev/null || true
rustc -V 2>/dev/null || echo 'rustc missing'

if [[ '$MODE' == 'develop' ]]; then
  echo '==> make develop (可能很久)'
  python3 -m pip install -q maturin
  make develop 2>&1 | tail -80
fi

# 优先已安装的 probing；否则尝试 pip 可编辑/帮助路径
if command -v probing >/dev/null 2>&1; then
  probing --help 2>&1 | head -40
  echo PROBING_HELP_OK
elif [[ -f pyproject.toml ]]; then
  echo '==> try pip install -e . (may fail without rust)'
  set +e
  python -m pip install -e . 2>&1 | tee /tmp/probing_pip.log | tail -40
  PIP_RC=\${PIPESTATUS[0]}
  set -e
  if command -v probing >/dev/null 2>&1; then
    probing --help 2>&1 | head -40
    echo PROBING_HELP_OK
  else
    echo \"PROBING_INSTALL_BLOCKED rc=\$PIP_RC\" 
    tail -20 /tmp/probing_pip.log || true
    # 至少验证源码树可读
    test -f README.md && echo PROBING_TREE_OK
  fi
else
  echo PROBING_TREE_MISSING
  exit 1
fi
" | tee "$LOG_DIR/probing_plus.remote.log"

echo "==> Probing_plus 冒烟结束 → $LOG_DIR"
