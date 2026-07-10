#!/usr/bin/env bash
# 在 master pod 上冒烟 Probing_plus
# MODE=smoke（默认）: 校验源码树 + 工具链，缺 rust 则明确阻塞、不长时间 pip
# MODE=develop: 尝试 make develop（需 rustc/maturin，可能很久）
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
MODE="${MODE:-smoke}"
echo "==> LOG_DIR=$LOG_DIR"
echo "==> PP_DIR=$PP_DIR MODE=$MODE POD=$CLUSTER_POD"

cluster_pod_exec "$CLUSTER_POD" "
set -euo pipefail
cd '$PP_DIR'
pwd
test -f README.md && test -f Cargo.toml && test -f pyproject.toml
echo PROBING_TREE_OK
ls -la | head -20
# submodule 可能无完整 .git；有则打印
git rev-parse --abbrev-ref HEAD 2>/dev/null || true
git log -1 --oneline 2>/dev/null || true

echo '==> toolchain'
python -V
# 优先 AFS 上的 rust（install_rust_afs.sh 安装）
if [[ -f /afs-a3-241ceshi-shared/montyyin/toolchains/rust-env.sh ]]; then
  # shellcheck disable=SC1091
  source /afs-a3-241ceshi-shared/montyyin/toolchains/rust-env.sh
  echo 'sourced AFS rust-env.sh'
fi
if command -v rustc >/dev/null 2>&1; then
  rustc -V
  echo RUSTC_OK
else
  echo 'PROBING_BLOCKED: rustc missing (run install_rust_afs.sh or use rust image)'
fi
command -v cargo >/dev/null 2>&1 && cargo -V || true
command -v maturin >/dev/null 2>&1 && maturin --version || true
command -v probing >/dev/null 2>&1 && probing --help 2>&1 | head -20 && echo PROBING_HELP_OK || echo 'probing binary not installed'

if [[ '$MODE' == 'develop' ]]; then
  if ! command -v rustc >/dev/null 2>&1; then
    echo 'MODE=develop but rustc missing; abort develop'
    exit 2
  fi
  python3 -m pip install -q maturin
  make develop 2>&1 | tail -80
  probing --help 2>&1 | head -40
  echo PROBING_DEVELOP_OK
fi
echo PROBING_SMOKE_DONE
" | tee "$LOG_DIR/probing_plus.remote.log"

echo "==> Probing_plus 冒烟结束 → $LOG_DIR"
