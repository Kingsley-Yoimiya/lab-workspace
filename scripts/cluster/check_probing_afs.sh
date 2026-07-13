#!/usr/bin/env bash
# 经 weibozhen + vcctl 检查 AFS 上 rustc / probing 就绪状态，写本地 status 文件。
# 不长时间编译；仅探测。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/probing-check-$STAMP}"
mkdir -p "$LOG_DIR"
STATUS_FILE="${STATUS_FILE:-$LOG_DIR/probing_afs_status.txt}"
AFS_RUST_ENV="${AFS_RUST_ENV:-/afs-a3-weight-share/yinjinrun.p-huawei/toolchains/rust-env.sh}"
PP_DIR="${AFS_WORKSPACE}/projects/Probing_plus"
PROBING_DIR="${AFS_WORKSPACE}/projects/probing"

echo "==> LOG_DIR=$LOG_DIR"
echo "==> STATUS_FILE=$STATUS_FILE POD=$CLUSTER_POD"

REMOTE_OUT="$(cluster_pod_exec "$CLUSTER_POD" "
set +e
echo '=== probing AFS readiness ==='
echo \"host=\$(hostname)\"
echo \"date=\$(date -Iseconds)\"

RUSTC_OK=0
CARGO_OK=0
PROBING_BIN_OK=0
PROBING_PLUS_TREE_OK=0
PROBING_TREE_OK=0
RUST_ENV_OK=0

if [[ -f '$AFS_RUST_ENV' ]]; then
  echo \"rust_env=present:$AFS_RUST_ENV\"
  # shellcheck disable=SC1091
  source '$AFS_RUST_ENV'
  RUST_ENV_OK=1
else
  echo \"rust_env=missing:$AFS_RUST_ENV\"
fi

if command -v rustc >/dev/null 2>&1; then
  echo \"rustc=\$(rustc -V 2>&1)\"
  echo \"rustc_path=\$(command -v rustc)\"
  RUSTC_OK=1
else
  echo 'rustc=MISSING'
fi

if command -v cargo >/dev/null 2>&1; then
  echo \"cargo=\$(cargo -V 2>&1)\"
  CARGO_OK=1
else
  echo 'cargo=MISSING'
fi

if command -v probing >/dev/null 2>&1; then
  echo \"probing_bin=\$(command -v probing)\"
  probing --help >/dev/null 2>&1 && PROBING_BIN_OK=1
  echo \"probing_bin_ok=\$PROBING_BIN_OK\"
else
  echo 'probing_bin=MISSING (expected until make develop)'
fi

if [[ -f '$PP_DIR/Cargo.toml' && -f '$PP_DIR/pyproject.toml' ]]; then
  PROBING_PLUS_TREE_OK=1
  echo \"probing_plus_tree=OK:$PP_DIR\"
else
  echo \"probing_plus_tree=MISSING:$PP_DIR\"
fi

if [[ -f '$PROBING_DIR/Cargo.toml' ]]; then
  PROBING_TREE_OK=1
  echo \"probing_tree=OK:$PROBING_DIR\"
else
  echo \"probing_tree=MISSING:$PROBING_DIR\"
fi

# 综合门禁：rustc + Probing_plus 树即可进入 develop；probing 二进制可选
READY=0
if [[ \$RUSTC_OK -eq 1 && \$PROBING_PLUS_TREE_OK -eq 1 ]]; then
  READY=1
fi

echo \"RUST_ENV_OK=\$RUST_ENV_OK\"
echo \"RUSTC_OK=\$RUSTC_OK\"
echo \"CARGO_OK=\$CARGO_OK\"
echo \"PROBING_BIN_OK=\$PROBING_BIN_OK\"
echo \"PROBING_PLUS_TREE_OK=\$PROBING_PLUS_TREE_OK\"
echo \"PROBING_TREE_OK=\$PROBING_TREE_OK\"
echo \"PROBING_AFS_READY=\$READY\"
if [[ \$READY -eq 1 ]]; then
  echo 'STATUS=READY'
else
  echo 'STATUS=BLOCKED'
  echo 'HINT: run ./scripts/cluster/install_rust_afs.sh then MODE=develop ./scripts/cluster/run_probing_plus.sh'
fi
")"

printf '%s\n' "$REMOTE_OUT" | tee "$LOG_DIR/remote.log" | tee "$STATUS_FILE"

if grep -q 'STATUS=READY' <<<"$REMOTE_OUT"; then
  echo "PROBING_CHECK_OK → $STATUS_FILE"
  exit 0
fi
echo "PROBING_CHECK_BLOCKED → $STATUS_FILE"
exit 1
