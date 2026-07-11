#!/usr/bin/env bash
# 双集群 kubeconfig 状态 / 确保独立文件存在 —— 绝不覆盖默认 config
#
# 用法:
#   ./scripts/cluster/switch_kube_context.sh status
#   ./scripts/cluster/switch_kube_context.sh ensure
#
# 真正「切换」靠 source 对应 env（设 CLUSTER_KUBECONFIG），例如:
#   source scripts/cluster/huawei.env   # vcctl → config.huawei-a3-241ceshi
#   source scripts/cluster/muxi.env     # vcctl → config.muxi-mohe
#
# 默认 ~/.kube/config 建议保持华为，供其他同事/会话；本仓库脚本不依赖它。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

TARGET="${1:-status}"

HUAWEI_CFG="/root/.kube/config.huawei-a3-241ceshi"
MUXI_CFG="/root/.kube/config.muxi-mohe"

remote_status() {
  cluster_ssh '
    set -euo pipefail
    echo "=== default ~/.kube/config (勿被脚本覆盖) ==="
    if [[ -f ~/.kube/config ]]; then
      grep -E "current-context|^\s+server:" ~/.kube/config | head -6
    else
      echo "(missing)"
    fi
    echo "=== profile files ==="
    ls -la ~/.kube/config.huawei-a3-241ceshi ~/.kube/config.muxi-mohe 2>&1 || true
    echo "=== huawei file context ==="
    grep -E "current-context|^\s+server:" ~/.kube/config.huawei-a3-241ceshi 2>/dev/null | head -4 || echo missing
    echo "=== muxi file context ==="
    grep -E "current-context|^\s+server:" ~/.kube/config.muxi-mohe 2>/dev/null | head -4 || echo missing
  '
}

ensure_files() {
  cluster_ssh "
    set -euo pipefail
    mkdir -p ~/.kube
    # 若缺华为独立文件：从默认或 bak 复制
    if [[ ! -f $HUAWEI_CFG ]]; then
      if [[ -f ~/.kube/config ]] && grep -q vc-a3-241ceshi ~/.kube/config; then
        cp -a ~/.kube/config $HUAWEI_CFG
        echo created $HUAWEI_CFG from default
      elif ls ~/.kube/config.huawei-a3-241ceshi.bak* >/dev/null 2>&1; then
        cp -a \$(ls -1t ~/.kube/config.huawei-a3-241ceshi.bak* | head -1) $HUAWEI_CFG
        echo created $HUAWEI_CFG from bak
      else
        echo 'WARN: cannot create huawei kubeconfig' >&2
      fi
    fi
    if [[ ! -f $MUXI_CFG ]]; then
      echo 'WARN: missing $MUXI_CFG — 从冒烟文档放入 muxi kubeconfig' >&2
      exit 1
    fi
    # 若默认 config 被误写成 muxi，恢复为华为（不删 muxi 文件）
    if [[ -f ~/.kube/config ]] && grep -q vc-c550-mohe ~/.kube/config \
       && [[ -f $HUAWEI_CFG ]]; then
      echo 'NOTE: default config points to muxi; restoring default to huawei profile file'
      cp -a $HUAWEI_CFG ~/.kube/config
    fi
    echo ensure_ok
  "
}

case "$TARGET" in
  status)
    echo "==> dual-cluster kube status (no mutation)"
    remote_status
    echo "==> local tip: source huawei.env | muxi.env  (sets CLUSTER_KUBECONFIG)"
    ;;
  ensure)
    echo "==> ensure profile kubeconfig files exist"
    ensure_files
    remote_status
    ;;
  muxi|huawei)
    cat <<EOF
已废弃「覆盖切换」。请改用:

  source scripts/cluster/${TARGET}.env
  ./scripts/cluster/job_helpers.sh pods

这只会让本 shell 的 vcctl 走独立 KUBECONFIG，不影响默认 ~/.kube/config，
也不会 kill 另一集群上的会话。
EOF
    exit 2
    ;;
  *)
    cat <<USAGE
用法: $0 {status|ensure}
  status  查看默认 config + 两个 profile 文件
  ensure  补齐独立文件；若默认被误写成 muxi 则恢复为华为

切换集群: source scripts/cluster/{huawei,muxi}.env
USAGE
    exit 1
    ;;
esac
