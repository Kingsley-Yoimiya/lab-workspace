#!/usr/bin/env bash
# 把 rustup 装到 AFS，供当前 job 立刻使用（不换镜像）
# 流程: 本机 Clash 反代 → 登录机安装 toolchain → tar 灌进 AFS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

REMOTE_PROXY_PORT="${REMOTE_PROXY_PORT:-18080}"
LOCAL_PROXY="${LOCAL_PROXY:-http://127.0.0.1:7897}"
AFS_RUST="${AFS_RUST:-/afs-a3-241ceshi-shared/montyyin/toolchains/rust}"
USE_EGRESS="${USE_EGRESS:-1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/cluster-rust-afs-$STAMP}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/install.log") 2>&1

RUSTUP_INIT="$SCRIPT_DIR/docker/rustup-init"
mkdir -p "$(dirname "$RUSTUP_INIT")"
if [[ ! -f "$RUSTUP_INIT" ]]; then
  echo "==> 本机下载 rustup-init"
  curl -fsSL --connect-timeout 10 --max-time 180 -x "$LOCAL_PROXY" \
    -o "$RUSTUP_INIT" \
    https://static.rust-lang.org/rustup/dist/aarch64-unknown-linux-gnu/rustup-init
fi
chmod +x "$RUSTUP_INIT"
ls -la "$RUSTUP_INIT"

if [[ "$USE_EGRESS" == "1" ]]; then
  "$SCRIPT_DIR/egress_tunnel.sh" start
  "$SCRIPT_DIR/egress_tunnel.sh" test | tee "$LOG_DIR/egress-test.log"
fi

PROXY_URL="http://127.0.0.1:${REMOTE_PROXY_PORT}"
echo "==> 上传 rustup-init 到登录机"
scp -o BatchMode=yes -o ConnectTimeout=20 "$RUSTUP_INIT" "${CLUSTER_SSH_HOST}:/tmp/rustup-init.afs"

echo "==> 登录机经反代安装 stable toolchain"
cluster_ssh "bash -s" <<EOF
set -euo pipefail
export http_proxy='$PROXY_URL' https_proxy='$PROXY_URL'
export HTTP_PROXY='$PROXY_URL' HTTPS_PROXY='$PROXY_URL'
export RUSTUP_HOME=/tmp/rust-stage/rustup
export CARGO_HOME=/tmp/rust-stage/cargo
rm -rf /tmp/rust-stage /tmp/rust-stage.tar
mkdir -p /tmp/rust-stage
chmod +x /tmp/rustup-init.afs
/tmp/rustup-init.afs -y --default-toolchain stable --profile minimal
"\$CARGO_HOME/bin/rustc" -V
"\$CARGO_HOME/bin/cargo" -V
tar -C /tmp/rust-stage -cf /tmp/rust-stage.tar rustup cargo
ls -lh /tmp/rust-stage.tar
EOF

echo "==> 灌进 AFS: $AFS_RUST"
# 登录机 cat tar | vcctl -i pod tar xf
ssh -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER_SSH_HOST" \
  "bash -c 'vcctl pod exec -i ${CLUSTER_POD} -- bash -c \"mkdir -p ${AFS_RUST} && tar -xpf - -C ${AFS_RUST}\" < /tmp/rust-stage.tar'"

cluster_pod_exec "$CLUSTER_POD" "
set -euo pipefail
cat > /afs-a3-241ceshi-shared/montyyin/toolchains/rust-env.sh <<'ENV'
export RUSTUP_HOME=/afs-a3-241ceshi-shared/montyyin/toolchains/rust/rustup
export CARGO_HOME=/afs-a3-241ceshi-shared/montyyin/toolchains/rust/cargo
export PATH=\"\$CARGO_HOME/bin:\$RUSTUP_HOME/bin:\$PATH\"
ENV
# shellcheck disable=SC1091
source /afs-a3-241ceshi-shared/montyyin/toolchains/rust-env.sh
rustc -V
cargo -V
which rustc
du -sh '$AFS_RUST'
echo AFS_RUST_OK
" | tee "$LOG_DIR/verify.log"

echo "==> 完成。pod 内: source /afs-a3-241ceshi-shared/montyyin/toolchains/rust-env.sh"
