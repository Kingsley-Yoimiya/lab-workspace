#!/usr/bin/env bash
# 在 weibozhen 上 docker build（经本机 Clash 反代装 rustup）
# 用法:
#   ./scripts/cluster/build_image.sh
#   PUSH=1 ./scripts/cluster/build_image.sh
#   BASE_IMAGE=registry2.../mindspeed-llm:... ./scripts/cluster/build_image.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

BASE_IMAGE="${BASE_IMAGE:-registry2.d.pjlab.org.cn/lepton-trainingjob/a3-cann:8.3.rc2-a3-openeuler24.03-py3.11}"
IMAGE_REPO="${IMAGE_REPO:-registry2.d.pjlab.org.cn/ccr-yangxiaolei}"
IMAGE_NAME="${IMAGE_NAME:-lab-workspace-env}"
IMAGE_TAG="${IMAGE_TAG:-v0.1.0-rust}"
FULL_IMAGE="${IMAGE_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
PUSH="${PUSH:-0}"
PULL_SECRET="${PULL_SECRET:-huawei-dev2}"
REMOTE_PROXY_PORT="${REMOTE_PROXY_PORT:-18080}"
USE_EGRESS="${USE_EGRESS:-1}"
LOCAL_PROXY="${LOCAL_PROXY:-http://127.0.0.1:7897}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/cluster-image-$STAMP}"
DOCKER_DIR="$SCRIPT_DIR/docker"
mkdir -p "$LOG_DIR" "$DOCKER_DIR"
exec > >(tee -a "$LOG_DIR/build.log") 2>&1

DOCKERFILE="$DOCKER_DIR/Dockerfile"
RUSTUP_INIT="$DOCKER_DIR/rustup-init"

echo "==> BASE=$BASE_IMAGE"
echo "==> TARGET=$FULL_IMAGE"
echo "==> LOG_DIR=$LOG_DIR"

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "缺少 $DOCKERFILE"
  exit 1
fi

# 确保有 aarch64 rustup-init（本机经 Clash 下载）
if [[ ! -f "$RUSTUP_INIT" ]]; then
  echo "==> 本机下载 rustup-init (aarch64)"
  curl -fsSL --connect-timeout 10 --max-time 180 -x "$LOCAL_PROXY" \
    -o "$RUSTUP_INIT" \
    https://static.rust-lang.org/rustup/dist/aarch64-unknown-linux-gnu/rustup-init
  chmod +x "$RUSTUP_INIT"
fi
ls -la "$RUSTUP_INIT"
file "$RUSTUP_INIT" || true

# 启动本机→weibozhen 反代
if [[ "$USE_EGRESS" == "1" ]]; then
  "$SCRIPT_DIR/egress_tunnel.sh" start
  "$SCRIPT_DIR/egress_tunnel.sh" test | tee "$LOG_DIR/egress-test.log"
fi

PROXY_URL="http://127.0.0.1:${REMOTE_PROXY_PORT}"
REMOTE_DIR="/tmp/lab-workspace-image-$STAMP"

echo "==> 上传 build context"
ssh -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER_SSH_HOST" "mkdir -p '$REMOTE_DIR'"
scp -o BatchMode=yes -o ConnectTimeout=20 \
  "$DOCKERFILE" "$RUSTUP_INIT" \
  "${CLUSTER_SSH_HOST}:${REMOTE_DIR}/"

echo "==> docker build --network=host (经反代)"
cluster_ssh "bash -s" <<EOF
set -euo pipefail
cd '$REMOTE_DIR'
chmod +x rustup-init
echo "==> df"; df -h / | tail -1
echo "==> docker build"
docker build --network=host \
  --build-arg BASE_IMAGE='$BASE_IMAGE' \
  --build-arg HTTP_PROXY='$PROXY_URL' \
  --build-arg HTTPS_PROXY='$PROXY_URL' \
  --build-arg http_proxy='$PROXY_URL' \
  --build-arg https_proxy='$PROXY_URL' \
  -t '$FULL_IMAGE' \
  .
docker images '$FULL_IMAGE'
echo "==> verify rust in image"
docker run --rm --network=host '$FULL_IMAGE' bash -lc 'rustc -V; cargo -V; which rustc'
if [[ '$PUSH' == '1' ]]; then
  echo "==> docker push"
  docker push '$FULL_IMAGE'
  echo "==> 可选预热: vcctl image load -i $FULL_IMAGE --imagepullsecret $PULL_SECRET"
fi
echo BUILD_OK
EOF

echo "==> 完成。换 job 镜像示例:"
echo "  source scripts/cluster/job_helpers.sh"
echo "  cluster_job_clone my-128 $FULL_IMAGE"
echo "==> 默认 PUSH=0；确认 registry 写权限后: PUSH=1 $0"
