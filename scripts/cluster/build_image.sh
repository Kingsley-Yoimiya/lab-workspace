#!/usr/bin/env bash
# 在 weibozhen 上 docker build/push 自定义镜像（骨架）
# 用法:
#   ./scripts/cluster/build_image.sh
#   IMAGE_TAG=my-env:v1 ./scripts/cluster/build_image.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

BASE_IMAGE="${BASE_IMAGE:-$CLUSTER_IMAGE}"
IMAGE_REPO="${IMAGE_REPO:-registry2.d.pjlab.org.cn/ccr-yangxiaolei}"
IMAGE_NAME="${IMAGE_NAME:-lab-workspace-env}"
IMAGE_TAG="${IMAGE_TAG:-v0.1.0}"
FULL_IMAGE="${IMAGE_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
PUSH="${PUSH:-0}"
PULL_SECRET="${PULL_SECRET:-huawei-dev2}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/cluster-image-$STAMP}"
mkdir -p "$LOG_DIR" "$SCRIPT_DIR/docker"
exec > >(tee -a "$LOG_DIR/build.log") 2>&1

DOCKERFILE="$SCRIPT_DIR/docker/Dockerfile"
if [[ ! -f "$DOCKERFILE" ]]; then
  cat > "$DOCKERFILE" <<EOF
# 基于现有 Ascend 训练镜像，叠加工作区常用工具（骨架，按需改）
FROM ${BASE_IMAGE}
USER root
RUN pip install --no-cache-dir pyyaml || true
WORKDIR /workspace
EOF
fi

echo "==> BASE=$BASE_IMAGE"
echo "==> TARGET=$FULL_IMAGE"
echo "==> LOG_DIR=$LOG_DIR"

# 把 Dockerfile 拷到登录机再 build（登录机有 docker）
REMOTE_DIR="/tmp/lab-workspace-image-$STAMP"
scp -o BatchMode=yes -o ConnectTimeout=20 "$DOCKERFILE" "${CLUSTER_SSH_HOST}:${REMOTE_DIR}.Dockerfile"
cluster_ssh "bash -s" <<EOF
set -euo pipefail
mkdir -p '$REMOTE_DIR'
mv '${REMOTE_DIR}.Dockerfile' '$REMOTE_DIR/Dockerfile'
cd '$REMOTE_DIR'
echo "==> docker build"
docker build -t '$FULL_IMAGE' .
docker images '$FULL_IMAGE'
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
