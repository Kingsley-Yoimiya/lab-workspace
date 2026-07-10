#!/usr/bin/env bash
# vcctl / weibozhen 常用封装（在本机 source 或直接调用）
set -euo pipefail

CLUSTER_SSH_HOST="${CLUSTER_SSH_HOST:-weibozhen}"
CLUSTER_JOB="${CLUSTER_JOB:-huawei-8node-copy}"
CLUSTER_POD="${CLUSTER_POD:-${CLUSTER_JOB}-master-0}"
CLUSTER_IMAGE="${CLUSTER_IMAGE:-registry2.d.pjlab.org.cn/ccr-yangxiaolei/mindspeed-llm:openeuler22.03-mindspeed-llm-2.3.0-a3-arm}"
AFS_WORKSPACE="${AFS_WORKSPACE:-/afs-a3-241ceshi-shared/montyyin/lab-workspace}"

cluster_ssh() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER_SSH_HOST" "$@"
}

# 在登录机上跑 vcctl
cluster_vcctl() {
  cluster_ssh "vcctl $*"
}

# 在指定 pod 里非交互执行 bash -lc '...'
# 用法: cluster_pod_exec [pod] 'command'
cluster_pod_exec() {
  local pod="$CLUSTER_POD"
  if [[ $# -ge 2 ]]; then
    pod="$1"
    shift
  fi
  local cmd="$1"
  cluster_ssh "vcctl pod exec ${pod} -- bash -lc $(printf '%q' "$cmd")"
}

# 带 stdin 的 exec（上传 tar 等）；本机 stdin → ssh → vcctl -i
# 用法: cluster_pod_exec_i [pod] 'command'  < input
cluster_pod_exec_i() {
  local pod="$CLUSTER_POD"
  if [[ $# -ge 2 ]]; then
    pod="$1"
    shift
  fi
  local cmd="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER_SSH_HOST" \
    "vcctl pod exec -i ${pod} -- bash -c $(printf '%q' "$cmd")"
}

cluster_pod_list() {
  cluster_vcctl "pod get -j ${CLUSTER_JOB}"
}

cluster_job_clone() {
  local new_name="${1:?usage: cluster_job_clone <new-name> [image]}"
  local image="${2:-$CLUSTER_IMAGE}"
  cluster_vcctl "job clone -n ${new_name} -i ${image} ${CLUSTER_JOB}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-}" in
    pods) cluster_pod_list ;;
    exec)
      shift
      cluster_pod_exec "$@"
      ;;
    clone)
      shift
      cluster_job_clone "$@"
      ;;
    *)
      cat <<USAGE
用法: $0 {pods|exec|clone}
  pods
  exec [pod] '<bash -lc 命令>'
  clone <new-job-name> [image]

环境变量: CLUSTER_SSH_HOST CLUSTER_JOB CLUSTER_POD CLUSTER_IMAGE AFS_WORKSPACE
USAGE
      exit 1
      ;;
  esac
fi
