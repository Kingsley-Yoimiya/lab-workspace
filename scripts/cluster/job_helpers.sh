#!/usr/bin/env bash
# vcctl / 跳板常用封装（在本机 source 或直接调用）
#
# 双集群并存（重要）:
#   - 绝不覆盖跳板默认 ~/.kube/config
#   - 通过 CLUSTER_KUBECONFIG 指向独立文件，vcctl 前缀 KUBECONFIG=...
#   - source huawei.env 或 muxi.env 后再调本文件的函数
#   - 默认跳板 ais-cf3e61a5（见 docs/AIS_JUMP_CLUSTER.md）；旧 weibozhen 暂不可用
#
# 例:
#   source scripts/cluster/muxi.env
#   cluster_pod_list
#   cluster_pod_exec 'mx-smi | head'
set -euo pipefail

CLUSTER_SSH_HOST="${CLUSTER_SSH_HOST:-ais-cf3e61a5}"
CLUSTER_JOB="${CLUSTER_JOB:-huawei-8node-copy}"
CLUSTER_POD="${CLUSTER_POD:-${CLUSTER_JOB}-master-0}"
CLUSTER_IMAGE="${CLUSTER_IMAGE:-registry2.d.pjlab.org.cn/ccr-yangxiaolei/mindspeed-llm:openeuler22.03-mindspeed-llm-2.3.0-a3-arm}"
AFS_WORKSPACE="${AFS_WORKSPACE:-/afs-a3-241ceshi-shared/montyyin/lab-workspace}"
AFS_RESULTS="${AFS_RESULTS:-/afs-a3-241ceshi-shared/montyyin/results}"

# 独立 kubeconfig（跳板上的路径）。空 = 用登录机默认 ~/.kube/config
# 华为: ~/.kube/config.huawei-a3-241ceshi
# 沐曦: ~/.kube/config.muxi-mohe
CLUSTER_KUBECONFIG="${CLUSTER_KUBECONFIG:-}"

DEVICES_PER_NODE="${DEVICES_PER_NODE:-16}"
CLUSTER_N_WORKERS="${CLUSTER_N_WORKERS:-6}"
# 扇出时本机→weibozhen 的并发 SSH 上限（过大易 Connection closed by UNKNOWN port）
CLUSTER_FANOUT_PARALLEL="${CLUSTER_FANOUT_PARALLEL:-6}"

# 远端 vcctl 命令前缀：绑定独立 KUBECONFIG，不碰默认 config
_cluster_vcctl_prefix() {
  if [[ -n "${CLUSTER_KUBECONFIG}" ]]; then
    printf 'KUBECONFIG=%q vcctl' "$CLUSTER_KUBECONFIG"
  else
    printf 'vcctl'
  fi
}

cluster_ssh() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER_SSH_HOST" "$@"
}

# 在登录机上跑 vcctl（自动带 KUBECONFIG）
cluster_vcctl() {
  local prefix
  prefix="$(_cluster_vcctl_prefix)"
  cluster_ssh "${prefix} $*"
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
  local prefix
  prefix="$(_cluster_vcctl_prefix)"
  cluster_ssh "${prefix} pod exec ${pod} -- bash -lc $(printf '%q' "$cmd")"
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
  local prefix
  prefix="$(_cluster_vcctl_prefix)"
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER_SSH_HOST" \
    "${prefix} pod exec -i ${pod} -- bash -c $(printf '%q' "$cmd")"
}

cluster_pod_list() {
  # 兼容旧 -j 与新 --job
  if ! cluster_vcctl "pod get --job ${CLUSTER_JOB}" 2>/dev/null; then
    cluster_vcctl "pod get -j ${CLUSTER_JOB}"
  fi
}

# 列出 Running pod 名（stdout 一行一个），兼容 macOS bash 3.2
cluster_pods_running() {
  cluster_ssh "$(_cluster_vcctl_prefix) pod get --job ${CLUSTER_JOB} 2>/dev/null" \
    | awk 'NR>1 && $3=="Running" {print $1}' \
    | sort
}

cluster_job_clone() {
  local new_name="${1:?usage: cluster_job_clone <new-name> [image]}"
  local image="${2:-$CLUSTER_IMAGE}"
  cluster_vcctl "job clone -n ${new_name} -i ${image} ${CLUSTER_JOB}"
}

# 有界并行跑函数：run_fn pod_name；全局数组 CLUSTER_FANOUT_FAIL_PODS
cluster_fanout_run() {
  local run_fn="$1"
  shift
  local pods=("$@")
  local fail_pods=()
  local pids=()
  local pod_of_pid=()
  local running=0
  local pod pid i rc

  for pod in "${pods[@]}"; do
    while [[ "$running" -ge "$CLUSTER_FANOUT_PARALLEL" ]]; do
      # wait 任意一个（bash 3.2 无 wait -n：轮询）
      for i in "${!pids[@]}"; do
        pid="${pids[$i]}"
        if ! kill -0 "$pid" 2>/dev/null; then
          rc=0
          wait "$pid" || rc=$?
          if [[ "$rc" -ne 0 ]]; then
            fail_pods+=("${pod_of_pid[$i]}")
          fi
          unset "pids[$i]"
          unset "pod_of_pid[$i]"
          running=$((running - 1))
        fi
      done
      # 压缩数组空洞
      pids=("${pids[@]}")
      pod_of_pid=("${pod_of_pid[@]}")
      [[ "$running" -ge "$CLUSTER_FANOUT_PARALLEL" ]] && sleep 0.5
    done
    "$run_fn" "$pod" &
    pids+=("$!")
    pod_of_pid+=("$pod")
    running=$((running + 1))
  done
  for i in "${!pids[@]}"; do
    rc=0
    wait "${pids[$i]}" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
      fail_pods+=("${pod_of_pid[$i]}")
    fi
  done
  CLUSTER_FANOUT_FAIL_PODS=("${fail_pods[@]}")
  [[ ${#fail_pods[@]} -eq 0 ]]
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-}" in
    pods) cluster_pod_list ;;
    running)
      cluster_pods_running
      ;;
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
用法: $0 {pods|running|exec|clone}
  pods
  running
  exec [pod] '<bash -lc 命令>'
  clone <new-job-name> [image]

环境变量:
  CLUSTER_SSH_HOST CLUSTER_JOB CLUSTER_POD CLUSTER_IMAGE
  CLUSTER_KUBECONFIG   # 跳板上独立 kubeconfig，勿覆盖默认 config
  AFS_WORKSPACE AFS_RESULTS DEVICES_PER_NODE CLUSTER_FANOUT_PARALLEL
USAGE
      exit 1
      ;;
  esac
fi
