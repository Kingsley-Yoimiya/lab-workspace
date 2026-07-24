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
AFS_ROOT="${AFS_ROOT:-/afs-a3-weight-share}"
AFS_USER="${AFS_USER:-yinjinrun.p-huawei}"
AFS_HOME="${AFS_HOME:-${AFS_ROOT}/${AFS_USER}}"
AFS_WORKSPACE="${AFS_WORKSPACE:-${AFS_HOME}/lab-workspace}"
AFS_RESULTS="${AFS_RESULTS:-${AFS_HOME}/results}"

# 写盘守卫（afs_assert_under_home）；约定见 AFS_LAYOUT.md
_JH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=afs_guard.sh
source "${_JH_DIR}/afs_guard.sh"
unset _JH_DIR

# 独立 kubeconfig（跳板上的路径）。空 = 用登录机默认 ~/.kube/config
# 华为: ~/.kube/config.huawei-a3-241ceshi
# 沐曦 h3c: ~/.kube/config-vc-c550-h3c-test.yaml；旧 mohe 见 muxi.env
CLUSTER_KUBECONFIG="${CLUSTER_KUBECONFIG:-}"
# kubectl=本机直连；vcctl=经跳板（默认）
CLUSTER_EXEC_MODE="${CLUSTER_EXEC_MODE:-vcctl}"

DEVICES_PER_NODE="${DEVICES_PER_NODE:-16}"
CLUSTER_N_WORKERS="${CLUSTER_N_WORKERS:-6}"
# 扇出时本机→weibozhen 的并发 SSH 上限（过大易 Connection closed by UNKNOWN port）
CLUSTER_FANOUT_PARALLEL="${CLUSTER_FANOUT_PARALLEL:-6}"

_cluster_kubectl_env() {
  # 本机 Clash → aTrust；必须清掉 NO_PROXY=10/8
  export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
  export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"
  export https_proxy="$HTTPS_PROXY"
  export http_proxy="$HTTP_PROXY"
  export NO_PROXY="127.0.0.1,localhost"
  export no_proxy="$NO_PROXY"
  unset ALL_PROXY all_proxy 2>/dev/null || true
  if [[ -n "${CLUSTER_KUBECONFIG}" ]]; then
    export KUBECONFIG="$CLUSTER_KUBECONFIG"
  fi
}

# 远端 vcctl 命令前缀：绑定独立 KUBECONFIG，不碰默认 config
_cluster_vcctl_prefix() {
  if [[ -n "${CLUSTER_KUBECONFIG}" ]]; then
    printf 'KUBECONFIG=%q vcctl' "$CLUSTER_KUBECONFIG"
  else
    printf 'vcctl'
  fi
}

# CLUSTER_SSH_CONTROL_PATH 非空时，所有 ssh 复用同一 ControlMaster 连接
# （跳板经多层 ProxyCommand，逐次新建连接慢且触发 sshd 限流；多路复用是正解）。
cluster_ssh() {
  if [[ -n "${CLUSTER_SSH_CONTROL_PATH:-}" ]]; then
    ssh -o BatchMode=yes -o ConnectTimeout=20 \
        -o ControlPath="$CLUSTER_SSH_CONTROL_PATH" "$CLUSTER_SSH_HOST" "$@"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER_SSH_HOST" "$@"
  fi
}

# 建立/关闭复用主连接。driver 起止各调一次。
cluster_ssh_mux_start() {
  [[ -n "${CLUSTER_SSH_CONTROL_PATH:-}" ]] || return 0
  # 已存在活连接则复用
  if ssh -o ControlPath="$CLUSTER_SSH_CONTROL_PATH" -O check "$CLUSTER_SSH_HOST" 2>/dev/null; then
    return 0
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=20 \
      -o ControlMaster=yes -o ControlPath="$CLUSTER_SSH_CONTROL_PATH" \
      -o ControlPersist="${CLUSTER_SSH_CONTROL_PERSIST:-300}" \
      -fN "$CLUSTER_SSH_HOST"
}
cluster_ssh_mux_stop() {
  [[ -n "${CLUSTER_SSH_CONTROL_PATH:-}" ]] || return 0
  ssh -o ControlPath="$CLUSTER_SSH_CONTROL_PATH" -O exit "$CLUSTER_SSH_HOST" 2>/dev/null || true
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
  if [[ "$CLUSTER_EXEC_MODE" == "kubectl" ]]; then
    _cluster_kubectl_env
    kubectl exec "$pod" -- bash -lc "$cmd"
    return
  fi
  if [[ "$CLUSTER_EXEC_MODE" == "vcctl_local" ]]; then
    # driver 已在跳板上：直接本地 vcctl，无 ssh（消除 Mac↔跳板连接 churn）
    KUBECONFIG="${CLUSTER_KUBECONFIG:-$KUBECONFIG}" "${VCCTL_BIN:-vcctl}" pod exec "${pod}" -- bash -lc "$cmd"
    return
  fi
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
  if [[ "$CLUSTER_EXEC_MODE" == "kubectl" ]]; then
    _cluster_kubectl_env
    kubectl exec -i "$pod" -- bash -c "$cmd"
    return
  fi
  if [[ "$CLUSTER_EXEC_MODE" == "vcctl_local" ]]; then
    KUBECONFIG="${CLUSTER_KUBECONFIG:-$KUBECONFIG}" "${VCCTL_BIN:-vcctl}" pod exec -i "${pod}" -- bash -c "$cmd"
    return
  fi
  local prefix
  prefix="$(_cluster_vcctl_prefix)"
  cluster_ssh "${prefix} pod exec -i ${pod} -- bash -c $(printf '%q' "$cmd")"
}

cluster_pod_list() {
  if [[ "$CLUSTER_EXEC_MODE" == "kubectl" ]]; then
    _cluster_kubectl_env
    kubectl get pods -l "volcano.sh/job-name=${CLUSTER_JOB}" -o wide
    return
  fi
  # 兼容旧 -j 与新 --job
  if ! cluster_vcctl "pod get --job ${CLUSTER_JOB}" 2>/dev/null; then
    cluster_vcctl "pod get -j ${CLUSTER_JOB}"
  fi
}

# 列出 Running pod 名（stdout 一行一个），兼容 macOS bash 3.2
cluster_pods_running() {
  if [[ "$CLUSTER_EXEC_MODE" == "kubectl" ]]; then
    _cluster_kubectl_env
    kubectl get pods -l "volcano.sh/job-name=${CLUSTER_JOB}" --no-headers 2>/dev/null \
      | awk '$3=="Running" {print $1}' \
      | sort
    return
  fi
  if [[ "$CLUSTER_EXEC_MODE" == "vcctl_local" ]]; then
    KUBECONFIG="${CLUSTER_KUBECONFIG:-$KUBECONFIG}" "${VCCTL_BIN:-vcctl}" pod get --job "${CLUSTER_JOB}" 2>/dev/null \
      | awk 'NR>1 && $3=="Running" {print $1}' \
      | sort
    return
  fi
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
      # 压缩数组空洞（bash 3.2 + set -u：空数组需兜底）
      if [[ ${#pids[@]} -gt 0 ]]; then
        pids=("${pids[@]}")
        pod_of_pid=("${pod_of_pid[@]}")
      else
        pids=()
        pod_of_pid=()
      fi
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
  if [[ ${#fail_pods[@]} -gt 0 ]]; then
    CLUSTER_FANOUT_FAIL_PODS=("${fail_pods[@]}")
  else
    CLUSTER_FANOUT_FAIL_PODS=()
  fi
  [[ ${#fail_pods[@]} -eq 0 ]]
}

if [[ "${BASH_SOURCE[0]:-}" == "$0" ]]; then
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
  AFS_ROOT AFS_USER AFS_HOME AFS_WORKSPACE AFS_RESULTS
  DEVICES_PER_NODE CLUSTER_FANOUT_PARALLEL
  # 破坏性写盘前: afs_assert_under_home <path>（见 afs_guard.sh / AFS_LAYOUT.md）
USAGE
      exit 1
      ;;
  esac
fi
