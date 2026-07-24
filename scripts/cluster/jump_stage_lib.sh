#!/usr/bin/env bash
# jump_stage_lib.sh — 分阶段有界 fanout + 重试原语（在本机 source）
#
# 抽取自 fire_nccl_scale_muxi_jump.sh 的 run_stage（3 重试 / auth 或 rc40-41
# 永久失败）语义，但并发改为有界（CLUSTER_FANOUT_PARALLEL），供无 AFS 的
# QP/ECMP 驱动复用。依赖 job_helpers.sh 的 cluster_pod_exec[_i]。
#
# bash 3.2 兼容（macOS 登录机）：无关联数组、无 wait -n；per-pod rc 落临时文件。
#
# 主接口：
#   staged_fanout <stage> <run_fn> <max_attempts> <stage_dir> <pod...>
#     run_fn 契约: run_fn <pod> <attempt> <stage_dir>
#       - 自行把日志写到 $stage_dir/<san_pod>.attempt<N>.log
#       - return 0=成功; 40|41=永久失败(鉴权/权限); 其它=可重试
#     结果: 置全局数组 STAGED_FANOUT_FAIL_PODS（永久失败或重试耗尽的 pod）
#           返回 0 当且仅当全部成功
#
#   is_auth_error_log <logfile>   # run_fn 可用来把鉴权类错误归为 rc 40
#   sanitize_pod <pod>            # pod 名 → 安全文件名片段

set -uo pipefail

STAGE_FANOUT_PARALLEL="${STAGE_FANOUT_PARALLEL:-${CLUSTER_FANOUT_PARALLEL:-8}}"

sanitize_pod() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

is_auth_error_log() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  grep -Eiq 'Forbidden|Unauthorized|permission denied|certificate|credentials|authentication|not authorized' "$f"
}

# 有界并行跑一批 pod 的 run_fn，per-pod rc 落 $stage_dir/rc.<san_pod>
# 内部使用；不做重试。
_stage_batch() {
  local run_fn="$1" attempt="$2" stage_dir="$3"; shift 3
  local pods=("$@")
  local pids=() pod_of_pid=() running=0
  local pod pid i rc san

  for pod in "${pods[@]}"; do
    while [[ "$running" -ge "$STAGE_FANOUT_PARALLEL" ]]; do
      for i in "${!pids[@]}"; do
        pid="${pids[$i]}"
        if ! kill -0 "$pid" 2>/dev/null; then
          rc=0; wait "$pid" || rc=$?
          san="$(sanitize_pod "${pod_of_pid[$i]}")"
          printf '%s' "$rc" >"$stage_dir/rc.$san"
          unset "pids[$i]"; unset "pod_of_pid[$i]"
          running=$((running - 1))
        fi
      done
      if [[ ${#pids[@]} -gt 0 ]]; then
        pids=("${pids[@]}"); pod_of_pid=("${pod_of_pid[@]}")
      else
        pids=(); pod_of_pid=()
      fi
      [[ "$running" -ge "$STAGE_FANOUT_PARALLEL" ]] && sleep 0.3
    done
    ( "$run_fn" "$pod" "$attempt" "$stage_dir" ) &
    pids+=("$!"); pod_of_pid+=("$pod"); running=$((running + 1))
  done
  for i in "${!pids[@]}"; do
    rc=0; wait "${pids[$i]}" || rc=$?
    san="$(sanitize_pod "${pod_of_pid[$i]}")"
    printf '%s' "$rc" >"$stage_dir/rc.$san"
  done
}

# 分阶段：首轮全部 pending；后续只重试非永久失败者。
staged_fanout() {
  local stage="$1" run_fn="$2" max_attempts="$3" stage_dir="$4"; shift 4
  local pods=("$@")
  mkdir -p "$stage_dir"
  local -a pending=("${pods[@]}")
  local -a permanent=()
  local attempt pod san rc

  echo "STAGE_BEGIN stage=$stage pods=${#pods[@]} parallel=$STAGE_FANOUT_PARALLEL max_attempts=$max_attempts at=$(date -Iseconds)"
  for ((attempt=1; attempt<=max_attempts; attempt++)); do
    [[ ${#pending[@]} -gt 0 ]] || break
    echo "STAGE_ATTEMPT stage=$stage attempt=$attempt pending=${#pending[@]}"
    _stage_batch "$run_fn" "$attempt" "$stage_dir" "${pending[@]}"

    local -a retry=()
    for pod in "${pending[@]}"; do
      san="$(sanitize_pod "$pod")"
      rc="$(cat "$stage_dir/rc.$san" 2>/dev/null || echo 99)"
      if [[ "$rc" -eq 0 ]]; then
        continue
      elif [[ "$rc" -eq 40 || "$rc" -eq 41 ]] || is_auth_error_log "$stage_dir/$san.attempt${attempt}.log"; then
        permanent+=("$pod:$rc")
        echo "STAGE_PERMANENT stage=$stage pod=$pod rc=$rc (auth/permission — 不重试)"
      elif [[ "$attempt" -lt "$max_attempts" ]]; then
        retry+=("$pod")
      else
        permanent+=("$pod:$rc")
      fi
    done
    pending=("${retry[@]:-}")
    # 清掉可能的空串（retry 为空时 :- 兜底会塞一个空元素）
    if [[ ${#pending[@]} -eq 1 && -z "${pending[0]}" ]]; then pending=(); fi
    [[ ${#pending[@]} -gt 0 ]] && sleep "$attempt"
  done

  if [[ ${#permanent[@]} -gt 0 ]]; then
    STAGED_FANOUT_FAIL_PODS=("${permanent[@]}")
  else
    STAGED_FANOUT_FAIL_PODS=()
  fi
  echo "STAGE_END stage=$stage fail=${STAGED_FANOUT_FAIL_PODS[*]:-none} at=$(date -Iseconds)"
  [[ ${#permanent[@]} -eq 0 ]]
}
