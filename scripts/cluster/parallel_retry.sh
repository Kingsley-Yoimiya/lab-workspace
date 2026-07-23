#!/usr/bin/env bash
# Bash 3.2-compatible bounded parallel retry scheduler.
#
# Callback contract:
#   callback <rank> <attempt> <attempt_log>
# It must write stderr/stdout evidence to attempt_log and return the step rc.

parallel_retry_is_transient() {
  local log="$1"
  grep -Eiq \
    '(^|[^A-Za-z])EOF([^A-Za-z]|$)|error sending request|connection reset by peer|TLS handshake timeout|i/o timeout|Unable to connect to the server|connection closed by foreign host' \
    "$log"
}

parallel_retry_run() {
  local stage="$1"
  local count="$2"
  local parallelism="$3"
  local max_attempts="$4"
  local log_dir="$5"
  local callback="$6"

  if ! [[ "$count" =~ ^[1-9][0-9]*$ ]] ||
     ! [[ "$parallelism" =~ ^[1-9][0-9]*$ ]] ||
     ! [[ "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
    echo "parallel_retry_run: invalid numeric argument" >&2
    return 2
  fi

  mkdir -p "$log_dir"
  local summary="$log_dir/${stage}_retry_summary.log"
  : >"$summary"
  printf 'STAGE=%s COUNT=%s PARALLELISM=%s MAX_ATTEMPTS=%s STARTED_AT=%s\n' \
    "$stage" "$count" "$parallelism" "$max_attempts" "$(date -Iseconds)" >>"$summary"

  local -a pending=()
  local -a permanent=()
  local rank attempt base limit i rc log
  for ((rank = 0; rank < count; rank++)); do
    pending+=("$rank")
  done

  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    [[ ${#pending[@]} -gt 0 ]] || break
    printf 'ATTEMPT=%s PENDING=%s RANKS=%s AT=%s\n' \
      "$attempt" "${#pending[@]}" "${pending[*]}" "$(date -Iseconds)" >>"$summary"
    local -a retry_next=()

    for ((base = 0; base < ${#pending[@]}; base += parallelism)); do
      limit=$((base + parallelism))
      [[ "$limit" -gt "${#pending[@]}" ]] && limit="${#pending[@]}"
      local -a pids=()
      local -a ranks=()
      for ((i = base; i < limit; i++)); do
        rank="${pending[$i]}"
        log="$log_dir/${stage}.rank${rank}.attempt${attempt}.log"
        "$callback" "$rank" "$attempt" "$log" &
        pids+=("$!")
        ranks+=("$rank")
      done

      for i in "${!pids[@]}"; do
        rank="${ranks[$i]}"
        log="$log_dir/${stage}.rank${rank}.attempt${attempt}.log"
        rc=0
        wait "${pids[$i]}" || rc=$?
        printf 'RESULT ATTEMPT=%s RANK=%s RC=%s LOG=%s\n' \
          "$attempt" "$rank" "$rc" "$log" >>"$summary"
        if [[ "$rc" -eq 0 ]]; then
          continue
        fi
        if parallel_retry_is_transient "$log" && [[ "$attempt" -lt "$max_attempts" ]]; then
          retry_next+=("$rank")
          printf 'RETRY ATTEMPT=%s RANK=%s BACKOFF=%ss\n' \
            "$attempt" "$rank" "$attempt" >>"$summary"
        else
          permanent+=("$rank:$rc")
        fi
      done
    done

    if [[ ${#retry_next[@]} -gt 0 ]]; then
      pending=("${retry_next[@]}")
    else
      pending=()
    fi
    if [[ ${#pending[@]} -gt 0 ]]; then
      sleep "$attempt"
    fi
  done

  if [[ ${#pending[@]} -gt 0 ]]; then
    for rank in "${pending[@]}"; do
      permanent+=("$rank:retry_exhausted")
    done
  fi
  printf 'ENDED_AT=%s PERMANENT_FAILURES=%s\n' \
    "$(date -Iseconds)" "${permanent[*]:-none}" >>"$summary"
  [[ ${#permanent[@]} -eq 0 ]]
}
