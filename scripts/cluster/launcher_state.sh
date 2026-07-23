#!/usr/bin/env bash
# Pure Bash 3.2-compatible launcher marker/state helpers.

launcher_atomic_marker() {
  local path="$1"
  shift
  local tmp="${path}.tmp.$$"
  printf '%s\n' "$@" >"$tmp"
  mv -f "$tmp" "$path"
}

launcher_marker_matches_run() {
  local path="$1"
  local run_id="$2"
  [[ -f "$path" ]] && grep -qx "RUN_ID=$run_id" "$path"
}

launcher_count_markers() {
  local dir="$1"
  local stem="$2"
  local suffix="$3"
  local run_id="$4"
  local count=0 path
  for path in "$dir"/"${stem}".*."${suffix}"; do
    [[ -e "$path" ]] || continue
    launcher_marker_matches_run "$path" "$run_id" && count=$((count + 1))
  done
  echo "$count"
}

# Count diagnostic flags without treating zero matches as a shell error under
# `set -euo pipefail` (unlike `rg ... | wc -l`).
launcher_count_log_flag() {
  local dir="$1"
  local pattern="$2"
  local count=0 path
  for path in "$dir"/*.log; do
    [[ -e "$path" ]] || continue
    grep -Eq "$pattern" "$path" && count=$((count + 1))
  done
  echo "$count"
}

# Prints one machine-readable line:
# ACTION=<WAIT|PROGRESS|COMPLETE|FAIL|TIMEOUT> LAST_PROGRESS=<epoch>
launcher_state_decide() {
  local nnodes="$1"
  local started="$2"
  local done_count="$3"
  local fail_count="$4"
  local missing="$5"
  local live_launcher="$6"
  local live_torchrun="$7"
  local now="$8"
  local last_progress="$9"
  shift 9
  local prev_started="$1"
  local prev_done="$2"
  local prev_fail="$3"
  local timeout_s="$4"

  local action=WAIT
  if [[ "$fail_count" -gt 0 ]]; then
    action=FAIL
  elif [[ "$done_count" -eq "$nnodes" && "$fail_count" -eq 0 ]]; then
    action=COMPLETE
  elif [[ "$started" -gt "$prev_started" ||
          "$done_count" -gt "$prev_done" ||
          "$fail_count" -gt "$prev_fail" ]]; then
    action=PROGRESS
    last_progress="$now"
  elif [[ $((now - last_progress)) -ge "$timeout_s" ]]; then
    action=TIMEOUT
  fi

  printf 'ACTION=%s LAST_PROGRESS=%s STARTED=%s DONE=%s FAIL=%s MISSING=%s LIVE_LAUNCHER=%s LIVE_TORCHRUN=%s\n' \
    "$action" "$last_progress" "$started" "$done_count" "$fail_count" \
    "$missing" "$live_launcher" "$live_torchrun"
}
