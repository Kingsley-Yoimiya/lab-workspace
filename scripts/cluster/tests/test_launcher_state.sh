#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ROOT/scripts/cluster/launcher_state.sh"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
run_id=new-run

# 原子 marker 与旧 run 隔离。
launcher_atomic_marker "$tmp/scale_64.node_0.started" \
  "RUN_ID=$run_id" "PID=100" "STARTED_AT=t0"
launcher_atomic_marker "$tmp/scale_64.node_1.started" \
  "RUN_ID=old-run" "PID=101" "STARTED_AT=old"
[[ "$(launcher_count_markers "$tmp" scale_64 started "$run_id")" -eq 1 ]]

# 诊断日志零匹配必须返回0而不是触发pipefail。
mkdir -p "$tmp/live"
echo "LIVE_LAUNCHER=0 LIVE_TORCHRUN=0" >"$tmp/live/rank0.log"
echo "LIVE_LAUNCHER=1 LIVE_TORCHRUN=0" >"$tmp/live/rank1.log"
[[ "$(launcher_count_log_flag "$tmp/live" 'LIVE_LAUNCHER=1')" -eq 1 ]]
[[ "$(launcher_count_log_flag "$tmp/live" 'LIVE_TORCHRUN=1')" -eq 0 ]]

# 延迟出现进程：pgrep/live=0 不能立即失败；started 增长算进展。
line="$(launcher_state_decide 2 1 0 0 1 1 0 101 100 0 0 0 180)"
grep -q 'ACTION=PROGRESS' <<<"$line"
grep -q 'LAST_PROGRESS=101' <<<"$line"

# 快速完成后 pgrep 为空：done齐全直接完成。
line="$(launcher_state_decide 2 2 2 0 0 0 0 105 101 2 1 0 180)"
grep -q 'ACTION=COMPLETE' <<<"$line"

# 部分 done 逐步增长刷新 last_progress。
line="$(launcher_state_decide 4 4 2 0 0 2 2 120 100 4 1 0 180)"
grep -q 'ACTION=PROGRESS' <<<"$line"
grep -q 'LAST_PROGRESS=120' <<<"$line"

# 明确 fail 立即失败。
line="$(launcher_state_decide 4 4 2 1 0 1 1 121 120 4 2 0 180)"
grep -q 'ACTION=FAIL' <<<"$line"

# live/pgrep 暂空但未超过 last_progress 超时，只等待。
line="$(launcher_state_decide 4 4 2 0 0 0 0 200 120 4 2 0 180)"
grep -q 'ACTION=WAIT' <<<"$line"

# 无进展超过阈值才超时。
line="$(launcher_state_decide 4 4 2 0 0 0 0 301 120 4 2 0 180)"
grep -q 'ACTION=TIMEOUT' <<<"$line"

echo "test_launcher_state: OK"
