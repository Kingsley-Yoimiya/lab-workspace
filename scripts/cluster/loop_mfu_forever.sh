#!/usr/bin/env bash
# 永不停止：反复执行 loop_mfu_one_round.sh，每轮结束打 sentinel 唤醒 Cursor agent。
# 轮间短歇 30s，避免打爆集群；单轮失败不退出。
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "AGENT_LOOP_TICK_mfu_opt {\"prompt\":\"MFU forever loop started\",\"event\":\"boot\"}"
while true; do
  STATE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/reports/rounds/mfu_loop_state"
  if [[ -f "$STATE_DIR/PAUSE" ]] && [[ ! -s "$STATE_DIR/next_job.json" ]]; then
    # 暂停期静默等待，不打 AGENT_LOOP_TICK（避免刷屏）
    echo "==> forever paused, sleep 1800s"
    sleep 1800
    continue
  fi
  "$SCRIPT_DIR/loop_mfu_one_round.sh" || echo "WARN round failed rc=$?"
  echo "AGENT_LOOP_TICK_mfu_opt {\"prompt\":\"一轮结束：读 ledger + 最新 log，写 mfu_opt_rN 报告，必要时改 next_job.json 插队；然后等下一 tick\",\"event\":\"round_done\",\"ts\":\"$(date -Iseconds)\"}"
  sleep 120
done
