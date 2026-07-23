#!/usr/bin/env bash
# C2 注入窗内：对训练进程做 Probing SQL 落盘（进程必须存活）。
#
# 环境：
#   OUT_DIR   = …/C2_probing（写 probing/ 子目录）
#   CASE      = P1-EXT-A|P1-EXT-B|P3-EXT-A|…
#   CODE_DIR  = /workspace/probe-bundle（含 pydeps）
#   VICTIM_LOCAL_RANK 默认 7
set -uo pipefail

OUT_DIR="${OUT_DIR:?need OUT_DIR}"
CASE="${CASE:-unknown}"
CODE_DIR="${CODE_DIR:-/workspace/probe-bundle}"
VICTIM_LOCAL_RANK="${VICTIM_LOCAL_RANK:-7}"
PYTHONPATH="${CODE_DIR}/pydeps${PYTHONPATH:+:$PYTHONPATH}"
export PATH="${CODE_DIR}/pydeps/bin:/opt/conda/bin:${PATH:-}"
export PYTHONPATH

DUMP="$OUT_DIR/probing"
mkdir -p "$DUMP"
MANIFEST="$DUMP/query_manifest.json"
TS=$(date -Iseconds)

# 找训练 worker：优先 victim local_rank 的 python；否则任意 train_bench
find_pid() {
  local prefer="$1"
  local p
  # torchrun worker cmdline 含 local_rank
  p=$(ps -eo pid,args | awk -v lr="$prefer" '
    /train_bench_probe|\/tmp\/tbp\.py/ && $0 !~ /awk/ {
      if ($0 ~ ("local[_-]rank[= ]*" lr) || $0 ~ ("LOCAL_RANK=" lr)) { print $1; exit }
    }')
  if [ -n "${p:-}" ]; then echo "$p"; return 0; fi
  p=$(pgrep -n -f '/tmp/tbp.py' 2>/dev/null | head -1 || true)
  if [ -n "${p:-}" ]; then echo "$p"; return 0; fi
  p=$(pgrep -n -f 'train_bench_probe' 2>/dev/null | head -1 || true)
  echo "${p:-}"
}

PID=$(find_pid "$VICTIM_LOCAL_RANK")
echo "dump_probing_sql case=$CASE out=$DUMP pid=${PID:-none} ts=$TS" | tee "$DUMP/dump.log"

run_q() {
  local name="$1" sql="$2"
  local f="$DUMP/query_${name}.txt"
  echo "SQL: $sql" >"$f"
  echo "----" >>"$f"
  if [ -z "${PID:-}" ]; then
    echo "error=no_training_pid" >>"$f"
    echo "$name|error=no_training_pid" >>"$DUMP/status.tsv"
    return 1
  fi
  set +e
  probing -t "$PID" query "$sql" >>"$f" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "error=query_rc_$rc" >>"$f"
    echo "$name|error=query_rc_$rc" >>"$DUMP/status.tsv"
    return 1
  fi
  if grep -qiE "table .* not found|QueryError|Error during planning" "$f"; then
    echo "$name|error=table_or_query" >>"$DUMP/status.tsv"
    return 1
  fi
  echo "$name|ok" >>"$DUMP/status.tsv"
  return 0
}

: >"$DUMP/status.tsv"

if [ -n "${PID:-}" ]; then
  probing -t "$PID" query "SHOW TABLES" >"$DUMP/tables.txt" 2>&1 || echo "error=show_tables" >>"$DUMP/tables.txt"
  # 尝试热开 GPU 采样（失败仅记日志；cpu.utilization 仍可用）
  {
    probing -t "$PID" query "SET probing.gpu.sample_interval=1000" 2>&1 || true
    probing -t "$PID" query "SET probing.gpu.gpu_sample_interval_ms=1000" 2>&1 || true
    sleep 2
    probing -t "$PID" query "SHOW TABLES" 2>&1 || true
  } >"$DUMP/gpu_enable.log" || true
else
  echo "error=no_training_pid" >"$DUMP/tables.txt"
fi

# 通用
run_q show_tables "SHOW TABLES" || true
run_q torch_trace_tail \
  "SELECT timestamp, step, module, stage, duration, allocated FROM python.torch_trace ORDER BY timestamp DESC LIMIT 50" || true
run_q gpu_util \
  "SELECT ts, device_id, name, gpu_util_pct, mem_used_pct, used_bytes FROM gpu.utilization ORDER BY ts DESC LIMIT 100" || true
run_q cpu_util \
  "SELECT ts, scope, cpu_total_pct, rss_kb, thread_count, comm FROM cpu.utilization ORDER BY ts DESC LIMIT 100" || true

# case 文档里的表（多数预期 TABLE_MISSING，如实落盘）
run_q process_gpu_users "SELECT * FROM process.gpu_users LIMIT 20" || true
run_q process_cpu_stats "SELECT * FROM process.cpu_stats LIMIT 20" || true

case "$CASE" in
  P1-EXT-A|P1-EXT-B)
    run_q p1_gpu_window \
      "SELECT ts, device_id, gpu_util_pct, mem_used_pct FROM gpu.utilization ORDER BY ts DESC LIMIT 200" || true
    ;;
  P3-EXT-A|P3-EXT-B)
    run_q p3_cpu_window \
      "SELECT ts, scope, cpu_total_pct, comm FROM cpu.utilization WHERE scope = 'process' ORDER BY ts DESC LIMIT 200" || true
    ;;
esac

# 解析表是否存在
has_table() {
  local t="$1"
  grep -qiE "[[:space:]]${t//./[[:space:]]*}|[[:space:]]$t[[:space:]]" "$DUMP/tables.txt" 2>/dev/null \
    || grep -qF "$t" "$DUMP/tables.txt" 2>/dev/null
}

python3 - <<PY
import json, os, re
from pathlib import Path
dump = Path("$DUMP")
tables_txt = (dump / "tables.txt").read_text(errors="ignore")
status = {}
for line in (dump / "status.tsv").read_text(errors="ignore").splitlines():
    if "|" in line:
        k, v = line.split("|", 1)
        status[k] = v
needed = {
    "gpu.utilization": "gpu.utilization" in tables_txt,
    "cpu.utilization": "cpu.utilization" in tables_txt,
    "python.torch_trace": "torch_trace" in tables_txt,
    "process.gpu_users": "gpu_users" in tables_txt,
    "process.cpu_stats": "cpu_stats" in tables_txt,
}
missing = [k for k, ok in needed.items() if not ok]
manifest = {
    "case": "$CASE",
    "pid": "${PID:-}",
    "ts": "$TS",
    "victim_local_rank": int("$VICTIM_LOCAL_RANK"),
    "tables_present": needed,
    "tables_missing": missing,
    "query_status": status,
    "dump_dir": str(dump),
}
(dump / "query_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(manifest, ensure_ascii=False))
PY

echo "dump_probing_sql done → $DUMP"
