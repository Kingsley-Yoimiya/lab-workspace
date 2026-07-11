#!/usr/bin/env bash
# 在无 AFS 的 Running job 上：本机 kubectl 直连发射 constitution128（8 节点并行）
# 避开 weibozhen 默认 kubeconfig 与沐曦互踩。
#
# 用法:
#   export KUBECONFIG=~/.kube/config-vc-a3-241ceshi
#   export CLUSTER_JOB=whj4stu-copy-copy-copy
#   ./scripts/cluster/launch_constitution_kubectl.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CLUSTER_JOB="${CLUSTER_JOB:?set CLUSTER_JOB}"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config-vc-a3-241ceshi}"
export LOCAL_KUBECONFIG="$KUBECONFIG"
export CLUSTER_EXEC_MODE=kubectl

# 容器内必须用 conda py3.10+；系统 python3.8 会炸 dict|dict / 部分语法
REMOTE_PY="${REMOTE_PY:-/root/miniconda3/envs/llm_test/bin/python3}"
WS="${REMOTE_WS:-/data/montyyin/lab-workspace}"
CS="$WS/projects/CARD_SCREEN"
RES_ROOT="${REMOTE_RESULTS:-/data/montyyin/results}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${STAMP}-constitution128"
OUT_DIR="$RES_ROOT/card_screen-${RUN_ID}"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/card-constitution-128-${RUN_ID}}"
mkdir -p "$LOG_DIR/pod_logs"
echo "$OUT_DIR" >"$LOG_DIR/out_dir.txt"
echo "==> LOG_DIR=$LOG_DIR"
echo "==> OUT_DIR=$OUT_DIR"
echo "==> CLUSTER_JOB=$CLUSTER_JOB KUBECONFIG=$KUBECONFIG"

PODS=(master-0 worker-0 worker-1 worker-2 worker-3 worker-4 worker-5 worker-6)

# --- upload CARD_SCREEN ---
PACK="$LOG_DIR/payload"
rm -rf "$PACK"
mkdir -p "$PACK/projects"
cp -a "$OPS_ROOT/projects/CARD_SCREEN" "$PACK/projects/CARD_SCREEN"
rm -rf "$PACK/projects/CARD_SCREEN/.git"
TAR="$LOG_DIR/payload.tar.gz"
COPYFILE_DISABLE=1 tar -C "$PACK" -czf "$TAR" projects
echo "==> upload $(du -h "$TAR" | awk '{print $1}')"

for p in "${PODS[@]}"; do
  kubectl exec -i -n default "${CLUSTER_JOB}-${p}" -- /bin/bash -c \
    "mkdir -p '$WS' '$RES_ROOT' && tar -xzf - -C '$WS' && test -f '$CS/screen.py' && echo OK" \
    <"$TAR" >"$LOG_DIR/upload_${p}.log" 2>&1 &
done
wait
grep -h OK "$LOG_DIR"/upload_*.log | wc -l | awk '{print "upload_ok",$1"/8"}'

# --- durable start (never pkill by cmdline containing screen.py in launcher) ---
start_one() {
  local p="$1"
  local pod="${CLUSTER_JOB}-${p}"
  kubectl exec -n default "$pod" -- /bin/bash -c "
set -e
mkdir -p '$OUT_DIR'
# kill prior by PID only
pids=\$(ps -eo pid,cmd | awk '/[p]ython3 screen.py/{print \$1}')
if [ -n \"\$pids\" ]; then kill -9 \$pids || true; sleep 1; fi
'$REMOTE_PY' - <<'PY'
from pathlib import Path
p = '''$p'''
out = '''$OUT_DIR'''
py = '''$REMOTE_PY'''
cs = '''$CS'''
script = f'''#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# set_env 可能把 python 指回 3.8；强制 conda
export PATH=/root/miniconda3/envs/llm_test/bin:\$PATH
cd {cs}
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
exec {py} screen.py --device all --config config.constitution128.yaml \\
  --sdc-rounds 5 --gemm-n 8192 --sustained-s 30 \\
  --idle-max-memory-mib 1024 --out {out}/{p}.jsonl --no-plot
'''
Path(f'/data/montyyin/run_{p}.sh').write_text(script)
print('wrote', p)
PY
chmod +x /data/montyyin/run_${p}.sh
setsid nohup /data/montyyin/run_${p}.sh > '$OUT_DIR/${p}.run.log' 2>&1 < /dev/null &
echo STARTED_\$!
sleep 2
ps -eo etime,pcpu,cmd | awk '/[p]ython3 screen.py/ && !/awk/ {print; found=1} END{if(!found) exit 1}'
"
}

echo "==> start 8 pods"
for p in "${PODS[@]}"; do
  start_one "$p" >"$LOG_DIR/start_${p}.log" 2>&1 &
done
wait
for p in "${PODS[@]}"; do
  echo -n "$p: "
  grep -E 'STARTED|python3 screen|Error|error' "$LOG_DIR/start_${p}.log" | tail -2 | tr '\n' ' '
  echo
done

# --- monitor ---
cat >"$LOG_DIR/monitor.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
export KUBECONFIG="$KUBECONFIG"
LOG_DIR="$LOG_DIR"
OUT_DIR="$OUT_DIR"
JOB="$CLUSTER_JOB"
PODS=(master-0 worker-0 worker-1 worker-2 worker-3 worker-4 worker-5 worker-6)
while true; do
  ts=\$(date -Iseconds)
  echo "==== \$ts ====" | tee -a "\$LOG_DIR/monitor.log"
  run=0
  for p in "\${PODS[@]}"; do
    info=\$(kubectl exec -n default \${JOB}-\$p -- /bin/bash -c \
      "py=\$(ps -eo etime,pcpu,cmd | awk '/[p]ython3 screen.py/ && !/awk/ {print \\\$1\\\" cpu\\\"\\\$2; exit}'); sz=\$(stat -c%s \$OUT_DIR/*.jsonl 2>/dev/null | awk '{s+=\\\$1} END{print s+0}'); echo py=\${py:-DEAD} bytes=\$sz" 2>/dev/null | tail -1)
    echo "\$p \$info" | tee -a "\$LOG_DIR/monitor.log"
    echo "\$info" | grep -q 'py=DEAD' || run=\$((run+1))
  done
  echo "running=\$run" | tee -a "\$LOG_DIR/monitor.log"
  if [[ \$run -eq 0 ]]; then
    echo ALL_DONE | tee -a "\$LOG_DIR/monitor.log"
    break
  fi
  sleep 90
done
EOS
chmod +x "$LOG_DIR/monitor.sh"
nohup bash "$LOG_DIR/monitor.sh" >"$LOG_DIR/monitor.nohup" 2>&1 &
echo $! >"$LOG_DIR/monitor.pid"
echo "==> monitor pid=$(cat "$LOG_DIR/monitor.pid")"
echo "CONSTITUTION_LAUNCHED → $LOG_DIR"
echo "预计 10–40 min（16 卡/节点 × 8 节点并行）；看 monitor.log / *.run.log"
