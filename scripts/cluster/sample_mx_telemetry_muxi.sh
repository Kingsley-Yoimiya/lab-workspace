#!/usr/bin/env bash
# 全节点周期采样 mx-smi JSON → AFS telemetry/
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_FORCE_JUMP="${CLUSTER_FORCE_JUMP:-1}"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

AFS_TELEM="${AFS_TELEM:?}"
INTERVAL="${INTERVAL:-5}"
DURATION="${DURATION:-900}"
JOB="$CLUSTER_JOB"
pods=("${JOB}-master-0")
for i in $(seq 0 14); do pods+=("${JOB}-worker-$i"); done

cluster_pod_exec "${JOB}-master-0" "mkdir -p '$AFS_TELEM'"

idx=0
for pod in "${pods[@]}"; do
  node=$idx
  cluster_pod_exec "$pod" "
set +e
export PATH=/opt/conda/bin:\$PATH
nohup python3 - <<'PY' >'$AFS_TELEM/node${node}.sampler.log' 2>&1 &
import json, subprocess, time, os
out = '$AFS_TELEM/node${node}.jsonl'
interval = float('$INTERVAL')
end = time.time() + float('$DURATION')
node = int('$node')
host = os.uname().nodename
while time.time() < end:
    ts = time.time()
    try:
        p = subprocess.run(['mx-smi', '-j'], capture_output=True, text=True, timeout=15)
        raw = p.stdout
        i = raw.find('{')
        data = json.loads(raw[i:]) if i >= 0 else {'raw': raw[:2000]}
    except Exception as e:
        data = {'error': str(e)}
    rec = {'ts': ts, 'node': node, 'hostname': host, 'smi': data}
    with open(out, 'a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    time.sleep(interval)
print('TELEM_DONE')
PY
echo TELEM_FIRED node=$node pid=\$!
" || echo "WARN telem $pod"
  idx=$((idx + 1))
  sleep 0.2
done
echo "TELEM_ALL_FIRED out=$AFS_TELEM interval=$INTERVAL duration=$DURATION"
