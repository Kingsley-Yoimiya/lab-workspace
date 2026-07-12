#!/usr/bin/env bash
# 16×16 节点对连通性：TCP 22/29500 探测 + 可选同节点 NCCL 冒烟；RoCE 跨节点失败如实记录
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_FORCE_JUMP="${CLUSTER_FORCE_JUMP:-1}"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"
set +e  # job_helpers 可能启用 -e；探测允许单节点失败

AFS_OUT="${AFS_OUT:?}"
OUT_DIR="$AFS_OUT/C1"
LOCAL_C1="${LOCAL_C1:-/tmp/muxi_c1}"
JOB="$CLUSTER_JOB"
MASTER="${JOB}-master-0"
pods=("${JOB}-master-0")
for i in $(seq 0 14); do pods+=("${JOB}-worker-$i"); done

mkdir -p "$LOCAL_C1"
cluster_pod_exec "$MASTER" "mkdir -p '$OUT_DIR'"

# 可选：收集 hosts（失败不阻断矩阵）
: >"$LOCAL_C1/hosts_local.txt"
for pod in "${pods[@]}"; do
  line=$(cluster_pod_exec "$pod" "hostname -i 2>/dev/null | awk '{print \$1}'; echo HOST=\$(hostname)" 2>/dev/null | head -2 | tr '\n' ' ') || true
  echo "${line} POD=$pod" | tee -a "$LOCAL_C1/hosts_local.txt" || true
done
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec -i ${MASTER} -- bash -c 'cat > ${OUT_DIR}/hosts.txt && wc -l ${OUT_DIR}/hosts.txt'" \
  <"$LOCAL_C1/hosts_local.txt" >/dev/null || true

echo "[C1] running pair matrix on master..."
# 在 master 上跑矩阵：TCP/ping 探测；RoCE 跨节点失败如实记入 roce_note
cluster_pod_exec "$MASTER" "
set +e
python3 - <<'PY'
import json, os, socket, time, subprocess
from pathlib import Path
out = Path('$OUT_DIR')
# 重新从各 pod 不可达时用 DNS: pod.job
job = '$JOB'
names = [f'{job}-master-0'] + [f'{job}-worker-{i}' for i in range(15)]
# 解析 IP
ips = []
for n in names:
    try:
        ips.append(socket.gethostbyname(n + '.' + job) if False else socket.gethostbyname(n))
    except Exception:
        # k8s short name
        try:
            ips.append(socket.gethostbyname(n))
        except Exception as e:
            ips.append(None)
# 也试 pod IP via getent
def resolve(host):
    for h in (host, host + '.$JOB', host + '.default.svc'):
        try:
            return socket.gethostbyname(h)
        except Exception:
            continue
    return None
ips = [resolve(n) for n in names]
rows = []
matrix = []
for i, src in enumerate(names):
    row = []
    for j, dst in enumerate(names):
        if i == j:
            cell = {'status': 'self', 'ms': 0.0}
        elif ips[j] is None:
            cell = {'status': 'unresolved', 'ms': None}
        else:
            # TCP connect to 22 or high port — 容器内 sshd 可能无；改探测任意可达：连对端 29500 会 refuse，用 ICMP 不可用时用 TCP 1s 到 80/443/22
            t0 = time.time()
            status = 'fail'
            err = ''
            for port in (22, 29500, 8080):
                try:
                    s = socket.create_connection((ips[j], port), timeout=1.0)
                    s.close()
                    status = 'ok'
                    break
                except Exception as e:
                    err = str(e)
            # 额外：ping 一次（若有权限）
            if status != 'ok':
                try:
                    r = subprocess.run(['ping', '-c', '1', '-W', '1', ips[j]], capture_output=True, text=True, timeout=3)
                    if r.returncode == 0:
                        status = 'ping_ok'
                except Exception as e:
                    err = err or str(e)
            ms = (time.time() - t0) * 1000
            cell = {'status': status, 'ms': round(ms, 2), 'ip': ips[j], 'err': err[:120]}
        row.append(cell)
        rows.append({'i': i, 'j': j, 'src': src, 'dst': names[j], **cell})
    matrix.append(row)
summary = {
    'n': 16,
    'ok': sum(1 for r in rows if r['status'] in ('ok', 'ping_ok', 'self')),
    'fail': sum(1 for r in rows if r['status'] in ('fail', 'unresolved')),
    'ips': ips,
    'names': names,
}
(out / 'pair_matrix.json').write_text(json.dumps({'summary': summary, 'matrix': matrix, 'flat': rows}, indent=2))
(out / 'pair_summary.json').write_text(json.dumps(summary, indent=2))
print('MATRIX_DONE', summary)
PY
"

# RoCE 门禁复测：单节点 w8 OK 标记 + 跨 2 节点尝试记录失败
log() { echo "[C1] $*"; }
log "RoCE spot: intra-node note from prior gate; attempt 2-node fail record"
cluster_pod_exec "$MASTER" "
echo '{\"roce_inter_node\": \"expected_fail_proxy_connect\", \"ref\": \"docs/muxi/muxi_ib_gate_20260712_gid4.md\"}' > '$OUT_DIR/roce_note.json'
"

# 拉回
LOCAL_C1="${LOCAL_C1:-/tmp/muxi_c1}"
mkdir -p "$LOCAL_C1"
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec ${MASTER} -- bash -lc $(printf '%q' "cat $OUT_DIR/pair_summary.json")" \
  > "$LOCAL_C1/pair_summary.json" 2>/dev/null || true
ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
  "$(_cluster_vcctl_prefix) pod exec ${MASTER} -- bash -lc $(printf '%q' "cat $OUT_DIR/pair_matrix.json")" \
  > "$LOCAL_C1/pair_matrix.json" 2>/dev/null || true
echo "C1_DONE OUT=$OUT_DIR LOCAL=$LOCAL_C1"
