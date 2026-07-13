#!/usr/bin/env bash
# 卡间拓扑 best-effort 探测（烤机后、collective 前）
#
# 每节点 raw：npu-smi / hccn.conf / hccn_tool(若有) / ranktable 搜索 / HCCL env
# 本机汇总 topo_summary.json；缺工具不崩。
#
# 用法:
#   ./scripts/cluster/probe_hccl_topology.sh
#   CLUSTER_JOB=huawei-8node-copy2 ./scripts/cluster/probe_hccl_topology.sh
#   LOCAL_ONLY=1 ./scripts/cluster/probe_hccl_topology.sh
#
# 方案: reports/research/research_comm_topology_r0.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
AFS_OUT="${AFS_RESULTS:-/afs-a3-weight-share/yinjinrun.p-huawei/results}/hccl-topo-${STAMP}"
OPS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$OPS_ROOT/../../logs/hccl-topo-${STAMP}}"
mkdir -p "$LOG_DIR/raw" "$LOG_DIR/results"
exec > >(tee -a "$LOG_DIR/topo.log") 2>&1

LOCAL_ONLY="${LOCAL_ONLY:-0}"
MAX_DEV="${MAX_DEV:-15}"

PODS=(
  "${CLUSTER_JOB}-master-0"
  "${CLUSTER_JOB}-worker-0"
  "${CLUSTER_JOB}-worker-1"
  "${CLUSTER_JOB}-worker-2"
  "${CLUSTER_JOB}-worker-3"
  "${CLUSTER_JOB}-worker-4"
  "${CLUSTER_JOB}-worker-5"
  "${CLUSTER_JOB}-worker-6"
)

# 节点侧探测命令（字符串；在目标 bash -lc 中执行）
node_probe_cmd() {
  local out_dir="$1"
  cat <<EOF
set +e
OUT='$out_dir'
MAX_DEV='$MAX_DEV'
mkdir -p "\$OUT"
HOST=\$(hostname 2>/dev/null || echo unknown)
RAW="\$OUT/\${HOST}.raw.txt"
{
  echo HOST=\$HOST
  echo TS=\$(date -Iseconds 2>/dev/null || date)
  echo UNAME=\$(uname -a 2>/dev/null)
  echo ''
  echo '=== which npu-smi ==='
  which npu-smi 2>&1
  echo '=== npu-smi info ==='
  npu-smi info 2>&1
  echo '=== npu-smi info -l ==='
  npu-smi info -l 2>&1
  echo '=== npu-smi info -m ==='
  npu-smi info -m 2>&1
  echo '=== npu-smi info -t topo ==='
  npu-smi info -t topo 2>&1
  echo '=== npu-smi info -t health ==='
  for i in \$(seq 0 \$MAX_DEV); do
    echo "-- device \$i --"
    npu-smi info -t health -i \$i 2>&1
  done
  echo '=== /etc/hccn.conf ==='
  if [ -f /etc/hccn.conf ]; then cat /etc/hccn.conf 2>&1; else echo 'MISSING /etc/hccn.conf'; fi
  echo '=== /etc/ascend_install.info ==='
  if [ -f /etc/ascend_install.info ]; then cat /etc/ascend_install.info 2>&1; else echo 'MISSING /etc/ascend_install.info'; fi
  echo '=== find hccn_tool ==='
  HCCN=\$(find /usr/local/Ascend /usr/local/bin /usr/bin -name hccn_tool 2>/dev/null | head -1)
  echo HCCN_TOOL=\$HCCN
  if [ -n "\$HCCN" ]; then
    for i in \$(seq 0 \$MAX_DEV); do
      echo "=== hccn device \$i link ==="
      "\$HCCN" -i \$i -link -g 2>&1
      echo "=== hccn device \$i speed ==="
      "\$HCCN" -i \$i -speed -g 2>&1
      echo "=== hccn device \$i stat ==="
      "\$HCCN" -i \$i -stat -g 2>&1
      echo "=== hccn device \$i ip ==="
      "\$HCCN" -i \$i -ip -g 2>&1
      echo "=== hccn device \$i net_health ==="
      "\$HCCN" -i \$i -net_health -g 2>&1
    done
  else
    echo 'hccn_tool not found'
  fi
  echo '=== driver topo dir ==='
  ls -la /usr/local/Ascend/driver/topo 2>&1 | head -40
  ls -la /usr/local/Ascend/driver/tools 2>&1 | head -20
  echo '=== search ranktable / hccl json ==='
  find /etc /home /tmp /opt /afs-a3-241ceshi-shared \\( -name 'rank_table*.json' -o -name 'ranktable*.json' -o -name 'hccl*.json' \\) 2>/dev/null | head -40
  echo '=== HCCL / Ascend env ==='
  env 2>/dev/null | grep -E '^(HCCL_|ASCEND_|RANK_TABLE|RANKTABLE|WORLD_SIZE|LOCAL_RANK|RANK|MASTER_)' | sort
} >"\$RAW" 2>&1
ls -la "\$RAW"
echo PROBE_NODE_DONE host=\$HOST
EOF
}

summarize_all() {
  python3 - "$LOG_DIR" "$STAMP" <<'PY'
import json, re, sys
from pathlib import Path

log_dir = Path(sys.argv[1])
stamp = sys.argv[2]
results = log_dir / "results"
results.mkdir(parents=True, exist_ok=True)

def summarize_raw(raw_path: Path) -> dict:
    text = raw_path.read_text(errors="replace")
    host_m = re.search(r"^HOST=(.*)$", text, re.M)
    host = (host_m.group(1).strip() if host_m else raw_path.stem.replace(".raw", ""))

    npu_ids = sorted(set(re.findall(r"^\|\s*(\d+)\s+Ascend", text, re.M)), key=lambda x: int(x))
    health_ok = len(re.findall(r"Health\s*:\s*OK", text, re.I))
    health_bad = len(re.findall(r"Health\s*:\s*(?!OK)\w+", text, re.I))

    hccn_m = re.search(r"^HCCN_TOOL=(.*)$", text, re.M)
    hccn_path = (hccn_m.group(1).strip() if hccn_m else "")
    hccn_found = bool(hccn_path) and hccn_path.startswith("/")

    topo_block = ""
    m = re.search(r"=== npu-smi info -t topo ===\n(.*?)(?=\n=== |\Z)", text, re.S)
    if m:
        topo_block = m.group(1).strip()
    topo_nonempty = (
        bool(topo_block)
        and "command not found" not in topo_block.lower()
        and len(topo_block) > 20
    )
    hccs_count = len(re.findall(r"\bHCCS\b", topo_block))

    conf_missing = "MISSING /etc/hccn.conf" in text
    addrs = dict(re.findall(r"^address_(\d+)=(\S+)", text, re.M))

    rank_files = []
    sec = re.search(r"=== search ranktable.*===\n(.*?)(?=\n=== |\Z)", text, re.S)
    if sec:
        for line in sec.group(1).splitlines():
            line = line.strip()
            if line.startswith("/") and ".json" in line:
                rank_files.append(line)

    env_keys = []
    esec = re.search(r"=== HCCL / Ascend env ===\n(.*?)(?=\n=== |\Z)", text, re.S)
    if esec:
        for line in esec.group(1).splitlines():
            if "=" in line:
                env_keys.append(line.split("=", 1)[0])

    npu_smi_ok = ("npu-smi" in text) and ("command not found" not in text.split("=== npu-smi info ===")[-1][:200].lower() if "=== npu-smi info ===" in text else False)
    # simpler: Ascend table present
    npu_smi_ok = bool(re.search(r"Ascend\d+|npu-smi\s+\d", text))

    node = {
        "host": host,
        "tools_available": {
            "npu_smi": npu_smi_ok,
            "npu_smi_topo": topo_nonempty,
            "hccn_conf": (not conf_missing) and bool(addrs),
            "hccn_tool": hccn_found,
            "ranktable_files": bool(rank_files),
        },
        "npu_ids_guess": npu_ids,
        "npu_count_guess": len(npu_ids) if npu_ids else None,
        "health_ok_mentions": health_ok,
        "health_non_ok_mentions": health_bad,
        "topo": {
            "nonempty": topo_nonempty,
            "hccs_cell_count": hccs_count,
            "preview": topo_block[:1200] if topo_nonempty else "",
        },
        "hccn_tool_path": hccn_path if hccn_found else None,
        "hccn_conf_addresses": addrs,
        "ranktable_candidates": rank_files[:20],
        "env_keys": env_keys[:80],
        "raw_file": str(raw_path),
    }
    sum_path = results / f"{host}.summary.json"
    sum_path.write_text(json.dumps(node, ensure_ascii=False, indent=2) + "\n")
    return node

nodes = []
for raw in sorted(results.glob("*.raw.txt")):
    try:
        nodes.append(summarize_raw(raw))
    except Exception as e:
        nodes.append({"host": raw.name, "error": str(e)})

# 也扫 LOG_DIR 根下误放的 raw
for raw in sorted(log_dir.glob("*.raw.txt")):
    if not any(n.get("raw_file") == str(raw) for n in nodes):
        try:
            nodes.append(summarize_raw(raw))
        except Exception as e:
            nodes.append({"host": raw.name, "error": str(e)})

tools = {
    "npu_smi": any(n.get("tools_available", {}).get("npu_smi") for n in nodes),
    "npu_smi_topo": any(n.get("tools_available", {}).get("npu_smi_topo") for n in nodes),
    "hccn_conf": any(n.get("tools_available", {}).get("hccn_conf") for n in nodes),
    "hccn_tool": any(n.get("tools_available", {}).get("hccn_tool") for n in nodes),
    "ranktable_files": any(n.get("tools_available", {}).get("ranktable_files") for n in nodes),
}
n_nodes = len(nodes)
tier_hint = {1: "1n16", 2: "2n32", 4: "4n64", 8: "8n128"}.get(n_nodes, f"{n_nodes}n?")

out = {
    "probe_id": f"topo-{stamp}",
    "stamp": stamp,
    "node_count": n_nodes,
    "topo_tier_hint": tier_hint,
    "tools_available_any_node": tools,
    "nodes": nodes,
    "notes": [
        "Health=OK 不等于链路健康；无 hccn_tool 时无法做 link/speed 交叉。",
        "机内矩阵见各节点 topo.preview；跨机依赖 hccn_conf addresses + 后续 P2P。",
        "详见 reports/research/research_comm_topology_r0.md",
    ],
}
path = log_dir / "topo_summary.json"
path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(f"CLUSTER_SUMMARY → {path}")
print("tools_available_any_node=", json.dumps(tools, ensure_ascii=False))
PY
}

echo "==> probe_hccl_topology stamp=$STAMP LOCAL_ONLY=$LOCAL_ONLY"

if [[ "$LOCAL_ONLY" == "1" ]]; then
  echo "==> LOCAL_ONLY on $(hostname)"
  bash -lc "$(node_probe_cmd "$LOG_DIR/results")" || true
else
  echo "==> AFS_OUT=$AFS_OUT"
  cluster_pod_exec "${PODS[0]}" "mkdir -p '$AFS_OUT'" || true
  for pod in "${PODS[@]}"; do
    echo "==> probe $pod"
    logf="$LOG_DIR/raw/${pod}.log"
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER_SSH_HOST" \
      "vcctl pod exec ${pod} -- bash -lc $(printf '%q' "$(node_probe_cmd "$AFS_OUT")")" \
      >"$logf" 2>&1 || echo "WARN pod_probe_failed pod=$pod" | tee -a "$LOG_DIR/failures.txt"
  done
  echo "==> pull results"
  ssh -o BatchMode=yes "$CLUSTER_SSH_HOST" \
    "vcctl pod exec -i ${PODS[0]} -- bash -c 'tar -C $AFS_OUT -cf - .' " \
    >"$LOG_DIR/results.tar" || true
  if [[ -s "$LOG_DIR/results.tar" ]]; then
    tar -xf "$LOG_DIR/results.tar" -C "$LOG_DIR/results" 2>/dev/null || true
  fi
fi

summarize_all

echo ""
echo "TOPO_PROBE_DONE → $LOG_DIR"
echo "summary: $LOG_DIR/topo_summary.json"
if [[ -f "$LOG_DIR/failures.txt" ]]; then
  echo "failures:"
  cat "$LOG_DIR/failures.txt"
fi
[[ -f "$LOG_DIR/topo_summary.json" ]]
