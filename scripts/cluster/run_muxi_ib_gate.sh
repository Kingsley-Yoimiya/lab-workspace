#!/usr/bin/env bash
# Muxi IB 门禁：D0 诊断 + w8/w16 All-Reduce 对照（默认 IB_HCA=xscale）
# 用法: ./run_muxi_ib_gate.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
AFS_OUT="${AFS_OUT:-/afs-a3-weight-share/montyyin/results/muxi-ib-gate-${STAMP}}"
LOG_DIR="${LOG_DIR:-/Users/yinjinrun/random-thing/logs/muxi-ib-gate-${STAMP}}"
REPORT="${REPORT:-$SCRIPT_DIR/../../reports/rounds/muxi_ib_gate_${STAMP}.md}"
NCCL_IB_HCA="${NCCL_IB_HCA:-xscale}"
MCCL_IB_HCA="${MCCL_IB_HCA:-xscale}"
export NCCL_IB_HCA MCCL_IB_HCA NCCL_DEBUG="${NCCL_DEBUG:-INFO}" MCCL_DEBUG="${MCCL_DEBUG:-INFO}"
mkdir -p "$LOG_DIR"
echo "{\"ts\":\"$(date -Iseconds)\",\"phase\":\"P0\",\"status\":\"ib_gate_start\",\"afs\":\"$AFS_OUT\"}" \
  >> /Users/yinjinrun/random-thing/logs/muxi-watchdog-20260712/check.jsonl

echo "==> D0 verbs/topo on master"
cluster_pod_exec "$CLUSTER_POD" "
ibv_devinfo 2>&1 | egrep 'hca_id:|state:|link_layer:' | head -40
echo '---'
mx-smi topo -n 2>&1 | tail -30
echo '---'
ls /sys/class/infiniband/ 2>/dev/null || true
env | egrep 'NCCL_|MCCL_' | sort || true
" | tee "$LOG_DIR/d0_diag.log"

echo "==> fire w8 then w16 with IB_HCA=$NCCL_IB_HCA"
export AFS_OUT LOG_DIR
"$SCRIPT_DIR/fire_nccl_scale_muxi.sh" 8 29641
"$SCRIPT_DIR/poll_nccl_scale_muxi.sh" 8 || true
"$SCRIPT_DIR/fire_nccl_scale_muxi.sh" 16 29642
"$SCRIPT_DIR/poll_nccl_scale_muxi.sh" 16 || true

python3 - <<PY | tee "$REPORT"
from pathlib import Path
import json, statistics, re
afs = Path("$AFS_OUT")
log = Path("$LOG_DIR")
rows = []
for w in (8, 16):
    p = afs / f"scale_{w}.jsonl"
    bws = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                o = json.loads(line)
            except Exception:
                continue
            if str(o.get("op","")).lower() in ("all_reduce","allreduce") and "256" in str(o.get("size","")):
                for k in ("bus_bw","bus_bw_gbps","alg_bw"):
                    if k in o and o[k] is not None:
                        try: bws.append(float(o[k])); break
                        except Exception: pass
    med = statistics.median(bws) if bws else None
    rows.append((w, med, len(bws)))
w8 = rows[0][1]; w16 = rows[1][1]
keep = (w16/w8*100) if (w8 and w16 and w8>0) else None
gate = keep is not None and keep >= 50.0
print(f"# Muxi IB gate {Path('$STAMP').name if False else '$STAMP'}")
print()
print(f"- IB_HCA: `$NCCL_IB_HCA`")
print(f"- AFS: `$AFS_OUT`")
print(f"- w8 AR@256M median bus_bw: {w8}")
print(f"- w16 AR@256M median bus_bw: {w16}")
print(f"- keep% (w16/w8): {keep}")
print(f"- GATE: {'PASS' if gate else 'FAIL'} (need keep>=50%)")
print()
print("## D0 excerpt")
print("```")
print((log/"d0_diag.log").read_text()[:2000])
print("```")
open("/Users/yinjinrun/random-thing/logs/muxi-watchdog-20260712/check.jsonl","a").write(
  json.dumps({"ts":__import__("datetime").datetime.now().isoformat(),"phase":"P0","status":"PASS" if gate else "FAIL","keep":keep})+"\n"
)
PY

echo "==> report → $REPORT"
