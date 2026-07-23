#!/usr/bin/env bash
# 修复 tar 污染后，仅重跑 master LLDP + 聚合报告。
set -euo pipefail

RUN_ID="${RUN_ID:?}"
BUNDLE_DIR="${BUNDLE_DIR:?}"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config-vc-c550-h3c-test.yaml}"
VCCTL="${VCCTL:-/usr/local/bin/vcctl}"
JOB="${JOB:-yinjinrun-cs512-20260716-221823}"
LLDP_DURATION="${LLDP_DURATION:-70}"
WORK="${WORK:-/tmp/muxi-topology-$RUN_ID}"
AFS_OUT="${AFS_OUT:-/afs-a3-weight-share/yinjinrun.p/results/muxi-topology-facts/$RUN_ID}"
MASTER="$JOB-master-0"

export KUBECONFIG
mkdir -p "$WORK/lldp/pods" "$WORK/report"
exec > >(tee -a "$WORK/lldp_rerun.log") 2>&1

echo "LLDP_RERUN_BEGIN run_id=$RUN_ID at=$(date -Iseconds)"
master_lldp_dir="$WORK/lldp/pods/$MASTER"
rm -rf "$master_lldp_dir"
mkdir -p "$master_lldp_dir"

"$VCCTL" pod exec -i "$MASTER" -- bash -lc "
  set -euo pipefail
  cat > /tmp/muxi_lldp_listen.py
  rm -rf /tmp/lldp_${RUN_ID}
  mkdir -p /tmp/lldp_${RUN_ID}
  python3 /tmp/muxi_lldp_listen.py \
    --pod '$MASTER' \
    --ifaces eth0,net1,net2,net3,net4 \
    --duration $LLDP_DURATION \
    --out-dir /tmp/lldp_${RUN_ID} \
    >/tmp/lldp_${RUN_ID}/listener.stdout 2>/tmp/lldp_${RUN_ID}/listener.stderr
  tar -C /tmp/lldp_${RUN_ID} -cf - .
" <"$BUNDLE_DIR/muxi_lldp_listen.py" >"$WORK/lldp/master_bundle.tar" 2>"$WORK/lldp/master_exec.err"

if tar -tf "$WORK/lldp/master_bundle.tar" >/dev/null 2>&1; then
  tar -C "$master_lldp_dir" -xf "$WORK/lldp/master_bundle.tar"
  echo "TAR_OK files=$(find "$master_lldp_dir" -type f | wc -l)"
else
  echo "TAR_BAD size=$(wc -c <"$WORK/lldp/master_bundle.tar")"
  "$VCCTL" pod exec "$MASTER" -- bash -lc "cat /tmp/lldp_${RUN_ID}/summary.json" \
    >"$master_lldp_dir/summary.json" || true
  # 尽量拉回各 iface result
  for iface in eth0 net1 net2 net3 net4; do
    mkdir -p "$master_lldp_dir/$iface"
    "$VCCTL" pod exec "$MASTER" -- bash -lc \
      "test -f /tmp/lldp_${RUN_ID}/$iface/result.json && cat /tmp/lldp_${RUN_ID}/$iface/result.json" \
      >"$master_lldp_dir/$iface/result.json" 2>/dev/null || true
  done
fi

python3 - <<PY
import json
from pathlib import Path
p = Path("$master_lldp_dir/summary.json")
print("summary_exists", p.exists())
if p.exists():
    s = json.loads(p.read_text())
    print("MASTER_LLDP any=%s total=%s" % (s.get("any_lldp"), s.get("total_lldp_frames")))
    print("cap", s.get("cap"))
    for i in s.get("ifaces", []):
        print(
            "IFACE",
            i.get("iface"),
            "lldp",
            i.get("lldp_frames"),
            "raw",
            i.get("raw_frames_seen"),
            "dur",
            i.get("duration_s"),
            "status",
            i.get("status"),
        )
PY

expand=0
if [[ -f "$master_lldp_dir/summary.json" ]]; then
  expand=$(python3 -c "import json;print(1 if json.load(open('$master_lldp_dir/summary.json')).get('any_lldp') else 0)")
fi

if [[ "$expand" -eq 1 ]]; then
  echo "EXPAND64 begin"
  pods=("$MASTER")
  for i in $(seq 0 62); do pods+=("$JOB-worker-$i"); done
  run_one() {
    local pod="$1"
    local dest="$WORK/lldp/pods/$pod"
    mkdir -p "$dest"
    "$VCCTL" pod exec -i "$pod" -- bash -lc "
      set -euo pipefail
      cat > /tmp/muxi_lldp_listen.py
      rm -rf /tmp/lldp_${RUN_ID}
      mkdir -p /tmp/lldp_${RUN_ID}
      python3 /tmp/muxi_lldp_listen.py \
        --pod '$pod' --ifaces eth0,net1,net2,net3,net4 \
        --duration $LLDP_DURATION --out-dir /tmp/lldp_${RUN_ID} \
        >/tmp/lldp_${RUN_ID}/listener.stdout 2>/tmp/lldp_${RUN_ID}/listener.stderr
      tar -C /tmp/lldp_${RUN_ID} -cf - .
    " <"$BUNDLE_DIR/muxi_lldp_listen.py" >"$dest/bundle.tar" 2>"$dest/exec.err" || return $?
    tar -C "$dest" -xf "$dest/bundle.tar"
  }
  pids=()
  for pod in "${pods[@]}"; do
    run_one "$pod" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  echo "EXPAND64 end"
else
  echo "NO_EXPAND master_any_lldp=0"
fi

python3 "$BUNDLE_DIR/parse_muxi_lldp.py" --lldp-root "$WORK/lldp/pods" --out-dir "$WORK/lldp"
python3 "$BUNDLE_DIR/aggregate_topology_report.py" --work-dir "$WORK" --out-dir "$WORK/report"

# 写 prior evidence 备注
cat >"$WORK/report/prior_evidence_search.md" <<'EOF'
# 已有报告 / AFS 中的拓扑命名检索

检索范围：lab-workspace/reports、myportal/results/muxi-h3c、AFS `yinjinrun.p/results`。

## 明确存在的直接配置/主机静态证据（本轮前）

- net1..4 有 172.23–26.x IPv4，与 GID4/5 一致（`MUXI_MULTI_UNLOCK_EVIDENCE_20260712.md`）
- ARP 指向网关 MAC，非 peer NIC（同上；`MUXI_ROCE_PLATFORM_REQUEST`）
- W2 inventory：xscale_i→net{i+1}、200G、MTU4096、无 lldpctl

## 未找到的厂商环境硬证据

- 无具体 leaf/spine 设备主机名或配线表
- 无交换机 CLI 快照、无正式 port-channel/MLAG/ECMP 文档落入个人 AFS
- 前序报告中的 leaf/spine 字样均为待请求项或禁止命名的边界说明，不是已恢复拓扑

EOF

"$VCCTL" pod exec "$MASTER" -- bash -lc "mkdir -p '$AFS_OUT'"
tar -C "$WORK" -cf - lldp report manifest.yaml lldp_rerun.log control_plane logical \
  | "$VCCTL" pod exec -i "$MASTER" -- bash -lc "tar -C '$AFS_OUT' -xf -"

{
  echo "lldp_rerun_at: $(date -Iseconds)"
  echo "lldp_expanded_64_rerun: $expand"
} >>"$WORK/manifest.yaml"

echo "LLDP_RERUN_END at=$(date -Iseconds) expand=$expand afs=$AFS_OUT"
