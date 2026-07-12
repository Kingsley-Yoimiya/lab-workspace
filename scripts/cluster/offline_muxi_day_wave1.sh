#!/usr/bin/env bash
# Wave1 结束后：拉回 A1/A2、解析 A3、画热力图 A4
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_FORCE_JUMP=1
source "$SCRIPT_DIR/muxi.env"
source "$SCRIPT_DIR/job_helpers.sh"
STAMP=$(cat /tmp/muxi_day_stamp.txt)
DAY_ROOT=$(cat /tmp/muxi_day_root.txt)
AFS_OUT=$(cat /tmp/muxi_day_afs.txt)
LOCAL=$DAY_ROOT/results
MASTER=${CLUSTER_JOB}-master-0
mkdir -p "$LOCAL"/{A2/exp1_independent,A4,analysis}

pull_dir() {
  local remote="$1" localdir="$2"
  mkdir -p "$localdir"
  ssh -o BatchMode=yes -o ConnectTimeout=90 ais-cf3e61a5 \
    "KUBECONFIG=/root/.kube/config.muxi-mohe vcctl pod exec ${MASTER} -- bash -lc $(printf '%q' "cd $remote && tar czf - step_times_rank*.jsonl meta_rank*.json done_rank*.txt 2>/dev/null")" \
    > "$localdir.tgz"
  tar xzf "$localdir.tgz" -C "$localdir" 2>/dev/null || true
  echo "pulled $remote -> $(ls "$localdir"/step_times_rank*.jsonl 2>/dev/null | wc -l) files"
}

# A1 各节点
for tag in A1_master A1_worker3 A1_worker10; do
  for mode in exp0_real exp0_virtual; do
    pull_dir "$AFS_OUT/$tag/$mode" "$LOCAL/$tag/$mode"
  done
  # 校准解析
  python3 "$SCRIPT_DIR/parse_virtual_sync.py" "$LOCAL/$tag" --drop-first 100 \
    --out "$LOCAL/analysis/${tag}" || true
done

# A2
pull_dir "$AFS_OUT/A2/exp1_independent" "$LOCAL/A2/exp1_independent"
# 把 A2 数据放到带 exp1_independent 结构的根以便 parse
mkdir -p "$LOCAL/A2_root/exp1_independent"
cp -f "$LOCAL/A2/exp1_independent"/step_times_rank*.jsonl "$LOCAL/A2_root/exp1_independent/" 2>/dev/null || \
  cp -f "$LOCAL/A2/exp1_independent"/step_times_rank*.jsonl "$LOCAL/A2_root/" 2>/dev/null || true
# parse 期望 root/exp1_independent 或 root 直接有 jsonl
if ls "$LOCAL/A2/exp1_independent"/step_times_rank*.jsonl >/dev/null 2>&1; then
  # 直接对 A2 目录：若文件在子目录
  ROOT_FOR_PARSE="$LOCAL/A2"
  if [[ ! -f "$LOCAL/A2/step_times_rank000.jsonl" ]]; then
    # parse looks for exp1_independent subdir OR files in root
    :
  fi
  python3 "$SCRIPT_DIR/parse_virtual_sync.py" "$LOCAL/A2/exp1_independent" \
    --drop-first 100 --n-random 20 --sizes 8,16,32,64,128 \
    --out "$LOCAL/analysis/A2" || true
fi

python3 "$SCRIPT_DIR/../../reports/plot_virtual_sync_gap.py" \
  --csv "$LOCAL/analysis/A2/gap_by_scale.csv" \
  --summary "$LOCAL/analysis/A2/gap_summary.json" \
  --out "$LOCAL/analysis/A2/gap_vs_scale.svg" || true

# A4 telem pull + plot
mkdir -p "$LOCAL/A4/telemetry"
ssh -o BatchMode=yes ais-cf3e61a5 \
  "KUBECONFIG=/root/.kube/config.muxi-mohe vcctl pod exec ${MASTER} -- bash -lc $(printf '%q' "cd $AFS_OUT/A4/telemetry && tar czf - node*.jsonl 2>/dev/null")" \
  > "$LOCAL/A4/telem.tgz" || true
tar xzf "$LOCAL/A4/telem.tgz" -C "$LOCAL/A4/telemetry" 2>/dev/null || true
python3 "$SCRIPT_DIR/../../reports/plot_muxi_power_heatmap.py" \
  --telem-dir "$LOCAL/A4/telemetry" --metric power \
  --out "$LOCAL/analysis/A4_power_heatmap.svg" || true
python3 "$SCRIPT_DIR/../../reports/plot_muxi_power_heatmap.py" \
  --telem-dir "$LOCAL/A4/telemetry" --metric clock \
  --out "$LOCAL/analysis/A4_clock_heatmap.svg" || true

echo A3_A4_OFFLINE_DONE > "$DAY_ROOT/A3.done"
echo OFFLINE_DONE
ls -la "$LOCAL/analysis/"
