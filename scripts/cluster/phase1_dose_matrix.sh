#!/usr/bin/env bash
# Phase1 剂量矩阵：单卡扫 kind × dose（0/light/mid/heavy）
# 固定 duty 初值：0 / 0.1 / 0.3 / 0.6
#
# 用法:
#   PHYS_DEVICE=7 HOST_CS=$HOME/CARD_SCREEN ./phase1_dose_matrix.sh
#   PHYS_DEVICE=7 KINDS="cube vector" ./phase1_dose_matrix.sh
#
# 产物（HOST_CS/results/$MATRIX_RUN_ID/）:
#   dose_trials.jsonl   每次试验一行 record=dose_trial
#   dose_table.json     汇总 factor→dose_label→{median_drop_pct,params,n}
#   <kind>_<dose>/      各次 run 子目录（sentinel + meta + inject.log）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIB_SH="$SCRIPT_DIR/run_phase1_dose_calib.sh"
PARSE_PY="$SCRIPT_DIR/parse_phase1_dose.py"

PHYS_DEVICE="${PHYS_DEVICE:?PHYS_DEVICE required}"
HOST_CS="${HOST_CS:-$HOME/CARD_SCREEN}"
HOST_SCRIPTS="${HOST_SCRIPTS:-$SCRIPT_DIR}"
MATRIX_RUN_ID="${MATRIX_RUN_ID:-$(date +%Y%m%d_%H%M%S)-phase1-dose-d${PHYS_DEVICE}}"
KINDS="${KINDS:-cpu cube vector hbm_mte}"

# dose_label → duty
dose_duty() {
  case "$1" in
    0|off|placebo|zero) echo 0 ;;
    light) echo 0.1 ;;
    mid) echo 0.3 ;;
    heavy) echo 0.6 ;;
    *) echo "unknown dose label: $1" >&2; return 1 ;;
  esac
}

DOSES="${DOSES:-0 light mid heavy}"
ROOT="$HOST_CS/results/$MATRIX_RUN_ID"
mkdir -p "$ROOT"
TRIALS_JSONL="$ROOT/dose_trials.jsonl"
: >"$TRIALS_JSONL"

echo "[$(date -Is)] phase1 dose matrix PHYS=$PHYS_DEVICE RUN=$MATRIX_RUN_ID" | tee "$ROOT/matrix.log"
echo "KINDS=$KINDS DOSES=$DOSES" | tee -a "$ROOT/matrix.log"

# 每因素先跑 dose=0 建立 baseline，再跑其余档
declare -A BASELINE_METRIC=()

for kind in $KINDS; do
  for dose in $DOSES; do
    duty="$(dose_duty "$dose")"
    # dose=0 用 placebo 语义（同 kind 记 factor，inject 用 duty=0 或 kind=placebo）
    inj_kind="$kind"
    if [[ "$duty" == "0" || "$duty" == "0.0" ]]; then
      inj_kind="placebo"
    fi
    sub_id="${kind}_${dose}"
    run_id="${MATRIX_RUN_ID}/${sub_id}"
    echo "[$(date -Is)] >>> kind=$kind dose=$dose duty=$duty inj=$inj_kind" | tee -a "$ROOT/matrix.log"

    PHYS_DEVICE="$PHYS_DEVICE" \
      KIND="$inj_kind" \
      DOSE_LABEL="$dose" \
      DUTY="$duty" \
      RUN_ID="$run_id" \
      HOST_CS="$HOST_CS" \
      HOST_SCRIPTS="$HOST_SCRIPTS" \
      bash "$CALIB_SH" | tee -a "$ROOT/matrix.log"

    # meta.factor 应记真实因素（placebo 注入时仍归到 kind）
    sub_dir="$HOST_CS/results/$run_id"
    python3 - <<PY
import json
from pathlib import Path
p = Path("$sub_dir/meta.json")
meta = json.loads(p.read_text(encoding="utf-8"))
meta["factor"] = "$kind"
meta["dose_label"] = "$dose"
meta["inject_kind"] = "$inj_kind"
p.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

    base_arg=()
    if [[ -n "${BASELINE_METRIC[$kind]:-}" ]]; then
      base_arg=(--baseline-metric "${BASELINE_METRIC[$kind]}")
    fi

    card_glob="$sub_dir/sentinel*.jsonl"
    trial_line="$(
      python3 "$PARSE_PY" --emit-trial \
        --card "$card_glob" \
        --meta "$sub_dir/meta.json" \
        "${base_arg[@]+"${base_arg[@]}"}"
    )"
    echo "$trial_line" >>"$TRIALS_JSONL"
    echo "$trial_line" | tee -a "$ROOT/matrix.log" >/dev/null

    # 更新 baseline：首个 dose=0 / placebo
    if [[ "$duty" == "0" || "$duty" == "0.0" ]]; then
      BASELINE_METRIC[$kind]="$(
        python3 -c 'import json,sys; t=json.loads(sys.argv[1]); print(t.get("metric_value") or "")' "$trial_line"
      )"
      echo "  baseline[$kind]=${BASELINE_METRIC[$kind]}" | tee -a "$ROOT/matrix.log"
    fi
  done
done

# 用目录全量重算 drop（更稳：median baseline），覆盖 dose_trials + dose_table
python3 "$PARSE_PY" \
  --from-dirs "$ROOT" \
  --write-trials "$TRIALS_JSONL" \
  --out "$ROOT/dose_table.json" \
  | tee -a "$ROOT/matrix.log"

echo "[$(date -Is)] DONE matrix → $ROOT" | tee -a "$ROOT/matrix.log"
echo "$ROOT"
