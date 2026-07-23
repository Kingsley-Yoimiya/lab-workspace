#!/usr/bin/env bash
# Phase1 单次剂量试验：docker 内 inject sidecar + CARD_SCREEN sentinel
#
# 必填/常用环境变量:
#   PHYS_DEVICE   物理卡号（ASCEND_RT_VISIBLE_DEVICES）
#   KIND          cpu|cube|vector|hbm_mte|placebo
#   DOSE_LABEL    0|light|mid|heavy|...
#   DUTY          0–1
#   RUN_ID        结果目录名
#   HOST_CS       主机 CARD_SCREEN 路径（默认 $HOME/CARD_SCREEN）
#
# 可选:
#   HOST_SCRIPTS  主机 scripts/cluster（默认本脚本目录）
#   IMG           镜像（默认 vllm-ascend:v0.19.1rc1）
#   PERIOD_MS SIZE ELEMS MB DTYPE VECTOR_OP CPU_THREADS
#   INJECT_SECONDS  sidecar 最长秒数（默认 3600，sentinel 结束后会 kill）
#
# 产物:
#   $HOST_CS/results/$RUN_ID/
#     sentinel*.jsonl  meta.json  inject.log  run.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PHYS_DEVICE="${PHYS_DEVICE:?PHYS_DEVICE required}"
KIND="${KIND:?KIND required}"
DOSE_LABEL="${DOSE_LABEL:?DOSE_LABEL required}"
DUTY="${DUTY:?DUTY required}"
RUN_ID="${RUN_ID:?RUN_ID required}"
HOST_CS="${HOST_CS:-$HOME/CARD_SCREEN}"
HOST_SCRIPTS="${HOST_SCRIPTS:-$SCRIPT_DIR}"
IMG="${IMG:-quay.io/ascend/vllm-ascend:v0.19.1rc1}"

PERIOD_MS="${PERIOD_MS:-200}"
SIZE="${SIZE:-4096}"
ELEMS="${ELEMS:-67108864}"
MB="${MB:-512}"
DTYPE="${DTYPE:-fp16}"
VECTOR_OP="${VECTOR_OP:-fma}"
CPU_THREADS="${CPU_THREADS:-4}"
INJECT_SECONDS="${INJECT_SECONDS:-3600}"
CONFIG_NAME="${CONFIG_NAME:-config.phase1_sentinel.yaml}"

WORKDIR=/workspace/CARD_SCREEN
INJECT_MNT=/workspace/inject
OUT_HOST="$HOST_CS/results/$RUN_ID"
mkdir -p "$OUT_HOST"

META_JSON="$OUT_HOST/meta.json"
python3 - <<PY
import json
meta = {
    "record": "dose_meta",
    "phase": 1,
    "run_id": "$RUN_ID",
    "phys_device": int("$PHYS_DEVICE"),
    "visible_device": 0,
    "factor": "$KIND",
    "dose_label": "$DOSE_LABEL",
    "inject_kind": "$KIND",
    "inject_mode": "process",
    "inject_params": {
        "duty": float("$DUTY"),
        "period_ms": float("$PERIOD_MS"),
        "size": int("$SIZE"),
        "elems": int("$ELEMS"),
        "mb": int("$MB"),
        "dtype": "$DTYPE",
        "vector_op": "$VECTOR_OP",
        "cpu_threads": int("$CPU_THREADS"),
    },
    "placebo": ("$KIND" == "placebo") or (float("$DUTY") <= 0.0),
    "config": "$CONFIG_NAME",
    "image": "$IMG",
    "host_cs": "$HOST_CS",
}
with open("$META_JSON", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

# 挂全部 davinci*（与全卡 CARD_SCREEN 一致）；用 ASCEND_RT_VISIBLE_DEVICES 收窄到 PHYS
DEVICES=()
for i in $(seq 0 15); do
  DEVICES+=(--device="/dev/davinci${i}")
done
DEVICES+=(
  --device=/dev/davinci_manager
  --device=/dev/devmm_svm
  --device=/dev/hisi_hdc
)

DOCKER_NAME="phase1-$(echo "$RUN_ID" | tr '/:' '__')"

echo "[$(date -Is)] phase1 dose calib RUN_ID=$RUN_ID PHYS=$PHYS_DEVICE KIND=$KIND DOSE=$DOSE_LABEL DUTY=$DUTY" \
  | tee "$OUT_HOST/run.log"

# shellcheck disable=SC2086
sudo -n docker run --rm \
  --name "$DOCKER_NAME" \
  --network=host --ipc=host \
  "${DEVICES[@]}" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$HOST_CS":"$WORKDIR" \
  -v "$HOST_SCRIPTS":"$INJECT_MNT":ro \
  -w "$WORKDIR" \
  -e "ASCEND_RT_VISIBLE_DEVICES=${PHYS_DEVICE}" \
  -e PYTHONUNBUFFERED=1 \
  "$IMG" \
  bash -lc "
set -euo pipefail
OUT='results/${RUN_ID}'
mkdir -p \"\$OUT\"
python3 ${INJECT_MNT}/npu_component_inject.py \
  --kind '${KIND}' --device 0 \
  --duty '${DUTY}' --period-ms '${PERIOD_MS}' \
  --size '${SIZE}' --elems '${ELEMS}' --mb '${MB}' \
  --dtype '${DTYPE}' --vector-op '${VECTOR_OP}' \
  --cpu-threads '${CPU_THREADS}' \
  --seconds '${INJECT_SECONDS}' \
  > \"\$OUT/inject.log\" 2>&1 &
INJ=\$!
cleanup() { kill \$INJ 2>/dev/null || true; wait \$INJ 2>/dev/null || true; }
trap cleanup EXIT
# 等 sidecar 打出 INJECT_START
for i in \$(seq 1 60); do
  grep -q INJECT_START \"\$OUT/inject.log\" 2>/dev/null && break
  sleep 0.5
done
python3 screen.py --device 0 \
  --config '${CONFIG_NAME}' \
  --no-require-idle --no-plot \
  --out \"\$OUT/sentinel.jsonl\"
cleanup
trap - EXIT
echo SENTINEL_DONE
" 2>&1 | tee -a "$OUT_HOST/run.log"

echo "[$(date -Is)] done RUN_ID=$RUN_ID → $OUT_HOST" | tee -a "$OUT_HOST/run.log"
echo "$OUT_HOST"
