#!/usr/bin/env bash
# 多维 NPU 干扰敏感度图谱。
#
# Examples:
#   PHYS=11 MODE=smoke bash run_sensitivity_atlas.sh
#   PHYS=11 MODE=screen bash run_sensitivity_atlas.sh
set -euo pipefail

PHYS="${PHYS:?error: PHYS is required, e.g. PHYS=11 MODE=smoke bash $0}"
MODE="${MODE:-smoke}"
HOST_CS="${HOST_CS:-$HOME/CARD_SCREEN}"
HOST_SCRIPTS="${HOST_SCRIPTS:-$HOME/lab-workspace/scripts/cluster}"
IMG="${IMG:-quay.io/ascend/vllm-ascend:v0.19.1rc1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)-sensitivity-atlas-d${PHYS}-${MODE}}"

case "$MODE" in
  smoke|screen|targeted) ;;
  *)
    echo "error: MODE must be smoke, screen, or targeted" >&2
    exit 2
    ;;
esac

OUT_HOST="$HOST_CS/results/$RUN_ID"
LOG_DIR="$HOST_CS/logs/$RUN_ID"
LOG="$LOG_DIR/run.log"
mkdir -p "$OUT_HOST" "$LOG_DIR"

DEVICES=()
for i in $(seq 0 15); do
  DEVICES+=(--device="/dev/davinci${i}")
done
DEVICES+=(
  --device=/dev/davinci_manager
  --device=/dev/devmm_svm
  --device=/dev/hisi_hdc
)

if [[ "$MODE" == "screen" ]]; then
  MATRIX_RUNNER=sensitivity_atlas_screen_npu.py
  MATRIX_ARGS=(--device 0 --out-dir "/workspace/CARD_SCREEN/results/$RUN_ID")
else
  MATRIX_RUNNER=run_sensitivity_atlas_matrix.py
  MATRIX_ARGS=(--mode "$MODE" --device 0 --out-dir "/workspace/CARD_SCREEN/results/$RUN_ID")
fi

{
  echo "RUN_ID=$RUN_ID"
  echo "PHYS=$PHYS MODE=$MODE"
  echo "IMAGE=$IMG"
  date -Is
} | tee "$LOG"

set +e
sudo -n docker run --rm \
  --name "sensitivity-atlas-${RUN_ID}" \
  --network=host \
  --ipc=host \
  "${DEVICES[@]}" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$HOST_CS":/workspace/CARD_SCREEN \
  -v "$HOST_SCRIPTS":/workspace/atlas \
  -w /workspace/atlas \
  -e "ASCEND_RT_VISIBLE_DEVICES=${PHYS}" \
  -e PYTHONUNBUFFERED=1 \
  "$IMG" \
  python3 "$MATRIX_RUNNER" "${MATRIX_ARGS[@]}" \
  2>&1 | tee -a "$LOG"
RC="${PIPESTATUS[0]}"
set -e

cat >"$OUT_HOST/RUN_META.md" <<EOF
# RUN_META

- run_id: \`$RUN_ID\`
- physical_device: $PHYS
- mode: $MODE
- image: \`$IMG\`
- injectors: cube / vector / hbm_mte / hbm_vector / small_ops
- victims: gemm / attention / norm / elementwise / block / transformer
- profiles: small / large
- patterns: periodic / poisson
- screen_doses: 0 / 0.1 / 0.3 / 0.5
- screen_window: 2 秒
- targeted_repeats: 5
- exit_code: $RC
EOF

echo "result_path: $OUT_HOST" | tee -a "$LOG"
exit "$RC"
