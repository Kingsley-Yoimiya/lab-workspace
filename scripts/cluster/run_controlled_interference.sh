#!/usr/bin/env bash
# 可控 NPU 干扰剂量扫描：预热常驻 sidecar + 随机剂量窗口。
#
# Examples:
#   PHYS=11 MODE=smoke bash run_controlled_interference.sh
#   PHYS=11 MODE=full  bash run_controlled_interference.sh
set -euo pipefail

PHYS="${PHYS:?error: PHYS is required, e.g. PHYS=11 MODE=smoke bash $0}"
MODE="${MODE:-full}"
HOST_CS="${HOST_CS:-$HOME/CARD_SCREEN}"
HOST_SCRIPTS="${HOST_SCRIPTS:-$HOME/lab-workspace/scripts/cluster}"
IMG="${IMG:-quay.io/ascend/vllm-ascend:v0.19.1rc1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)-controlled-interference-d${PHYS}-${MODE}}"

case "$MODE" in
  smoke)
    DOSES="${DOSES:-0,0.2,0.5}"
    WINDOW_S="${WINDOW_S:-2}"
    REPEATS="${REPEATS:-1}"
    ;;
  full)
    DOSES="${DOSES:-0,0.1,0.2,0.3,0.4,0.5}"
    WINDOW_S="${WINDOW_S:-8}"
    REPEATS="${REPEATS:-10}"
    ;;
  *)
    echo "error: MODE must be smoke or full" >&2
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

{
  echo "RUN_ID=$RUN_ID"
  echo "PHYS=$PHYS MODE=$MODE DOSES=$DOSES WINDOW_S=$WINDOW_S REPEATS=$REPEATS"
  echo "IMAGE=$IMG"
  date -Is
} | tee "$LOG"

sudo -n docker run --rm \
  --name "controlled-interference-${RUN_ID}" \
  --network=host \
  --ipc=host \
  "${DEVICES[@]}" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$HOST_CS":/workspace/CARD_SCREEN \
  -v "$HOST_SCRIPTS":/workspace/inject \
  -w /workspace/inject \
  -e "ASCEND_RT_VISIBLE_DEVICES=${PHYS}" \
  -e PYTHONUNBUFFERED=1 \
  "$IMG" bash -lc "
set -euo pipefail
OUT=/workspace/CARD_SCREEN/results/${RUN_ID}

python3 controlled_interference_bench_npu.py \
  --device 0 \
  --workload gemm \
  --inject-kind cube \
  --doses '${DOSES}' \
  --window-s '${WINDOW_S}' \
  --repeats '${REPEATS}' \
  --period-ms 100 \
  --out \"\$OUT/cube_gemm.jsonl\"

python3 controlled_interference_bench_npu.py \
  --device 0 \
  --workload block \
  --inject-kind hbm_mte \
  --doses '${DOSES}' \
  --window-s '${WINDOW_S}' \
  --repeats '${REPEATS}' \
  --period-ms 100 \
  --out \"\$OUT/hbm_mte_block.jsonl\"

echo CONTROLLED_INTERFERENCE_DONE
" 2>&1 | tee -a "$LOG"

cat >"$OUT_HOST/RUN_META.md" <<EOF
# RUN_META

- run_id: \`$RUN_ID\`
- phys_device: $PHYS
- mode: $MODE
- doses: \`$DOSES\`
- window_s: $WINDOW_S
- repeats: $REPEATS
- period_ms: 100
- image: \`$IMG\`
- protocol: 预热常驻 sidecar；JSONL IPC START/STOP；每个小批次后 NPU synchronize；剂量顺序随机。
- pairings: Cube → GEMM；HBM/MTE copy → MLP Block fwd+bwd。
EOF

echo "result_path: $OUT_HOST" | tee -a "$LOG"
