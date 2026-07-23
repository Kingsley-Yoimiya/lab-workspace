#!/usr/bin/env bash
# Phase2：代表卡上跑必跑算子/Block（可选 inject）
# 用法（在 npu-dev-1 上，docker 内或主机经 docker run）:
#   PHYS=11 bash run_phase2_ops.sh
set -euo pipefail
PHYS="${PHYS:?PHYS required}"
HOST_CS="${HOST_CS:-$HOME/CARD_SCREEN}"
HOST_SCRIPTS="${HOST_SCRIPTS:-$HOME/lab-workspace/scripts/cluster}"
IMG="${IMG:-quay.io/ascend/vllm-ascend:v0.19.1rc1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)-phase2-ops-d${PHYS}}"
OUT_HOST="$HOST_CS/results/$RUN_ID"
mkdir -p "$OUT_HOST"

DEVICES=()
for i in $(seq 0 15); do DEVICES+=(--device="/dev/davinci${i}"); done
DEVICES+=(--device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc)

WORKLOADS="${WORKLOADS:-gemm_ffn_up gemm_ffn_down ln block_fwd_bwd block_small_ops}"
# 无注入 + cube mid（若有 dose_table 可改）
FACTORS="${FACTORS:-none cube}"
DOSES_NONE="0"
DOSES_CUBE="${DOSES_CUBE:-0 mid}"

sudo -n docker run --rm --name "phase2-${RUN_ID}" --network=host --ipc=host \
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
mkdir -p \"\$OUT\"
# baseline (no inject)
for wl in ${WORKLOADS}; do
  python3 op_block_bench_npu.py --workload \"\$wl\" --device 0 --iters 30 --warmup 5 \
    --factor none --dose 0 --out \"\$OUT/ops.jsonl\"
done
# cube mid inject（duty=1.0 保证短算子窗口内持续争用）
for wl in ${WORKLOADS}; do
  python3 op_block_bench_npu.py --workload \"\$wl\" --device 0 --iters 30 --warmup 5 \
    --factor cube --dose mid --inject-kind cube --inject-duty 1.0 \
    --out \"\$OUT/ops.jsonl\" || true
done
# hbm_mte mid inject (Phase1 Top 效应)
for wl in ${WORKLOADS}; do
  python3 op_block_bench_npu.py --workload \"\$wl\" --device 0 --iters 30 --warmup 5 \
    --factor hbm_mte --dose mid --inject-kind hbm_mte --inject-duty 1.0 \
    --out \"\$OUT/ops.jsonl\" || true
done
python3 parse_op_sensitivity.py --jsonl \"\$OUT/ops.jsonl\" --out \"\$OUT/summary.json\" || true
echo PHASE2_DONE
"
echo "$OUT_HOST"
