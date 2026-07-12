#!/usr/bin/env bash
# Muxi MoE 缩小版真训练（proxy→真 EP）：experts 默认 8、topk 2，单机 EP=8 可冒烟
# 全尺寸 128 experts / topk8 等 R0 显存验证后再抬。
set -uo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export MCCL_SOCKET_IFNAME="${MCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-xscale}"
export MCCL_IB_HCA="${MCCL_IB_HCA:-xscale}"
export PATH="/opt/conda/bin:${PATH:-/usr/bin}"

CU_BRIDGE_BIN="${CU_BRIDGE_BIN:-/opt/maca/tools/cu-bridge/bin}"
if [[ -x "${CU_BRIDGE_BIN}/cucc" && ! -e "${CU_BRIDGE_BIN}/nvcc" ]]; then
  ln -sfn "${CU_BRIDGE_BIN}/cucc" "${CU_BRIDGE_BIN}/nvcc" || true
fi
export CUDA_HOME="${CUDA_HOME:-/opt/maca/tools/cu-bridge}"

MEGATRON_ROOT="${MEGATRON_ROOT:-/afs-a3-weight-share/workspace/Megatron-LM}"
cd "$MEGATRON_ROOT"
export PYTHONPATH="${MEGATRON_ROOT}:${PYTHONPATH:-}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-30301}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
TP="${TP:-1}"
PP="${PP:-1}"
EP="${EP:-8}"
MBS="${MBS:-1}"
GBS="${GBS:-64}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
NUM_LAYERS="${NUM_LAYERS:-8}"
HIDDEN="${HIDDEN:-2048}"
FFN_HIDDEN="${FFN_HIDDEN:-6144}"
HEADS="${HEADS:-16}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
MOE_TOPK="${MOE_TOPK:-2}"
MOE_FFN="${MOE_FFN:-768}"
MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-1}"
RUN_DIR="${RUN_DIR:-/tmp/train-mfu-moe-muxi}"
mkdir -p "$RUN_DIR"

RECOMPUTE_ARGS=()
if [[ "${RECOMPUTE:-1}" == "1" ]]; then
  RECOMPUTE_ARGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)
fi

torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  pretrain_gpt.py \
  --tensor-model-parallel-size "$TP" \
  --pipeline-model-parallel-size "$PP" \
  --expert-model-parallel-size "$EP" \
  --num-layers "$NUM_LAYERS" \
  --hidden-size "$HIDDEN" \
  --ffn-hidden-size "$FFN_HIDDEN" \
  --num-attention-heads "$HEADS" \
  --seq-length "$SEQ_LENGTH" \
  --max-position-embeddings "$SEQ_LENGTH" \
  --micro-batch-size "$MBS" \
  --global-batch-size "$GBS" \
  --train-iters "$TRAIN_ITERS" \
  --lr 1e-4 \
  --min-lr 1e-5 \
  --lr-decay-style cosine \
  --lr-warmup-iters 0 \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --bf16 \
  --disable-bias-linear \
  --mock-data \
  --tokenizer-type NullTokenizer \
  --vocab-size 32000 \
  --num-experts "$NUM_EXPERTS" \
  --moe-router-topk "$MOE_TOPK" \
  --moe-ffn-hidden-size "$MOE_FFN" \
  --moe-layer-freq "$MOE_LAYER_FREQ" \
  --moe-router-load-balancing-type aux_loss \
  --moe-aux-loss-coeff 0.01 \
  --moe-token-dispatcher-type alltoall \
  --no-load-optim \
  --no-load-rng \
  --no-save-optim \
  --no-save-rng \
  --save-interval 10000 \
  --eval-interval 10000 \
  --eval-iters 0 \
  --log-interval 1 \
  --log-throughput \
  --timing-log-level 0 \
  --distributed-backend nccl \
  --use-distributed-optimizer \
  --transformer-impl local \
  --attention-backend unfused \
  "${RECOMPUTE_ARGS[@]}" \
  2>&1 | tee "$RUN_DIR/train.log"
ec=${PIPESTATUS[0]}

if [[ "$ec" -eq 0 ]]; then
  echo TRAIN_MOE_MUXI_DONE | tee -a "$RUN_DIR/train.log"
else
  echo TRAIN_MOE_MUXI_FAIL ec=$ec | tee -a "$RUN_DIR/train.log"
fi
exit "$ec"
