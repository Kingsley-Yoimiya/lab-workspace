#!/usr/bin/env bash
# Muxi 真训练 MFU 冒烟：tiny GPT + --mock-data（对标 Ascend wrapper 的最小可跑版）
# 在 /afs-a3-weight-share/workspace/Megatron-LM 根目录由 torchrun 拉起。
set -uo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export MCCL_SOCKET_IFNAME="${MCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export PATH="/opt/conda/bin:${PATH:-/usr/bin}"

# MetaX cu-bridge 无 nvcc；Megatron fused_kernels 硬查 CUDA_HOME/bin/nvcc
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
MASTER_PORT="${MASTER_PORT:-30101}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
TP="${TP:-1}"
PP="${PP:-1}"
MBS="${MBS:-1}"
GBS="${GBS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
RUN_DIR="${RUN_DIR:-/tmp/train-mfu-muxi}"
mkdir -p "$RUN_DIR"

# 极小模型，保证冒烟可在数分钟内出 TFLOP 日志
NUM_LAYERS="${NUM_LAYERS:-4}"
HIDDEN="${HIDDEN:-1024}"
HEADS="${HEADS:-16}"

torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  pretrain_gpt.py \
  --tensor-model-parallel-size "$TP" \
  --pipeline-model-parallel-size "$PP" \
  --num-layers "$NUM_LAYERS" \
  --hidden-size "$HIDDEN" \
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
  --mock-data \
  --tokenizer-type NullTokenizer \
  --vocab-size 32000 \
  --no-load-optim \
  --no-load-rng \
  --no-save-optim \
  --no-save-rng \
  --save-interval 10000 \
  --eval-interval 10000 \
  --eval-iters 0 \
  --log-interval 1 \
  --timing-log-level 0 \
  --distributed-backend nccl \
  --use-distributed-optimizer \
  --transformer-impl local \
  --attention-backend unfused \
  2>&1 | tee "$RUN_DIR/train.log"
ec=${PIPESTATUS[0]}

if [[ "$ec" -eq 0 ]]; then
  echo TRAIN_TINY_MUXI_DONE | tee -a "$RUN_DIR/train.log"
else
  echo TRAIN_TINY_MUXI_FAIL ec=$ec | tee -a "$RUN_DIR/train.log"
fi
exit "$ec"
