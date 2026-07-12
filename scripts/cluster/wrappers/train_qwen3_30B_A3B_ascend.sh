#!/usr/bin/env bash
# Ascend 适配：muxi-Megatron train_qwen3_30B_A3B.sh（MoE，NUM_EXPERTS=128）
# 在 Megatron-LM-0.12.3 根目录执行；由 run_train_mfu_scale.sh 注入分布式环境变量。
# 修复点：去掉 MACA；/afs-grj→真实 AFS；NPUS_PER_NODE=16；torchrun | tee 行续接。
# 注意：PT_qwen3_32B.sh 也是 MoE，不可当 dense 用。
set -uo pipefail
export PATH=/root/miniconda3/envs/llm_test/bin:$PATH
export PYTHONPATH=/MindSpeed-LLM/MindSpeed:${PYTHONPATH:-}
# Do not source set_env.sh here — it may `exit` the shell. Expect ASCEND_* already set,
# or use: eval "$(bash -c 'source ...; export -p' | grep -E '^export (ASCEND|LD_|PATH|ATB)')"

export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-30000}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-3000}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_INTRA_PCIE_ENABLE="${HCCL_INTRA_PCIE_ENABLE:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1

DATA_ROOT="${DATA_ROOT:-/afs-a3-241ceshi-shared/geruijun}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${DATA_ROOT}/Qwen3-32B}"
VOCAB_FILE="${VOCAB_FILE:-${DATA_ROOT}/Qwen3-32B/vocab.json}"
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/dataset/data_text_document}"

RUN_DIR="${RUN_DIR:-.}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_DIR}/tb}"
CKPT_SAVE_DIR="${CKPT_SAVE_DIR:-${RUN_DIR}/ckpt}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/}"
mkdir -p "$TENSORBOARD_DIR" "$CKPT_SAVE_DIR" "$LOG_DIR"

NNODES="${NNODES:-${WORLD_SIZE:-1}}"
WORLD_SIZE="$NNODES"
RANK="${RANK:-${NODE_RANK:-0}}"
NODE_RANK="$RANK"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-24670}"
NPUS_PER_NODE="${NPUS_PER_NODE:-16}"
GPUS_PER_NODE="$NPUS_PER_NODE"
PROC_PER_NODE="$NPUS_PER_NODE"

export WORLD_SIZE NNODES RANK NODE_RANK MASTER_ADDR MASTER_PORT NPUS_PER_NODE GPUS_PER_NODE

# 16 卡/节点：TP*EP*PP=16 → 单节点 DP=1
TP="${TP:-1}"
EP="${EP:-4}"
ETP="${ETP:-1}"
PP="${PP:-4}"
MBS="${MBS:-1}"
GBS="${GBS:-128}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-40960}"
SKIP_SAVE="${SKIP_SAVE:-1}"
SKIP_PROFILE="${SKIP_PROFILE:-1}"
SKIP_TB="${SKIP_TB:-1}"

if [[ -f warmup.sh ]]; then
  bash warmup.sh || true
fi

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-rotary-position-embeddings \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1e-5 \
    --lr-decay-style cosine \
    --min-lr 1.e-6 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction 0.02 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --norm-epsilon 1e-6
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel
"

GPT_ARGS="
    --use-mcore-models \
    --qk-layernorm \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --vocab-file ${VOCAB_FILE} \
    --kv-channels 128 \
    --num-layers 48 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 4 \
    --cross-entropy-loss-fusion \
    --no-persist-layer-norm
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5000 \
    --eval-interval 5000 \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --log-throughput
"
if [[ "${SKIP_TB:-1}" != "1" ]]; then
  OUTPUT_ARGS="${OUTPUT_ARGS} \\
    --tensorboard-dir ${TENSORBOARD_DIR}
"
fi

RECOMPUTE_ARGS="
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1
"

MOE_INTERMEDIATE_SIZE="${MOE_INTERMEDIATE_SIZE:-768}"
ROUTER_TOPK="${ROUTER_TOPK:-8}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
moe_options="
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --expert-tensor-parallel-size ${ETP} \
    --expert-model-parallel-size ${EP} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.001
"

EXTRA_ARGS=(
  --distributed-timeout-minutes 60
  --ckpt-format torch
  --distributed-backend nccl
)
if [[ "$SKIP_SAVE" != "1" ]]; then
  EXTRA_ARGS+=(--save "${CKPT_SAVE_DIR}")
fi
if [[ "$SKIP_PROFILE" != "1" ]]; then
  EXTRA_ARGS+=(
    --profile
    --use-pytorch-profiler
    --profile-step-start 5
    --profile-step-end 7
    --profile-ranks 0
  )
fi

LOG_FILE="${LOG_DIR}/train_mcore_qwen3_30b_a3b_rank${RANK}.log"
echo "[wrapper moe] NNODES=${NNODES} RANK=${RANK} NPUS=${NPUS_PER_NODE} MASTER=${MASTER_ADDR}:${MASTER_PORT} ITERS=${TRAIN_ITERS} EP=${EP}"
echo "[wrapper moe] log → ${LOG_FILE}"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    $moe_options \
    $RECOMPUTE_ARGS \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"
rc=${PIPESTATUS[0]}
echo "TRAIN_MOE_RANK_${RANK}_DONE rc=${rc}"
exit "${rc}"
