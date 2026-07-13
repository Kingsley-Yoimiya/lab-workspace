#!/usr/bin/env bash
set -uo pipefail
export TP=4 PP=2 EP=1 ETP=1 MBS=1 GBS=320 SEQ_LENGTH=4096
export SKIP_TB=1 SKIP_SAVE=1 SKIP_PROFILE=1 TRAIN_ITERS=30
export PROBING=0 FAILSLOW_STEP_LOG=1
export DELAY_INJECT=1 DELAY_STAGE=1 DELAY_MS=1500 DELAY_EVERY=3 DELAY_BURST=1
export DELAY_RANKS='12,13,14,15'
export PP_SIZE=2 WORLD_SIZE_NPUS=16
export PATH=/root/miniconda3/envs/llm_test/bin:$PATH
export PYTHONPATH=/afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster/hooks:/MindSpeed-LLM/MindSpeed:\ foresPYTHONPATH:-}
export WORLD_SIZE=1 NNODES=1 RANK=0 NODE_RANK=0
export MASTER_ADDR=montyyin-moe96-r2-worker-1.montyyin-moe96-r2 MASTER_PORT=26410
export NPUS_PER_NODE=16 GPUS_PER_NODE=16
export DATA_ROOT=/afs-a3-241ceshi-shared/geruijun
export RUN_DIR=/afs-a3-241ceshi-shared/montyyin/results/exp45_parallel/20260713_135142_p1/exp3_rank_inject LOG_DIR=/afs-a3-241ceshi-shared/montyyin/results/exp45_parallel/20260713_135142_p1/exp3_rank_inject/
export TENSORBOARD_DIR=/afs-a3-241ceshi-shared/montyyin/results/exp45_parallel/20260713_135142_p1/exp3_rank_inject/tb CKPT_SAVE_DIR=/afs-a3-241ceshi-shared/montyyin/results/exp45_parallel/20260713_135142_p1/exp3_rank_inject/ckpt
export HCCL_IF_BASE_PORT=28410
mkdir -p \"$RUN_DIR\" \"$LOG_DIR\" \"$TENSORBOARD_DIR\" \"$CKPT_SAVE_DIR\"
SP=$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)
if [[ -n \"$SP\" && -d \"$SP\" ]]; then
  printf '%s\nimport failslow_step_timer\n' \"/afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster/hooks\" > \"$SP/zz_failslow_step.pth\"
fi
cd /afs-a3-241ceshi-shared/geruijun/Megatron-LM-0.12.3
bash /afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster/wrappers/train_qwen3_8B_ascend.sh 2>&1 | tee /afs-a3-241ceshi-shared/montyyin/results/exp45_parallel/20260713_135142_p1/exp3_rank_inject/rank0.log
rc=${PIPESTATUS[0]}
echo TRAIN_RANK_0_DONE rc=$rc | tee -a /afs-a3-241ceshi-shared/montyyin/results/exp45_parallel/20260713_135142_p1/exp3_rank_inject/rank0.log
exit $rc
