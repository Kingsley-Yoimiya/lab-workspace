#!/usr/bin/env bash
# 本机 durable 启动：MoE 弱扩展 32/64/96（Phase1 无 Probing）
set +eu
cd /Users/yinjinrun/random-thing/project/lab-workspace
source scripts/cluster/huawei.env
export CLUSTER_JOB=montyyin-moe-scale-96
export CLUSTER_JOB_OVERRIDE=montyyin-moe-scale-96
export CLUSTER_POD=montyyin-moe-scale-96-master-0
export CLUSTER_FORCE_JUMP=1
export CLUSTER_KUBECONFIG=/root/.kube/config.huawei-a3-241ceshi
export CLUSTER_SSH_HOST=ais-cf3e61a5
export MODE=moe SOURCE=wrapper
export TP=1 PP=4 EP=4 ETP=1
export GBS="${GBS:-1920}" MBS=1 SEQ_LENGTH=4096 SKIP_TB=1
export TRAIN_ITERS="${TRAIN_ITERS:-8}"
# 默认只补跑 96；全矩阵用 SCALES=32+64,96
export SCALES="${SCALES:-96}"
export PROBING="${PROBING:-0}"
export SCALE_TIMEOUT_SEC="${SCALE_TIMEOUT_SEC:-3600}"
export MASTER_ADDR=montyyin-moe-scale-96-master-0.montyyin-moe-scale-96
export MASTER_PORT="${MASTER_PORT:-25000}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export LOG_DIR="${LOG_DIR:-/Users/yinjinrun/random-thing/logs/moe-phase1-${STAMP}}"
export RUN_ROOT="${RUN_ROOT:-/afs-a3-241ceshi-shared/montyyin/results/mfu_moe_scale/${STAMP}}"
mkdir -p "$LOG_DIR"
echo "$LOG_DIR" > /tmp/moe_phase1_logdir.txt
echo "START $(date -Iseconds) LOG=$LOG_DIR RUN_ROOT=$RUN_ROOT SCALES=$SCALES PROBING=$PROBING"
bash scripts/cluster/run_train_mfu_scale.sh
echo "PHASE1_SCRIPT_EXIT=$? $(date -Iseconds)"
