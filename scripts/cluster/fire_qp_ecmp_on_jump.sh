#!/usr/bin/env bash
# fire_qp_ecmp_on_jump.sh — Mac 侧启动器：把 QP/ECMP 战役搬到跳板上跑。
#
# 为什么: 跳板经多层反向隧道，Mac 高频 ssh(scan/upload/fire/poll/collect ×多臂)
# 会把 sshd 挤爆(Connection closed 255)。解法——driver 整个搬上跳板本地跑，
# vcctl 全部本地调用；Mac 只做 3 件事: scp 代码上去、nohup 起、回拉结果。
# 全程 Mac↔跳板 ssh 连接数是个位数。
#
# 用法:
#   bash fire_qp_ecmp_on_jump.sh                       # 起全战役(4阶段)
#   PHASES="1 2" bash fire_qp_ecmp_on_jump.sh          # 只跑 256/512 AllReduce
#   ACTION=pull CAMPAIGN=<名> bash fire_qp_ecmp_on_jump.sh   # 回拉结果
#   ACTION=status CAMPAIGN=<名> bash fire_qp_ecmp_on_jump.sh # 看进度
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

JUMP="${CLUSTER_SSH_HOST:-ais-cf3e61a5}"
JUMP_BASE="${JUMP_BASE:-/root/qp-ecmp}"          # 跳板上代码 + 结果根
CM="/tmp/qp-ecmp-onjump-cm-%r-%h.sock"           # Mac↔跳板单条复用连接
ACTION="${ACTION:-run}"
STAMP="$(date +%Y%m%d_%H%M%S)"
CAMPAIGN="${CAMPAIGN:-qp-ecmp-campaign-$STAMP}"
LOCAL_PULL="${LOCAL_PULL:-$REPO_ROOT/results/muxi-h3c}"

FILES=(
  fire_qp_ecmp_muxi.sh run_qp_ecmp_campaign.sh jump_stage_lib.sh
  job_helpers.sh muxi.env afs_guard.sh aggregate_qp_ecmp.py
  nccl_torch_bench.py nccl_torch_bench_metrics.py nccl_p2p_bench.py
)

jssh() { ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath="$CM" -o ControlPersist=120 "$JUMP" "$@"; }

case "$ACTION" in
  run)
    echo "=== 部署 QP-ECMP 战役到跳板 $JUMP:$JUMP_BASE/$CAMPAIGN ==="
    CODE_DIR="$JUMP_BASE/code"
    jssh "mkdir -p '$CODE_DIR' '$JUMP_BASE/results'"
    echo "--- scp ${#FILES[@]} 个文件上跳板 ---"
    ( cd "$SCRIPT_DIR" && tar -cf - "${FILES[@]}" ) \
      | jssh "tar -C '$CODE_DIR' -xf - && echo UNPACKED"
    # 起战役：ON_JUMP=1，结果落 $JUMP_BASE/results/$CAMPAIGN，nohup 脱离
    run_env="ON_JUMP=1 CAMPAIGN='$CAMPAIGN' CAMPAIGN_ROOT='$JUMP_BASE/results' PHASES='${PHASES:-1 2 3 4}' REP_AR256='${REP_AR256:-3}' REP_AR512='${REP_AR512:-5}' REP_IC='${REP_IC:-2}' INCLUDE_INCAST='${INCLUDE_INCAST:-1}'"
    RUN_LOG="$JUMP_BASE/results/$CAMPAIGN.nohup.log"
    jssh "mkdir -p '$JUMP_BASE/results'; cd '$CODE_DIR' && setsid nohup env $run_env bash run_qp_ecmp_campaign.sh >'$RUN_LOG' 2>&1 & echo STARTED_PID=\$!"
    echo
    echo "=== 已在跳板后台启动。CAMPAIGN=$CAMPAIGN ==="
    echo "看进度:   ACTION=status CAMPAIGN=$CAMPAIGN bash $0"
    echo "回拉结果: ACTION=pull   CAMPAIGN=$CAMPAIGN bash $0"
    echo "跳板日志: $JUMP:$RUN_LOG"
    ;;

  status)
    echo "=== 战役 $CAMPAIGN 跳板进度 ==="
    jssh "tail -40 '$JUMP_BASE/results/$CAMPAIGN.nohup.log' 2>/dev/null; echo; echo '--- 阶段索引 ---'; cat '$JUMP_BASE/results/$CAMPAIGN/phase_index.tsv' 2>/dev/null; echo '--- 是否还在跑 ---'; pgrep -af run_qp_ecmp_campaign | head -3 || echo '(campaign 进程已退出)'"
    ;;

  pull)
    echo "=== 回拉 $CAMPAIGN → $LOCAL_PULL/$CAMPAIGN ==="
    mkdir -p "$LOCAL_PULL/$CAMPAIGN"
    # 跳板无 rsync，用 tar over ssh（本机 tar 解包）。只拉结果目录。
    jssh "tar -C '$JUMP_BASE/results/$CAMPAIGN' -cf - . 2>/dev/null" \
      | tar -C "$LOCAL_PULL/$CAMPAIGN" -xf - \
      && echo "回拉完成: $LOCAL_PULL/$CAMPAIGN"
    jssh "cat '$JUMP_BASE/results/$CAMPAIGN.nohup.log'" > "$LOCAL_PULL/$CAMPAIGN/campaign.nohup.log" 2>/dev/null || true
    echo "--- 各阶段 SUMMARY ---"
    find "$LOCAL_PULL/$CAMPAIGN" -name SUMMARY.md 2>/dev/null | sort
    ;;

  *)
    echo "未知 ACTION=$ACTION (run|status|pull)"; exit 1 ;;
esac
