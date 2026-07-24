#!/usr/bin/env bash
# run_qp_ecmp_campaign.sh — QP/ECMP 全战役串行编排
#
# 关键: 四阶段必须【串行】——实验测的就是 fabric 带宽，两个 run 同时压同一批
# 上行链路会互相拥塞污染数据。每阶段跑完（含自身 cleanup）再起下一阶段。
#
# 阶段:
#   1. 256 卡 AllReduce  QP{default,1,2,4,8,16,32} × REP_AR256   核心曲线
#   2. 512 卡 AllReduce  QP{1,4,16}                × REP_AR512   塌陷最重
#   3. 256 卡 incast     QP{1,4,16}                × REP_IC      交叉验证
#   4. 512 卡 incast     QP{1,4,16}                × REP_IC
#
# 用法: bash run_qp_ecmp_campaign.sh
#   PHASES="1 2 3 4"（默认全跑；可只跑子集，如 PHASES="1 2"）
#   INCLUDE_INCAST=1（默认；置 0 跳过 3/4，等价 PHASES="1 2"）
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRE="$SCRIPT_DIR/fire_qp_ecmp_muxi.sh"
ON_JUMP="${ON_JUMP:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
CAMPAIGN="${CAMPAIGN:-qp-ecmp-campaign-$STAMP}"
# ON_JUMP: 结果落跳板本地 /tmp（Mac 侧启动器回拉）；否则落 myportal/results。
if [[ "$ON_JUMP" == "1" ]]; then
  CROOT="${CAMPAIGN_ROOT:-/tmp/qp-ecmp-results}"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
  CROOT="${CAMPAIGN_ROOT:-$REPO_ROOT/results/muxi-h3c}"
fi
CDIR="$CROOT/$CAMPAIGN"
mkdir -p "$CDIR"
exec > >(tee -a "$CDIR/campaign.log") 2>&1

REP_AR256="${REP_AR256:-3}"
REP_AR512="${REP_AR512:-5}"
REP_IC="${REP_IC:-2}"
INCLUDE_INCAST="${INCLUDE_INCAST:-1}"
PHASES="${PHASES:-1 2 3 4}"

echo "=== QP-ECMP CAMPAIGN $CAMPAIGN (ON_JUMP=$ON_JUMP) ==="
echo "phases=$PHASES include_incast=$INCLUDE_INCAST reps: ar256=$REP_AR256 ar512=$REP_AR512 ic=$REP_IC"
echo "结果根: $CDIR"
echo "开始: $(date -Iseconds)"

# 单阶段执行；透传结果 RUN_ID 到 campaign 索引
run_phase() {
  local name="$1" nnodes="$2" mode="$3" arms="$4" reps="$5" sizes="$6" stall="$7"
  local run_id="qp-ecmp-${name}-$(date +%Y%m%d_%H%M%S)"
  echo
  echo "########## 阶段 $name: nnodes=$nnodes mode=$mode arms=$arms reps=$reps ##########"
  echo "$(date -Iseconds) START $name run_id=$run_id"
  local t0; t0="$(date +%s)"
  # 512 卡 init 慢，放宽 stall + 硬超时；256 用默认。ON_JUMP + LOCAL_ROOT 透传，
  # 让每阶段结果都落在 campaign 目录下，便于一次性回拉。
  NNODES="$nnodes" MODE="$mode" QP_ARMS="$arms" REPEATS="$reps" \
    SIZES="$sizes" INCAST_SIZES="${INCAST_SIZES:-16M}" \
    MCCL_DEBUG="${PHASE_DEBUG:-WARN}" ITERS="${ITERS:-20}" WARMUP="${WARMUP:-5}" \
    ARM_TIMEOUT="${8:-1800}" ARM_STALL="$stall" \
    ON_JUMP="$ON_JUMP" LOCAL_ROOT="$CDIR/$run_id" \
    RUN_ID="$run_id" \
    bash "$FIRE"
  local rc=$? el=$(( $(date +%s)-t0 ))
  echo "$(date -Iseconds) END $name rc=$rc elapsed=${el}s run_id=$run_id"
  printf '%s\t%s\t%s\trc=%s\telapsed=%ss\n' "$name" "$run_id" "$mode" "$rc" "$el" \
    >>"$CDIR/phase_index.tsv"
  # 每阶段留一次呼吸，确保上一 run 的 pod 进程/端口彻底释放，fabric 归零
  sleep 20
  return 0
}

for ph in $PHASES; do
  case "$ph" in
    1) run_phase "ar256" 32 all_reduce "default,1,2,4,8,16,32" "$REP_AR256" 256M 300 1800 ;;
    2) run_phase "ar512" 64 all_reduce "1,4,16"                "$REP_AR512" 256M 600 2700 ;;
    3) [[ "$INCLUDE_INCAST" == "1" ]] && run_phase "ic256" 32 incast "1,4,16" "$REP_IC" 16M 300 1800 ;;
    4) [[ "$INCLUDE_INCAST" == "1" ]] && run_phase "ic512" 64 incast "1,4,16" "$REP_IC" 16M 600 2700 ;;
    *) echo "跳过未知阶段 $ph" ;;
  esac
done

echo
echo "=== CAMPAIGN 各阶段 run_id ==="
cat "$CDIR/phase_index.tsv" 2>/dev/null
echo
echo "=== CAMPAIGN DONE $(date -Iseconds) ==="
echo "各阶段 SUMMARY 在各自 results/muxi-h3c/<run_id>/SUMMARY.md"
echo "索引: $CDIR/phase_index.tsv"
