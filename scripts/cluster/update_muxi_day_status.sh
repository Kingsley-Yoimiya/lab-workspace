#!/usr/bin/env bash
# 统一状态机：写入 DAY_ROOT/STATUS.md
set -uo pipefail
DAY_ROOT="${DAY_ROOT:-$(cat /tmp/muxi_day_root.txt 2>/dev/null)}"
STAMP="${STAMP:-$(cat /tmp/muxi_day_stamp.txt 2>/dev/null)}"
AFS_OUT="${AFS_OUT:-$(cat /tmp/muxi_day_afs.txt 2>/dev/null)}"
WAVE="${WAVE:-Wave0}"
PHASE="${PHASE:-init}"
NOTE="${NOTE:-}"
STATUS_FILE="${DAY_ROOT}/STATUS.md"
mkdir -p "$DAY_ROOT"

# 可选：从集群抽样
DONE_INFO=""
if [[ "${PROBE_CLUSTER:-0}" == "1" ]]; then
  export CLUSTER_FORCE_JUMP=1
  # shellcheck source=/dev/null
  source /Users/yinjinrun/random-thing/project/lab-workspace/scripts/cluster/muxi.env
  # shellcheck source=/dev/null
  source /Users/yinjinrun/random-thing/project/lab-workspace/scripts/cluster/job_helpers.sh
  MASTER="${CLUSTER_JOB}-master-0"
  DONE_INFO=$(cluster_pod_exec "$MASTER" "
    printf 'a1_master_real=%s a2_done=%s a4_samples=%s b1_done=%s c1_pairs=%s\n' \
      \"\$(ls ${AFS_OUT}/A1_master/exp0_real/done_rank*.txt 2>/dev/null|wc -l)\" \
      \"\$(ls ${AFS_OUT}/A2/exp1_independent/done_rank*.txt 2>/dev/null|wc -l)\" \
      \"\$(ls ${AFS_OUT}/A4/telemetry/*.jsonl 2>/dev/null|wc -l)\" \
      \"\$(test -f ${AFS_OUT}/B1/DONE && echo 1 || echo 0)\" \
      \"\$(test -f ${AFS_OUT}/C1/pair_matrix.json && echo 1 || echo 0)\"
  " 2>/dev/null || echo "probe_fail")
fi

cat > "$STATUS_FILE" <<EOF
# MUXI Phase0 一天战役状态

- 更新时间: $(date '+%Y-%m-%d %H:%M:%S %Z')
- Stamp: \`${STAMP}\`
- AFS: \`${AFS_OUT}\`
- 本地: \`${DAY_ROOT}\`
- 当前 Wave: **${WAVE}**
- 当前 Phase: **${PHASE}**
- 备注: ${NOTE}

## 集群抽样
\`\`\`
${DONE_INFO:-n/a}
\`\`\`

## 清单进度

| ID | 状态 | 说明 |
|----|------|------|
| A1 | ${A1_STATUS:-pending} | 三节点 Exp0×3000 |
| A2 | ${A2_STATUS:-pending} | 128 卡 independent×3000 |
| A3 | ${A3_STATUS:-pending} | 离线加深解析 |
| A4 | ${A4_STATUS:-pending} | 功耗频率热力图 |
| B1 | ${B1_STATUS:-pending} | PP 切片掩蔽 |
| B2 | ${B2_STATUS:-pending} | 外部抢占 |
| C1 | ${C1_STATUS:-pending} | 16×16 连通性 |
| D1 | ${D1_STATUS:-pending} | 综合报告 |

## 最近日志尾
\`\`\`
$(tail -20 "${DAY_ROOT}/orchestrator.log" 2>/dev/null || echo "(no orch log yet)")
\`\`\`
EOF
echo "WROTE $STATUS_FILE"
