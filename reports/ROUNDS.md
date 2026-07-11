# 轮次产物约定（128 卡长期迭代）

## 目录

| 位置 | 用途 |
|------|------|
| `reports/research/` | 调研门禁文档（`research_*_rN.md`） |
| `reports/rounds/` | 每轮 Grok 报告（`card_screen_diff_rN.md` / `hccl_cluster_rN.md` / `mfu_opt_rN.md`） |
| `reports/rounds/*_figs/` | 该轮图（禁止覆盖历史轮） |
| `../../logs/<track>-rN-<ts>/` | 原始 JSONL/日志（时间戳目录，永不覆盖） |
| AFS `/afs-a3-241ceshi-shared/montyyin/results/<track>-rN-<ts>/` | 集群侧结果 |

## 命名

- Track A：`card-diff-rN-<ts>` / `research_card_diff_rN.md` / `card_screen_diff_rN.md`
- Track B：`hccl-cluster-rN-<ts>` / `research_nccl_verify_rN.md` / `hccl_cluster_rN.md`
- Track C：`mfu-unblock-r0` → `mfu-opt-rN-<ts>` / `mfu_unblock_r0.md` / `mfu_opt_rN.md`

## 门禁

调研文档未过父 agent 审查前，不得大规模 128 重跑。每轮结束后必须有二次调研笔记（可附在报告末节）。
Subagent 统一使用 Grok 4.5（`grok-4.5-fast-xhigh`）做调研、写代码与测试。

## 体质可视化（Track A）

方案：`reports/research/research_constitution_viz_r0.md`  
脚本：`reports/plot_card_constitution.py`（`gen_card_constitution_report.py` 为兼容入口）

### 仅旧字段冒烟（已有 `logs/card-screen-128-*`）

旧 JSONL 通常只有 `func_tflops` / `hbm_gbps` / `sustained_tflops`（及可选 shape）。缺 vector/mte/power 时脚本会跳过并在 md 注明，不应崩溃。

```bash
cd /Users/yinjinrun/random-thing/project/lab-workspace

# 对已有 128 卡 screen 结果出一版分布报告
python3 reports/plot_card_constitution.py \
  --data-dir /Users/yinjinrun/random-thing/logs/card-screen-128-20260710_224218/results \
  --out-dir reports/rounds

# 产物：
#   reports/rounds/card_constitution_<stamp>.md
#   reports/rounds/card_constitution_<stamp>_figs/
#     hist_*.png / heatmap_relmed_*.png / box_by_host_*.png / sorted_bar_*.png
```

期望：旧三指标的 hist / heatmap / box_by_host / sorted_bar 均能出；散点（Cube×Vector 等）在跳过说明中列出。
