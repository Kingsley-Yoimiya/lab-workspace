# Ascend 终局同步（kill 前拉取）

- 同步时间：2026-07-13 15:18–15:19 +0800  
- 本地目录：`reports/rounds/ascend_final_sync_20260713_151847/`  
- 主包：`ascend_campaign_20260713/`（CSV / JSON / SVG / CAMPAIGN_FINAL.md）  
- Job：`montyyin-moe96-r2` **已删除**（`vcctl job delete` 成功）

## 写报告/出图优先用这些文件

| 用途 | 文件 |
|---|---|
| Block A 分解主表 | `network_contrib_final.csv`、`gap_indep.csv`、`gap_real_gbsprop.csv`、`gap_real16.csv` |
| Block A 图 | `blockA_gap_decomp.svg` |
| Block C | `heatmap_means.csv`、`gap_real96_blockC.csv`、`blockC_power_aicore_heatmap.svg` |
| Block E | `dual_signal.json`、`exp45_*/exp4_preempt/` |
| Block D | `exp3_delayed_iter_analysis.json`、`pp_inject_ab.json`、`rank_contrast.json` |
| MoE 对照（早先完整 32） | `moe_partial/gap_moe_003312.csv` |
| 叙事底稿 | `CAMPAIGN_FINAL.md` |

## 未完成 / 勿当终局

- MoE 32+64 串行战役：中途停掉，半截勿当主结果  
- Dense fast 短窗：spawn 后随 kill 终止，可忽略  
