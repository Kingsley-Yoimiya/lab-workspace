# HCCL 通信战役报告 · 20260711

> 生成时间：2026-07-11 14:00  
> 数据源：`/Users/yinjinrun/random-thing/logs/pipeline-comm-20260711_134811`

## 摘要

本报告汇总 All-Reduce / All-Gather / Reduce-Scatter / Broadcast 四算子在 world=16/32/64/128 下的 Bus 带宽曲线，256 MB 大消息的扩展性阶梯图与保持率，Rank 级分布，以及 P2P 边级带宽对比。

### 256 MB 保持率（相对 world=16）

| 算子 | w=16 | w=32 | w=64 | w=128 |
|------|------|------|------|-------|
| All-Reduce | 100.0% | 96.8% | 94.9% | 89.4% |
| All-Gather | 100.0% | 88.0% | 64.2% | 54.0% |
| Reduce-Scatter | 100.0% | 91.8% | 71.0% | 46.4% |
| Broadcast | 100.0% | 91.4% | 86.8% | 86.8% |

### 256 MB 平均 Bus 带宽 (GB/s)

| 算子 | w=16 | w=32 | w=64 | w=128 |
|------|------|------|------|-------|
| All-Reduce | 154.80 | 149.79 | 146.97 | 138.44 |
| All-Gather | 119.28 | 105.03 | 76.59 | 64.42 |
| Reduce-Scatter | 110.46 | 101.43 | 78.41 | 51.26 |
| Broadcast | 92.41 | 84.49 | 80.21 | 80.18 |

- HCCL 记录数：1024
- P2P 去重边数：122
- 拓扑：已解析 `master-0.raw.txt`

## 1. Collective · Bus 带宽 vs 消息大小

### All-Reduce

![All-Reduce](hccl_campaign_20260711_figs/hccl_bus_bw_vs_size_all_reduce.png)

### All-Gather

![All-Gather](hccl_campaign_20260711_figs/hccl_bus_bw_vs_size_all_gather.png)

### Reduce-Scatter

![Reduce-Scatter](hccl_campaign_20260711_figs/hccl_bus_bw_vs_size_reduce_scatter.png)

### Broadcast

![Broadcast](hccl_campaign_20260711_figs/hccl_bus_bw_vs_size_broadcast.png)

## 2. 256 MB 大消息扩展性

![阶梯图](hccl_campaign_20260711_figs/hccl_256mb_step_bus_bw.png)

![分算子阶梯](hccl_campaign_20260711_figs/hccl_256mb_step_per_op.png)

![保持率](hccl_campaign_20260711_figs/hccl_256mb_retention_bar.png)

## 3. Rank 分布（256 MB）

### All-Reduce

![violin](hccl_campaign_20260711_figs/hccl_rank_violin_256mb_all_reduce.png)

### All-Gather

![violin](hccl_campaign_20260711_figs/hccl_rank_violin_256mb_all_gather.png)

### Reduce-Scatter

![violin](hccl_campaign_20260711_figs/hccl_rank_violin_256mb_reduce_scatter.png)

### Broadcast

![violin](hccl_campaign_20260711_figs/hccl_rank_violin_256mb_broadcast.png)

![箱线图汇总](hccl_campaign_20260711_figs/hccl_rank_box_256mb_all_ops.png)

### world=16

![hist w16](hccl_campaign_20260711_figs/hccl_rank_hist_w16_256mb.png)

### world=32

![hist w32](hccl_campaign_20260711_figs/hccl_rank_hist_w32_256mb.png)

### world=64

![hist w64](hccl_campaign_20260711_figs/hccl_rank_hist_w64_256mb.png)

### world=128

![hist w128](hccl_campaign_20260711_figs/hccl_rank_hist_w128_256mb.png)

## 4. P2P

![p2p_bw_violin_by_kind_size.png](hccl_campaign_20260711_figs/p2p_bw_violin_by_kind_size.png)

![p2p_box_compare_w16_w128_65536.png](hccl_campaign_20260711_figs/p2p_box_compare_w16_w128_65536.png)

![p2p_box_compare_w16_w128_16777216.png](hccl_campaign_20260711_figs/p2p_box_compare_w16_w128_16777216.png)

![p2p_slow_edges_top15_16mb.png](hccl_campaign_20260711_figs/p2p_slow_edges_top15_16mb.png)

![p2p_fast_edges_top15_16mb.png](hccl_campaign_20260711_figs/p2p_fast_edges_top15_16mb.png)

![p2p_kind_mean_compare_16mb.png](hccl_campaign_20260711_figs/p2p_kind_mean_compare_16mb.png)

## 5. 机内拓扑

![HCCS](hccl_campaign_20260711_figs/topo_hccs_heatmap_master0.png)

## 附录 · 图文件清单

- `hccl_256mb_retention_bar.png`
- `hccl_256mb_step_bus_bw.png`
- `hccl_256mb_step_per_op.png`
- `hccl_bus_bw_vs_size_all_gather.png`
- `hccl_bus_bw_vs_size_all_reduce.png`
- `hccl_bus_bw_vs_size_broadcast.png`
- `hccl_bus_bw_vs_size_reduce_scatter.png`
- `hccl_rank_box_256mb_all_ops.png`
- `hccl_rank_hist_w128_256mb.png`
- `hccl_rank_hist_w16_256mb.png`
- `hccl_rank_hist_w32_256mb.png`
- `hccl_rank_hist_w64_256mb.png`
- `hccl_rank_violin_256mb_all_gather.png`
- `hccl_rank_violin_256mb_all_reduce.png`
- `hccl_rank_violin_256mb_broadcast.png`
- `hccl_rank_violin_256mb_reduce_scatter.png`
- `p2p_box_compare_w16_w128_16777216.png`
- `p2p_box_compare_w16_w128_65536.png`
- `p2p_bw_violin_by_kind_size.png`
- `p2p_fast_edges_top15_16mb.png`
- `p2p_kind_mean_compare_16mb.png`
- `p2p_slow_edges_top15_16mb.png`
- `topo_hccs_heatmap_master0.png`
