# HCCL 集群画像 R0（P2P 抽样）

> 作者：Cursor Grok 4.5  
> 调研：`reports/research/research_nccl_verify_r0.md`  
> 数据：`logs/hccl-cluster-r0-20260710_235358/results`（world=16，ring+star，64K/16M）  
> 图：`reports/rounds/hccl_cluster_r0_figs/`  
> 日期：2026-07-11

## 1. 本轮切片

按调研 P0：最小 P2P（`isend/irecv`）+ payload 校验 + **每 rank 写 JSONL**（打破 rank0-only）。

| 项 | 取值 |
|----|------|
| 策略 | ring 邻接 + star→rank0 |
| sizes | 64KiB（延迟）、16MiB（带宽） |
| world=16 | **成功**：176 条边记录，`ok=true` 全通过 |
| world=128 | **失败**：`SIGSEGV`（exit -11，local_rank=10），ring+star 并发边过多；ring-only 重试进行中 |

## 2. 16 卡画像（差异优先）

已生成：

- `heatmap_host_host_lat.png` / `heatmap_rank_rank_lat.png`
- `heatmap_host_host_bw.png` / `heatmap_rank_rank_bw.png`
- `bar_slow_edges_topk_lat.png` / `bar_slow_edges_topk_bw.png`
- `stats.json`

相对上一轮 `hccl_128.md`（仅 collective × scale 单点带宽），本轮第一次给出 **边级** 延迟/带宽矩阵，可直接看慢边 TopK。

## 3. 二次调研问题

1. 128 卡 P2P SIGSEGV：是否 HCCL P2P 在多节点 + 多并发 pair 下的已知限制？串行化 pair / 仅 ring 是否稳定？
2. `hccn_tool` 仍缺失时，慢边能否与 npu-smi 拓扑交叉验证？
3. P1 保底：per-rank collective 计时热力图是否足以在 P2P 128 修好前支撑「通信 diff」叙事？

## 4. 代码

- `scripts/cluster/hccl_p2p_bench.py`
- `scripts/cluster/run_hccl_p2p_128.sh`
- `reports/gen_hccl_cluster_r0_figs.py`
