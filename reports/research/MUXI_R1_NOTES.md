# MUXI R1 调研笔记 · 2026-07-12

## 决策

| 项 | 结论 |
|----|------|
| 跳板 | **ais-cf3e61a5**（weibozhen TCP 黑洞）；见 `docs/AIS_JUMP_CLUSTER.md` |
| 编排移植 | fire/poll 模式 + hosts_full/clean；campaign 复用华为 queue 逻辑，peak=279.9，DEVICES=8 |
| Dense | `train_gpt_dense_muxi.sh`：20L/H4096，GBS=2048，SEQ=4096，local/unfused，`--log-throughput` |
| MoE | `train_gpt_moe_muxi.sh`：缩小版 experts=8 topk=2 EP=8；与 Dense **分 ledger** |
| IB | 默认 `*_IB_HCA=xscale`（verbs）；SOCKET 仍 eth0 |
| Peak | 279.9 |

## Dense 队列（初稿）

拓扑 × scale × hostset(clean 优先，full 对照)：
- TP4PP2 @ 8/16/32/64/104(clean)/128(full)
- TP4PP1, TP2PP2, TP8PP1 @ 主要 scale

## MoE 队列（初稿）

- 冒烟：TP1PP1EP8 @8 clean
- 扩展：TP1PP2EP4 @8/16/32
- EP 扫：world=8，EP∈{1,2,4,8}
