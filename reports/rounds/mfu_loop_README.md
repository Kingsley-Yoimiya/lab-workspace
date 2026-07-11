# MFU 永不停止优化环

> 启动：2026-07-11 · Cursor Grok 4.5  
> 执行器：`scripts/cluster/loop_mfu_forever.sh` → `loop_mfu_one_round.sh`  
> 账本：`reports/rounds/mfu_loop_ledger.md`  
> Peak：292.79 TFLOPS/卡

## 机制

1. 队列 `mfu_loop_state/queue.jsonl` 驱动假设（可插队写 `next_job.json`）
2. 每轮跑指定 scale（16/32/64/128 子集或全量）+ TP/PP/MBS/GBS/SEQ
3. 解析稳态 TFLOP/s/GPU → 追加账本 → 打 `AGENT_LOOP_TICK_mfu_opt` 唤醒分析
4. 队列耗尽后自动 recycle 变体，**不停止**

## 已排队列（首轮）

| id | 假设 |
|----|------|
| r1_mbs2 | TP1PP1 + MBS=2 @16 |
| r1b_scale | TP1PP1 扩 32/64 |
| r2_gbs256 | GBS=256 |
| r3–r6 | TP/PP 组合扫 |
| r7–r8 | SEQ 2048/8192 |
| r9_best128 | 最佳并行打满 128 |
| r10_moe16 | MoE 冒烟 |

## 基线（R0，环外）

- TP2PP2 @16：~47.7% MFU
- TP1PP1 @16：~58.3% MFU
