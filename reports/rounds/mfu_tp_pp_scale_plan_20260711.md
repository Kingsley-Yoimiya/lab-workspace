# 华为 128 卡：固定 TP×PP 扩 DP 的 MFU 弱扩展战役

> 日期：2026-07-11  
> 目标：测「有意义并行拓扑」下 MFU 随卡数变化，**不**再刷 TP1PP1 单点最优  
> Job：`montyyin-mfu-scale`（自 `huawei-8node-copy` clone）  
> Peak：292.79 TFLOPS/卡  
> 执行器：`scripts/cluster/run_mfu_tp_pp_scale_campaign.sh`

## 设计原则

1. **固定** `(TP, PP)`（Dense）或 `(TP, PP, EP)`（MoE），模型参数不变  
2. **只扩 DP**：`DP = world / (TP × PP × CP)`，`world ∈ {16,32,64,128}`  
3. **固定 GBS=2048**（与 R29 弱扩展同控制变量，便于对比通信摊销；必要时再开「GBS∝DP」对照）  
4. MBS=1，SEQ=4096，ITERS=5（稳态取 drop-first=1）  
5. 禁止 TP=1 PP=1 进本战役主矩阵（可作对照一行，不作为主结论）

## Dense 矩阵（Qwen3-8B wrapper：20L / H4096 / heads=32 / GQA groups=8）

约束：`TP | 32` 且 `TP | 8`（GQA）→ TP∈{1,2,4,8}；`PP | 20` → PP∈{1,2,4,5,10,20}。

| 优先级 | TP | PP | 可测 world | DP 序列 | 说明 |
|-------:|---:|---:|------------|------|------|
| P0 | 8 | 1 | 16/32/64/128 | 2/4/8/16 | 用户最关心 |
| P0 | 8 | 2 | 16/32/64/128 | 1/2/4/8 | TP8 + 有 PP |
| P1 | 4 | 2 | 16/32/64/128 | 2/4/8/16 | 中等 TP |
| P1 | 4 | 4 | 16/32/64/128 | 1/2/4/8 | 节点内较满 |
| P1 | 2 | 4 | 16/32/64/128 | 2/4/8/16 | 偏 PP |
| P2 | 8 | 4 | 32/64/128 | 1/2/4 | 无 16（model_par=32） |
| 对照 | 2 | 2 | 16/32/64/128 | 4/8/16/32 | 旧基线扩展 |

每条拓扑先跑 **16 → 若 OK 再 32/64/128**，单 scale 预计 8–20 min（含 torchrun 拉起）。

## Sparse / MoE 矩阵（30B-A3B：48L / experts=128）

约束：优先节点内可填满 `TP×PP×EP` 因子；EP | 128。

| 优先级 | TP | PP | EP | 可测 world | 说明 |
|-------:|---:|---:|---:|------------|------|
| P0 | 1 | 4 | 4 | 16/32/64/128 | wrapper 默认族，扩 DP |
| P1 | 2 | 2 | 4 | 16/32/64/128 | 带一点 TP |
| P1 | 1 | 2 | 8 | 16/32/64/128 | 更大 EP |
| P2 | 1 | 4 | 8 | 32/64/128 | model 并行因子更大 |

MoE 绝对 MFU 不与 Dense 直接比；只看同拓扑弱扩展斜率。

## GBS 对齐规则

`unit = MBS × DP`，`GBS` 向上取整到 `unit` 倍数。本战役目标 GBS=2048，在 DP≤2048 时均可整除。

## 产出

- 账本：`reports/rounds/mfu_tp_pp_scale_ledger.md`  
- 状态：`reports/rounds/mfu_tp_pp_scale_state/`  
- 本地日志：`logs/mfu-tp-pp-scale-<stamp>/`  
- AFS：`/afs-a3-241ceshi-shared/montyyin/logs/train-tp-pp-*`  
- 汇总表/曲线：战役结束后 `mfu_tp_pp_scale_summary_*.md` + SVG（plot_style）

## 调度现实

- 克隆时集群上已有 `whj4stu-copy-copy-copy`（Running, 8 replicas）占满 128 卡  
- `montyyin-mfu-scale` 可能 Pending，等资源释放后自动起 pod  
- 旧优化环 `mfu_loop_state/PAUSE` 保持暂停，互不抢队列
