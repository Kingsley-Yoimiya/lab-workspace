# MFU 优化环 R1b：TP1PP1 扩 32/64

> 作者：Cursor Grok 4.5  
> 基线 16 卡 TP1PP1 MBS=1：~170.6 TFLOP/s/GPU，MFU≈**58.3%**  
> 日志：`logs/mfu-loop-r6-r1b_scale-20260711_004532`

## 假设

纯 DP（TP=1 PP=1）在 32/64 卡上仍能接近 16 卡 MFU。

## 证据

| scale | 稳态 TFLOP/s/GPU（估） | MFU% | 备注 |
|------:|----------------------:|-----:|------|
| 16（R0） | ~170.6 | **58.3** | 对照 |
| 32 | ~**146.0**（median 159.7） | **~49.9** | 成功；相对 16 卡弱扩展效率 ≈ 49.9/58.3 ≈ **85%** |
| 64 | — | — | **FAIL**：master `banner exchange…port 65535`；worker ssh Broken pipe（并发 vcctl 打挂跳板） |

**结论：** 32 卡假设部分成立——能跑，但 per-GPU MFU 从 58% 掉到 ~50%。64 卡本次为**启动基础设施失败**，非算法结论；已加 rank 启动错开 2s，待重试。

## 二次假设

1. 错开启动后重试 **64→128**  
2. **GBS=256** @16/32 看能否抬回 MFU  
3. 扫 TP/PP；Probing 量化跨节点 allreduce
