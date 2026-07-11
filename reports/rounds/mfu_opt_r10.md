# MFU 优化环 R10：MoE 30B-A3B 冒烟

> 作者：Cursor Grok 4.5  
> 配置：mode=moe @16，请求 EP=8（wrapper 默认可能另有 PP）  
> 日志：`logs/mfu-loop-r16-r10_moe16-20260711_013103`

## 证据

| iter | TFLOP/s/GPU |
|-----:|------------:|
| 1（冷） | 30.1 |
| 2–5 | 48.2 / 43.0 / 40.7 / 40.8 |

稳态约 **41 TFLOP** → 相对 dense peak 292.79 的名义 MFU **~14%**。

**结论：** MoE 冒烟成功；名义 MFU 远低于 dense（专家通信 + FLOP 口径不同，不可直接与 8B dense 比）。首轮队列已走完，进入 recycle；应插队 **TP1PP1 + 更大 GBS @64/128**。
