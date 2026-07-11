# MFU 优化环 R17：统一 GBS=2048 矩阵

> 日志：`logs/mfu-loop-r28-r17_best_matrix-20260711_023109`

| scale | 结果 |
|------:|------|
| 16 | FAIL（跳板 ssh） |
| 32 | FAIL（跳板 ssh） |
| 64 | **~54.4%**（复现 R14） |
| 128 | FAIL |

已在 scale 间加 15s 歇息；R29 正在重跑同矩阵。历史最佳仍以分 scale 成功轮为准。
