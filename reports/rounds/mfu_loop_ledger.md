# MFU 永不停止优化环 · 账本

> peak = 292.79 TFLOPS/卡（card_screen medians.func_tflops）  
> 基线 R0：TP2PP2 ≈47.7% · TP1PP1 ≈58.3%（16 卡）

| round | id | scale | TP/PP/MBS/GBS/SEQ | steady TFLOP/s/GPU | MFU% | status | log | note |
|------:|----|------:|-------------------|-------------------:|-----:|--------|-----|------|
| 1 | `r1_mbs2` | 16 | 1/1/2/128/4096 | - | - | nometrics | `/Users/yinjinrun/random-thing/logs/mfu-loop-r1-r1_mbs2-20260711_004019` | R1: TP1PP1 上抬 MBS=2 |
| 2 | `r1_mbs2` | 16 | 1/1/2/128/4096 | - | - | nometrics | `/Users/yinjinrun/random-thing/logs/mfu-loop-r2-r1_mbs2-20260711_004108` | R1 retry: TP1PP1 MBS=2 |
| 3 | `r1b_scale` | 32 | 1/1/1/128/4096 | - | - | nometrics | `/Users/yinjinrun/random-thing/logs/mfu-loop-r3-r1b_scale-20260711_004153` | R1b: 最佳并行扩 32/64 |
| 3 | `r1b_scale` | 64 | 1/1/1/128/4096 | - | - | nometrics | `/Users/yinjinrun/random-thing/logs/mfu-loop-r3-r1b_scale-20260711_004153` | R1b: 最佳并行扩 32/64 |
| 4 | `r1_mbs2` | 16 | 1/1/2/128/4096 | - | - | nometrics | `/Users/yinjinrun/random-thing/logs/mfu-loop-r4-r1_mbs2-20260711_004232` | R1: TP1PP1 MBS=2 (launcher fix) |
| 5 | `r1_mbs2` | 16 | 1/1/2/128/4096 | - | - | nometrics | `/Users/yinjinrun/random-thing/logs/mfu-loop-r5-r1_mbs2-20260711_004343` | R1: TP1PP1 MBS=2 (pyboot+no set_env) |
| 6 | `r1b_scale` | 32 | 1/1/1/128/4096 | 138.28 | 47.23 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r6-r1b_scale-20260711_004532` | R1b: TP1PP1 MBS=1 扩 32/64（跳过 OOM 的 MBS=2） |
| 6 | `r1b_scale` | 64 | 1/1/1/128/4096 | 138.28 | 47.23 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r6-r1b_scale-20260711_004532` | R1b: TP1PP1 MBS=1 扩 32/64（跳过 OOM 的 MBS=2） |
| 6 | `r1b_scale` | 64 | 1/1/1/128/4096 | - | - | FAIL_ssh | `logs/mfu-loop-r6-r1b_scale-20260711_004532` | 并发vcctl/ssh挂；指标勿信上一行复用 |
| 7 | `r1c_64` | 64 | 1/1/1/128/4096 | 104.94 | 35.84 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r7-r1c_64-20260711_005912` | R1c: 64卡重试（rank启动错开） |
| 8 | `r2_gbs256` | 16 | 1/1/1/256/4096 | 163.43 | 55.82 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r8-r2_gbs256-20260711_010135` | R2: GBS=256 |
| 8 | `r2_gbs256` | 32 | 1/1/1/256/4096 | 153.61 | 52.46 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r8-r2_gbs256-20260711_010135` | R2: GBS=256 |
| 9 | `r3_tp2pp1` | 16 | 2/1/1/128/4096 | 143.99 | 49.18 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r9-r3_tp2pp1-20260711_010614` | R3: TP=2 PP=1 |
| 9 | `r3_tp2pp1` | 32 | 2/1/1/128/4096 | - | - | fail | `/Users/yinjinrun/random-thing/logs/mfu-loop-r9-r3_tp2pp1-20260711_010614` | R3: TP=2 PP=1 |
| 10 | `r4_tp4pp1` | 16 | 4/1/1/128/4096 | 125.62 | 42.90 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r10-r4_tp4pp1-20260711_010838` | R4: TP=4 PP=1 |
| 10 | `r4_tp4pp1` | 32 | 4/1/1/128/4096 | 120.44 | 41.13 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r10-r4_tp4pp1-20260711_010838` | R4: TP=4 PP=1 |
| 11 | `r5_tp1pp2` | 16 | 1/2/1/128/4096 | 135.08 | 46.13 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r11-r5_tp1pp2-20260711_011232` | R5: TP=1 PP=2 |
| 11 | `r5_tp1pp2` | 32 | 1/2/1/128/4096 | 121.03 | 41.34 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r11-r5_tp1pp2-20260711_011232` | R5: TP=1 PP=2 |
| 12 | `r6_tp2pp2` | 16 | 2/2/1/128/4096 | 127.76 | 43.63 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r12-r6_tp2pp2-20260711_011635` | R6: TP2PP2 scale |
| 12 | `r6_tp2pp2` | 32 | 2/2/1/128/4096 | 105.80 | 36.14 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r12-r6_tp2pp2-20260711_011635` | R6: TP2PP2 scale |
| 12 | `r6_tp2pp2` | 64 | 2/2/1/128/4096 | 102.49 | 35.00 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r12-r6_tp2pp2-20260711_011635` | R6: TP2PP2 scale |
| 13 | `r7_seq2k` | 16 | 1/1/1/128/2048 | 130.94 | 44.72 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r13-r7_seq2k-20260711_012209` | R7: SEQ=2048 |
| 13 | `r7_seq2k` | 32 | 1/1/1/128/2048 | 121.63 | 41.54 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r13-r7_seq2k-20260711_012209` | R7: SEQ=2048 |
| 14 | `r8_seq8k` | 16 | 1/1/1/64/8192 | - | - | nometrics | `/Users/yinjinrun/random-thing/logs/mfu-loop-r14-r8_seq8k-20260711_012545` | R8: SEQ=8192 |
| 15 | `r9_best128` | 128 | 1/1/1/128/4096 | 83.40 | 28.48 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r15-r9_best128-20260711_012753` | R9: 128卡 |
| 16 | `r10_moe16` | 16 | 1/1/1/128/4096 | 41.00 | 14.00 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r16-r10_moe16-20260711_013103` | R10: MoE冒烟 |
| 17 | `r11_gbs512_s64` | 64 | 1/1/1/512/4096 | 140.61 | 48.02 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r17-r11_gbs512_s64-20260711_013423` | R11: TP1PP1 GBS=512 补弱扩展（插队） |
| 17 | `r11_gbs512_s64` | 128 | 1/1/1/512/4096 | 131.92 | 45.05 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r17-r11_gbs512_s64-20260711_013423` | R11: TP1PP1 GBS=512 补弱扩展（插队） |
| 18 | `r12_gbs1024_s128` | 128 | 1/1/1/1024/4096 | 138.98 | 47.47 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r18-r12_gbs1024_s128-20260711_013917` | R12: TP1PP1 GBS=1024 @128 继续摊销 |
| 19 | `recycle_20260711_013103_full` | 16 | 1/1/1/128/4096 | 156.02 | 53.29 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r19-recycle_20260711_013103_full-20260711_014233` | recycle: re-sweep best parallel across all scales |
| 19 | `recycle_20260711_013103_full` | 32 | 1/1/1/128/4096 | 136.52 | 46.63 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r19-recycle_20260711_013103_full-20260711_014233` | recycle: re-sweep best parallel across all scales |
| 19 | `recycle_20260711_013103_full` | 64 | 1/1/1/128/4096 | 118.35 | 40.42 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r19-recycle_20260711_013103_full-20260711_014233` | recycle: re-sweep best parallel across all scales |
| 19 | `recycle_20260711_013103_full` | 128 | 1/1/1/128/4096 | - | - | nometrics | `/Users/yinjinrun/random-thing/logs/mfu-loop-r19-recycle_20260711_013103_full-20260711_014233` | recycle: re-sweep best parallel across all scales |
| 20 | `r13_gbs2048_s128` | 128 | 1/1/1/2048/4096 | 145.80 | 49.80 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r20-r13_gbs2048_s128-20260711_014834` | R13: GBS=2048 @128 探边际 |
| 21 | `recycle_20260711_013103_mbs2` | 16 | 1/1/2/256/4096 | - | - | nometrics | `/Users/yinjinrun/random-thing/logs/mfu-loop-r21-recycle_20260711_013103_mbs2-20260711_015217` | recycle: MBS=2 GBS=256 |
| 21 | `recycle_20260711_013103_mbs2` | 32 | 1/1/2/256/4096 | - | - | nometrics | `/Users/yinjinrun/random-thing/logs/mfu-loop-r21-recycle_20260711_013103_mbs2-20260711_015217` | recycle: MBS=2 GBS=256 |
| 22 | `r14_gbs2048_s64` | 64 | 1/1/1/2048/4096 | 158.06 | 53.98 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r22-r14_gbs2048_s64-20260711_015528` | R14: GBS=2048 @64 对照128 |
| 23 | `recycle_20260711_015528_full` | 16 | 1/1/1/128/4096 | 156.22 | 53.36 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r23-recycle_20260711_015528_full-20260711_015939` | recycle: re-sweep best parallel across all scales |
| 23 | `recycle_20260711_015528_full` | 32 | 1/1/1/128/4096 | 135.92 | 46.42 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r23-recycle_20260711_015528_full-20260711_015939` | recycle: re-sweep best parallel across all scales |
| 23 | `recycle_20260711_015528_full` | 64 | 1/1/1/128/4096 | 118.63 | 40.52 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r23-recycle_20260711_015528_full-20260711_015939` | recycle: re-sweep best parallel across all scales |
| 23 | `recycle_20260711_015528_full` | 128 | 1/1/1/128/4096 | 87.45 | 29.87 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r23-recycle_20260711_015528_full-20260711_015939` | recycle: re-sweep best parallel across all scales |
| 24 | `r15_gbs2048_s32` | 32 | 1/1/1/2048/4096 | - | - | fail | `/Users/yinjinrun/random-thing/logs/mfu-loop-r24-r15_gbs2048_s32-20260711_020708` | R15: GBS=2048 @32 补齐排行榜 |
| 25 | `r15b_gbs2048_s32` | 32 | 1/1/1/2048/4096 | 167.13 | 57.08 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r25-r15b_gbs2048_s32-20260711_020758` | R15b: GBS=2048 @32 重试 |
| 26 | `r16_gbs4096_s128` | 128 | 1/1/1/4096/4096 | - | - | nometrics | `/Users/yinjinrun/random-thing/logs/mfu-loop-r26-r16_gbs4096_s128-20260711_021346` | R16: GBS=4096 @128 探平台上限 |
| 27 | `r16b_gbs4096_s128` | 128 | 1/1/1/4096/4096 | 152.69 | 52.15 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r27-r16b_gbs4096_s128-20260711_022548` | R16b: GBS=4096 @128 重试（ssh） |
| 28 | `r17_best_matrix` | 16 | 1/1/1/2048/4096 | - | - | fail | `/Users/yinjinrun/random-thing/logs/mfu-loop-r28-r17_best_matrix-20260711_023109` | R17: 统一 GBS=2048 弱扩展矩阵 |
| 28 | `r17_best_matrix` | 32 | 1/1/1/2048/4096 | - | - | fail | `/Users/yinjinrun/random-thing/logs/mfu-loop-r28-r17_best_matrix-20260711_023109` | R17: 统一 GBS=2048 弱扩展矩阵 |
| 28 | `r17_best_matrix` | 64 | 1/1/1/2048/4096 | 157.45 | 53.78 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r28-r17_best_matrix-20260711_023109` | R17: 统一 GBS=2048 弱扩展矩阵 |
| 28 | `r17_best_matrix` | 128 | 1/1/1/2048/4096 | - | - | nometrics | `/Users/yinjinrun/random-thing/logs/mfu-loop-r28-r17_best_matrix-20260711_023109` | R17: 统一 GBS=2048 弱扩展矩阵 |
| 29 | `recycle_20260711_023109_gbs2k_matrix` | 16 | 1/1/1/2048/4096 | 175.03 | 59.78 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r29-recycle_20260711_023109_gbs2k_matrix-20260711_024716` | recycle: TP1PP1 GBS=2048 弱扩展矩阵 |
| 29 | `recycle_20260711_023109_gbs2k_matrix` | 32 | 1/1/1/2048/4096 | 169.48 | 57.88 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r29-recycle_20260711_023109_gbs2k_matrix-20260711_024716` | recycle: TP1PP1 GBS=2048 弱扩展矩阵 |
| 29 | `recycle_20260711_023109_gbs2k_matrix` | 64 | 1/1/1/2048/4096 | 157.01 | 53.63 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r29-recycle_20260711_023109_gbs2k_matrix-20260711_024716` | recycle: TP1PP1 GBS=2048 弱扩展矩阵 |
| 29 | `recycle_20260711_023109_gbs2k_matrix` | 128 | 1/1/1/2048/4096 | 141.48 | 48.32 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r29-recycle_20260711_023109_gbs2k_matrix-20260711_024716` | recycle: TP1PP1 GBS=2048 弱扩展矩阵 |
| 30 | `recycle_20260711_023109_gbs2k_matrix` | 16 | 1/1/1/2048/4096 | 174.69 | 59.66 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r30-recycle_20260711_023109_gbs2k_matrix-20260711_030833` | recycle: TP1PP1 GBS=2048 弱扩展矩阵 |
| 30 | `recycle_20260711_023109_gbs2k_matrix` | 32 | 1/1/1/2048/4096 | - | - | nometrics | `/Users/yinjinrun/random-thing/logs/mfu-loop-r30-recycle_20260711_023109_gbs2k_matrix-20260711_030833` | recycle: TP1PP1 GBS=2048 弱扩展矩阵 |
| 30 | `recycle_20260711_023109_gbs2k_matrix` | 64 | 1/1/1/2048/4096 | 160.27 | 54.74 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r30-recycle_20260711_023109_gbs2k_matrix-20260711_030833` | recycle: TP1PP1 GBS=2048 弱扩展矩阵 |
| 30 | `recycle_20260711_023109_gbs2k_matrix` | 128 | 1/1/1/2048/4096 | 145.32 | 49.63 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r30-recycle_20260711_023109_gbs2k_matrix-20260711_030833` | recycle: TP1PP1 GBS=2048 弱扩展矩阵 |
| 31 | `r31_gbs4096_s128` | 128 | 1/1/1/4096/4096 | 157.23 | 53.70 | ok | `/Users/yinjinrun/random-thing/logs/mfu-loop-r31-r31_gbs4096_s128-20260711_033635` | R31: GBS=4096 @128 复测一次后应转向 Probing |
