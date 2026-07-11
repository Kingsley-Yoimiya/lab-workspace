# MFU 优化环 R31：GBS=4096 @128 复测

> 日志：`logs/mfu-loop-r31-r31_gbs4096_s128-20260711_033635`

稳态约 **159 TFLOP** → MFU **~54.4%**，与 R16b（~53%）一致。

128 卡 GBS 平台已确认；环已 **PAUSE**，等 Probing / 新假设（写 `next_job.json` 或删 `PAUSE` 恢复）。
