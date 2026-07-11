# MFU 优化环 R23：低价值 recycle（GBS=128 全 scale）

> 日志：`logs/mfu-loop-r23-recycle_20260711_015528_full-20260711_015939`

复测曲线 16/32/64/128 ≈ **53% / 46% / 41% / 30%**，与历史一致。已改 recycle 模板，不再自动塞 GBS=128 / MBS=2 / TP2。
