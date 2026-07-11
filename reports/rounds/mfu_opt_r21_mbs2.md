# MFU 优化环 R21：MBS=2 再确认

> 配置：TP1PP1 MBS=2 GBS=256 @16/32  
> 日志：`logs/mfu-loop-r21-recycle_20260711_013103_mbs2-20260711_015217`

**结果：双 scale 均 FAIL**（与 R1 一致，显存墙）。放弃 MBS>1；靠 **GBS** 摊销通信。
