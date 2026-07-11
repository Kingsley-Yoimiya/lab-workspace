# MFU 优化环 R8：SEQ=8192

> 作者：Cursor Grok 4.5  
> 配置：TP1PP1 MBS=1 GBS=64 SEQ=8192 @16  
> 日志：`logs/mfu-loop-r14-r8_seq8k-20260711_012545`

## 证据

启动后 **NPU 显存分配失败**（`aclnnFlashAttentionScore` / PTA memory error），无稳态 MFU。

**结论：** 当前 8B dense + 全量激活下 **8k 序列不可行**（至少需重计算/更小 MBS 或切 CP）。默认保持 **SEQ=4096**。
