# MFU 优化环 R1：TP1PP1 + MBS=2 → OOM

> 作者：Cursor Grok 4.5  
> 基线 R0：TP1PP1 MBS=1 → ~170.6 TFLOP/s/GPU，MFU≈**58.3%**  
> 本轮：同并行抬 **MBS=2** @16 卡  
> 日志：`logs/mfu-loop-r5-r1_mbs2-20260711_004343`

## 假设

在 TP=1 PP=1 上增大 micro-batch 可提高算力强度、抬 MFU。

## 证据

训练启动成功（launcher/`python3 -` 路径已打通），首步 forward 即 **NPU OOM**（约需再分配 2.32 GiB，空闲 <1 GiB）。无稳态 TFLOP 可读。

**结论：假设不成立（显存墙）。** 该配置下 MBS=2 不可行。

## 启动链路修复（本轮附带）

1. 勿在 `set -e` 下 `source set_env.sh`（会直接 exit）  
2. 长命令勿直接塞 `vcctl pod exec`（易 Usage）→ AFS launcher + `python3 -` subprocess  
3. 训练失败须检查 `PIPESTATUS` / OOM 关键字，避免假 OK

## 二次假设（入队）

1. **放弃 MBS=2**；保持 MBS=1，扩 **32/64/128** 看弱扩展  
2. 试 **GBS=256**（不增 MBS）摊销通信  
3. 扫 TP/PP 组合；Probing 量化 comm/compute（待装二进制）
