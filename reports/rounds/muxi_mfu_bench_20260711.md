# Muxi MFU 微基准（G8）

时间: 2026-07-11  
AFS: `/afs-a3-weight-share/montyyin/results/mfu-dense-20260711_152200`  
峰值分母: constitution GEMM func median **279 TFLOPS/卡**

## dense 全档

| mode | world | MFU | achieved/peak TFLOPS | toks/s | step_ms |
|------|------:|----:|---------------------:|-------:|--------:|
| dense | 8 | **26.7%** | 596 / 2232 | 186k | 88 |
| dense | 16 | **0.32%** | 14 / 4464 | 4.4k | 7391 |
| dense | 32 | **0.28%** | 25 / 8928 | 7.9k | 8285 |
| dense | 64 | **0.24%** | 43 / 17856 | 13.6k | 9662 |
| dense | 128 | **0.22%** | 78 / 35712 | 24.5k | 10697 |

说明：跨节点受 eth0 NCCL 限制，MFU 骤降；**单机 8 卡 26.7%** 为有效算力参考。IB/`net*` 修好后应重测 16+。

## moe

| mode | world | MFU | achieved/peak TFLOPS | toks/s | step_ms |
|------|------:|----:|---------------------:|-------:|--------:|
| moe | 8 | **15.0%** | 334 / 2232 | 48.8k | 336 |

## 脚本

`mfu_train_bench_nccl.py` · `fire_mfu_bench_muxi.sh` · `poll_mfu_bench_muxi.sh`

