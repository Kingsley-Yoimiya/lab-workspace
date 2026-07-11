# Muxi G9：真训练 MFU 冒烟（tiny GPT）

时间: 2026-07-11  
集群: `vc-c550-mohe-241` / `yushan-muxi-card-screen-128-cp-copy`  
结果: `/afs-a3-weight-share/montyyin/results/train-mfu-muxi-20260711_155824/`  
本地: `logs/muxi-train-tiny-20260711_155824/`

## 结论

**单机 8 卡 tiny GPT 真训练冒烟通过**（5 iter，`--mock-data`）。

| 指标 | 值 |
|------|-----|
| 稳态 iter 耗时（median，iter2–5） | **53.9 ms** |
| 估算聚合吞吐 | **~101 TFLOPS** |
| 单卡吞吐 | **~12.7 TFLOPS** |
| MFU（相对 G2 peak 279 TFLOPS/卡） | **~4.5%** |

> 注：模型极小（4L/H1024）+ `--transformer-impl local` + `--attention-backend unfused`，MFU 远低于 G8 GEMM 微基准（dense@8=26.7%）属预期；本项验收是「Megatron 真训练链路可跑通并出 iteration 日志」。

## 配置

- Megatron: `/afs-a3-weight-share/workspace/Megatron-LM`
- Wrapper: `scripts/cluster/wrappers/train_gpt_tiny_muxi.sh`
- Fire/Poll: `fire_train_tiny_muxi.sh` / `poll_train_tiny_muxi.sh`
- 4 layers / hidden 1024 / 16 heads / seq 1024 / GBS 8 / bf16 / NullTokenizer / mock-data

## 踩坑与修复

1. **缺 `nvcc`**：`fused_kernels` 硬查 `$CUDA_HOME/bin/nvcc`；cu-bridge 只有 `cucc`。  
   → `ln -sfn cucc …/bin/nvcc`（wrapper 启动时幂等创建）。
2. **TE fused attn 崩**：`transformer_engine_torch` 无 `get_fused_attn_backend`。  
   → `--transformer-impl local --attention-backend unfused`。
3. **假成功**：`torchrun | tee` 后无条件 `echo DONE` 导致 exit 0。  
   → 用 `PIPESTATUS[0]` 传播失败。

## 迭代日志摘要

```
iter1: 17832 ms（编译/warmup）
iter2: 54.8 ms
iter3: 56.1 ms
iter4: 53.0 ms
iter5: 52.3 ms
loss: 10.59 → 9.74（下降正常）
```

## 复跑

```bash
source scripts/cluster/muxi.env
AFS_OUT=/afs-a3-weight-share/montyyin/results/train-mfu-muxi-$(date +%Y%m%d_%H%M%S) \
  ./scripts/cluster/fire_train_tiny_muxi.sh
AFS_OUT=... ./scripts/cluster/poll_train_tiny_muxi.sh
```
