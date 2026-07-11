# 体质筛查 · 分布优先方案（R3）

**日期**: 2026-07-11  
**用户意图（覆盖 R2.5 判定争论）**:  
不关心「哪些指标能判坏卡」；要的是 **128 卡各部件/各指标的分布尽量详细、尽量多**，直接拿到数据。

**前置**: R2.5 合并里的 **遥测修复**（命令/`-c`/正则）仍是阻塞——毒 `temp_c=2` 会让热/功耗/频率分布全废。

---

## 1. 原则

| 旧重心 | 新重心 |
|--------|--------|
| slow_frac / 换卡单 / 主键争论 | **全指标落库 + 分布图 + 分位数/CV/host×device** |
| verdict_neutral 争论 | 全部探针 **只采集**；verdict 仅保留 dead/bad/SDC 硬故障 |
| 少探针、快发射 | **在合理时长内开满正交探针**，宁可单卡多几分钟 |

## 2. 要采的分布面（尽量全）

| 面 | 指标 | 来源 |
|----|------|------|
| Cube | func_tflops, sustained_tflops, shape 可选关 | Stage A |
| Vector | vector_gflops | Stage C |
| Scalar 代理 | scalar_elems_per_s | Stage C |
| SFU | sfu_gflops（新增） | Stage C |
| HBM | hbm_gbps | Stage A |
| MTE/DMA | mte_copy_gbps（新增） | Stage C |
| Cube↔Vector | cube_vector_tflops（新增） | Stage C |
| Launch | sync/tiny/host_overhead p50/p99 + burst（增强） | Stage C |
| 热/功耗/频 | temp / hbm_temp / power / aicore_freq / util（修遥测后） | npu-smi |
| 健康计数 | ecc / pcie-err / health（快照） | npu-smi |
| 元数据 | host, device, card_id, chip_id, driver, firmware | 写入 JSONL |

## 3. 报告交付物（拿到数据后）

每指标：median / mean / std / CV / p5–p95 / min–max  
相对中位数偏差；host×device 热力图；正交散点（Cube×Vector、HBM×MTE、launch×ctrlcpu 等）  
原始 JSONL 全保留，不覆盖旧 run。

## 4. 发射

```
修遥测 → sync AFS → 16 卡冒烟验分布非空
→ 128 fanout（constitution 全开观察探针）
→ 拉回 logs + 出分布报告
```

MFU 环保持 PAUSE。
