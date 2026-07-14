# 128 卡体质 + 通信采集战役 · 最终汇报

**日期**: 2026-07-11  
**Job**: `whj4stu-copy-copy-copy`（8×16 Ascend910 = 128 卡）  
**对照计划**: `reports/research/research_run_plan_ready.md`  
**稳定性对照结论（昇腾 vs 沐曦）**: [`COMPARE_ASCEND_MUXI_STABILITY_20260713.md`](COMPARE_ASCEND_MUXI_STABILITY_20260713.md)  
**本批 128 卡卡间差异（含 launch latency 专节）**: [`WITHIN_CLUSTER_CARD_VARIATION_20260713.md`](WITHIN_CLUSTER_CARD_VARIATION_20260713.md)

---

## 1. 结论（相对计划）

| 计划项 | 状态 | 说明 |
|--------|------|------|
| 128 卡体质分布 | **完成** | 128/128 GOOD；Cube（矩阵计算单元：主计算核内专做大规模矩阵乘加、提供主算力的部件）/HBM（High Bandwidth Memory，器件高带宽外存）/Vector（向量计算单元：做逐元素/向量运算与部分数学函数，灵活度高于矩阵单元、峰值算力通常更低）/Scalar（标量与控制单元：负责循环/分支，并为矩阵/向量/搬运指令计算地址与参数）/SFU（特殊函数类吞吐代理；本探针默认 torch.exp，按 1 op/元素计，公开叙述常归在向量计算能力面）/MTE（Memory Transfer Engine，片上 Buffer 与 Global Memory 之间的数据搬运引擎；本字段多用 Tensor.copy_ 作纯搬运带宽代理，并非直接读该引擎指令计数器）/pipeline/launch/SDC |
| 10 shape BNMK | **完成**（补采） | 10 labels × 128 卡 = **1280** 条 `gemm_bnmk_sample` |
| 遥测温/功耗/util | **完成**（补采） | 修复 `npu-smi -t` 解析后重跑；功耗×性能图已出 |
| 出图 | **完成** | 主图 112 + 增强 12 + BNMK 3 + HCCL 23 + 机间 1 ≈ **151 SVG**（默认 `plot_style`） |
| HCCL 拓扑 | **完成** | 8 节点 HCCS 矩阵；`topo_summary` JSON/MD；无 hccn（平台未挂） |
| Collective 16→128 | **完成** | 四算子×四消息；保持率已算 |
| P2P 16→128 | **完成** | 128 ring 通；慢边 TopK 已出 |

**计划达成度：≈ 95%+。** 剩余仅平台侧缺口（hccn.conf / hccn_tool 本 job 不可用），不影响主结论。

---

## 2. 关键数字

### 体质（fillgap 重跑，含遥测+BNMK）

| 指标 | 中位 | 覆盖 |
|------|------|------|
| Cube（矩阵计算单元：主计算核内专做大规模矩阵乘加、提供主算力的部件） func TFLOPS | **292.4** | 128/128 |
| Sustained TFLOPS | ~306–310 | 128/128 |
| HBM（High Bandwidth Memory，器件高带宽外存） GB/s | ~1240 | 128/128 |
| Vector（向量计算单元：做逐元素/向量运算与部分数学函数，灵活度高于矩阵单元、峰值算力通常更低） GFLOPS | **98.8** | 128/128 |
| SFU（特殊函数类吞吐代理；本探针默认 torch.exp，按 1 op/元素计，公开叙述常归在向量计算能力面） GFLOPS | ~156 | 128/128 |
| MTE（Memory Transfer Engine，片上 Buffer 与 Global Memory 之间的数据搬运引擎；本字段多用 Tensor.copy_ 作纯搬运带宽代理，并非直接读该引擎指令计数器） GB/s | ~1267 | 128/128 |
| 空闲功耗 health_power_w（流程早期轻载时刻的 npu-smi Real-time Power；health 是采样阶段标签，不是健康分） | **168 W** | 128/128 |
| 满载功耗 power_w | **872 W** | 128/128 |
| 健康温度 health_temp_c（流程早期轻载/开测温度快照，与负载中 board_temp 不同时刻） | **40 °C** | 128/128 |
| 板温 board_temp_c（满载采样） | **66 °C** | 128/128 |
| BNMK peak TFLOPS | **310.7** | 128/128 |
| 判定 | **128 GOOD / 0 BAD** | |

### 通信（all_reduce @ 256MB bus_bw 保持率 vs w16）

| 算子 | w32 | w64 | w128 |
|------|-----|-----|------|
| All-Reduce | 96.8% | 94.9% | **89.4%** |
| Broadcast | 91.4% | 86.8% | **86.8%** |
| All-Gather | 88.0% | 64.2% | **54.0%** |
| Reduce-Scatter | 91.8% | 71.0% | **46.4%** |

→ All-Reduce / Broadcast 扩展健康；AG / RS 在 128 卡掉得明显，是后续通信优化重点。

---

## 3. 产物索引（看图从这里进）

> 出图默认 `reports/plot_style.py`（大字号 / 去顶右边框 / y 点线网格 / hatch 柱 / **SVG**）。  
> **图注优先讲清：字段人话含义 + 底层 API/命令/算子**（不是「怎么画直方图」）。语义手册：[`METRIC_SEMANTICS_20260711.md`](METRIC_SEMANTICS_20260711.md)；采集链路审计：[`FIGURE_PROVENANCE_AUDIT_20260711.md`](FIGURE_PROVENANCE_AUDIT_20260711.md)。

### 体质主报告（含功耗/温度）
- [`card_constitution_20260711.md`](card_constitution_20260711.md)
- [`card_constitution_20260711_figs/`](card_constitution_20260711_figs/)（**112** svg）
  - 注意：`timeseries_sustained_p05_p50.svg` = **跨卡** p05/p50（按 iter 对齐），不是代表卡时序

### 体质增强
- [`constitution_extra_fillgap_20260711.md`](constitution_extra_fillgap_20260711.md)
- [`constitution_extra_fillgap_20260711_figs/`](constitution_extra_fillgap_20260711_figs/)（12）

### BNMK 10 shape
- [`bnmk_shapes_20260711.md`](bnmk_shapes_20260711.md) / [`bnmk_shapes_20260711_figs/`](bnmk_shapes_20260711_figs/)（3；样本 1280）

### HCCL + P2P + 拓扑
- [`hccl_campaign_20260711.md`](hccl_campaign_20260711.md) / [`hccl_campaign_20260711_figs/`](hccl_campaign_20260711_figs/)（23）
- [`topo_summary_20260711.md`](topo_summary_20260711.md)

### 机间带宽
- [`inter_bw_20260711.md`](inter_bw_20260711.md) / [`INTER_BW_PROBE_20260711.md`](INTER_BW_PROBE_20260711.md)

### 原始数据
- 体质 fillgap: `logs/card-fillgap-20260711_140301/results/constitution128.merged.jsonl`
- 通信: `logs/pipeline-comm-20260711_134811/`
- 机间: `logs/inter-bw-20260711_141922/`

## 4. 本轮修过的坑（便于复现）

1. **双集群 kubeconfig**：本机华为用独立 `KUBECONFIG` + kubectl，避免 weibozhen 覆盖。
2. **Python 3.8**：容器 `set_env` 后系统 python 无 `dict|dict` → 强制 conda 3.10。
3. **遥测**：禁止 `npu-smi info -i`；改用 `-t power/temp/usages -i card -c chip`。
4. **BNMK**：配置有 `bnmk_sweep` 但缺 probe → 已注册 `BnmkSweep` + `gemm_bnmk_sweep`。
5. **长任务**：pod 内 `setsid nohup`，勿对含 `screen.py` 字样的启动脚本 `pkill -f`。

---

## 5. 一眼叙事

这批 128 卡 **算力/带宽一致性很好**（多数 CV < 4%），判定全 GOOD。  
通信上 **All-Reduce 到 128 仍保留约 89%**，但 **All-Gather / Reduce-Scatter 腰斩级下滑**，与机内 HCCS 健康、跨机扩展变差的图像一致。  
功耗画像已打通：空闲 ~168 W，满载采样中位 ~872 W，可与 Cube（矩阵计算单元：主计算核内专做大规模矩阵乘加、提供主算力的部件）/HBM（High Bandwidth Memory，器件高带宽外存） 做交叉散点。

> 索引更新时间：2026-07-11 20:18
