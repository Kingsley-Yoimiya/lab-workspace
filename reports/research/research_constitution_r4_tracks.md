# 体质 / 烤机后续任务拆分（R4）

**日期**: 2026-07-11  
**原则**: 分布优先、尽量多采；不纠结坏卡判定。集群暂缓，先把方案与脚本备齐。

## Track A — 可视化交付（Grok）
- 列出图清单（hist / heatmap / scatter / box_by_host / power-perf / timeseries）
- 实现/增强 `gen_card_constitution_report.py`（或拆出 `plot_constitution.py`）
- 输出可复现命令；为 push 准备

## Track B — Shape 矩阵扩展（Grok 调研 + 配置）
- 不止现有 4 个 BNMK；查训练/FFN/Attention/tall-skinny/batched 等典型 shape
- 给出可开的 shape/bnmk 配置草案（时长可控）

## Track C — HBM 多模式（小开）
- 在单一 `src*2` 之外加少量模式（stride / 读写比 / 块大小）
- 配置开关，默认 constitution 轻量开

## Track D — 功耗×性能分布（大任务，Grok + Sonnet）
- 烤机过程中采 power 与 perf 的联合分布（非仅时间维）
- 每卡 power–tflops / power–gbps 曲线或云图；跨卡对比

## Track E — 卡间通信体质（Grok + Sonnet）
- 建模：该测什么（collective / P2P / 消息尺寸 / 拓扑）
- 现状评估：现有 hccl_128 / p2p 初级，如何升级成「通信体质报告」
- 模式是否正常的判定框架（分布/异常边，非简单阈值）

## 并行派遣
| ID | 模型 | 任务 |
|----|------|------|
| A | grok-4.5 | 可视化方案 + 脚本 |
| B | grok-4.5 | Shape 矩阵调研 + 配置建议 |
| C | grok-4.5 | HBM 多模式小开设计+骨架 |
| D1 | grok-4.5 | 功耗×性能采集/作图方案 |
| D2 | sonnet-5 | 功耗×性能方案评审与补全 |
| E1 | grok-4.5 | 卡间通信体质调研 |
| E2 | sonnet-5 | 卡间通信体质调研（独立） |
