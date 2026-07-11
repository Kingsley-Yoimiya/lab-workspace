# GOAL 进度板：华为 → Muxi 全迁移

最后更新: 2026-07-11 21:00 CST

## 总览

| ID | 项 | 状态 | 结果 |
|----|-----|------|------|
| G0–G10 | 全项 | **done** | 见下；**正式总汇报** [`CAMPAIGN_FINAL_MUXI_20260711.md`](../rounds/CAMPAIGN_FINAL_MUXI_20260711.md) |

| ID | 项 | 状态 | 结果 |
|----|-----|------|------|
| G0 | 双集群 KUBECONFIG | **done** | `huawei.env` / `muxi.env` |
| G1 | 快慢卡冒烟 128 | **done** | good=106 / slow=19 / bad=1 |
| G2 | 体质 constitution | **done** | median≈279；[`card_constitution_muxi_20260711.md`](../rounds/card_constitution_muxi_20260711.md) |
| G3 | 拓扑 | **done** | `muxi_topo_20260711.md` |
| G4 | NCCL collective | **done** | 单机≈191 / 跨节点≈0.2；[`nccl_campaign_muxi_20260711.md`](../rounds/nccl_campaign_muxi_20260711.md) |
| G5 | NCCL P2P | **done** | 机内≈30–33；跨节点≈0.35 |
| G6 | 链路健康 | **done** | 16/16 |
| G7 | 一键流水线 | **done** | `run_constitution_then_comm_muxi.sh` |
| G8 | MFU 微基准 | **done** | dense@8=26.7%；moe@8=15.0% |
| G9 | 真训练 MFU | **done** | tiny GPT；MFU≈4.5% |
| G10 | 报告对齐 | **done** | 总汇报 + [`METRIC_SEMANTICS_MUXI_20260711.md`](../rounds/METRIC_SEMANTICS_MUXI_20260711.md) |

## 关键结论

- **GOAL 主路径已收口**；报告体例已对齐昇腾（总汇报 / 语义手册 / 含义优先分报告）。
- 机内可用；跨节点被 eth0（~0.2 GB/s）拖死。
- G9：`nvcc→cucc` + `local/unfused`。

## 可选后续（非 GOAL 阻塞）

- 切 IB/`net*` 重测跨节点 collective / MFU
- 更大模型 + TE 可用时再冲高真训练 MFU
- metax 遥测 util/board_temp 接线后补体质图
