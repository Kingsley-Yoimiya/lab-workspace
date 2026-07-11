# GOAL 进度板：华为 → Muxi 全迁移

最后更新: 2026-07-11 16:00 CST

## 总览

| ID | 项 | 状态 | 结果 |
|----|-----|------|------|
| G0 | 双集群 KUBECONFIG | **done** | `huawei.env` / `muxi.env` |
| G1 | 快慢卡冒烟 128 | **done** | good=106 / slow=19 / bad=1 |
| G2 | 体质 constitution | **done** | median≈279 TFLOPS；`card_constitution_20260711_141605.md` |
| G3 | 拓扑 | **done** | `muxi_topo_20260711.md` |
| G4 | NCCL collective | **done** | 单机 AR≈191 GB/s；跨节点≈0.2 GB/s |
| G5 | NCCL P2P | **done** | 机内≈30–33 GB/s；跨节点≈0.35 |
| G6 | 链路健康 | **done** | 16/16；`muxi_link_health_20260711.md` |
| G7 | 一键流水线 | **done** | `run_constitution_then_comm_muxi.sh` |
| G8 | MFU 微基准 | **done** | dense@8=26.7%；moe@8=15.0% |
| G9 | 真训练 MFU | **done** | tiny GPT 8 卡 5iter 通；稳态≈54ms；MFU≈4.5%（local/unfused）；`muxi_train_mfu_20260711.md` |
| G10 | 报告对齐 | **done** | `muxi_vs_huawei_align_20260711.md` |

## 关键结论

- **GOAL 主路径已收口**：G0–G10 均有可复跑脚本 + 落盘结果。
- 机内（MetaXLink / 单机 NCCL / 单机训练）可用；跨节点被 eth0（~0.2 GB/s）拖死。
- G9 需 `nvcc→cucc` shim + `local/unfused` 绕开 TE fused attn 缺口。

## 可选后续（非 GOAL 阻塞）

- 切 IB/`net*` 重测跨节点 collective / MFU
- 更大模型 + TE 可用时再冲高真训练 MFU
