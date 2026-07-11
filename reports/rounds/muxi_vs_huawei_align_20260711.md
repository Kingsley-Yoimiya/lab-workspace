# 华为 ↔ Muxi 迁移对照总表（G10）

时间: 2026-07-11  
集群: Muxi `vc-c550-mohe-241` / `yushan-muxi-card-screen-128-cp-copy`（16×8 C550）

## 能力对照

| ID | 能力 | Muxi 状态 | 关键产物 |
|----|------|-----------|----------|
| G0 | 双集群隔离 | **done** | `muxi.env` / `huawei.env`，独立 kubeconfig |
| G1 | 快慢卡冒烟 | **done** | good=106 slow=19 bad=1（128 卡） |
| G2 | 体质 | **done** | good=108 slow=19 bad=1；median ≈279 TFLOPS |
| G3 | 拓扑 | **done** | 机内 MetaXLink；NIC mlx5 + xscale |
| G4 | NCCL collective | **done** | 单机 8 卡 AR@256M ≈191 GB/s；跨节点 eth0 ≈0.2 GB/s |
| G5 | NCCL P2P | **done** | 机内 16M ≈30–33 GB/s；跨节点 ≈0.35 GB/s |
| G6 | 链路健康 | **done** | `run_link_health_muxi.sh` | 16/16 节点健康文本 |
| G7 | 一键流水线 | **done** | `run_constitution_then_comm_muxi.sh` |
| G8 | MFU 微基准 | **done** | dense@8=26.7%；跨节点~0.2%；moe@8=15.0% |
| G9 | 真训练 MFU | **done** | tiny GPT 8 卡通；稳态≈54ms；MFU≈4.5%（local/unfused）；`muxi_train_mfu_20260711.md` |
| G10 | 报告对齐 | **done** | 本文件 + rounds 专项 |

## 关键结论

1. **机内**（MetaXLink / 单机 NCCL / 真训练）能力正常：体质、P2P、单机 collective、单机 MFU、Megatron 冒烟均可用。
2. **跨节点**当前强制 `NCCL/MCCL_SOCKET_IFNAME=eth0`，带宽约 **0.2–0.35 GB/s**，拖垮多机 collective / MFU；拓扑显示有 **mlx5/xscale**，后续应切 IB 重测。
3. 运维：本机长 SSH 易死 → **pod 内 setsid nohup + tick poll/fire** 为标准模式。
4. **G9**：需 `nvcc→cucc` shim；TE fused attn 缺符号 → `local/unfused`。

## rounds 索引

- 体质: `card_constitution_20260711_141605.md`
- NCCL scale: `muxi_nccl_scale_20260711.md`
- 拓扑: `muxi_topo_20260711.md`
- P2P: `muxi_nccl_p2p_20260711.md`
- MFU 微基准: `muxi_mfu_bench_20260711.md`
- 真训练: `muxi_train_mfu_20260711.md`
- 冒烟: `muxi_smoke_20260711.md`（若存在）
