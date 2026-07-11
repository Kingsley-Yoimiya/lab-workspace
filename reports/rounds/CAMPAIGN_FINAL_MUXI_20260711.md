# 128 卡体质 + 通信采集战役 · Muxi 最终汇报

**日期**: 2026-07-11  
**Job**: `yushan-muxi-card-screen-128-cp-copy`（16×8 MetaX C550-PL = 128 卡）  
**集群**: `vc-c550-mohe-241`（kube 隔离：`scripts/cluster/muxi.env` → `CLUSTER_KUBECONFIG`）  
**对照计划**: [`../research/GOAL_MUXI_MIGRATE_STATUS.md`](../research/GOAL_MUXI_MIGRATE_STATUS.md)  
**对标昇腾总汇报**: [`CAMPAIGN_FINAL_20260711.md`](CAMPAIGN_FINAL_20260711.md)

---

## 1. 结论（相对 GOAL G0–G10）

| 计划项 | 状态 | 说明 |
|--------|------|------|
| G0 双集群隔离 | **完成** | `huawei.env` / `muxi.env`；不覆盖默认 kubeconfig |
| G1 快慢卡冒烟 128 | **完成** | good=106 / slow=19 / bad=1 / contended=2 |
| G2 体质 constitution | **完成** | 127/128 有效；func 中位 **279.3 TFLOPS**；含义优先图 91 SVG |
| G3 拓扑 | **完成** | 16/16 `mx-smi`；机内 MetaXLink；NIC mlx5+xscale |
| G4 NCCL collective | **完成** | 8→128；单机 AR@256M ≈**190.5 GB/s**；跨节点保持率 ~**0.13%** |
| G5 NCCL P2P | **完成** | ring 16/128；机内 16M ≈30–33 GB/s；跨节点 ≈0.35 |
| G6 链路健康 | **完成** | 16/16 mx-smi + ibv 文本 |
| G7 一键流水线 | **完成** | `run_constitution_then_comm_muxi.sh` |
| G8 MFU 微基准 | **完成** | dense@8=**26.7%**；跨节点 ~0.2%；moe@8=15.0% |
| G9 真训练 MFU | **完成** | tiny GPT 8 卡 5iter 通；稳态≈54ms；估算 MFU≈4.5% |
| G10 报告对齐 | **完成** | 本文件 + 语义手册 + 溯源 + 含义优先分报告 |

**计划达成度：GOAL 主路径 100%。**  
剩余为 **可选增强**：跨节点切 IB/`net*` 重测；TE fused attn 符号补齐后冲高真训练 MFU；metax 遥测 util / board_temp 接线；BNMK sample。

---

## 2. 关键数字

### 体质（constitution128 merged，现算中位）

| 指标 | 中位 | 覆盖 |
|------|------|------|
| 方阵 GEMM func TFLOPS | **279.3** | 127/128 |
| Sustained TFLOPS | **279.5** | 127/128 |
| HBM GB/s | **1470** | 127/128 |
| Vector GFLOPS | **122.1** | 127/128 |
| SFU（Gops/s 量级） | **176.8** | 127/128 |
| MTE GB/s | **1387** | 127/128 |
| Cube+Vector TFLOPS | **195.1** | 127/128 |
| 空闲功耗 health_power_w | **94.83 W** | 128/128 |
| 满载功耗 power_w | **467 W** | 127/128 |
| 功耗墙 power_limit_w | **550 W** | 127/128 |
| 健康温度 health_temp_c | **38.75 °C** | 128/128 |
| 冒烟判定 | good **106** / slow 19 / bad **1** | 128 |
| BNMK | **无本批 sample** | — |

### 通信（All-Reduce @ 256MB bus_bw 保持率 vs **w8**，现算）

| world | bus_bw (GB/s) | 保持率 vs w8 |
|------:|--------------:|-------------:|
| 8 | **190.5** | 100% |
| 16 | 0.2563 | **0.13%** |
| 32 | 0.2691 | **0.14%** |
| 64 | 0.2557 | **0.13%** |
| 128 | 0.2409 | **0.13%** |

四算子保持率（256MB，相对 w8）：

| 算子 | w16 | w32 | w64 | w128 |
|------|-----|-----|-----|------|
| All-Reduce | 0.13% | 0.14% | 0.13% | 0.13% |
| Broadcast | 0.69% | 0.26% | 0.24% | 0.23% |
| All-Gather | 0.25% | 0.25% | 0.24% | 0.22% |
| Reduce-Scatter | 0.25% | 0.25% | 0.25% | 0.22% |

→ **断崖在 8→16（首次跨节点）**；其后几乎持平。机内健康，跨节点 eth0 打穿。

### MFU / 训练

| 项 | 值 |
|----|-----|
| dense MFU @8 | **26.7%**（peak=279×8） |
| dense MFU @16–128 | **0.22–0.32%** |
| moe MFU @8 | **15.0%** |
| 真训练 tiny GPT @8 | 稳态 **53.9 ms/iter**；估算 MFU **~4.5%**（local/unfused） |

---

## 3. 产物索引（看图从这里进）

> 出图默认 `reports/plot_style.py`（大字号 / 去顶右边框 / y 点线网格 / hatch 柱 / **SVG**）。  
> **图注优先讲清：字段人话含义 + 底层 API/命令/算子**（禁止画图空话）。  
> 语义手册：[`METRIC_SEMANTICS_MUXI_20260711.md`](METRIC_SEMANTICS_MUXI_20260711.md)；采集链路溯源：[`FIGURE_PROVENANCE_MUXI_20260711.md`](FIGURE_PROVENANCE_MUXI_20260711.md)。

### 体质主报告
- [`card_constitution_muxi_20260711.md`](card_constitution_muxi_20260711.md)
- [`card_constitution_muxi_20260711_figs/`](card_constitution_muxi_20260711_figs/)（**91** svg）
  - 注意：`timeseries_sustained_p05_p50.svg` = **跨卡** p05/p50（按 iter 对齐），不是代表卡时序

### 体质增强
- [`constitution_extra_muxi_20260711.md`](constitution_extra_muxi_20260711.md)
- [`constitution_extra_muxi_20260711_figs/`](constitution_extra_muxi_20260711_figs/)（12）

### NCCL/MCCL + P2P + 拓扑
- [`nccl_campaign_muxi_20260711.md`](nccl_campaign_muxi_20260711.md) / [`nccl_campaign_muxi_20260711_figs/`](nccl_campaign_muxi_20260711_figs/)（23）
- 短版：[`muxi_nccl_scale_20260711.md`](muxi_nccl_scale_20260711.md) · [`muxi_nccl_p2p_20260711.md`](muxi_nccl_p2p_20260711.md)
- 拓扑：[`muxi_topo_20260711.md`](muxi_topo_20260711.md) · 链路：[`muxi_link_health_20260711.md`](muxi_link_health_20260711.md)

### MFU / 真训练
- [`muxi_mfu_bench_20260711.md`](muxi_mfu_bench_20260711.md)
- [`muxi_train_mfu_20260711.md`](muxi_train_mfu_20260711.md)

### 对照与总览
- [`muxi_vs_huawei_align_20260711.md`](muxi_vs_huawei_align_20260711.md)
- 昇腾总汇报：[`CAMPAIGN_FINAL_20260711.md`](CAMPAIGN_FINAL_20260711.md)

### 原始数据
- 体质: `logs/muxi-constitution-20260711_140024-muxi-constitution128/results/constitution128.merged.jsonl`
- NCCL 本地: `logs/muxi-nccl-campaign-20260711/nccl-results/`
- NCCL AFS: `/afs-a3-weight-share/montyyin/results/nccl-20260711_142129`
- 冒烟: `logs/muxi-card-screen-20260711_133828-muxi-smoke/`

---

## 4. 本轮修过的坑（便于复现）

1. **双集群 kubeconfig**：只用 `CLUSTER_KUBECONFIG`，永不覆盖 weibozhen 默认 config。
2. **长任务**：本机 `nohup`/长 SSH 易被 IDE 杀掉 → **pod 内 `setsid nohup` + 短连 fire/poll**。
3. **`pod exec -i` 上传与启动分离**：合并会导致空文件。
4. **多机 NCCL**：必须 `*_SOCKET_IFNAME=eth0`，否则 Proxy Connect 失败。
5. **扇出并发**：16 路过猛会 SSH 踢人 → `CLUSTER_FANOUT_PARALLEL≈4–6`。
6. **AFS 写结果**：经 pod 写；登录机假挂载不可用。
7. **G9 nvcc**：`CUDA_HOME/bin/nvcc` ← symlink `cucc`；TE fused attn 缺符号 → `local/unfused`。
8. **假成功**：`torchrun | tee` 必须用 `PIPESTATUS[0]`。
9. **遥测缺口**：本批无 board_temp/util 接线、无 BNMK sample——报告勿假装有图。

---

## 5. 一眼叙事

这批 128 张 C550 **机内算力很齐**（func 中位 **279.3 TFLOPS**，覆盖 127/128），HBM 中位高但有整节点掉速簇；冒烟有 1 张正确性坏卡。

通信上 **单机 All-Reduce@256MB 中位 ≈190.5 GB/s**，但 **一跨节点保持率就掉到 ~0.13%**（w128 ≈0.13%），
P2P/MFU 同步复现同一断崖——根因是当前走 **eth0 socket**，不是 MetaXLink 坏了。拓扑已看到 mlx5/xscale，下一步应切 IB 重测。

训练侧：微基准单机 MFU 26.7% 可用；Megatron tiny 冒烟已打通（需 nvcc shim + 避开 TE fused）。

与昇腾同日战役对照：昇腾跨节点 All-Reduce 保持率仍约 89%，沐曦跨节点是 **路径问题**；机内两边都健康，可并行作为双集群基线。

> 索引更新时间：2026-07-11（`rewrite_meaning_mds_muxi.py` 现算关键数字）
