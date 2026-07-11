# 沐曦（MetaX）架构对齐笔记 · 2026-07-11

> 用途：把「曦云 / XCORE / MACA / MCCL / mx-smi」公开术语钉死，并指导体质脚本与报告去昇腾化。  
> 依据：模力方舟 C500 文档、沐曦 mx-smi 手册、ms-swift Metax Support；C550 公开峰表稀缺，规格锚点以同系 C500 为参考。

---

## 1. 架构分层（第一性原理）

| 层 | 沐曦术语 | 公开含义 | 勿与昇腾混淆 |
|----|----------|----------|--------------|
| ISA / 计算核 | **XCORE** | 标量 + 矢量 + 张量单元 | ≠ AICore / Cube / Vector Core |
| 产品线 | **曦云 C 系列**（C500 / C550 / C588 / C600…） | 训练/通用 GPGPU | 曦彩 G=渲染；曦思 N=推理 |
| 机内互连 | **MetaXLink（MXLK）** | topo 图例 `MX` | ≠ HCCL / HCCS |
| 软件栈 | **MACA / MXMACA** | CUDA 兼容编程模型；`torch.cuda`、`cucc` | ≠ CANN / `torch_npu` |
| 集合通信 | **MCCL**（常叠 NCCL API） | `MCCL_SOCKET_IFNAME` / `MCCL_IB_HCA` | ≠ HCCL |
| 管理接口 | **mx-smi** | 温/功耗/利用率/拓扑/MXLK/RAS | ≠ `npu-smi` |

公开 C500 OAM 参考峰（仅作同系锚点，**非 C550 硬上限**）：BF16/FP16 **280 TFLOPS**，TF32 140，INT8 560 TOPS，64GB HBM2e，7× MetaXLink。本批 C550 `func_tflops` 中位 ~279.3，方向合理，勿直接写成「打满理论峰」。

---

## 2. 探针语义应对齐的人话（键名可双写）

| JSONL 旧键（昇腾壳） | 沐曦人话 | 真实测法 |
|----------------------|----------|----------|
| `func_tflops` | 方阵 GEMM / 主算力 TFLOPS | `a@b` + CUDA Event |
| `mte_gbps` | **纯 copy / DMA** GB/s | `Tensor.copy_`（非 Ascend MTE） |
| `cube_vector_tflops` | **GEMM + epilogue** TFLOPS | `a@b` + scale/bias |
| `vector_gflops` | 宽向量 FMA 代理 | `a*b+c` |
| `aicore_util_pct` | **GPU util %**（兼容键） | `mx-smi --show-usage` |
| `aicore_freq_mhz` | **XCORE_CLK**（兼容键） | `mx-smi -j` → `clocks.XCORE.XCORE_CLK` |
| `temp_c` / health temp | **hotspot / 结温** 代理 | `dmon` hotspot 列 |
| `board_temp_c` | 板温（若接线） | `--show-temperature` / `-j` 多传感器 |

推荐新键（双写）：`dma_copy_gbps`、`gemm_epilogue_tflops`、`gpu_util_pct`、`xcore_clk_mhz`、`hotspot_temp_c`。

---

## 3. mx-smi 能力对照（采集层）

| 官方命令 | 用途 | 本仓状态 |
|----------|------|----------|
| `dmon --show-temperature --show-board-power` | 高刷温/功耗 | ✅ |
| `--show-usage` / `dmon --show-usage` | GPU/VPU 利用率 | ✅ 已补线（TTL）；**历史 JSONL 无值需重跑** |
| `-j` → XCORE_CLK / RAS | 时钟 / ECC | ✅；负载路径时钟仍偏稀 |
| `--show-clk-tr` | 降频原因 | ✅ |
| `mxlk` / `topo -m/-n` | MetaXLink / NIC 拓扑 | 拓扑脚本有；未进体质 JSONL |
| `--show-hbm-bandwidth` | 厂商动态 HBM 带宽 | 未用（用 torch 探针） |

「本批无 util / board_temp」= **当时采集未接线或数据未重跑**，≠ 硬件无该量。

---

## 4. 通信路径

多机官方要求：`MCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `MCCL_IB_HCA`，用 `mx-smi topo -n` + `ifconfig` 选定。  
本批强制 `eth0` socket → 跨节点 ~0.2 GB/s **测量科学、路径可能非 IB 目标面**；机内 MetaXLink / 单机 AR ~190 GB/s 另论。

---

## 5. 修正原则

1. **算子与计时**：保持 CUDA/MACA Event，不回退到 NPU Event。  
2. **报告与图标签**：Cube/MTE/AICore → GEMM / 纯 copy / GPU·XCORE。  
3. **JSONL**：旧键保留 + 新键双写，避免断历史与昇腾对照。  
4. **缺口表述**：统一写「未接线 / 本批未采集」，禁止「硬件没有」。

Sources: [模力方舟 C500](https://ai.gitee.com/docs/compute/clusters_gpu/mx_gpu) · [mx-smi 命令介绍](https://developer.metax-tech.com/api/client/document/preview/1111/split_files/%E5%91%BD%E4%BB%A4%E4%BB%8B%E7%BB%8D.html) · [ms-swift Metax](https://swift.readthedocs.io/en/v4.0/BestPractices/Metax-support.html) · [开发者中心](https://developer.metax-tech.com/)
