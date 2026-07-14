# 沐曦 / MetaX 硬件词条（体质报告附录）· 20260711

> **怎么用**：正文已在关键缩写后括号附注；本文是**短对照附录**（不必边读边跳转）。  
> **写法约定**：只陈述硬件/工具客观事实，以及**本仓库探针实际调用了什么**；不替读者下「好/坏」「是否降频」的结论。  
> **适用范围**：沐曦曦云 C 系列（本批为 MetaX C550）。JSON 里同名 `aicore_*` / `cube_*` / `mte_*` 等是**昇腾同构壳键名**，**不是**昇腾硅部件。  
> **禁止**：不要把 `npu-smi`、Ascend Cube / AI Core / MTE 硅叙事套到本批。

详细对齐笔记（可并存）：[`../research/METAX_ARCH_ALIGNMENT_20260711.md`](../research/METAX_ARCH_ALIGNMENT_20260711.md)。  
配套度量字段手册：[`METRIC_SEMANTICS_MUXI_20260711.md`](METRIC_SEMANTICS_MUXI_20260711.md)。

---

## 总览：本报告相关的几层

| 层 | 沐曦名字 | 一句话事实 |
|----|----------|------------|
| 计算核 | **XCORE** | 沐曦主计算核（标量 + 矢量 + 张量单元）；≠ Ascend AI Core。 |
| 软件栈 | **MACA / MXMACA** | CUDA 兼容编程模型；本批算子走 `torch.cuda` + CUDA/MACA Event。 |
| 机内互连 | **MetaXLink（MXLK）** | 卡间高速互连；topo 图例常标 `MX`；≠ HCCL / HCCS。 |
| 集合通信 | **MCCL** | 常叠 NCCL API；环境变量如 `MCCL_SOCKET_IFNAME` / `MCCL_IB_HCA`。 |
| 管理接口 | **`mx-smi`** | 温/功耗/利用率/时钟/拓扑等；≠ `npu-smi`。 |
| 片外高带宽内存 | **HBM** | 器件高带宽外存；大块张量经此路径进出。 |

---

<a id="xcore"></a>
## XCORE

- **是什么**：沐曦 GPU 侧主计算核；公开叙述含标量 / 矢量 / 张量单元。
- **本报告怎么碰到它**：几乎所有 GEMM / 向量算子最终跑在 XCORE 相关路径上。
- **兼容键**：`aicore_util_pct` → **GPU util %**（`mx-smi --show-usage`）；`aicore_freq_mhz` → **XCORE_CLK**（`mx-smi -j` → `clocks.XCORE.XCORE_CLK`）。推荐别名：`gpu_util_pct`、`xcore_clk_mhz`。

---

<a id="maca"></a>
## MACA

- **是什么**：沐曦 CUDA 兼容软件栈；本批体质探针用 `torch.cuda` Event 计时，不是 NPU Event。
- **本报告字段**：`vector_gflops` / `scalar_elems_per_s` / `sfu_gflops` 等是 **MACA 上的软件探针路径**，不是昇腾 Vector / Scalar / SFU 硅部件计数器。

---

<a id="metaxlink"></a>
## MetaXLink（MXLK）

- **是什么**：机内卡间高速互连；`mx-smi` topo / mxlk 可查。
- **读数边界**：本批跨节点 NCCL/MCCL 若走 `eth0` socket，测到的断崖反映的是**路径选择**，不能直接写成「MetaXLink 坏了」。

---

<a id="mccl"></a>
## MCCL

- **是什么**：沐曦集合通信库（常与 NCCL API 叠用）。
- **本报告**：通信战役字段按 NCCL/MCCL bench 口径；环境与网卡选择见对齐笔记。

---

<a id="mx-smi"></a>
## `mx-smi`

- **是什么**：MetaX 系统管理命令行（类比 `nvidia-smi`）。
- **本报告常用**：功耗/温度、`--show-usage`（GPU util）、`-j`（XCORE_CLK 等）、topo / MetaXLink。
- **明确**：本批遥测**只**走 `mx-smi`；不要套 `npu-smi -t usages/power` 的字段叙事。

---

<a id="hbm"></a>
## HBM（短注）

- **是什么**：High Bandwidth Memory，器件高带宽外存。
- **本报告 `hbm_gbps`**：`dst = src * 2.0`（含一次逐元素乘）的访存+轻算代理，不是纯 DMA，也不是 `mx-smi` 带宽占用率。纯 copy 见 `mte_gbps`（别名 `dma_copy_gbps`）。

---

## 同构键名对照（壳名 → 沐曦人话）

| JSONL 壳键（昇腾同名） | 沐曦人话 | 真实测法 |
|------------------------|----------|----------|
| `aicore_util_pct` | GPU util % | `mx-smi --show-usage` |
| `aicore_freq_mhz` | XCORE clk (MHz) | `mx-smi -j` → `XCORE_CLK` |
| `func_tflops` / `sustained_tflops` | 方阵 GEMM 主算力 | `a@b` + CUDA/MACA Event |
| `cube_vector_tflops` | **GEMM + epilogue** | `a@b` + scale/bias；**不是** Ascend Cube |
| `mte_gbps` | **纯 copy / DMA** | `Tensor.copy_`；**不是** Ascend MTE |
| `vector_gflops` | MACA 向量 FMA 探针 | `a*b+c` |
| `scalar_elems_per_s` | MACA 串行/同步敏感探针 | `torch.cumsum` |
| `sfu_gflops` | MACA 一元特殊函数探针 | 默认 `torch.exp`（Gops/s 量级） |

---

<a id="health"></a>
## `health_*`：阶段标签 ≠ 健康分

- `health_power_w` / `health_temp_c`：constitution **早期轻载**快照（`mx-smi`）。
- `power_w` / `board_temp_c` 等：多取**负载路径**另一时刻。
- 键名里的 `health` **只标识采样阶段**，不是健康评分，也不是 TDP；两次功耗/温度差值 **≠**「降频幅度」定义式。

---

## 字段名 ↔ 词条（速查）

| 报告字段 | 先读词条 |
|----------|----------|
| `func_tflops` / `sustained_tflops` / BNMK | [XCORE](#xcore) + GEMM 行 |
| `cube_vector_tflops` | 上表「GEMM + epilogue」 |
| `mte_gbps` | 上表「纯 copy」 |
| `vector_*` / `scalar_*` / `sfu_*` | [MACA](#maca) |
| `aicore_util_pct` / `aicore_freq_mhz` | [XCORE](#xcore) |
| `hbm_gbps` | [HBM](#hbm) |
| `health_power_w` / `health_temp_c` | [health_*](#health) |
| 通信 / topo | [MetaXLink](#metaxlink) · [MCCL](#mccl) · [mx-smi](#mx-smi) |

---

## Sources

- [`METAX_ARCH_ALIGNMENT_20260711.md`](../research/METAX_ARCH_ALIGNMENT_20260711.md)
- [模力方舟 C500](https://ai.gitee.com/docs/compute/clusters_gpu/mx_gpu)（同系锚点，非 C550 硬上限）
- [mx-smi 命令介绍](https://developer.metax-tech.com/api/client/document/preview/1111/split_files/%E5%91%BD%E4%BB%8B%E7%BB%8D.html)
- [ms-swift Metax Support](https://swift.readthedocs.io/en/v4.0/BestPractices/Metax-support.html)
