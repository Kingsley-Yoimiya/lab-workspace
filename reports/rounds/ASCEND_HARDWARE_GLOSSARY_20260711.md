# 昇腾硬件词条（体质报告引用）· 20260711

> **怎么用**：体质报告正文已在关键缩写后直接括号附注；本文是**完整对照附录**（不必边读边跳转）。  
> **写法约定**：只陈述硬件/工具文档里的客观事实，以及**本仓库探针实际调用了什么**；不替读者下「好/坏」「是否降频」的结论。  
> **适用范围**：昇腾 DaVinci / AI Core 叙述（本批卡为 Ascend910 系）。沐曦报告里同名 JSON 键是**兼容壳**，硬件含义见 [`../research/METAX_ARCH_ALIGNMENT_20260711.md`](../research/METAX_ARCH_ALIGNMENT_20260711.md)。

配套度量字段手册：[`METRIC_SEMANTICS_20260711.md`](METRIC_SEMANTICS_20260711.md)。

---

## 总览：一块 NPU 里和本报告相关的几层

| 层 | 常见名字 | 一句话事实 |
|----|----------|------------|
| 计算核 | **AI Core**（亦称 DaVinci Core） | 昇腾面向 AI 负载的专用计算核心，内含多类计算单元与专用存储层次，而不是通用 GPU SM 的同构复制。 |
| 计算单元 | **Cube / Vector / Scalar** | AI Core 内三条主要计算流水：矩阵、向量、标量控制。 |
| 搬运 | **MTE**（Memory Transfer Engine） | 负责片上 Buffer 与 Global Memory 之间的数据搬运（及格式转换）的引擎；有独立指令队列。 |
| 片外高带宽内存 | **HBM** | 芯片外挂的高带宽存储器；算子数据最终多经 Global Memory / HBM 路径。 |
| 片上控制面 CPU | **CtrlCPU（Control CPU）** | 设备侧控制类 CPU；`npu-smi` 暴露其占用率字段。 |
| 片上 AI CPU | **AICPU（AI CPU）** | 设备侧可跑部分算子/任务的 CPU 资源；与 AI Core 上的 Cube/Vector 不是同一执行体。 |
| 管理接口 | **`npu-smi`** | 驱动提供的 NPU 系统管理命令行（类比 `nvidia-smi`），可查功耗、温度、usages 等。 |

公开架构叙述（Cube / Vector / Scalar / MTE 队列）见华为云社区对 AI Core 的整理：[AI Core硬件架构剖析：Cube、Vector、Scalar三核协同机制](https://bbs.huaweicloud.com/blogs/471408)（文中图示标注来源为昇腾官方文档）。  
`npu-smi` usages / power 命令格式见昇腾社区命令参考，例如：[查询指定芯片统计信息（`-t usages`）](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/24.1.RC3/re/npu/npusmi_020.html)、华为企业文档对 [`-t power` 查芯片功耗](https://support.huawei.com/enterprise/en/doc/EDOC1100079295/c8f5b2f7/introduction-to-the-npu-smi-command-for-version-1011-1015) 的说明。

---

<a id="ai-core"></a>
## AI Core

- **是什么（硬件）**：昇腾 AI 处理器上的主计算核心；面向深度学习常见的矩阵/向量/控制流模式做特定域设计（DSA），而不是通用 CPU/GPU 的等价物。
- **内部与本报告相关的部件**：Cube、Vector、Scalar 计算单元；L0/L1/Unified Buffer 等片上存储；MTE 搬运引擎；多条指令队列（含 Scalar / Vector / Matrix / MTE1–3）。
- **本报告怎么碰到它**：几乎所有 `torch_npu` GEMM / 向量算子最终都跑在 AI Core 相关执行路径上；利用率字段 `aicore_util_pct` 来自 `npu-smi … -t usages` 的 **Aicore Usage Rate (%)**。
- **来源**：[AI Core 三核协同机制](https://bbs.huaweicloud.com/blogs/471408)；usages 字段见 [`npu-smi -t usages`](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/24.1.RC3/re/npu/npusmi_020.html) 与公开样例输出中的 `Aicore Usage Rate(%)`（如 [Atlas 200 DK 使用笔记](https://blog.csdn.net/m0_37605642/article/details/137472243)）。

---

<a id="cube"></a>
## Cube（矩阵计算单元）

- **是什么（硬件）**：AI Core 内专门做大规模矩阵乘加的单元；公开叙述称其为 AI Core **主要算力**来源。典型深度学习全连接、卷积（im2col 后）、注意力里的矩阵乘都落在这类能力上。
- **公开规格叙述（DaVinci 系常见写法）**：一次可完成 FP16 的 16×16×16 类矩阵乘形态；INT8 等有对应更大/不同形状的一次乘加规模（具体代际以该卡数据手册为准）。
- **专用存储**：L0A / L0B（输入矩阵）、L0C（结果/中间结果）。
- **本报告字段**：
  - `func_tflops` / `sustained_tflops`：软件侧用 `c = a @ b`（bf16）+ NPU Event 计时，按 `2·N³` 折算 TFLOPS——**测的是「矩阵乘主路径吞吐代理」**，不是手册里某一条 Cube 指令的周期计数器。
  - 字段名里的 “Cube” 表示**意图对齐的硬件语义**，不是 CANN 里某个叫 Cube 的 Python API。
- **来源**：[Cube 单元说明](https://bbs.huaweicloud.com/blogs/471408)。

---

<a id="vector"></a>
## Vector（向量计算单元）

- **是什么（硬件）**：AI Core 内做向量级运算的单元；算力通常低于 Cube，但指令更灵活。公开列举能力包括向量加减乘除、若干数学函数（倒数、平方根、指数、对数等）、比较/逻辑、类型转换等。
- **存储约束（公开叙述）**：源/目的数据在 **Unified Buffer（UB）**，并有对齐要求（文中写 32 Byte）。
- **本报告字段**：
  - `vector_gflops`：`a * b + c` 逐元素 FMA，按 2 flops/elem 计——软件代理。
  - `sfu_gflops`：默认 `torch.exp`；字段名沿用 “SFU/特殊函数” 习惯，**实现上是一次一元逐元素 op**，按 1 op/elem 计成 “gflops”，量纲更接近 Gops/s。公开 AI Core 文章常把 exp/sqrt 等归在 Vector 能力面，本仓库**未**单独读出一块名为 SFU 的硬件计数器。
- **来源**：[Vector 单元说明](https://bbs.huaweicloud.com/blogs/471408)。

---

<a id="scalar"></a>
## Scalar（标量计算单元）

- **是什么（硬件）**：AI Core 内负责标量运算与**程序流控制**的单元（循环、分支、给 Cube/Vector/MTE 算地址与参数等）。公开叙述将其比作 AI Core 内的小型控制 CPU 角色。
- **本报告字段**：`scalar_elems_per_s` 用 `torch.cumsum` 测长依赖串行链元素吞吐——是**软件侧串行/同步敏感代理**，不是 Scalar 指令吞吐计数值。
- **来源**：[Scalar 单元说明](https://bbs.huaweicloud.com/blogs/471408)。

---

<a id="mte"></a>
## MTE（Memory Transfer Engine / 内存搬运引擎）

- **是什么（硬件）**：AI Core 内负责 **不同存储层次之间数据搬运（及格式转换）** 的引擎。公开指令队列划分包括：
  - **MTE1**：L1 → L0A/L0B/UB
  - **MTE2**：Global Memory（GM）→ L1/L0A/L0B/UB
  - **MTE3**：UB → GM  
  与 Scalar / Vector / Matrix（Cube）队列并列，可在满足依赖时与计算队列并行。
- **本报告字段 `mte_gbps`**：
  - 实际调用：`Tensor.copy_`（设备侧纯拷贝），流量按读+写字节计，Event 取中位 GB/s。
  - **事实边界**：这是「纯搬运带宽」的**软件探针命名**；并**没有**在报告里直接下发 Ascend C 的 MTE1/2/3 指令或读 MTE 硬件性能计数器。把结果读成 “MTE 峰值手册规格” 会过拟合。
- **来源**：[MTE 与六队列表](https://bbs.huaweicloud.com/blogs/471408)。

---

<a id="hbm"></a>
## HBM（High Bandwidth Memory）

- **是什么（硬件）**：器件使用的高带宽外存；算子大块张量通常经 Global Memory / HBM 路径进出片上 Buffer。
- **本报告字段**：
  - `hbm_gbps`：`dst = src * 2.0`（fp32，**含一次逐元素乘**），按 R+W 计 GB/s——是「访存 + 轻算」混合代理，**不是**纯 DMA，也不是 `npu-smi` 的带宽利用率。
  - `mem_bw_util_pct`：`npu-smi -t usages` 的 **HBM Bandwidth Usage Rate (%)**（瞬时占用率字段名以工具输出为准）。
- **来源**：usages 输出字段见公开 `npu-smi info -t usages` 样例（含 Memory / 带宽类 Usage Rate）如 [此笔记](https://blog.csdn.net/m0_37605642/article/details/137472243)；命令入口见 [昇腾 `npu-smi -t usages`](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/24.1.RC3/re/npu/npusmi_020.html)。

---

<a id="aicpu"></a>
## AICPU（AI CPU）

- **是什么（硬件/系统）**：NPU 器件上的 **AI CPU** 资源，可承担部分不适合或不放在 AI Core 上的任务（具体可调度范围随 CANN/产品型号而变）。与 Cube/Vector **不是**同一个执行引擎。
- **本报告字段 `aicpu_util_pct`**：解析自 `npu-smi info -t usages` 的 **Aicpu Usage Rate (%)**（公开命令输出字段名）。
- **读数事实**：本批体质里该字段常为 **0**。0 表示**该次采样时刻工具报告的占用率为 0**；不能单独证明「芯片没有 AICPU」或「探针失败」。
- **来源**：[`npu-smi -t usages`](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/24.1.RC3/re/npu/npusmi_020.html)；输出字段样例见 [Atlas 200 DK 笔记](https://blog.csdn.net/m0_37605642/article/details/137472243)。产品侧亦可见 AI CPU / control CPU 数量配置相关说明（如 [Atlas 200I DK A2 笔记中的 AI CPU / control CPU number](https://blog.csdn.net/m0_37605642/article/details/137585875)）。

---

<a id="ctrlcpu"></a>
## CtrlCPU（Control CPU / 控制 CPU）

- **是什么（硬件/系统）**：器件侧 **控制 CPU**。公开工具与 Profiling 文档使用 Ctrl CPU / Ctrlcpu 命名（例如 usages 里的占用率、以及部分产品 Profiling 对 Ctrl CPU 缓存等计数的说明）。
- **本报告字段 `ctrlcpu_util_pct`**：解析自 `npu-smi info -t usages` 的 **Ctrlcpu Usage Rate (%)**。
- **与 host 的关系**：这是 **NPU 侧** 工具字段，不是宿主机 `top` 里的 CPU%。与本报告 `launch_*`（host 侧 `perf_counter` + `synchronize`）不在同一观测面。
- **来源**：usages 字段样例 [同上](https://blog.csdn.net/m0_37605642/article/details/137472243)；Profiling 侧命名例：[Ctrl CPU 三级缓存使用量数据说明（CANN 文档）](https://www.hiascend.com/document/detail/zh/canncommercial/5046/devtools/auxiliarydevtool/atlasprofiling_16_0097.html)。

---

<a id="health-power"></a>
## `health_power_w` / `power_w`（功耗字段）

- **工具事实**：二者底层都可以来自 `npu-smi info -t power -i <id> -c <chip>` 一类查询；企业文档写明该命令用于 **query the power of a chip**（查芯片功耗）。解析字段为本仓库实现中的 **Real-time Power**（实时功耗读数，单位 W）。
- **本报告采样时刻（客观差异）**：
  - `health_power_w`：constitution 流程里 **较早的 health/轻载快照**（本批中位约百瓦级）。
  - `power_w`：多取 **vector_fma 等负载探针末轮** 同步采样（本批中位约八百余瓦）。
- **不是什么**：
  - 不是字段名里的 “health 分数”，也不是厂商标定的 TDP / `power_limit` 本身（若需墙值，应看工具是否单独暴露 power limit 字段；沐曦侧另有 `power_limit_w`）。
  - 两次读数的差值 **不等于**「降频幅度」或「浪费功率」的定义式；它们只是**不同时刻、不同负载**下的两次 Real-time Power。
- **温度同理**：`health_temp_c` 与 `board_temp_c` 同样是不同时刻的温度类读数，不能当成同一热稳态的两个别名。
- **来源**：[npu-smi power 命令说明](https://support.huawei.com/enterprise/en/doc/EDOC1100079295/c8f5b2f7/introduction-to-the-npu-smi-command-for-version-1011-1015)；本仓解析路径见 `METRIC_SEMANTICS_20260711.md` 对应节。

---

<a id="npu-smi"></a>
## `npu-smi`

- **是什么**：昇腾 NPU 的系统管理接口命令行（NPU System Management Interface）。
- **本报告常用子命令**：
  - `-t power`：芯片功耗
  - `-t temp` / board 相关温度查询（以本仓 `telemetry.py` 实际解析名为准）
  - `-t usages`：Aicore / Aicpu / Ctrlcpu / 内存与 HBM 带宽占用率等统计字段
- **来源**：[昇腾社区 npu-smi 信息查询](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/24.1.RC3/re/npu/npusmi_020.html)。

---

## 字段名 ↔ 词条（速查）

| 报告字段 | 先读词条 |
|----------|----------|
| `func_tflops` / `sustained_tflops` / `shape_sweep_peak_tflops` / BNMK | [Cube](#cube) |
| `vector_gflops` / `sfu_gflops` | [Vector](#vector) |
| `scalar_elems_per_s` | [Scalar](#scalar) |
| `mte_gbps` | [MTE](#mte) |
| `hbm_gbps` / `mem_bw_util_pct` / `hbm_mode_*` | [HBM](#hbm) |
| `cube_vector_tflops` | [Cube](#cube) + [Vector](#vector) |
| `aicore_util_pct` | [AI Core](#ai-core) |
| `aicpu_util_pct` | [AICPU](#aicpu) |
| `ctrlcpu_util_pct` | [CtrlCPU](#ctrlcpu) |
| `health_power_w` / `power_w` / `health_temp_c` / `board_temp_c` | [功耗字段](#health-power) |

---

## Sources

- [AI Core硬件架构剖析：Cube、Vector、Scalar三核协同机制（华为云社区）](https://bbs.huaweicloud.com/blogs/471408)（2025-12-24）
- [查询指定芯片统计信息 `npu-smi info -t usages`（昇腾社区）](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/24.1.RC3/re/npu/npusmi_020.html)
- [Introduction to the npu-smi Command · power 查询说明（Huawei Support）](https://support.huawei.com/enterprise/en/doc/EDOC1100079295/c8f5b2f7/introduction-to-the-npu-smi-command-for-version-1011-1015)
- [Ctrl CPU 三级缓存使用量数据说明（CANN Profiling）](https://www.hiascend.com/document/detail/zh/canncommercial/5046/devtools/auxiliarydevtool/atlasprofiling_16_0097.html)
- [npu-smi usages 输出字段样例（社区笔记）](https://blog.csdn.net/m0_37605642/article/details/137472243)
- [AI CPU / control CPU number 配置相关笔记](https://blog.csdn.net/m0_37605642/article/details/137585875)
