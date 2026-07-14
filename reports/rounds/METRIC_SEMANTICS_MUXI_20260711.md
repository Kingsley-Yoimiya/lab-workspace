# 指标语义审计 · Muxi / MetaX C550 · 20260711

> **用途**：为沐曦侧体质筛查（`record=card`，`backend=metax`）与通信战役（`record=nccl_bench` / `nccl_p2p`）提供字段级语义字典。  
> **对标**：昇腾 [`METRIC_SEMANTICS_20260711.md`](METRIC_SEMANTICS_20260711.md)（同构字段名；后端/遥测命令不同）。  
> **代码锚点**：`projects/CARD_SCREEN/.../stage_a.py`、`stage_c.py`、`io/jsonl.py`、`MetaxAdapter` / `MxSmiProvider`；`scripts/cluster/nccl_torch_bench.py`、`nccl_p2p_bench.py`、`mfu_train_bench_nccl.py`。  
> **本批数据路径**：  
> - 体质 merged：`logs/muxi-constitution-20260711_232400-muxi-constitution128/results/constitution128.merged.jsonl`  
> - NCCL 本地：`logs/muxi-nccl-campaign-20260711/nccl-results/scale_*.jsonl`  
> - NCCL AFS：`/afs-a3-weight-share/montyyin/results/nccl-20260711_142129`  
> - P2P AFS：`…/results/nccl-p2p-20260711_150700`  
> **遥测统一命令**：`mx-smi`（温度/功耗/拓扑/MetaXLink）；**禁止**把昇腾 `npu-smi -t …` 套到沐曦。  
> **计时**：CUDA/MACA Event（`torch.cuda`），不是 NPU Event。  
> **架构对齐**：[`../research/METAX_ARCH_ALIGNMENT_20260711.md`](../research/METAX_ARCH_ALIGNMENT_20260711.md)（XCORE / MACA / MCCL / mx-smi）；报告附录 [`METAX_HARDWARE_GLOSSARY_20260711.md`](METAX_HARDWARE_GLOSSARY_20260711.md)。  
> **本批状态**：有 BNMK；board_temp / GPU util / XCORE clk 已落盘。

---

## 一、体质字段（`record=card`，backend=metax）

卡级字段由 `jsonl.py` 从各探针 `perf` 子树扁平化写入。探针逻辑与昇腾同构（torch 算子 + Event 计时），设备路径走 **CUDA/MACA**（`torch.cuda` Event），不是 NPU Event。

---

### `func_tflops`（方阵 GEMM / MetaX 主算力 TFLOPS）

- **是什么**：单卡方阵 GEMM 峰值吞吐代理（TFLOPS）。沐曦上反映 MetaX 主算力路径在方阵乘下的瞬时能力。
- **怎么得到**：探针 `func_perf`。`c = a @ b`（bf16）；理论 FLOPs = `2·N³`；CUDA/MACA Event 计时；卡级取各轮 tflops **中位数**。
- **关键参数**：`N=8192`，warmup=20，iters=50（`config.constitution128.yaml` + `launch_one.sh`：`--gemm-n 8192 --sdc-rounds 5 --sustained-s 30`）。
- **本批中位**：**279.9 TFLOPS**（覆盖 127/128）。
- **注意**：短窗峰值；与 `sustained_tflops` 对比看热稳态。键名可对照昇腾；沐曦实测走 MetaX/MACA 的 `a @ b`（GEMM）路径。勿把两侧数值直接当「谁更快」——硬件代差与探针参数需对齐后再比。

---

### `sustained_tflops`（稳态 GEMM TFLOPS）

- **是什么**：连续烤机后的稳态方阵 GEMM 吞吐（TFLOPS）。
- **怎么得到**：循环 `a @ b`，按时间窗聚合；卡级字段取 **最后一个时间窗**（非中位）。CUDA Event；~30s。
- **关键参数**：`--sustained-s 30`，每窗 50 次 GEMM，N=8192 bf16。
- **本批中位**：**280 TFLOPS**（127/128）。
- **注意**：末窗可能略高于/低于 func；含自洽检查。

---

### `hbm_gbps`（HBM（高带宽外存） GB/s）

- **是什么**：HBM（高带宽外存）有效带宽代理（GB/s）。`dst = src * 2.0`（fp32，含一次乘法，非纯 DMA）；流量按读+写计。
- **怎么得到**：探针 `hbm`；Event 中位；默认 1024MB，w20/i50。
- **关键参数**：1024 MB fp32。
- **本批中位**：**1469 GB/s**（127/128）；分布有双峰（部分节点掉到 ~1000–1050）。
- **注意**：慢卡簇多与整节点 HBM 掉速相关（冒烟已见 worker-7/14）。

---

### `vector_gflops` / `scalar_elems_per_s` / `mte_gbps` / `sfu_gflops` / `cube_vector_tflops`

- **是什么**：同构 JSON 键名遗留；沐曦实测走 MetaX/MACA 路径——`vector_gflops`=`a*b+c` FMA，`scalar_elems_per_s`=`cumsum`，`mte_gbps`=`Tensor.copy_`，`sfu_gflops`=`torch.exp`，`cube_vector_tflops`=GEMM+epilogue。
- **怎么得到**：对应 stage_c 探针；CUDA/MACA Event 中位。
- **关键参数**：Vector/SFU 64M fp32；Scalar 16M；DMA copy 512MB；pipeline N=4096 bf16。
- **本批中位**：Vector **122.2** GFLOPS；DMA copy **1387** GB/s；SFU **177.4**；GEMM+epilogue **195.2** TFLOPS；Scalar **1.209e+11** elems/s。
- **注意**：键名可对照昇腾，但不是 Ascend Cube/Vector/Scalar/MTE 硬件；见 `METAX_ARCH_ALIGNMENT_20260711.md`。`sfu_gflops` 按 1 op/元素计，实质偏 Gops/s；勿与 Vector FMA 按 2× 换算。

---

### Launch 延迟族（`launch_sync_*` / `launch_host_overhead_*` / `launch_burst_*`）

- **是什么**：空 sync、host−device 差分、burst 发射成本（µs）。
- **怎么得到**：`launch_latency`；CPU `perf_counter` + `torch.cuda.synchronize` / Event 差分。
- **关键参数**：samples=500，warmup=50，burst_count=64，timing_method=event。
- **本批中位**：sync p50 **2.69 µs**；host overhead p50 **184 µs**；burst p50 **1318 µs**。
- **注意**：CV 明显高于算力字段——更适合看尾延迟/驱动抖动，不宜作为主判定指标。

---

### 遥测：温度 / 功耗 / GPU 利用率 / XCORE 时钟

- **是什么**：轻载与负载阶段的功耗/温度快照，以及负载阶段的板温、GPU 利用率和 XCORE 时钟；兼容键分别为 `board_temp_c`、`aicore_util_pct`、`aicore_freq_mhz`。
- **怎么得到（沐曦）**：统一解析 `mx-smi`（`MxSmiProvider`）：`--show-temperature` 取板温，`--show-usage` 取 GPU util，`-j` 的 `clocks.XCORE.XCORE_CLK` 取 XCORE clk。`aicore_*` 只是昇腾兼容 JSON 键，不表示沐曦使用 AICore 或 `npu-smi`。
- **关键参数**：轻载开测快照（health≠健康分）；负载末常取 vector_fma 末轮回填。
- **本批中位**：health temp **38.5 °C**；health power **94.84 W**；负载 power **471 W**；power limit **550 W**；board temp **54 °C**；GPU util **98%**；XCORE clk **1500 MHz**（后三项均覆盖 127/128）。
- **注意**：
  - **不要**把 health 与 load power 相减当降频证据。
  - 本批 JSONL 中 `board_temp_c`、`aicore_util_pct`、`aicore_freq_mhz` **均已采集并落盘**；板温、利用率与时钟是采样时刻快照，不等同于完整热稳态或长期平均。
  - 相对昇腾：空闲/满载功耗量级不同（昇腾满载中位 ~872 W；沐曦 ~467 W / 550 W 墙）。

---

### 判定字段（冒烟 vs 体质，勿混用）

- **是什么**：相对集群中位的偏离 + 正确性/计时质量门控。
- **怎么得到**：冒烟 job `logs/muxi-card-screen-20260711_133828-muxi-smoke/`；体质判定来自 constitution merged。
- **关键参数**：相对中位阈值 + `max_rel_err`（冒烟与体质规则不完全相同）。
- **注意**：本批冒烟 good=106 / slow=19 / bad=1 / contended=2；体质 good=119 / contended=8 / bad=1。两者采样阶段与规则不同，勿混读。

---

### `shape_sweep_peak_tflops`（名义 shape sweep · 本批=BNMK max）

- **是什么**：名义「shape sweep 峰值 TFLOPS」；**名不副实**——本批关闭方阵 `shape_sweep`、以 `bnmk_sweep` / `gemm_bnmk_sample` 为主，该字段实为 **BNMK 各形状中位吞吐的最大值**。
- **怎么得到**：`jsonl.py` 用 `max(BNMK tflops)` 回填此键。
- **本批中位**：**286 TFLOPS**（覆盖见 constitution 摘要）。
- **注意**：读图时按 **BNMK peak** 理解，不是方阵 2 幂 shape sweep。

## 二、通信字段（NCCL / MCCL）

### `alg_bw_GBps`（算法带宽）

- **是什么**：业务字节 / 平均 collective 耗时 → GB/s。
- **怎么得到**：`nccl_torch_bench.py`；`torch.distributed` + **NCCL**（MetaX 栈常叠 MCCL）；CPU `perf_counter` + `torch.cuda.synchronize`。
- **关键参数**：sizes 1M–256M；fp32；`SOCKET_IFNAME=eth0`。
- **注意**：与 `bus_bw` 差一个 NCCL-tests 折算因子。

---

### `bus_bw_GBps`（总线带宽）

- **是什么**：NCCL-tests 同构折算后的总线带宽（GB/s）——扩展叙事的核心指标。
- **怎么得到**：由 alg_bw 按公式折算（见下）；本批保持率用各 rank 中位。
- **关键参数 / 公式**：
  - All-Reduce：`alg × 2(n−1)/n`
  - All-Gather / Reduce-Scatter：`alg × (n−1)/n`
  - Broadcast：`= alg`
  - 沐曦基线世界大小：单节点 **8 卡** → 保持率 **`bus_N / bus_8`**（昇腾用 `/bus_16`）。
- **本批关键事实**：w8@256MB All-Reduce bus 中位 ≈ **190.5 GB/s**；w16 ≈ **0.2563 GB/s**（保持率 ≈ **0.13%**）。
- **注意**：跨节点数字是「功能通、链路错」的证据，不是 MetaXLink 机内上限。

---

### Collective Ops / P2P

- **是什么**：`all_reduce` / `all_gather` / `reduce_scatter` / `broadcast`；P2P 为 `isend/irecv` 单向有效带宽（**不用** bus_bw 公式）。
- **怎么得到**：`nccl_torch_bench.py` / `nccl_p2p_bench.py`。
- **关键参数**：多机必须 `*_SOCKET_IFNAME=eth0`；大 world P2P 默认 ring。
- **注意**：P2P 机内 16M ≈30–33 GB/s、跨节点 ≈0.35 GB/s，可与 collective 交叉验证断崖，但定义不同勿 1:1 对齐。

---

## 三、MFU 字段

### 微基准 MFU（G8，`mfu_train_bench_nccl.py`）

- **是什么**：合成 dense/moe 训练步的 Model FLOPs Utilization = `achieved_tflops / (peak_per_gpu × world)`。
- **怎么得到**：峰值分母取体质 `func_tflops` 中位 **279.9 TFLOPS/卡**。
- **关键参数**：dense/moe 合成步；world 8→128。
- **注意**：dense@8=**26.7%**；dense@16+≈**0.2–0.3%**（通信打穿）；moe@8=**15.0%**。

### 真训练 MFU（G9，tiny GPT）

- **是什么**：Megatron `pretrain_gpt` 冒烟上的估算吞吐 / 同峰值分母。
- **怎么得到**：稳态 ms/iter → 聚合 TFLOPS / (peak×world)。
- **关键参数**：4L/H1024 + `local/unfused`；需 `nvcc`←`cucc` shim。
- **注意**：验收是「链路跑通」，不是冲高 MFU；勿与 G8 GEMM 微基准直接比高低。本批稳态 ~54 ms/iter → MFU ≈ **4.5%**。

---

## 四、拓扑与链路健康

| 数据 | 含义 | 底层 |
|------|------|------|
| `mx-smi topo` / MetaXLink | 机内互联类型（MX / SYS） | `probe_muxi_topology.sh` |
| NIC `mlx5_*` + `xscale_*` | IB/加速网卡可见 | 同 topo + `ibv_devinfo` |
| link-health 文本 | 每节点温度/ECC/PCIe/链路摘要 | `run_link_health_muxi.sh` |

**解读**：设备健康 + IB 设备可见 ≠ 当前 NCCL 已走 IB。本批集体通信实测仍走 eth0 socket。

---

## 五、易混对照速查（沐曦特化）

| 对比 | 正确读法 |
|------|----------|
| 沐曦保持率基线 vs 昇腾 | 沐曦用 **w8**；昇腾用 **w16** |
| `func_tflops` 沐曦 vs 昇腾 | 本批 ~279.9 vs 昇腾 ~292；需同 N/dtype/计时再比 |
| `hbm_gbps` 沐曦 vs 昇腾 | 沐曦中位更高（~1469 vs ~1240），但有整节点掉速簇 |
| `health_power` vs `power_w` | ~94.84 W vs ~471 W；墙 550 W |
| G8 MFU vs G9 MFU | G8=合成 GEMM 步；G9=真 Megatron tiny；目的不同 |
| P2P 机内 30 GB/s vs AR 190.5 GB/s | 协议不同；AR 是 collective bus 折算饱和区 |
| eth0 0.2 GB/s vs MetaXLink | 跨节点当前路径 vs 机内路径；不是「卡不行」 |
| `nvcc` shim | cu-bridge 用 `cucc`；Megatron fused_kernels 硬查 `nvcc` |
| BNMK / board_temp / GPU util / XCORE clk | **本批均有数据并已落盘**；BNMK 为 `gemm_bnmk_sample`，遥测来自 `mx-smi` |

---

## 六、与昇腾语义手册的关系

- **字段名、公式、图注话术**尽量同构，便于对照阅读。
- **差异只写清楚**：后端（NCCL/MCCL vs HCCL）、遥测（沐曦 `mx-smi` vs 昇腾 `npu-smi`）、世界大小基线（8 vs 16）、计时设备（CUDA Event vs NPU Event），以及同构 JSON 键在两种架构上的真实含义；本批 BNMK、GPU util、board_temp 与 XCORE clk 均有数据。

*文档版本：2026-07-11 · 由 `rewrite_meaning_mds_muxi.py` 现算中位 · 配套 [`CAMPAIGN_FINAL_MUXI_20260711.md`](CAMPAIGN_FINAL_MUXI_20260711.md) · [`FIGURE_PROVENANCE_MUXI_20260711.md`](FIGURE_PROVENANCE_MUXI_20260711.md)*
