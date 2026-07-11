# 指标语义审计 · 20260711

> **用途**：为体质筛查（`record=card`）与通信战役（`record=hccl_bench` / `hccl_p2p` / `hccl_inter_bw`）提供字段级语义字典。  
> **代码锚点**：`projects/CARD_SCREEN/card_screen/probes/stage_a.py`、`stage_c.py`、`io/jsonl.py`、`telemetry.py`；`scripts/cluster/nccl_torch_bench.py`（HCCL 同构）、`hccl_inter_bw_probe.py`。  
> **本批数据**：`logs/card-fillgap-20260711_140301/results/constitution128.merged.jsonl`（128 卡）；`logs/pipeline-comm-20260711_134811/`（HCCL）；`logs/inter-bw-20260711_141922/`（机内/机间）。  
> **遥测统一命令**：`npu-smi info -t {temp|power|usages|board} -i <card> -c <chip>`（禁止裸 `npu-smi info -i`）。

---

## 一、体质字段（`record=card`）

卡级字段由 `jsonl.py` 从各探针 `perf` 子树扁平化写入；明细行见 `gemm_round`、`vector_fma_round`、`gemm_sustained_sample`、`gemm_bnmk_sample` 等。

---

### `func_tflops`（Cube func TFLOPS）

- **是什么**：单卡 Cube 矩阵乘峰值吞吐代理（TFLOPS）。反映 Ascend Cube 主算力路径在方阵 GEMM 下的瞬时能力；越高表示该卡 Cube 越强。
- **怎么得到**（底层 API）：探针 `func_perf`（`stage_a.py`）。`torch` 算子 `c = a @ b`（bf16）；理论 FLOPs = `2·N³`；用 NPU Event（`timing_context`）逐轮计时；卡级字段取 **各轮 tflops 的中位数**（非均值、非末轮）。
- **关键参数**：`N=8192`，`dtype=bf16`，`warmup=20`，`iters=50`；含 golden 正确性校验（`max_rel_err < tol`）。
- **注意**：这是短窗峰值探针，不代表热稳态；与 `sustained_tflops` 对比可看降频/争用。计时在设备 Event 内，不含 host 排队。

---

### `sustained_tflops`（Sustained TFLOPS）

- **是什么**：连续烤机后的稳态 Cube 吞吐（TFLOPS）。用于观察热稳态、降频、长时间争用后的可持续算力，不是瞬时峰值。
- **怎么得到**（底层 API）：探针 `sustained_tflops`（`stage_a.py`）。循环 `a @ b`，每窗连续执行 `window` 次 GEMM，NPU Event 计时；卡级 `sustained_tflops` = **`samples` 列表最后一个时间窗的 tflops**（`last_tflops`），**不是中位数**。
- **关键参数**：`N=8192`，`bf16`，`seconds=30`（本批 `--sustained-s 30`），`window=50`；明细行 `record=gemm_sustained_sample` 含 `t_s`、`iter`、`tflops` 时序。
- **注意**：探针同时记录 `tflops_median` / `tflops_min` / `tflops_max`，但 **card 字段只用末窗**。末窗可能高于 `func_tflops`（热身后频率爬升）。含自洽 SDC 检查（`self_check_*`）。

---

### `hbm_gbps`（HBM GB/s）

- **是什么**：HBM 有效带宽代理（GB/s）。反映高带宽内存读+写通路在带计算负载下的健康度。
- **怎么得到**（底层 API）：探针 `hbm_bandwidth`（`stage_a.py`）。设备侧 `dst = src * 2.0`（fp32，**含一次逐元素乘法，非纯 DMA**）；流量按 **读+写** 计：`nbytes = 2 × elems × 4`；NPU Event 计时；卡级取各轮 **中位** gbps。
- **关键参数**：`mb=1024`（1024 MB 缓冲），`warmup=20`，`iters=50`。
- **注意**：与 `mte_gbps`（纯 `copy_`）和 `hbm_mode_*`（四模式）互补，勿跨模式比绝对值当优劣分。非 memcpy 基准，更接近「访存+轻算」混合路径。

---

### `vector_gflops`（Vector GFLOPS）

- **是什么**：Vector 单元 FMA 吞吐（GFLOPS）。代理 Ascend Vector 宽并行能力，**不是 Cube**。
- **怎么得到**（底层 API）：探针 `vector_fma_perf`（`stage_c.py`）。逐元素 `a * b + c`；按 **2 flops/elem**（mul + add）；NPU Event 计时；卡级取各轮 **中位** gflops。
- **关键参数**：`elems = 64M`（`1 << 26`），`dtype=fp32`，`warmup=20`，`iters=50`。
- **注意**：`board_temp_c`、`*_util_pct`、`power_w` 等负载遥测多从本探针 **末轮 round** 回填到 card（见下文 util/power 节）。

---

### `scalar_elems_per_s`（Scalar elems/s）

- **是什么**：长依赖串行链吞吐（元素/秒）。更贴近 Scalar/控制流与同步屏障，不是 SIMD 峰值。
- **怎么得到**（底层 API）：探针 `scalar_chain_perf`（`stage_c.py`）。`torch.cumsum(x, dim=0)`；`elems_per_s = elems / dt`；NPU Event 计时；卡级取 **中位**。
- **关键参数**：`elems = 16M`（`1 << 24`），`warmup=10`，`iters=50`。
- **注意**：量纲是 **元素/秒**，不是 GFLOPS；勿与 `vector_gflops` 直接比倍速。cumsum 有前缀依赖，反映串行瓶颈。

---

### `mte_gbps`（MTE copy GB/s）

- **是什么**：纯拷贝带宽（GB/s）。代理 MTE/DMA 搬运通路，用于拆分「算+访存」与「纯搬运」。
- **怎么得到**（底层 API）：探针 `mte_copy_perf`（`stage_c.py`）。`dst.copy_(src)`；流量按 R+W：`nbytes = 2 × elems × 4`；NPU Event 计时；卡级取 **中位**。
- **关键参数**：`mb=512`，`warmup=20`，`iters=50`。
- **注意**：与 `hbm_gbps`（`src*2` 带乘）对比可粗看「纯搬运 vs 访存+算」差距；与 `hbm_mode_seq_copy` 同为 copy 但缓冲大小与 warmup 不同。

---

### `cube_vector_tflops`（Cube+Vector pipeline TFLOPS）

- **是什么**：Cube GEMM 后接 Vector epilogue（scale + bias）的端到端吞吐（TFLOPS）。考察 Cube→Vector 流水线衔接。
- **怎么得到**（底层 API）：探针 `cube_vector_pipeline`（`stage_c.py`）。`c = a @ b; out = c * scale + bias`；FLOPs = `2·N³ + 3·N²`；NPU Event 计时；卡级取 **中位**。
- **关键参数**：`N=4096`，`bf16`，`warmup=20`，`iters=50`。
- **注意**：数值通常 **低于** 纯 `func_tflops`（N=8192 方阵），因含 epilogue 且 N 更小；勿与 func 直接比绝对值。

---

### `sfu_gflops`（SFU GFLOPS）

- **是什么**：特殊函数单元（SFU）吞吐。字段名带 `gflops`，但实现按 **1 op/元素** 计数，实质更接近 **Gops/s**，易误解为 FMA GFLOPS。
- **怎么得到**（底层 API）：探针 `vector_sfu_perf`（`stage_c.py`）。默认 `torch.exp(x)`；`gflops ≈ elems / dt / 1e9`（1 flop/elem）；NPU Event 计时；卡级取 **中位**。
- **关键参数**：`elems = 64M`，`op=exp`（亦支持 `rsqrt`），`warmup=20`，`iters=50`。
- **注意**：与 SDC 正确性探针不是一回事；勿与 `vector_gflops`（2 flops/elem FMA）按 2× 换算。

---

### `hbm_mode_seq_copy_gbps`（HBM 顺序 copy）

- **是什么**：HBM 多模式探针之一——顺序 `copy_` 带宽（GB/s）。
- **怎么得到**（底层 API）：探针 `hbm_modes_perf` 模式 `seq_copy`（`stage_c.py`）。`dst.copy_(src)`；流量 R+W；NPU Event 中位。卡级键 `hbm_mode_seq_copy_gbps`。
- **关键参数**：`mb=512`，`stride=16`（本模式不用 stride），`warmup=10`，`iters=30`。
- **注意**：与 Stage A `hbm_gbps`（`src*2`、1024MB）正交；四模式 **勿跨模式比绝对值** 当卡质分数。

---

### `hbm_mode_strided_gbps`（HBM 跨步 copy）

- **是什么**：跨步访问有效带宽（GB/s）；只计实际触碰元素的字节量。
- **怎么得到**（底层 API）：`src[::stride]` → `dst[::stride]` 的 `copy_`；`nbytes = 2 × (elems/stride) × 4`；NPU Event 中位。
- **关键参数**：`mb=512`，`stride=16`，`warmup=10`，`iters=30`。
- **注意**：对 stride/缓存行敏感；与 `seq_copy` 绝对值不可直接比高低。

---

### `hbm_mode_read_heavy_gbps`（HBM 读密集）

- **是什么**：读密集路径带宽代理（GB/s）。
- **怎么得到**（底层 API）：`src.sum()`；流量按 **只读** 计：`nbytes = elems × 4`；NPU Event 中位。
- **关键参数**：`mb=512`，`warmup=10`，`iters=30`。
- **注意**：含归约语义，不是纯 DMA；与 `write_heavy` 配对看读写不对称。

---

### `hbm_mode_write_heavy_gbps`（HBM 写密集）

- **是什么**：写密集路径带宽代理（GB/s）。
- **怎么得到**（底层 API）：`dst.fill_(1.0)`；流量按 **只写** 计：`nbytes = elems × 4`；NPU Event 中位。
- **关键参数**：`mb=512`，`warmup=10`，`iters=30`。
- **注意**：与 `read_heavy` 的流量会计方式不同，跨模式比较需谨慎。

---

### `launch_sync_p50_us` / `launch_sync_p99_us`（Launch sync 延迟）

- **是什么**：空设备 `synchronize()` 往返延迟的分位数（µs）。反映驱动/设备响应基线，与 kernel 发射无关。
- **怎么得到**（底层 API）：探针 `launch_latency`（`stage_c.py`）。CPU `time.perf_counter()` 包裹 `adapter.sync(device)`（**sync 本身不用 NPU Event**）；对 `samples` 次测量取 p50/p99。
- **关键参数**：`samples=500`，`warmup=50`（tiny-kernel 预热，sync 桶独立采样）。
- **注意**：这是 **CPU 包一层 sync** 的墙钟，不是 device event 内的空转；p99 看调度抖动尾延迟。

---

### `launch_host_overhead_p50_us` / `launch_host_overhead_p99_us`（Host 发射开销）

- **是什么**：Host 侧发射开销（µs）≈ **wall − device event**。衡量 enqueue/驱动排队相对设备执行时间的额外成本。
- **怎么得到**（底层 API）：极小核 `x.add_(1.0)`；`host_overhead_us = max(0, wall_us - event_us)`；需 `timing_method=event` 才有意义。
- **关键参数**：`tiny_elems=1`，`samples=500`，`warmup=50`。
- **注意**：若 backend 无 Event 计时（`timing_method=wall_sync`），host_overhead 桶为空或无意义。与 `launch_sync_*` 不同维度。

---

### `launch_burst_p50_us` / `launch_burst_p99_us`（Burst 总时延）

- **是什么**：连续 enqueue `burst_count` 个极小核后 **一次 sync** 的总时延分位数（µs）。看队列深度下的批量发射成本。
- **怎么得到**（底层 API）：CPU 计时：sync → 循环 `burst_count` 次 `add_` → sync；`burst_samples = samples // 10`。
- **关键参数**：`burst_count=64`，`samples=500`。
- **注意**：测的是 burst+单次 sync 的总墙钟，不是单核时延。

---

### `launch_burst_per_kernel_p50_us` / `launch_burst_per_kernel_p99_us`（Burst 每核摊销）

- **是什么**：`burst_total_us / burst_count` 的分位数（µs）。将突发总时延摊到每个 kernel。
- **怎么得到**（底层 API）：由 `burst_us` 派生：`burst_per_kernel_us = total_us / burst_n`。
- **关键参数**：`burst_count=64`。
- **注意**：摊销假设各核同质；实际队列合并可能使 per-kernel 低于单次发射之和。

---

### `health_temp_c`（Health 温度）

- **是什么**：健康/开测路径温度快照（°C）。偏 **轻载或探针启动时** 的温度读数。
- **怎么得到**（底层 API）：探针 `health_counters`（`stage_c.py`）或 `health` 快照；`npu-smi info -t temp -i <card> -c <chip>` → `parse_npu_temps`。
- **关键参数**：开测阶段单次查询；与负载探针无固定时间对齐。
- **注意**：与 `board_temp_c`（负载末轮遥测）**不同时刻、不同工况**；勿混读为同一热状态。

---

### `health_power_w`（Health 功耗）

- **是什么**：健康/轻载路径实时功耗（W），常近 **空闲**（本批中位约 168 W）。
- **怎么得到**（底层 API）：`npu-smi info -t power -i <card> -c <chip>` → `parse_npu_power` → `Real-time Power`。
- **关键参数**：`health_counters` 探针在 constitution 流程早期执行。
- **注意**：**不要**与 `power_w`（负载末，可达 800–900 W）直接相减当降频证据；两者工况完全不同。

---

### `board_temp_c`（板温）

- **是什么**：板/NPU 温度（°C），来自负载探针期间的遥测缓存。
- **怎么得到**（底层 API）：`npu-smi info -t temp`（`telemetry.py` `NpuSmiProvider.sample`）解析 `Board/NPU Temperature`；卡级经 `jsonl.py` 的 `_last_telemetry_round(vf.rounds, "board_temp_c")` 回填。
- **关键参数**：本批多取 **vector_fma_perf 末轮** 遥测，非 sustained 烤机峰值时刻。
- **注意**：本批中位约 66 °C，显著高于 `health_temp_c`（约 40 °C），反映负载工况差异。

---

### `aicore_util_pct`（AICore 利用率）

- **是什么**：AICore 利用率（%），瞬时率而非时间平均。
- **怎么得到**（底层 API）：`npu-smi info -t usages -i <card> -c <chip>` → 解析 `Aicore Usage Rate (%)`；卡级取 **vector_fma 末轮** round 附带的遥测。
- **关键参数**：与 vector 探针同次采样；非 30s sustained 全程平均。
- **注意**：本批中位约 92%；是探针末瞬时值，不能代表整机日平均利用率。

---

### `aicpu_util_pct`（AICPU 利用率）

- **是什么**：AICPU 利用率（%）。
- **怎么得到**（底层 API）：`npu-smi info -t usages` → `Aicpu Usage Rate (%)`；回填路径同 `aicore_util_pct`。
- **关键参数**：vector_fma 末轮遥测。
- **注意**：本批常全 **0**；为 0 不表示探针失败，可能是该路径未忙。

---

### `ctrlcpu_util_pct`（CtrlCPU 利用率）

- **是什么**：CtrlCPU 利用率（%）。
- **怎么得到**（底层 API）：`npu-smi info -t usages` → `Ctrlcpu Usage Rate (%)`；回填路径同 vector 末轮。
- **关键参数**：vector_fma 末轮遥测。
- **注意**：本批中位约 7%；与 `launch_host_overhead_*` 散点图可看控制面争用关联（非因果）。

---

### `mem_bw_util_pct`（HBM 带宽利用率）

- **是什么**：HBM Bandwidth Usage Rate（%），内存带宽占用瞬时率。
- **怎么得到**（底层 API）：`npu-smi info -t usages` → `HBM Bandwidth Usage Rate (%)`；回填路径同 vector 末轮。
- **关键参数**：vector_fma 末轮遥测。
- **注意**：本批中位约 18%；与 `hbm_gbps` 探针结果不同源，勿数值对齐。

---

### `power_w`（负载功耗）

- **是什么**：负载探针时段实时功耗（W），常为数百瓦至近千瓦（本批中位约 872 W）。
- **怎么得到**（底层 API）：`npu-smi info -t power` → `Real-time Power`；卡级 `_last_telemetry_round(vector_fma.rounds, "power_w")`。
- **关键参数**：取 vector_fma **末轮** 同步采样。
- **注意**：与 `health_power_w`（~百瓦级健康快照）工况不同；报告里两套功耗 scatter 并存时勿混读。

---

### `shape_sweep_peak_tflops`（Shape sweep 峰值 · 本批回填）

- **是什么**：名义「方阵 shape sweep 峰值 TFLOPS」；**本批 constitution128 关闭 `shape_sweep`、开启 `bnmk_sweep`**，故该字段 **实为 BNMK 各形状中位吞吐的最大值**，名不副实。
- **怎么得到**（底层 API）：`jsonl.py`：`sw_tflops` 来自 `shape_sweep.samples`；若为空则用 `bnmk_sweep.samples` 的 `tflops` 列表；`shape_sweep_peak_tflops = max(sw_tflops)`。
- **关键参数**：本批 10 个 BNMK shape × 128 卡；最高中位 shape 约 `B1_M16384_N1024_K1024` ≈ 310.5 TFLOPS。
- **注意**：读直方图/热力图时应按 **BNMK peak** 理解，不是 128→16880 方阵 2 幂 sweep。真 sweep 明细在 `record=gemm_shape_sample`（本批 0 行）。

---

## 二、BNMK 形状探针（`record=gemm_bnmk_sample`）

### BNMK 字段族（`B`, `M`, `N`, `K`, `label`, `tflops`, …）

- **是什么**：按显式 `(B,M,N,K)` 做 batched GEMM 的训练层形状代理吞吐。`label` 形如 `B1_M16384_N1024_K1024`；`tflops` 为该 shape 在单卡上的稳态窗口 **中位** TFLOPS（bf16）。
- **怎么得到**（底层 API）：探针 `gemm_bnmk_sweep`（`stage_a.py`）。`c = a[B,M,K] @ b[B,K,N]`；FLOPs = `2·B·M·N·K`；每 shape 多窗 NPU Event 计时，取窗口 tflops **中位** 写入 sample；`peak_tflops` 为同 shape 窗内最大。
- **关键参数**：`dtype=bf16`，`layout=NN`，`warmup=10`，`window=50`，`min_seconds=2`，`min_windows=3`，`max_seconds=6`；本批 10 shape × 128 卡 = 1280 行。
- **注意**：与 `func_tflops`（N=8192 方阵）不可直接比；microbatch（如 `B1_M8_N8192_K8192`）会极低。`shape_sweep_peak_tflops` 取各 shape `tflops` 的 max，非 `peak_tflops` 字段的 max。

---

## 三、通信字段

### `alg_bw_GBps`（算法带宽）

- **是什么**：集体通信 **算法视角** 带宽（GB/s）= 参与通信的业务字节量 / 平均单次 collective 耗时。
- **怎么得到**（底层 API）：`hccl_torch_bench.py`（Ascend HCCL）或对标 `nccl_torch_bench.py`；`torch.distributed` + HCCL/NCCL；CPU `perf_counter` 包 `iters` 次 op 后 `synchronize`；`alg_bw = data_bytes / avg_s / 1e9`。
- **关键参数**：本批 `dtype=fp32`，消息 `1M–256M`，`warmup=5`，`iters=20`；`record=hccl_bench` 每 rank 一行。
- **注意**：`data_bytes` 语义随 op 变化（如 all_gather 用 per-rank chunk）；与 `bus_bw` 差一个 NCCL-tests 折算计因子。

---

### `bus_bw_GBps`（总线带宽）

- **是什么**：按 **NCCL-tests 同构公式** 把多跳通信折成可与物理链路对比的总线带宽（GB/s）。扩展叙事的核心指标。
- **怎么得到**（底层 API）：在 `alg_bw` 基础上按 op 缩放：
  - `all_reduce`：`bus_bw = alg_bw × 2×(n−1)/n`
  - `all_gather` / `reduce_scatter`：`bus_bw = alg_bw × (n−1)/n`
  - `broadcast`：`bus_bw = alg_bw`
- **关键参数**：`world_size` = n；与 `nbytes`、`op`、`rank` 同存一行。
- **注意**：保持率定义为 **`bus_bw_N / bus_bw_16`**（同 op、同 nbytes，通常 256MB），**不要**再除以 `N/16`。本批 w128@256MB：AR 89.4%，AG 54.0%，RS 46.4%，Broadcast 86.8%。

---

### HCCL Collective Ops（`all_reduce` / `all_gather` / `reduce_scatter` / `broadcast`）

- **是什么**：标准 `torch.distributed` 集体通信原语在 HCCL 后端上的微基准结果族；每行含 `op`、`world_size`、`rank`、`nbytes`、`avg_s`、`alg_bw_GBps`、`bus_bw_GBps`。
- **怎么得到**（底层 API）：`dist.all_reduce` / `all_gather` / `reduce_scatter` / `broadcast`；计时含 `torch.npu.synchronize()`（或 cuda 对标）；world 16→32→64→128 分档扫 size。
- **关键参数**：`HCCL_BUFFSIZE` 等环境由 launch 脚本注入；本批四算子 × 四档 world × 四档 size。
- **注意**：all_gather/reduce_scatter 的缓冲按 world 切分；读图时 y 轴多为同 (op,world,nbytes) **跨 rank 的 bus_bw 中位**。

---

### P2P（`record=hccl_p2p`）

- **是什么**：点对点 **单向** 有效带宽（GB/s），`isend/irecv` 严格串行单对测量；**不使用** bus_bw 公式。
- **怎么得到**（底层 API）：`hccl_p2p_bench.py`；`torch.distributed` P2P + HCCL；边类型含 **ring** / **star**（大 world 默认 ring）；字段含 `src`、`dst`、`nbytes`、`bw_GBps`、`world_size`、`edge_kind`。
- **关键参数**：本批 world 16 与 128 两档；size 含 64K–16M；与 collective _campaign 分文件存放 `p2p-results/`。
- **注意**：P2P 带宽与 `bus_bw` 量级可交叉验证，但定义不同；慢边 TopK 图看个别 `(src,dst)` 掉队。

---

### `inter` / `intra`（机间 vs 机内 P2P）

- **是什么**：**intra** = 同节点两卡（走机内 HCCS/SIO）；**inter** = 跨节点、**同 local_rank 对齐** 的一对（走机间 RoCE/UB 平面）。均为 HCCL P2P 有效带宽（GB/s）。
- **怎么得到**（底层 API）：`hccl_inter_bw_probe.py`；`torch.distributed.isend/irecv`；全员 `barrier` 下严格串行单对；默认 **流水线单向**（`inflight=4`）；`record=hccl_inter_bw`，字段 `kind` ∈ `{intra, inter}`，`bw_GBps`，`nbytes`，`src`，`dst`。
- **关键参数**：`torchrun --nnodes=8 --nproc_per_node=16`；`HCCL_BUFFSIZE=2048`；sizes `1M–256M`；汇总 `summarize_inter_bw.py` 取 recv 侧中位。
- **注意**：本批大包饱和区 intra≈122 GB/s、inter≈119 GB/s（256M 中位），互比值接近 1；与 AllReduce `bus_bw` **不可数值等同**，只能量级对照。可选 `--pingpong` 用 RTT/2 交叉验证。

---

## 四、遥测与落库约定

| 遥测类型 | `npu-smi` 子命令 | 解析函数 | 典型 card 字段 |
|----------|------------------|----------|----------------|
| 温度 | `-t temp` | `parse_npu_temps` | `board_temp_c`, `health_temp_c` |
| 功耗 | `-t power` | `parse_npu_power` | `power_w`, `health_power_w` |
| 利用率 | `-t usages` | `parse_npu_usages` | `aicore_util_pct`, `aicpu_util_pct`, `ctrlcpu_util_pct`, `mem_bw_util_pct` |
| 板信息 | `-t board` | `parse_npu_versions`, freq regex | `aicore_freq_mhz`（本批常空） |

- 卡/chip 映射：`device_to_card_chip`（默认 `auto`：`card=device//2`, `chip=device%2`）。
- 负载遥测附加在探针 round 上，**不进入计时窗**（`_attach_telemetry` 在 `timing_context` 外读缓存）。
- 合流后读图一律用 `constitution128.merged.jsonl`；BNMK 只计 merged 避免子目录 jsonl 双计。

---

## 五、易混对照速查

| 对比 | 正确读法 |
|------|----------|
| `func_tflops` vs `sustained_tflops` | func=短窗中位峰值；sustained=30s **末窗**（非中位） |
| `hbm_gbps` vs `mte_gbps` | hbm=`src*2` 含乘、1024MB；mte=纯 `copy_`、512MB |
| `health_power_w` vs `power_w` | health≈空闲快照；power≈vector 负载末，差数百瓦 |
| `health_temp_c` vs `board_temp_c` | health≈开测轻载；board≈vector 末轮负载 |
| `shape_sweep_peak_tflops` vs BNMK 图 | 本批 peak 字段 = **max(BNMK tflops)**，非真 shape sweep |
| `alg_bw` vs `bus_bw` | bus 按 NCCL-tests 公式折算；保持率用 bus 比 bus |
| P2P `bw_GBps` vs collective `bus_bw` | 不同测量协议，勿数值 1:1 对齐 |
| `sfu_gflops` vs `vector_gflops` | sfu=1 op/elem；vector=2 flops/elem FMA |

---

*文档版本：2026-07-11 · 对应战役 stamp `20260711` · 修订时请同步 `rewrite_meaning_mds.py` 的 `SEM` 字典。*
