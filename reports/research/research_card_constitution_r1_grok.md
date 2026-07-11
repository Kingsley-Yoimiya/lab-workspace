# Ascend 910 卡体质筛查增强方案（R1 / Grok）

**版本**: R1-grok  
**日期**: 2026-07-11  
**前置**: [`research_card_constitution_r0.md`](research_card_constitution_r0.md)  
**基线**: [`card_screen_128.md`](../card_screen_128.md)（128×910_9392，Cube/HBM/sustained）  
**状态**: 缺口补齐设计；**不改默认筛查路径**；Stage C 已有骨架保持 `verdict_neutral`

---

## 0. 摘要

R0 已补上 **Vector 吞吐 / Scalar 代理 / launch 三桶延迟 / npu-smi 扩展字段骨架**。对照 NVIDIA DCGM/gpu-burn 实践与 Ascend DaVinci（Cube / Vector / Scalar / **MTE**）第一性原理，R0 仍缺五类高价值能力：

1. **互联与可靠性计数器**（PCIe replay / `pcie-err` / ECC / Health）——DCGM 红旗指标的 Ascend 代理  
2. **供电-热-频率闭环**（功耗爬坡、多传感器温、电压、AI Core 频率稳态）——解释「同 Cube 不同 step」的降频根因  
3. **片上数据通路**（MTE/DMA、Cube↔Vector 经 HBM/L2 的 fixpipe）——昇腾特有、torch 层 GEMM/FMA 测不到  
4. **launch 细拆 + burst**——现有 sync / tiny / host_overhead 未分离 device queue，也未测连续发射  
5. **假阳性控制与正交可视化**——128 卡报告已暴露 `temp_c≈2°C` 占位；无节点协变量则无法区分本征慢卡 vs 机箱散热/争用

本轮建议 **P0 必上**：遥测机上标定 + 功耗/频率稳态采样、launch 增强（三分解 + burst）、健康/PCIe 快照、假阳性残差协议、体质报告散点。其余进 P1/P2。

---

## 1. 第一性原理：R0 覆盖了什么、还缺什么？

### 1.1 卡体质再陈述

固定型号（Ascend910_9392）与软件栈（CANN + torch_npu）下，**本征体质** = 硅片/封装/HBM/供电/片上互联的可重复偏移。必须从测量中剥离：

| 混淆源 | 表现 | 剥离手段 |
|--------|------|----------|
| 节点散热/风道 | 同节点多卡同向偏慢、温度相关 | host 残差、节点内 z-score |
| 进程争用 | preflight 漏网、HBM/util 异常 | `require_idle` + contended 标记 |
| 驱动/固件混部 | launch/host_overhead 整节点抬升 | `comparison_group` / driver 分桶 |
| 遥测误解析 | 温度恒为 2°C 等 | **机上标定**（R0.5，本轮升级为 P0） |

### 1.2 部件矩阵：NVIDIA 有 / Ascend 有 / R0 状态

| 部件 | NVIDIA（DCGM / diag） | Ascend 代理 | R0 | R1 缺口 |
|------|----------------------|-------------|----|---------|
| Cube / Tensor | GEMM / targeted stress | `func_perf` / sustained | ✅ | 小 shape 效率（P2） |
| Vector | —（并入 SM） | `vector_fma_perf` | ✅ 骨架 | 与 Cube 正交散点（分析） |
| Scalar | — | `scalar_chain_perf` | ✅ 骨架 | cumsum 快路径校验（P1） |
| SFU | — | 仅 `sdc_sfu_identity` | ❌ 吞吐 | **P1** `vector_sfu_perf` |
| HBM 带宽 | mem bandwidth | `hbm` | ✅ | 与 MTE 解耦（P1） |
| **MTE / DMA / copy** | `MEM_COPY_UTIL` / PCIe RX/TX | MTE 搬 GM↔UB；`Memory Bandwidth Usage` | ❌ | **P1** `mte_copy_perf` |
| **Cube↔Vector 流水** | 同 SM 融合 | 分核，经 HBM/L2 | ❌ | **P1** `cube_vector_pipeline` |
| Launch | event / sync | sync / tiny / host_overhead | ⚠️ 粗 | **P0** 三分解 + burst |
| 功耗爬坡 | `targeted_power` | power + limit | ❌ | **P0** `power_ramp` |
| 多温度 | GPU + Memory temp | `temp` / `sensors` / HBM temp | ⚠️ 正则未标定 | **P0** 标定 |
| 电压 | NVML 部分型号 | `npu-smi -t volt` | ❌ | **P0** 解析 |
| 频率稳态 | SM clock + throttle bits | `board` freq；无 throttle 位 | ⚠️ | **P0** 时序 + 推断 |
| PCIe / 互联 | PCIe replay / NVLink | `pcie-err` / `topo` / HCCS | ❌ | **P0** 快照；HCCS 带宽 **P2** |
| 可靠性 | ECC remap / Xid | `ecc` / `err-count` / Health | ⚠️ health 有 ECC | **P0** 结构化快照 |
| 假阳性 | 集群相对 + 健康红旗 | — | 弱 | **P0** 残差协议 |
| 报告 | Grafana / 箱线 | 三指标箱线/热力 | 部分 | **P0** 正交散点 |

**关键洞察（来自 128 卡基线）**：Cube CV≈2.4%、HBM CV≈4.0%、sustained CV≈1.3%；慢尾主要由 **hbm** 与 **func** 拉出；温度读数无效 → **任何「热降频」结论目前不可信**。要解释「Cube 接近但训练 step 不一致」，必须上 Vector/launch **且** 可信遥测。

### 1.3 Ascend 特有：为何 MTE 与 Cube↔Vector 是体质维度

DaVinci AI Core 中，**MTE** 与计算引擎分队列并行，负责 GM↔本地缓冲（UB/L1…）；Cube 与 Vector 在 910B 系为**分核**，核间交换常走全局内存/L2，代价高。因此：

- 纯 `matmul` 主要压 Cube + 其 MTE 路径  
- 纯 `a*b+c` 主要压 Vector + 其 MTE 路径  
- **训练真实图**（GEMM → bias/act/residual）反复跨 Cube↔Vector → 测「融合流水」才能抓住封装/NoC/L2 体质差  

R0 的 `vector_fma_perf` / `func_perf` **正交但未覆盖跨单元通路**。

---

## 2. 缺口清单（P0 / P1 / P2）

### 2.1 P0 — 本轮发射前必须具备（设计 + 最小实现/标定）

| ID | 缺口 | 价值 | 依赖 | 预估单卡耗时 |
|----|------|------|------|--------------|
| **G1** | npu-smi **机上标定**：`temp/sensors/volt/usages/board/power/memory` 正则 + golden | 否则遥测全废；128 卡已见假温度 | 真机 1 卡 10 min | 0（标定脚本） |
| **G2** | **功耗爬坡 + 频率稳态**采样（sustained 期间） | 对标 DCGM targeted_power；区分供电弱 vs 算力弱 | G1 | +0（挂在 sustained） |
| **G3** | **launch 三分解增强 + burst launch** | 小算子密集图 / MoE 尾延迟 | 现有 `launch_latency` | +10–20s |
| **G4** | **健康红旗快照**：`ecc` / `pcie-err` / `err-count` / Health | 对标 DCGM remap/Xid/PCIe replay | npu-smi | +1–2s |
| **G5** | **假阳性控制协议**（节点残差 + 双 pass 规则） | 避免整机箱误杀 | 分析层为主 | 0 |
| **G6** | **体质报告**：正交散点 + host×metric 热力 + PERF_KEYS 扩展 | Stage C 数据否则沉没 | aggregate/plot | 离线 |

### 2.2 P1 — 高价值，紧随 128 体质复跑后实现

| ID | 缺口 | 说明 |
|----|------|------|
| **G7** | `mte_copy_perf` | 大块 `copy_` / 非融合读写，代理 MTE/DMA；与 `hbm_gbps` 对照 |
| **G8** | `cube_vector_pipeline` | GEMM + 立即 Vector epilogue，测 fixpipe/跨核 |
| **G9** | `vector_sfu_perf` | `exp`/`log`/`rsqrt` 吞吐（非仅 SDC） |
| **G10** | Scalar 代理加固 | elems 缩放线性度检验；可选 host-loop 对照 |
| **G11** | `aicore_freq_trace` | 独立短探针：空闲→满载→回落频率曲线 |

### 2.3 P2 — 有信息量但成本高 / 需多卡

| ID | 缺口 | 说明 |
|----|------|------|
| **G12** | HCCS / 卡间带宽 | 需 2+ device；体质筛查默认单卡，可另 case |
| **G13** | `cube_micro_gemm` | 小 MNK Cube 效率曲线 |
| **G14** | `AclValueGate` | 去 enqueue 的真 device 计时 |
| **G15** | 与训练 step time 关联 | R3 ground truth |

---

## 3. 各缺口探针设计（函数签名级）

### 3.1 G1 — 遥测标定（非 perf 探针，是 provider 硬化）

```python
# scripts/cluster/npu_smi_calibrate.py（建议新增，本轮可先手工跑）
def calibrate_npu_smi(device: int, out_dir: str) -> dict:
    """对每种 -t type 落 raw + 解析结果；产出 golden JSON。
    types = temp, sensors, volt, usages, power, board, memory, ecc, pcie-err, health
    loads = idle | gemm_burst | vector_burst
    """
```

**必须锁定的字段映射**（解析失败保持 `None`，禁止填默认 0/2）：

| 键 | 命令线索 | 用途 |
|----|----------|------|
| `temp_c` / `board_temp_c` / `hbm_temp_c` | `-t temp` / `-t memory` / `-t sensors` | 热图、假阳性 |
| `volt_*`（多轨） | `-t volt` | 供电异常 |
| `aicore_util_pct` / `aicpu_util_pct` / `mem_bw_util_pct` | `-t usages` | 争用/通路 |
| `aicore_freq_mhz` | `-t board` | 降频推断 |
| `power_w` / `power_limit_w` | `-t power` | 功耗爬坡 |
| `pcie_err_*` / `ecc_*` | `-t pcie-err` / `-t ecc` | 红旗 |

**判定**：标定前禁止把 `temp_c` 写入 slow_cause；标定后写入 `telemetry.py` 单测 golden。

---

### 3.2 G2 — `power_ramp`（可并入 sustained 遥测窗）

```python
def power_ramp(
    device: int,
    seconds: float = 45.0,
    sample_hz: float = 2.0,
    gemm_n: int = 8192,
    dtype: str = "bf16",
    telemetry=None,
) -> dict:
    """满载 GEMM 期间采样 power_w / aicore_freq_mhz / temp_*。
    返回:
      power_w: {p50, p95, max, t_to_90pct_s}   # 爬到 0.9*max 的时间
      freq_mhz: {p50, p05, cv_pct, drop_frac}  # 相对窗口内峰值的跌幅
      temp_c / hbm_temp_c: {start, end, delta}
      power_limit_headroom_w: limit - max_power  (若 limit 可得)
    """
```

- **kind**: perf，`order=35`（紧挨 sustained 后或作为 sustained 的附属记录）  
- **verdict_neutral**: True  
- **判读**：`t_to_90pct_s` 异常长 → 供电爬坡慢；`freq drop_frac` 大且 `hbm_temp` 高 → 热限；`drop` 大但温度正常 → 查 volt/limit  

对标 DCGM `targeted_power`：不要求打到 TDP 百分比硬阈值，先做**集群相对分布**。

---

### 3.3 G3 — launch 增强：三分解 + burst

现有 `launch_latency` 三桶：

```
sync_us          = 空 synchronize 往返
tiny_kernel_event= 1-elem add 的 event 时间
host_overhead    = wall - event
```

**缺口**：未区分 **device 侧排队**；未测 **连续发射**（真实训练是 burst enqueue）。

```python
def launch_latency(
    device: int,
    samples: int = 500,
    warmup: int = 50,
    tiny_elems: int = 1,
    burst_n: int = 64,          # NEW
    burst_samples: int = 100,   # NEW
) -> dict:
    """扩展返回（在原有键之外）:
      enqueue_us:        仅 host 侧：record 前 wall 到 op 返回（不 sync）
      queue_depth_us:    burst 内「首 kernel event 起点 → 末 kernel event 终点」/ burst_n
                         减去单次 tiny_kernel_event 的超额 = 排队/调度膨胀
      burst_launch: {
          n, samples,
          total_wall_us: {p50, p99},
          per_kernel_wall_us: {p50, p99},   # total/n
          per_kernel_event_us: {p50, p99},
          host_amortized_us: {p50, p99},    # (wall - event_span)/n
      }
    """
```

**三分解定义（R1 规范）**：

| 分量 | 测法 | 捕获什么 |
|------|------|----------|
| **host enqueue** | op 返回时刻 − 调用前（无 sync） | Python→ACL 发射 |
| **device queue** | burst 下 event span/n − 单发 event | 流队列/调度器堆积 |
| **sync** | 空 `synchronize()` | 空闲等待/驱动同步路径 |

**判读**：

- 单发 `host_overhead` 高、burst `host_amortized` 正常 → 固定 per-call 开销（可接受或驱动问题）  
- burst `queue_depth` / `per_kernel_event` 尾部高 → device 调度体质/干扰  
- 仅 `sync_us` 高 → 同步原语或同节点干扰  

仍 **verdict_neutral**；集群对 `burst_launch.per_kernel_wall_us.p99` 做 > median×1.5 嫌疑标记。

---

### 3.4 G4 — `health_counters` 快照

```python
def health_counters(device: int) -> dict:
    """npu-smi 一次快照，不做压测。
    返回:
      health: str | None
      ecc_corrected: int | None
      ecc_uncorrected: int | None
      pcie_err_count: int | None      # -t pcie-err 汇总
      err_count: int | None           # -t err-count
      raw_ok: bool
    """
```

- **kind**: health 或 perf order=5（最早）  
- **判定**：`ecc_uncorrected>0` 或 `pcie_err` 递增 → 可直接 **bad/suspect**（与体质 slow 正交的红旗）  
- 对标 DCGM：remap / PCIe replay / Xid ——「健康卡这些应为 0」

---

### 3.5 G5 — 假阳性控制（分析协议，非单一探针）

```python
def constitution_residualize(cards: list[dict], metrics: list[str]) -> list[dict]:
    """对每个 metric:
      1. cluster_z = (x - median) / mad   (或 / std)
      2. host_median = median(cards on same host)
      3. within_host_z = (x - host_median) / host_mad
      4. label:
           intrinsic_slow  if within_host_z < -z0 and cluster_z < -z0
           node_thermal    if cluster_z < -z0 and within_host_z > -z0/2
                           and (temp or hbm_temp) 同节点偏高
           contended       if preflight/contended or util 异常
    """
```

**双 pass 规则（发射 checklist 强制）**：

1. Pass A：全集群 `slow_frac`（建议体质键 0.15）出嫌疑名单  
2. Pass B：仅保留 `intrinsic_slow`；`node_thermal` 进「机箱复查」不进换卡清单  
3. 同节点 ≥50% 卡同向变慢 → **默认节点问题**，禁止批量标本征慢卡  

**协变量**（G1 标定后）：`hbm_temp_c`、`board_temp_c`、`power_ramp.freq.drop_frac`、host_id。

---

### 3.6 G6 — 报告与可视化（信息量最大的图）

| 图 | 轴 / 编码 | 回答的问题 |
|----|-----------|------------|
| **散点 1** | `func_tflops` vs `vector_gflops` | Cube/Vector 正交？象限慢卡 |
| **散点 2** | `hbm_gbps` vs `vector_gflops` | 带宽绑 vs 算力绑 |
| **散点 3** | `hbm_gbps` vs `launch_host_overhead_p99` | 内存弱 vs 调度弱 |
| **散点 4** | `sustained_tflops` vs `power_ramp.freq.drop_frac` | 稳态掉算力是否伴随降频 |
| **散点 5** | `burst_p99` vs `sync_p99` | 排队 vs 同步 |
| **热力** | host × device，色=within_host_z(metric) | 机箱风道/槽位效应 |
| **箱线** | 各体质键跨卡 | 尾部厚度 |
| **配对** | 同卡 `func` 与 `hbm` 的 rank 相关 | 是否同一批「弱卡」 |

派生键：

```text
cube_vector_ratio = func_tflops * 1e3 / vector_gflops
hbm_launch_discordance = z(hbm) - z(launch_host_p99)   # 大负：带宽差但调度不差
```

---

### 3.7 G7 — `mte_copy_perf`（P1）

```python
def mte_copy_perf(
    device: int,
    mb: int = 1024,
    iters: int = 50,
    warmup: int = 20,
    telemetry=None,
) -> dict:
    """dst.copy_(src) 大块；报告 gbps = 2*bytes/t（读写各一）。
    与 hbm(src*2) 对照：若 copy << hbm 算子带宽，嫌疑在 MTE/DMA 路径。
    """
```

---

### 3.8 G8 — `cube_vector_pipeline`（P1）

```python
def cube_vector_pipeline(
    device: int,
    n: int = 4096,
    iters: int = 40,
    epilogue: str = "bias_relu",  # bias_relu | residual_add | gelu
    dtype: str = "bf16",
    telemetry=None,
) -> dict:
    """y = epilogue(x @ w)：强制 Cube 产出后立刻 Vector。
    指标:
      pipeline_tflops: 按 GEMM FLOPs / 端到端 event 时间
      pipeline_efficiency: pipeline_tflops / func_tflops_same_n
      epilogue_overhead_ms: 相对纯 matmul 增量
    """
```

**判读**：`func` 正常但 `pipeline_efficiency` 低 → Cube↔Vector/fixpipe/L2 通路弱（昇腾特有体质）。

---

### 3.9 G9 — `vector_sfu_perf`（P1）

```python
def vector_sfu_perf(
    device: int,
    elems: int = 1 << 25,
    iters: int = 40,
    op: str = "exp",  # exp | log | rsqrt | sigmoid
    telemetry=None,
) -> dict:
    """返回 elems_per_s / gflops_equiv；与 vector_fma_perf 比 SFU/ALU 比。"""
```

---

### 3.10 G11 — `aicore_freq_trace`（P1，可与 G2 合并）

```python
def aicore_freq_trace(
    device: int,
    phases: tuple = ("idle", "gemm", "idle"),
    phase_s: float = 10.0,
    sample_hz: float = 2.0,
) -> dict:
    """返回 phases[{name, freq_p50, freq_p05, temp_end, power_p95}]"""
```

---

## 4. 与现有 Stage C 集成

### 4.1 执行顺序（enabled 时）

```
health_counters(5)          # P0 G4，可选
func_perf(10) → hbm(20) → sustained(30) [+ power_ramp 附属]
→ vector_fma_perf(40) → scalar_chain_perf(50)
→ launch_latency(60)        # 含 burst
→ mte_copy_perf(70)         # P1
→ cube_vector_pipeline(80)  # P1
→ vector_sfu_perf(85)       # P1
→ shape_sweep(90)
sdc_* …
```

### 4.2 文件级改动（R1 落地时；本轮文档为主）

| 文件 | 改动 |
|------|------|
| `probes/stage_c.py` | 扩展 `launch_latency`；新增 `power_ramp` / `health_counters` /（P1）三探针 |
| `probes/builtin.py` | 注册；`default_enabled=False` |
| `config.py` + `config.constitution128.yaml` | 参数节与开关 |
| `telemetry.py` | G1 标定后硬化正则；增加 `volt_*`、`pcie_err` 键 |
| `io/jsonl.py` | 汇总新字段 |
| `cluster/aggregate.py` | `CONSTITUTION_PERF_KEYS` + `constitution_slow_frac` + residualize |
| `screening.py` | **零改动**（registry 驱动） |

### 4.3 配置开关草案（写入 constitution profile）

```yaml
# R1 追加（默认仍可 false；constitution128 建议逐步打开）
launch_latency:
  samples: 500
  warmup: 50
  tiny_elems: 1
  burst_n: 64
  burst_samples: 100

power_ramp:
  seconds: 45.0
  sample_hz: 2.0

health_counters:
  enabled_via_probes: true

cluster:
  slow_frac: 0.2
  constitution_slow_frac: 0.15
  constitution_perf_keys:
    - vector_gflops
    - scalar_elems_per_s
    - launch_host_overhead_p99_us
    - burst_per_kernel_wall_p99_us
  residualize: true
  within_host_z0: 2.0

probes:
  health_counters:     {enabled: true}   # P0
  power_ramp:          {enabled: true}   # P0，或并入 sustained
  # mte_copy_perf / cube_vector_pipeline / vector_sfu_perf: P1 默认 false
```

### 4.4 JSONL 新 record

| record | 内容 |
|--------|------|
| `card` | 增补 burst/power_ramp/health 摘要字段 |
| `power_ramp` | 功耗/频率/温度时序摘要 |
| `health_counters` | 红旗快照 |
| `launch_latency` | 扩展 burst 块（已有 record 名可复用） |

---

## 5. 发射前 Checklist（128 卡 constitution）

### 5.1 代码与配置

- [ ] AFS 已 sync 含 `stage_c.py` / `builtin.py` / `telemetry.py` 的 CARD_SCREEN  
- [ ] `config.constitution128.yaml` 在 AFS；Stage C 三探针 `enabled: true`  
- [ ] `shape_sweep` / `bnmk_sweep` 关闭（控时）  
- [ ] `require_idle` + `idle-max-memory-mib` 生效  
- [ ] 日志目录：`logs/card-constitution-128-<timestamp>/`  

### 5.2 遥测（阻塞项）

- [ ] **G1 完成**：至少 1 卡上对 `temp/sensors/volt/usages/board/power/memory` 采 idle+gemm 各 60s  
- [ ] 确认 `temp_c` **不再**出现恒定 2°C 占位；否则报告中禁止热结论  
- [ ] `hbm_temp_c` / `aicore_freq_mhz` 至少一项在满载下有合理变化  
- [ ] 解析失败字段保持 `null`，不写 0  

### 5.3 探针与时长

- [ ] 单卡预估：Stage A + SDC + Stage C ≈ 既有 + 1–2 min；若开 `power_ramp` 独立段再 +45s  
- [ ] 先 **1 节点 16 卡** 冒烟，再 128 卡 fanout  
- [ ] launch 期间可临时降低 `telemetry_interval_s`（如 1.0）减 jitter  

### 5.4 假阳性

- [ ] 聚合使用 `constitution_slow_frac=0.15` + residualize  
- [ ] 同节点大面积变慢 → 标 `node_thermal`，不进换卡单  
- [ ] 红旗（ECC uncorrect / pcie-err）与 perf slow **分列**  

### 5.5 交付物

- [ ] `*.jsonl` + `*.cluster.json`  
- [ ] 至少产出：散点1（Cube vs Vector）、散点3（HBM vs launch p99）、host×device within_host_z 热力  
- [ ] 更新 `reports/card_screen_constitution_128.md`（复跑后）  

---

## 6. 本轮建议：P0 必上清单（执行序）

1. **G1 遥测机上标定**（阻塞所有热/压/频结论）  
2. **G4 health_counters**（低成本红旗，对标 DCGM）  
3. **G3 launch burst + 三分解**（扩展现有探针，改动面小）  
4. **G2 power_ramp / freq 稳态**（挂 sustained 遥测窗即可）  
5. **G5+G6 残差协议 + 正交散点报告**（否则 Stage C 白跑）  

P1（`mte_copy` / `cube_vector_pipeline` / `sfu_perf`）放在首轮 128 体质数据回来后，按「Cube 正常但 step 慢」的残差卡集定向加测。

---

## 7. Top 5 新发现（相对 R0）

1. **128 卡温度无效是 P0 级事故**：报告中 `temp_c≈2°C` 说明 R0 遥测增强在未标定前不可用于降频归因；`sensors`/`volt`/`usages` 比继续堆 perf 探针更优先。  
2. **昇腾体质的「第四单元」是 MTE + Cube↔Vector 通路**：分核架构下纯 Cube/纯 Vector 测不到训练图真实瓶颈；R0 的 `mixed_pipeline` 应从「可选」升为 **P1 核心**。  
3. **DCGM 红旗三件套在 Ascend 有廉价代理**：`pcie-err` + `ecc` + `err-count/Health`，无需自研压测即可过滤「不可靠卡」与「慢卡」。  
4. **launch 必须测 burst**：单发 host_overhead 低估连续小算子场景；`queue_depth` 才是 device 调度体质。  
5. **假阳性主因是节点协变量，不是阈值松紧**：应用 `within_host_z` 残差；同节点集体变慢默认机箱问题——否则会把风道问题当成换卡清单。

---

## 附录 A：研究资料锚点

- Ascend AI Core / MTE / Cube–Vector 分核：[Parallel Scan on Ascend AI Accelerators](https://arxiv.org/html/2505.15112v1)  
- npu-smi `-t` 类型含 `sensors`/`volt`/`pcie-err`/`usages` 等：[npu-smi 用法摘录](https://blog.csdn.net/m0_37605642/article/details/137585875)  
- usages 字段（Aicore / Aicpu / Memory Bandwidth）：[华为文档 usages](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/24.1.RC3/re/npu/npusmi_020.html)  
- DCGM 指标（PCIe replay、mem temp、power、ECC）：[MetricFire DCGM 文](https://medium.com/@MetricFire/why-gpu-monitoring-matters-tracking-utilization-power-and-errors-with-dcgm-603de3c4742b)  
- DCGM targeted_power / targeted_stress：[NVIDIA DCGM Diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html)  
- MindSpore Profiler 可采 PCIe / HBM / AICore 指标：[Performance Profiling Ascend](https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling_ascend.html)  
- 本地检索缓存：`tmp/research/ascend-constitution-r1.json`、`nvidia-constitution-r1.json`、`ascend-profiling-r1.json`

## 附录 B：与 R0 路线图对齐

| 版本 | R0 原计划 | 本 R1-grok 调整 |
|------|-----------|-----------------|
| R0.5 | 机上标定正则 | **升为 P0 阻塞** |
| R1 | aggregate 体质键 + 复跑 | + launch burst + health + power_ramp + 残差 |
| R2 | AclValueGate + mixed_pipeline | mixed_pipeline 提前到 P1；AclValueGate 仍 P2 |
| R3 | step time 关联 | 不变 |
