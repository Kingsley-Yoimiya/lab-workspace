# Ascend 910 卡体质筛查增强方案（R0）

**版本**: R0  
**日期**: 2026-07-11  
**范围**: `project/lab-workspace/projects/CARD_SCREEN` + 128 卡集群分布分析  
**状态**: 设计完成 + 最小探针骨架已落地（默认关闭）

---

## 0. 摘要

现有 CARD_SCREEN 已覆盖 **Cube 峰值（func_tflops）**、**HBM 带宽**、**稳态算力**、**BNMK shape 曲线** 与 **SDC 五类正确性探针**，但 Stage A 的 Vector/SFU 仅有 SDC、无吞吐指标；NPU 遥测仅解析 `temp_c` / `power_w`，缺少 NVIDIA 侧的多传感器降频根因能力。

本方案从 **昇腾 AI Core 三单元（Cube / Vector / Scalar）** 的第一性原理出发，设计 **Stage C 体质探针矩阵**、**launch/sync 延迟分布探针**、**npu-smi 遥测增强**，并与现有 probe registry 干净集成。**所有新探针默认 `enabled: false`，不破坏现有筛查路径。**

---

## 1. 第一性原理：我们要筛什么？

### 1.1 卡体质的定义

在固定硬件型号（如 Ascend910_9392）与固定软件栈（CANN + torch_npu）下，**卡体质**指单卡在**本征硅片/封装/供电/散热**差异导致的可重复性能偏移，而非：

- 节点散热环境差异（可通过 sustained + 温度遥测分离）
- 外部进程争用（preflight + contended 标记）
- 驱动/固件混部（comparison_group 分桶）

参考 NVIDIA 多传感器筛卡思路：**多部件、多指标、分布统计、集群相对比较**，而非单点绝对阈值。

### 1.2 昇腾 AI Core 与 torch 算子映射

| AI Core 单元 | 硬件职责 | torch 层代理算子 | 现有覆盖 | 缺口 |
|-------------|---------|-----------------|---------|------|
| **Cube** | 矩阵乘累加（GEMM/Conv 核心） | `matmul` / `@` | func_tflops, sustained, shape_sweep, sdc_cube_gemm | 小 shape Cube 效率曲线 |
| **Vector** | 向量 ALU（逐元素算术） | `a*b+c`, `relu`, `add` | sdc_vector_fma（仅正确性） | **吞吐 GFLOPS 分布** |
| **Scalar** | 标量/控制流/地址生成 | 依赖链、规约、小循环 | sdc_reduce_chain（正确性） | **串行延迟代理** |
| **SFU** | 超越函数 | `exp`, `log` | sdc_sfu_identity | 吞吐（低优先级） |
| **HBM** | 片外带宽 | `src*2` 读写 | hbm_gbps, sdc_mem_pattern | 多温度传感器 |
| **Launch 路径** | Host→Device 调度 | 空 sync / 极小 kernel | 无 | **p50/p99 延迟分布** |

**关键洞察**：128 卡实测（见 `reports/card_screen_128.md`）Cube/HBM/sustained 的 CV 仅 1.3–4%，**慢卡尾部主要由 func_tflops 与 hbm_gbps 拉出**；要解释「同规格卡为何 Cube 相近但训练 step 不一致」，需要 Vector 吞吐、Scalar 代理、launch 尾延迟作为**正交维度**。

### 1.3 设计约束

1. **不破坏默认行为**：新探针 `default_enabled=False` + config 显式开启。
2. **registry 自描述**：kernel 在 `stage_c.py`，壳在 `builtin.py`，驱动零改动。
3. **NPU 计时现实**：无 CUDA stream-value gate，计时方法为 `npu_event`（含 host enqueue）；launch 探针显式拆分 wall vs event。
4. **判定中立先行**：Stage C 首批探针 `verdict_neutral=True`，仅落库分布；慢卡判定仍由 aggregate `slow_frac` 扩展键完成。

---

## 2. 指标定义

### 2.1 性能体质指标

| 指标键 | 单位 | 定义 | 探针 |
|--------|------|------|------|
| `func_tflops` | TFLOPS | median(2N³ / t)，N=8192 bf16 GEMM | func_perf（已有） |
| `hbm_gbps` | GB/s | median(2×bytes / t)，1GB fp32 读写 | hbm（已有） |
| `sustained_tflops` | TFLOPS | 热稳态 GEMM 末段/中位数 | sustained（已有） |
| `vector_gflops` | GFLOPS | median(2×elems / t)，fp32 FMA | **vector_fma_perf** |
| `scalar_elems_per_s` | elems/s | median(elems / t)，cumsum 依赖链 | **scalar_chain_perf** |
| `cube_vector_ratio` | 无量纲 | func_tflops×1e3 / vector_gflops | 派生（分析层） |
| `shape_sweep_peak_tflops` | TFLOPS | shape 曲线峰值 | shape_sweep（已有） |

### 2.2 延迟体质指标

| 指标键 | 单位 | 定义 |
|--------|------|------|
| `launch_sync_p50_us` | µs | 空 `synchronize()` 往返 p50 |
| `launch_sync_p99_us` | µs | 空 sync p99（尾延迟敏感） |
| `launch_tiny_kernel_p50_us` | µs | 1-element `add_` 设备 event 时间 p50 |
| `launch_host_overhead_p99_us` | µs | wall − event p99（host 发射开销尾部） |

### 2.3 遥测体质指标（NPU 增强）

| 指标键 | 来源 | 状态 |
|--------|------|------|
| `temp_c` | npu-smi info | 已有 |
| `power_w` | npu-smi info | 已有 |
| `hbm_temp_c` / `mem_temp_c` | npu-smi info -t temp | **需在机上验证** 正则 |
| `aicore_util_pct` | npu-smi info -t usages | **需在机上验证** |
| `aicore_freq_mhz` / `sm_clock_mhz` | npu-smi info -t board | **需在机上验证** |
| `power_limit_w` | npu-smi info -t power | **需在机上验证** |
| `throttle_*` | 无直接等价 | 保持 None；用温度+频率+功耗推断 |

### 2.4 分布统计（128 卡分析规范）

对每张卡、每个指标，在集群内计算：

- **median / mean / std / CV%**
- **相对中位数偏差** = `(x − median) / median × 100%`
- **p5 / p50 / p95**（跨卡分布，非单卡 round 内）
- **正交散点**：func_tflops vs vector_gflops；hbm_gbps vs launch_host_overhead_p99

慢卡判定（扩展）：

```
final_verdict = slow  当  verdict==good 且  ∃k∈PERF_KEYS: x_k < median_k × (1 − slow_frac)
```

建议 `slow_frac=0.15` 用于体质维度（比默认 0.2 更敏感），通过 `cluster.constitution_slow_frac` 单独配置（R1 实现）。

---

## 3. 探针矩阵与函数签名

### 3.1 架构：最干净的扩展路径

```
screening.run_card()
  └─ registry.enabled_probes(backend, cfg)   # 按 kind→order 排序
       └─ builtin.XXX(PerfProbe|SdcProbe)
            ├─ resolve_params(cfg) → kwargs
            ├─ run(ProbeContext, **kwargs) → stage_*.py
            └─ evaluate(result) → Outcome
```

**新增探针 checklist**：

1. 在 `stage_c.py` 实现 kernel 函数（返回 `{"probe": ..., ...}` dict）
2. 在 `builtin.py` 用 `@register` 声明 `PerfProbe` 子类
3. 在 `config.py` DEFAULTS + `config.yaml` 添加参数节与 `probes.<name>.enabled`
4. 在 `io/jsonl.py` 提取顶层汇总字段（供 aggregate/plot）
5. （可选）扩展 `aggregate.PERF_KEYS`

### 3.2 Stage C 探针清单（已实现骨架）

#### P1: `vector_fma_perf` — Vector 吞吐

```python
def vector_fma_perf(
    device: int,
    elems: int = 1 << 26,
    iters: int = 50,
    dtype: str = "fp32",
    telemetry=None,
    warmup: int = 20,
) -> dict:
    """返回 gflops, rounds[{round, iter_ms, gflops, timing, temp_c, ...}]"""
```

- **负载**：`out = a * b + c`，fp32，elems=64M（≈256MB 三缓冲）
- **指标**：median gflops
- **kind**：perf，order=40，`verdict_neutral=True`

#### P2: `scalar_chain_perf` — Scalar 代理

```python
def scalar_chain_perf(
    device: int,
    elems: int = 1 << 24,
    iters: int = 50,
    telemetry=None,
    warmup: int = 10,
) -> dict:
    """返回 elems_per_s, rounds[...]"""
```

- **负载**：`torch.cumsum(x)` — 前缀依赖链，抑制 Vector 全宽并行
- **指标**：median elems/s（不作为 GFLOPS，避免与 Vector 混淆）
- **kind**：perf，order=50

#### P3: `launch_latency` — 发射/同步延迟

```python
def launch_latency(
    device: int,
    samples: int = 500,
    warmup: int = 50,
    tiny_elems: int = 1,
) -> dict:
    """返回 sync_us{p50,p99}, tiny_kernel_event_us{...}, host_overhead_us{...}"""
```

- **sync_us**：连续两次 `adapter.sync()` 间 host 计时
- **tiny_kernel_event_us**：`x.add_(1)` + `timing_context` event
- **host_overhead_us**：`wall − event`（NPU 上 host enqueue 可观测）
- **kind**：perf，order=60，无 telemetry 快照（探针本身即延迟）

### 3.3 规划中的探针（R1，未实现）

| 名称 | 目的 | 签名草案 |
|------|------|---------|
| `cube_micro_gemm` | 小 M/N/K Cube 效率 | `cube_micro_gemm(device, shapes: list[tuple], iters)` |
| `vector_sfu_perf` | SFU 吞吐 | `vector_sfu_perf(device, elems, ops=['exp','log','sin'])` |
| `mixed_pipeline` | Cube+Vector 流水线 | `mixed_pipeline(device, n, elems, ratio)` |
| `acl_value_gate_timing` | 去 enqueue 的 NPU 计时 | 需 `gates.py` 增加 `AclValueGate` |

### 3.4 现有探针与体质维度映射

| 已有探针 | 体质维度 | 备注 |
|---------|---------|------|
| func_perf | Cube 峰值 | 正确性门控 |
| hbm | 内存子系统 | |
| sustained | 热稳态 Cube + 降频 | slow_cause 依赖 NVIDIA throttle bits；NPU 降级 |
| shape_sweep | Cube shape 敏感度 | 判定中立 |
| sdc_* | 静默错误 | 与体质正交但高价值 |

---

## 4. launch/sync 延迟探针设计细节

### 4.1 为什么要单独测？

Ascend 训练场景中，小算子密集图（MoE routing、频繁 allreduce 前后）对 **p99 launch latency** 敏感。128 卡若 Cube 峰值接近但 p99 host overhead 偏高 2×，表现为「同等算力、step time 抖动大」。

### 4.2 三分解

```
Host 调用 torch op
  ├─ [host_overhead]  Python→ACL→stream enqueue
  ├─ [tiny_kernel]    设备执行 1-element kernel
  └─ [sync]           空同步等待（scheduler/device idle）
```

### 4.3 计时方法

| 桶 | CUDA | NPU |
|----|------|-----|
| sync | `perf_counter` 包裹 `sync()` | 同左 |
| tiny_kernel event | `cuda_event` 或 stream-value gate | `npu_event`（含 enqueue） |
| host_overhead | wall − event | wall − event |

**NPU 限制**：`timing/gates.py` 中 TODO 已记录 `aclrtValueWait`/`aclrtValueWrite` 可做 enqueue 剥离（**需在机上验证**）。R0 用 wall−event 近似。

### 4.4 输出与判读

- 单卡：报告 p50/p99/median
- 集群：对 `launch_host_overhead_p99_us` 做跨卡箱线图；偏离 median >30% 标记「调度异常嫌疑」
- **不直接判 bad**：避免驱动版本差异导致假阳性

---

## 5. 遥测增强：npu-smi 可挖字段

### 5.1 已落地（`NpuSmiProvider._parse_extended`）

在 `sample()` 中追加 best-effort 解析：

```bash
npu-smi info -t temp  -i <id>   # HBM/Board 温度
npu-smi info -t usages -i <id>  # AI Core / Vector 利用率
npu-smi info -t power -i <id>   # 功耗上限
npu-smi info -t board -i <id>   # 频率
```

字段映射到 `TELEMETRY_KEYS` 扩展项；`hbm_temp_c` 同步镜像到 `mem_temp_c`，`aicore_freq_mhz` 镜像到 `sm_clock_mhz`（兼容 slow_cause 结构）。

### 5.2 需在机上验证的项

| 命令/字段 | 预期 | 风险 |
|----------|------|------|
| `info -t usages` AI Core % | 与负载相关性 | 输出格式因 CANN 版本而异 |
| HBM Temperature | 热点降频解释 | 正则可能匹配错误传感器 |
| AICore Freq MHz | 替代 sm_clock | 字段名不确定 |
| throttle 等价物 | 降频根因 | 可能不存在，需用温度+频率推断 |
| `npu-smi watch` 流式 | 降采样开销 | 无 `-lms`，polling 频率需权衡 |

### 5.3 建议的机上标定流程（R0.5）

1. 空闲 / func_perf / vector_fma_perf 三种负载下各采 60s `npu-smi info -t *`
2. 固化正则 → 写入 `telemetry.py` 单元测试（golden file）
3. 将验证后的字段接入 `slow_cause.classify` NPU 分支

---

## 6. 与 CARD_SCREEN 集成点

### 6.1 文件级映射

| 组件 | 路径 | 本方案改动 |
|------|------|-----------|
| Kernel | `card_screen/probes/stage_c.py` | **新增** 3 函数 |
| Registry | `card_screen/probes/builtin.py` | **新增** 3 PerfProbe |
| 默认配置 | `card_screen/config.py` | **新增** 参数节 + probes 开关 |
| 运行配置 | `config.yaml` | **新增** Stage C 节（默认 false） |
| JSONL | `card_screen/io/jsonl.py` | **新增** 汇总字段 + round 记录 |
| 遥测 | `card_screen/telemetry.py` | **增强** NpuSmiProvider |
| 驱动 | `screening.py` | 无改动 |
| 聚合 | `cluster/aggregate.py` | R1 扩展 PERF_KEYS |
| 计时 | `timing/gates.py` | R1 AclValueGate |

### 6.2 探针执行顺序（enabled 时）

```
perf: func_perf(10) → hbm(20) → sustained(30)
    → vector_fma_perf(40) → scalar_chain_perf(50) → launch_latency(60)
    → shape_sweep(90)
sdc: sdc_cube_gemm → ... → sdc_reduce_chain
```

`launch_latency` 放 sustained 之后：避免热状态未稳时的 sync 噪声（可配置）。

### 6.3 JSONL 记录类型

| record | 内容 |
|--------|------|
| `card` | 新增 `vector_gflops`, `scalar_elems_per_s`, `launch_sync_p99_us`, ... |
| `vector_fma_round` | 每轮 vector 吞吐 |
| `scalar_chain_round` | 每轮 scalar 代理 |
| `launch_latency` | 整卡延迟分布摘要 |

### 6.4 128 卡分析工作流

```bash
# 1. 默认筛查（不变）
python launch.py ... --config config.yaml

# 2. 体质增强（节点 config 覆盖）
# probes.vector_fma_perf.enabled: true
# probes.scalar_chain_perf.enabled: true
# probes.launch_latency.enabled: true

# 3. 聚合 + 报告
python -m card_screen.cluster.aggregate result.jsonl --slow-frac 0.15
```

---

## 7. 判定逻辑与假阳性风险

### 7.1 判定策略（分阶段）

| 阶段 | 行为 |
|------|------|
| R0 | Stage C 全部 `verdict_neutral`，仅采集 |
| R1 | aggregate 扩展 `PERF_KEYS` + `constitution_slow_frac` |
| R2 | 多变量联合：`cube正常 & vector慢` → `slow_vector` 子类 |

### 7.2 假阳性风险

| 风险 | 成因 | 缓解 |
|------|------|------|
| vector_gflops 误低 | fp32 未达 Cube 强度，DVFS 空闲态 | sustained 后执行；可选 `--gemm-n` 预热 |
| scalar 误低 | cumsum 实现走 Vector 快路径 | 对比 elems 缩放线性度；R1 加 host-loop scalar |
| launch p99 尖刺 | OS jitter / npu-smi 轮询干扰 | 探针期间可降 telemetry 频率；采前 sync |
| NPU 遥测误解析 | 正则不匹配 | 机上标定；解析失败保持 None |
| 跨驱动不可比 | comparison_group 未含 driver | 已有 `driver_version` 分桶字段预留 |

### 7.3 真阳性目标

- **本征 Vector 弱**：func_tflops 正常、vector_gflops < median×0.85、无高温
- **调度/驱动异常**：launch_host_overhead_p99 > median×1.5
- **内存子系统弱化**：hbm_gbps 低 + hbm_temp_c 偏高（需遥测验证）

---

## 8. 建议的配置开关

### 8.1 `config.yaml` 片段（体质筛查 profile）

```yaml
# --- constitution profile: 在默认筛查基础上追加 Stage C ---
vector_fma_perf:
  elems: 67108864
  iters: 50
  dtype: fp32

scalar_chain_perf:
  elems: 16777216
  iters: 50

launch_latency:
  samples: 500
  warmup: 50

probes:
  func_perf:         {enabled: true}
  hbm:               {enabled: true}
  sustained:         {enabled: true}
  vector_fma_perf:   {enabled: true}   # 开启体质 Vector
  scalar_chain_perf: {enabled: true}   # 开启体质 Scalar
  launch_latency:    {enabled: true}   # 开启延迟分布
  shape_sweep:       {enabled: false}  # 可与体质并行，注意时长

cluster:
  slow_frac: 0.2
  # R1: constitution_slow_frac: 0.15
  # R1: constitution_perf_keys: [vector_gflops, scalar_elems_per_s, launch_host_overhead_p99_us]
```

### 8.2 环境变量

| 变量 | 作用 |
|------|------|
| `CARD_SCREEN_DISABLE_STREAM_VALUE_GATE` | CUDA launch 计时对照 |
| `CARD_SCREEN_STREAM_VALUE_GATE_SELFTEST` | 开发调试 |

### 8.3 耗时估算（单卡，Atlas 910）

| 探针 | 额外耗时 |
|------|---------|
| vector_fma_perf | ~30–60s |
| scalar_chain_perf | ~15–30s |
| launch_latency | ~5–10s |
| **合计** | ~1–2 min（在 Stage A 之后） |

---

## 9. 128 卡实测基线（参考）

来源：`reports/card_screen_128.md`（2026-07-10，128×910_9392）

| 指标 | median | CV% | 尾部特征 |
|------|--------|-----|---------|
| func_tflops | 292.8 | 2.4% | 最慢 −8.6% |
| hbm_gbps | 1242.6 | 4.0% | 最慢 −18.5%（离散度最大） |
| sustained_tflops | 307.1 | 1.3% | 热稳态一致性好 |

**推论**：R0 体质增强应优先解释 **hbm 尾部** 与 **Cube/Vector 正交性**；launch 延迟作为第二梯队排查维度。

---

## 10. 后续路线图

| 版本 | 内容 |
|------|------|
| R0 | 本文 + stage_c 骨架 + NpuSmi 增强（**当前**） |
| R0.5 | 机上标定 npu-smi 正则 + golden tests |
| R1 | aggregate 体质键 + constitution 报告 + 128 卡复跑 |
| R2 | AclValueGate 去 enqueue；mixed_pipeline 探针 |
| R3 | 与训练 step time 关联验证（ground truth 标注） |

---

## 附录 A：registry 扩展示例

```python
@register
class VectorFmaPerf(PerfProbe):
    name = "vector_fma_perf"
    order = 40
    default_enabled = False
    verdict_neutral = True

    def resolve_params(self, cfg: dict) -> dict:
        return dict(cfg["vector_fma_perf"])

    def run(self, ctx: ProbeContext, **p) -> dict:
        return stage_c.vector_fma_perf(ctx.device, telemetry=ctx.telemetry, **p)
```

## 附录 B：相关文件

| 文件 | 状态 |
|------|------|
| `reports/research/research_card_constitution_r0.md` | 本文 |
| `projects/CARD_SCREEN/card_screen/probes/stage_c.py` | 新增 |
| `projects/CARD_SCREEN/card_screen/probes/builtin.py` | 修改 |
| `projects/CARD_SCREEN/card_screen/config.py` | 修改 |
| `projects/CARD_SCREEN/config.yaml` | 修改 |
| `projects/CARD_SCREEN/card_screen/io/jsonl.py` | 修改 |
| `projects/CARD_SCREEN/card_screen/telemetry.py` | 修改 |
| `projects/CARD_SCREEN/card_screen/probes/__init__.py` | 修改 |
