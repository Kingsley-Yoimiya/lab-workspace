# Ascend 910 卡体质筛查增强方案（R2 / Grok）

**版本**: R2-grok  
**日期**: 2026-07-11  
**前置**: [`research_card_constitution_r0.md`](research_card_constitution_r0.md) · [`research_card_constitution_r1_grok.md`](research_card_constitution_r1_grok.md)  
**基线**: [`card_screen_128.md`](../card_screen_128.md) · [`card_screen_diff_r1.md`](../rounds/card_screen_diff_r1.md)  
**代码锚点**: `projects/CARD_SCREEN`（stage_a/b/c、`NpuSmiProvider`、`config.constitution128.yaml`、`run_card_constitution_128.sh`）  
**研究缓存**: `tmp/research/{ascend,nvidia}-constitution-r2.json`、`ascend-profiling-r2.json`  
**状态**: 相对 R0/R1 **修正 + 收窄发射集**；文档为主，不改默认筛查路径

---

## 0. 摘要

R1 已正确指出：遥测未标定、DCGM 红旗代理、launch burst、功耗-频率闭环、节点残差、MTE/Cube↔Vector 通路。R2 用 **128 卡实数 + 正则复现 + npu-smi/DCGM/msprof 对照** 把「仍缺或需修正」收敛为：

1. **`temp_c≈2°C` 不是占位符，是 `I2C` 误匹配**——阻塞一切热/降频结论，且会污染 slow_cause。  
2. **扩展遥测字段名与真机 usages 不对齐**——`vector_util_pct` 正则大概率永远空；真字段是 Aicore / Aicpu / Ctrlcpu / Memory Bandwidth。  
3. **R1 P0 过宽**——发射前最小充分集应砍到「修温 + 健康红旗 + Stage C 已开 + 残差协议 + 遥测门禁」；burst / 独立 power_ramp / 全套散点可延后。  
4. **HBM 慢尾已呈节点聚集**（worker-4 / worker-1）——残差不是分析锦上添花，是换卡清单的前置条件。  
5. **BNMK 冒烟高 CV vs 全量近零 CV**——体质探针必须用足够 warmup/iters；短跑噪声会制造假慢卡。

---

## 1. 新发现（相对 R0 / R1）

### 1.1 `temp_c≡2.0` 根因：正则吃掉 `I2C`

**证据**：

- 128 卡 JSONL 全样本 `temp_c` 集合仅为 `{2.0}`（`logs/card-screen-128-20260710_224218`）。  
- `NpuSmiProvider.sample` 使用 `_first_float(info, r"(\d+(?:\.\d+)?)\s*C")`。  
- 本地复现：只要输出里出现 `I2C`，该正则**优先**匹配到 `2C`，真实 `Temperature: 45 C` 永远读不到。

```text
'I2C check: pass\nTemperature: 45 C'  →  解析得 2
'I2C' 单独                              →  匹配 2C
```

R0/R1 称「占位 / 未标定」**方向对、归因不够**：问题不是「没采到」，而是「采到了错误的第一个 `\d+C`」。  
`npu-smi info --help` 的 type 列表本身含 `i2c_check`，输出中出现 `I2C` 字符串是常态（见 [npu-smi 用法](https://blog.csdn.net/m0_37605642/article/details/137585875)）。

**修正原则**（P0）：

| 禁止 | 必须 |
|------|------|
| 裸匹配任意 `\d+\s*C` | 只匹配带标签字段：`Temperature` / `HBM` / `Board` / `AiCore` 等 |
| 解析失败填 0 或 2 | 保持 `null` |
| 未过门禁就写 slow_cause 热结论 | `temp_c < 10` 或全集群恒等 → **整轮遥测作废标记** |

---

### 1.2 usages 真字段 ≠ R0/R1 假设的 Vector%

华为文档 `npu-smi info -t usages` 输出为（[Atlas usages 文档](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/24.1.RC3/re/npu/npusmi_020.html)）：

- `Aicore Usage Rate(%)`  
- `Aicpu Usage Rate(%)`  
- `Ctrlcpu Usage Rate(%)`  
- `Memory Bandwidth Usage Rate(%)`  

**没有**名为 `Vector` 的利用率行。因此当前：

```python
"vector_util_pct": r"Vector.*?(\d+(?:\.\d+)?)\s*%"   # 大概率恒为 None
```

应改为映射：

| 键 | 来源 | 体质用途 |
|----|------|----------|
| `aicore_util_pct` | Aicore Usage | 满载是否真吃满 Cube |
| `aicpu_util_pct` | Aicpu Usage | 主机侧 AI CPU 争用 |
| `ctrlcpu_util_pct` | Ctrlcpu Usage | 控制面忙闲（launch/驱动代理） |
| `mem_bw_util_pct` | Memory Bandwidth Usage | 与 `hbm_gbps` 正交的通路占用 |

R1 的「多传感器」方向对，但 **字段字典必须按 usages 真表重写**，否则标定脚本会「绿过」却采不到 Vector%。

---

### 1.3 DCGM 多部件思路 → Ascend 代理（R2 精修映射）

对照 [DCGM Diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html) 与 [DCGM 指标实践](https://medium.com/@MetricFire/why-gpu-monitoring-matters-tracking-utilization-power-and-errors-with-dcgm-603de3c4742b)：

| DCGM / gpu-burn 能力 | Ascend 代理 | R1 状态 | R2 修正 |
|---------------------|-------------|---------|---------|
| 相对均值容差 `gflops_tolerance_pcnt` | `constitution_slow_frac` + 同 comparison_group | 提了 0.15 | **发射即用**；先相对、后绝对 |
| targeted_power 达 TDP 比例 | power_w / power_limit_w + 爬坡时间 | G2 独立探针 | **先挂 sustained 窗**；独立探针可延后 |
| PCIe replay / Xid / ECC remap | `pcie-err` / `err-count` / `ecc` / Health | G4 快照 | **压测中增量**也要记（见 1.5） |
| GPU + Memory 双温 | board/AiCore temp + HBM temp（`-t temp`/`sensors`） | 未标定 | **先修 I2C，再标标签字段** |
| throttle bits | 无直接位；用 freq drop + temp + power_limit 推断 | 正确 | 温无效时**禁止推断** |
| MEM_COPY_UTIL | `mem_bw_util_pct` + 未来 `mte_copy_perf` | P1 | util 可先上（零成本） |
| SM Stress / Targeted Stress | func + sustained + vector_fma | Stage C 已开 | 保持；勿再堆同质 GEMM |

gpu-burn 在现代数据中心已被厂商倾向用 DCGM 替代（[PNY 说明](https://pnysupport.freshdesk.com/support/solutions/articles/43000701755-should-i-use-gpu-burn-to-stress-test-my-quadro-datacenter-gpu-s-)）；Ascend 侧对应物是 **「多探针正交 + 健康红旗 + 集群相对」**，不是再写一个无限 GEMM。

---

### 1.4 部件覆盖：Cube / Vector / Scalar / MTE / SFU / launch / 热功耗频

DaVinci 分核事实（[Parallel Scan on Ascend](https://arxiv.org/html/2505.15112v1)、[MSDA on NPU](https://arxiv.org/html/2505.14022v1)）：AIC + 多 AIV，MTE 与计算分队列；Cube↔Vector 常经 GM/L2。

| 部件 | 现有 | R2 判定 |
|------|------|---------|
| Cube | func / sustained /（可选）bnmk | **充分**（本轮） |
| Vector | `vector_fma_perf`（constitution 已 enabled） | **充分采集**；判定仍 neutral |
| Scalar | `scalar_chain_perf`（cumsum 代理） | **可采**；线性度校验仍 P1 |
| SFU | 仅 SDC | **延后 P1**（吞吐） |
| MTE | 无专用探针；usages 有 MemBW% | **util 先上；copy 探针 P1** |
| Cube↔Vector 流水 | 无 | **P1**（解释「Cube 齐、step 不齐」） |
| launch-sync | 三桶无 burst | **本轮可发**；burst → P0.5/P1 |
| 热-功耗-频率 | 温坏；freq/limit 正则未证 | **修温 + 标 board/power 为 P0** |
| 健康红旗 | health 有 ECC；无 pcie-err 结构化 | **P0 补快照** |

msprof / MindSpore 可出 `FLOPS(cube|vec)`、`MTE2/MTE3`、`Scalar Ratio`（[ArithmeticUtilization](https://www.hiascend.com/document/detail/en/canncommercial/800/devaids/optool/atlasopdev_16_0093.html)、[MindSpore Profiling](https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling_ascend.html)）——**信息密度高但 128 卡扇出成本过高**，体质轮用 torch 代理；可疑卡再 msprof 深挖（P2）。

---

### 1.5 假阳性：节点散热 vs 本征慢卡（用 128 卡实锤）

**HBM 最慢 Top10** 高度集中在少数节点（worker-4、worker-1、master-0、worker-6），而各节点 **中位 HBM 仍接近全集群中位**——典型「槽位/风道/局部供电」尾部，而非整机箱全灭。

| 含义 | 动作 |
|------|------|
| 同 host 多卡同向偏慢 | 标 `node_thermal` / `node_env`，**不进换卡单** |
| 仅 within_host_z 显著偏低 | 标 `intrinsic_slow` |
| 同节点 ≥50% 卡同向慢 | **默认节点问题**（R1 规则保留，升为发射强制） |
| 温遥测作废时 | **禁止**用热归因；只能用 within_host 残差 + 健康红旗 |

BNMK Diff R1 额外警告：16 卡冒烟 FFN CV≈3%，128 全量 CV≈0.13%——**短跑噪声制造假慢卡**。体质 Stage C 已设 iters=50；冒烟节点不得用更短参数外推换卡结论。

---

### 1.6 「准备发射」：最小充分集 vs 可延后

R1 把 G1–G6 全标 P0，对「本周要跑 constitution128」过宽。R2 收窄：

**最小充分（Must）**

1. 修 `temp_c` 正则（去 I2C）+ 标签化 `-t temp`/`sensors`  
2. 遥测门禁：无效温 → `telemetry_trust=false`，报告禁止热结论  
3. 保持 `config.constitution128.yaml` 已开的 Stage C 三探针 + Stage A + 轻量 SDC  
4. health：ECC（已有）+ **pcie-err / err-count / Health 字符串**快照  
5. 聚合：`constitution_slow_frac=0.15` + **within_host 残差**（可先离线脚本）  
6. 先 1 节点 16 卡冒烟，再 128 fanout  

**可延后（Should / Later）**

| 项 | 理由 |
|----|------|
| launch burst / queue_depth | 单发三桶已能筛极端调度异常；burst 增益在 MoE 尾延迟 |
| 独立 `power_ramp` 探针 | sustained 已 `_attach_telemetry`；温/频修好后窗内即可算 drop |
| `mte_copy` / `cube_vector_pipeline` / `sfu_perf` | 首轮数据回来后，对「Cube 齐 step 慢」子集加测 |
| `AclValueGate` | 计时纯度；不阻塞相对比较 |
| HCCS 卡间带宽 | 多卡 case，非单卡体质默认路径 |
| 全套 5 散点 + 热力 | 最少：Cube×Vector 散点 + host×device within_host_z(HBM) |

---

## 2. 缺口清单（P0 / P1 / P2）

### 2.1 P0 — 本轮发射前必上

| ID | 项 | 相对 R1 | 动作 |
|----|-----|---------|------|
| **R2-G1** | 修复 I2C 温正则 + 标签字段 | **修正 R1-G1** | `telemetry.py` 小改；机上 1 卡 idle+gemm 60s 确认温∈合理区间 |
| **R2-G1b** | 遥测信任门禁 | **新增** | `telemetry_trust`；失败则报告大字警告 |
| **R2-G2** | usages 字段字典按真表 | **修正** | aicore/aicpu/ctrlcpu/mem_bw；删无效 Vector% 或改名 |
| **R2-G3** | health 红旗快照（ecc+pcie-err+err-count+Health） | 收窄 R1-G4 | 压测前/后各一次，记增量 |
| **R2-G4** | 残差协议落地（分析层） | 强化 R1-G5 | 换卡单只出 `intrinsic_slow` |
| **R2-G5** | constitution 聚合键 | 收窄 R1-G6 | PERF 扩展：`vector_gflops` 等；`slow_frac=0.15` |
| **R2-G6** | Stage C 保持 enabled | 已具备 | **不再加探针也可发射** |

### 2.2 P1 — 首轮 128 体质数据后

| ID | 项 |
|----|-----|
| R2-G7 | launch burst + enqueue/queue 三分解 |
| R2-G8 | sustained 窗内 power/freq 摘要（原 power_ramp 轻量版） |
| R2-G9 | `mte_copy_perf` |
| R2-G10 | `cube_vector_pipeline` |
| R2-G11 | `vector_sfu_perf` |
| R2-G12 | scalar elems 缩放线性度 |

### 2.3 P2 — 高成本 / 深挖

| ID | 项 |
|----|-----|
| R2-G13 | 可疑卡 msprof pipe（MTE2/3、Scalar/Vector ratio） |
| R2-G14 | HCCS / 卡间 |
| R2-G15 | AclValueGate |
| R2-G16 | 与训练 step time 关联（原 R3） |

---

## 3. 探针 / 函数签名（R2 增量）

### 3.1 遥测硬化（非 perf，但是发射阻塞）

```python
def parse_npu_temp(text: str) -> dict:
    """禁止裸 \\d+C。只匹配标签行。
    返回: {temp_c, board_temp_c, hbm_temp_c, aicore_temp_c} 缺失为 None
    """

def parse_npu_usages(text: str) -> dict:
    """键: aicore_util_pct, aicpu_util_pct, ctrlcpu_util_pct, mem_bw_util_pct"""

def telemetry_trust_gate(samples: list[dict]) -> dict:
    """若 median(temp_c)<10 或 unique(temp)≈1 且 temp<15 → trusted=False
    返回: {trusted: bool, reason: str|None}
    """
```

### 3.2 `health_counters`（P0，前后快照）

```python
def health_counters(device: int) -> dict:
    """npu-smi: health / ecc / pcie-err / err-count
    返回:
      health: str|None
      ecc_corrected / ecc_uncorrected: int|None
      pcie_err_count / err_count: int|None
      raw_ok: bool
    """

def health_delta(before: dict, after: dict) -> dict:
    """压测后增量；uncorrected 或 pcie 递增 → red_flag=True"""
```

### 3.3 残差（分析，可离线）

```python
def constitution_residualize(cards: list[dict], metrics: list[str],
                            z0: float = 2.0) -> list[dict]:
    """cluster_z + within_host_z → intrinsic_slow | node_env | contended
    同 host 同向慢占比 ≥0.5 → 强制 node_env
    """
```

### 3.4 已有 Stage C（发射沿用，签名不变）

```python
vector_fma_perf(device, elems=1<<26, iters=50, dtype="fp32", ...)
scalar_chain_perf(device, elems=1<<24, iters=50, ...)
launch_latency(device, samples=500, warmup=50, tiny_elems=1)  # burst 参数 P1 再开
```

### 3.5 P1 签名（保持 R1，不本轮实现）

```python
def launch_latency(..., burst_n: int = 64, burst_samples: int = 100) -> dict: ...
def power_ramp_summary(sustained_samples: list[dict]) -> dict: ...  # 从窗内派生，非新压测
def mte_copy_perf(device, mb=1024, iters=50, ...) -> dict: ...
def cube_vector_pipeline(device, n=4096, epilogue="bias_relu", ...) -> dict: ...
def vector_sfu_perf(device, elems=1<<25, op="exp", ...) -> dict: ...
```

---

## 4. 发射 Checklist（constitution128）

### 4.1 阻塞项（不勾不许 128 fanout）

- [ ] **R2-G1**：`temp_c` 不再恒为 2；idle 合理（约 30–50°C 量级，以机上为准），gemm 满载有上升  
- [ ] **R2-G1b**：报告写入 `telemetry_trust`；`false` 时禁止热/降频文字结论  
- [ ] **R2-G2**：usages 四字段至少 Aicore + MemBW 在满载下非空且随负载变化  
- [ ] AFS 已 sync 含 `stage_c.py` 的 CARD_SCREEN；`config.constitution128.yaml` 到位  
- [ ] 1 节点 16 卡冒烟：Stage C 三指标有分布、无大面积失败  

### 4.2 配置与时长

- [ ] Stage C 三探针 `enabled: true`；`shape_sweep`/`bnmk_sweep` `false`  
- [ ] `require_idle` + `idle-max-memory-mib`  
- [ ] 日志：`logs/card-constitution-128-<timestamp>/`  
- [ ] 单卡预估：既有 Stage A/SDC + Stage C ≈ +1–2 min；先冒烟看墙钟  

### 4.3 假阳性 / 交付

- [ ] 聚合 `constitution_slow_frac=0.15` + residualize  
- [ ] 换卡单 ⊆ `intrinsic_slow`；红旗（ECC/pcie）分列  
- [ ] 最少图：`func_tflops`×`vector_gflops` 散点；`within_host_z(hbm_gbps)` host×device 热力  
- [ ] 更新 `reports/card_screen_constitution_128.md`（跑完后）  

### 4.4 明确本轮不做

- [ ] 不实现 burst launch / 独立 power_ramp / MTE / pipeline / SFU 吞吐  
- [ ] 不对 128 卡开 msprof  
- [ ] 不在 `telemetry_trust=false` 时输出「无热节流」类结论（128 报告曾误写）  

---

## 5. 与代码现状对齐（2026-07-11）

| 组件 | 现状 | R2 含义 |
|------|------|---------|
| `stage_c.py` | vector / scalar / launch 三桶已实现 | 发射可开 |
| `config.constitution128.yaml` | Stage C enabled；burst/power 注释掉 | 符合最小集 |
| `NpuSmiProvider` | 扩展字段骨架 + **I2C 温 bug** | **必须先修** |
| `health.py` | ECC 门控；无 pcie-err | 补快照 |
| `aggregate.PERF_KEYS` | 仅三指标 | 体质轮需扩展或旁路脚本 |
| `run_card_constitution_128.sh` | fanout + aggregate slow_frac=0.2 | 冒烟后改 0.15 + 残差 |

---

## 6. Top 5 新发现 + 本轮必上 P0（执行序）

### Top 5

1. **`temp_c≡2` = 正则匹配 `I2C`，不是软占位**——不修则一切热结论有毒。  
2. **usages 真字段是 Aicore/Aicpu/Ctrlcpu/MemBW，没有 Vector%**——R0/R1 遥测字典需改。  
3. **HBM 慢尾已节点聚集**——残差协议是换卡前置，不是报告美化。  
4. **DCGM 精髓是「相对容差 + 红旗 + 多部件」**；Ascend 最小代理 = Stage C 正交 + ecc/pcie + within_host_z，不是再堆 GEMM。  
5. **R1 P0 过宽**：burst/独立 power_ramp/全散点可延后；**修温 + 门禁 + 红旗 + 残差 + 已开 Stage C** 即最小充分发射集。

### 本轮必上 P0（序）

1. 修温正则（去 I2C）+ 标签化 temp/sensors  
2. `telemetry_trust` 门禁  
3. usages 四字段对齐真表  
4. health 前后快照（ecc / pcie-err / err-count / Health）  
5. 残差协议 + `constitution_slow_frac=0.15`（换卡只出 intrinsic）  
6. 16 卡冒烟 → 128 constitution fanout（Stage C 保持开）

---

## 附录 A：资料锚点

- npu-smi `-t` 类型（含 sensors/volt/pcie-err/i2c_check）：[CSDN npu-smi 用法](https://blog.csdn.net/m0_37605642/article/details/137585875)  
- usages 字段表：[华为 Atlas usages](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/24.1.RC3/re/npu/npusmi_020.html)  
- 910B Cube/Vector/MTE 分核：[arXiv:2505.15112](https://arxiv.org/html/2505.15112v1) · [arXiv:2505.14022](https://arxiv.org/html/2505.14022v1)  
- msprof Cube/Vector ratio：[ArithmeticUtilization](https://www.hiascend.com/document/detail/en/canncommercial/800/devaids/optool/atlasopdev_16_0093.html)  
- DCGM targeted_power / 相对 gflops 容差：[DCGM Diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html)  
- DCGM 红旗指标：[MetricFire DCGM](https://medium.com/@MetricFire/why-gpu-monitoring-matters-tracking-utilization-power-and-errors-with-dcgm-603de3c4742b)  
- NVML 降频原因位（Ascend 无等价）：[NVML Clocks Event Reasons](https://docs.nvidia.com/deploy/archive/R535/nvml-api/group__nvmlClocksEventReasons.html)  
- 本地 R2 JSON：`tmp/research/ascend-constitution-r2.json` 等  

## 附录 B：路线图对齐

| 版本 | 内容 |
|------|------|
| R0 | Stage C 骨架 + 扩展遥测骨架 |
| R1 | P0 过宽清单（标定/burst/power/health/残差/报告） |
| **R2** | **根因修温；收窄发射最小集；usages 字典修正；残差升为强制** |
| R2.5 | 16→128 constitution 实跑 + 报告 |
| R3 | MTE/pipeline/SFU + step time 关联 |
