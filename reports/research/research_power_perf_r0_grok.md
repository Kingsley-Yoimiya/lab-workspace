# Track D1 · 功耗 × 性能分布（R0 / Grok）

**日期**: 2026-07-11  
**目标**: 烤机时得到「功耗 vs 性能」联合分布（曲线/云图），不只是时间维时序——要知道每张卡在不同功耗下的表现。  
**对齐**: R3 分布优先、R4 Track D；遥测修复（R2.5）为硬前置。

---

## 0. 现状结论（读代码 + 旧 128 数据）

| 事实 | 证据 |
|------|------|
| `sustained_tflops` **已**在每个 window sample 上 `_attach_telemetry` | `stage_a.py`：`samp` 含 `tflops` + `TELEMETRY_KEYS`（含 `power_w`/`temp_c`/`sm_clock_mhz`…） |
| JSONL 已落 `record=gemm_sustained_sample` | `jsonl.py` 逐 sample 写出 |
| 旧 128 跑 **power 不可用** | `power_w` 全集 `{None}`；`temp_c` 全集 `{2.0}`（毒数据） |
| 旧报告只画了 sustained **时序** | `card_screen_128.md` §7；无 power–perf 图 |
| 当前 `telemetry.py` 已按 R2.5 方向改 `-t` + card/chip | 但 **未在集群验收**前，power×perf 图一律标 `telemetry_untrusted` |

**一句话**：采集管道形状已齐，缺的是可信 `power_w`（及 freq/temp），不是新探针。

---

## 1. 采集设计

### 1.1 采什么点

每个样本点：

```
(power_w, perf, freq, temp, t_s, host, device, card_id, chip_id, probe)
```

| 字段 | 含义 | 来源 |
|------|------|------|
| `power_w` | 瞬时功耗 | `TelemetrySampler.latest()` → `npu-smi -t power` |
| `perf` | `tflops` 或 `gbps` | 探针窗口计时 |
| `freq` | `aicore_freq_mhz` / `sm_clock_mhz` | board / 映射 |
| `temp` | `temp_c` / `hbm_temp_c` / `board_temp_c` | `-t temp` |
| `t_s` | 窗内相对时间（辅） | sustained 已有 |

遥测在 timing 窗外 `_attach_telemetry`，不扰动 perf 测量（现有约定保持）。

### 1.2 在哪些探针窗口采

| 优先级 | 探针窗口 | 落库 record | 用途 |
|--------|----------|-------------|------|
| **P0** | `sustained_tflops` 每个 window sample | `gemm_sustained_sample` | 主曲线：热稳态过程中 power 自然漂移 × tflops |
| **P0** | `func_perf` 每 round | `gemm_round` | 峰值 Cube 工况点云 |
| **P0** | `hbm_bandwidth` 每 round | `hbm_round` | power × gbps |
| P1 | `vector_fma` / `mte_copy` / `cube_vector` rounds | 对应 `*_round` | 正交部件工况 |
| P2 | 独立 `power_ramp`（显式改 power limit） | 新 probe | **本轮不做**（见 §3） |

**不采**：launch 微秒级窗口（功耗采样周期 ~0.5s，对齐无意义）；SDC 正确性轮次（非 perf 表征）。

### 1.3 采样密度建议

| 阶段 | sustained | 预期每卡 sample 数 | 说明 |
|------|-----------|-------------------|------|
| 冒烟 16 | `seconds: 30`（现 constitution 配置） | ~数十–百 | 验 power 非空、有方差 |
| 128 | 建议 `seconds: 60`（或保持 30 若窗紧） | 更多稳态点 | 云图/分位带更稳 |

NPU 无 `-lms`，sampler 轮询；`interval_s≈0.5` 时 sustained window（~50×gemm）与遥测大致同量级，**一对一 attach 即可**，不必额外插值。

---

## 2. 图类型清单

### 2.1 每卡曲线（per-card）

| ID | 图 | 数据 | 读法 |
|----|----|------|------|
| C1 | **power–tflops 散点/轨迹** | sustained samples，按 `t_s` 着色或连线 | 单卡在烤机过程中的功耗–算力路径 |
| C2 | **power–gbps 散点** | hbm rounds | 带宽工况效率 |
| C3 | **tflops/W vs t_s**（可选） | sustained | 能效随时间 |

### 2.2 跨卡云图（fleet）

| ID | 图 | 数据 | 读法 |
|----|----|------|------|
| F1 | **全卡 power–tflops 云图** | 全部 sustained samples，点按 host 着色 | 集群联合分布；离群卡一目了然 |
| F2 | **全卡 power–gbps 云图** | hbm rounds | 同 F1，带宽维 |
| F3 | **分位数带（binned）** | 按 `power_w` 分箱，箱内 tflops 的 p25–p75 + median | 「该功耗下集群典型表现」；比裸云更稳 |
| F4 | **每卡中位 (power, tflops) 汇总散点** | 每卡 sustained 稳态段（后 50% 时间）中位 | 一张图比 128 条曲线 |

### 2.3 辅图（与 Track A 重叠，可复用）

| ID | 图 | 说明 |
|----|----|------|
| A1 | power / temp / freq 直方图 | 修遥测后的边缘分布（R3） |
| A2 | sustained 时序（已有） | **辅证**热爬升；不替代 power–perf |
| A3 | power vs temp / freq 散点 | 解释降频是否热/功耗相关 |

**交付优先级**: F1 + F3 + C1（代表慢/中/快各 1 卡）→ F2 → F4。

---

## 3. 与 sustained 合并 vs 独立 `power_ramp`

| 方案 | 做法 | 利 | 弊 |
|------|------|----|----|
| **A. 合并 sustained（推荐 R0）** | 修遥测后，既有 sustained/func/hbm 样本自然带 power | 零新探针、零额外时长；R2.5 已否决本轮独立 ramp | power 动态范围靠热爬升，非主动扫 limit |
| B. 独立 `power_ramp` | 阶梯改 `power_limit`，每档测 tflops | 可控横轴、真「功耗–性能曲线」 | 需 npu 限功率 API、时长×档数、权限；R2.5 明确延后 |

**R0 决策**: **合并 sustained（+ func/hbm rounds）**；独立 `power_ramp` 标为 Phase 3+ 大任务，不阻塞 16/128 分布轮。

若冒烟发现满载 power 几乎无方差（全卡贴 limit），再评估是否需要轻量 ramp（2–3 档）——届时另开 R1。

---

## 4. 依赖（阻塞）

1. **遥测修复验收**（R2.5 / R3 前置）  
   - `npu-smi info -t power|temp|board -i <card> -c <chip>`  
   - idle：`temp_c` ∈ 合理区间（~30–45℃），`power_w` 非空  
   - 满载 GEMM：`power_w` 上升、`temp` 上升（非恒定）  
2. `telemetry_trust`：零方差 / 毒 temp 时报告禁止画「无 throttling」类结论；power–perf 图标注不可信。  
3. JSONL：`card_id`/`chip_id` 写入（便于跨卡对齐）。

旧 128 数据 **不能** 回填 power–perf 图（`power_w≡None`）。

---

## 5. 代码落地（最小）

### 5.1 采集侧（已基本具备）

- `sustained_tflops` / `func_perf` / `hbm_bandwidth`：已 `_attach_telemetry` → 确保 `power_w` 在修遥测后非空即可。  
- **无需**改探针逻辑；冒烟验收字段非空率。  
- 可选增强（非阻塞）：card 行汇总 `sustained_power_w_median` / `sustained_tflops_per_watt`（从 samples 派生）。

### 5.2 报告侧 stub

见 `reports/gen_card_constitution_report.py`：

```python
def plot_power_perf(
    samples: list[dict],
    *,
    perf_key: str = "tflops",          # 或 "gbps"
    out_path: Path,
    mode: str = "cloud",                # "cloud" | "per_card" | "percentile_band"
    power_bins: int = 12,
    highlight: list[tuple[str, int]] | None = None,  # [(host, device), ...]
) -> Path | None:
    """从 gemm_sustained_sample / hbm_round 画 power×perf。

    - cloud: 全样本散点（跨卡云图 F1/F2）
    - per_card: 单卡或 highlight 轨迹（C1）
    - percentile_band: 按 power 分箱的 p25–p75 带（F3）
    缺 power_w 时跳过并返回 None。
    """
```

实现可与 Track A 的 `plot_constitution` 合并；R0 只要求签名 + 空实现/最小 cloud。

---

## 6. 大任务分阶段

```
Phase 0  遥测验收（1 卡）
         idle/满载：power_w、temp、freq 非空且有动态范围
              ↓
Phase 1  冒烟 16（单节点 constitution 配置）
         检查：gemm_sustained_sample 中 power_w 非空率 ≥ 95%
               画出 F1 草稿 + 1 张 C1；确认分位带非退化
              ↓
Phase 2  128 fanout（修遥测后的 constitution128）
         全量 F1/F2/F3/F4 + 代表卡 C1；写入分布报告
              ↓
Phase 3+ （可选）独立 power_ramp / 能效 SLA
         仅当 Phase1 显示 power 无方差或业务要限功率曲线时
```

MFU 环保持 PAUSE；体质轮独占空闲窗。

---

## 7. 验收清单（冒烟）

- [ ] `power_w is not None` 占比 ≥ 95%（sustained samples）  
- [ ] 单卡 sustained 内 `power_w` 有可观测方差（或明确贴 limit 的说明）  
- [ ] `temp_c` 不再恒为 2.0  
- [ ] 至少生成：跨卡云图 F1、分位数带 F3、1 张每卡曲线 C1  
- [ ] 报告注明数据 run 时间戳；不覆盖旧 logs  

---

## 8. 与 R3 / R4 的关系

| 文档 | 关系 |
|------|------|
| R3 | 热/功耗/频是分布面之一；本 Track 专做 **联合** 分布，非边缘 hist |
| R2.5 | 独立 power_ramp 延后；本方案遵守，走 sustained 合并 |
| R4 Track A | 图清单中的 power-perf 由本 Track 定义，A 可调用 `plot_power_perf` |
| R4 Track D2 | Sonnet 评审本稿 |

---

## 9. Sources

- `card_screen/telemetry.py` — `TELEMETRY_KEYS`、`NpuSmiProvider.sample`（`-t` + card/chip）  
- `card_screen/probes/stage_a.py` — `sustained_tflops` + `_attach_telemetry`  
- `reports/card_screen_128.md` — 旧 temp 有毒；无 power–perf  
- 旧 JSONL：`logs/card-screen-128-20260710_224218` — `power_w≡None`, `temp_c≡2.0`  
- `research_card_constitution_r3_distribution.md` / `research_constitution_r4_tracks.md`
