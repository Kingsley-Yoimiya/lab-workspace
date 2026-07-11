# 体质报告可视化方案（R0 · 分布优先）

**日期**: 2026-07-11  
**Track**: A（可视化）  
**原则**: 分布优先；不强调坏卡 / slow 判定。缺字段跳过并在报告注明。

**前置**: [R3 分布方案](research_card_constitution_r3_distribution.md) · 字段见 `card_screen/io/jsonl.py` · 采集面见 `config.constitution128.yaml`

---

## 1. 与旧 `card_screen_128_figs` 的关系

| 旧图 | 新图 | 关系 |
|------|------|------|
| `boxplot_three_metrics.png` | `box_by_host_<metric>.png` + 可选 `box_overview.png` | 旧图是全集群单箱；新图按 host 分箱，更贴分布面 |
| `heatmap_host_device_deviation.png` | `heatmap_relmed_<metric>.png` | 同语义（相对中位数 %），按指标拆文件，覆盖更多指标 |
| `bar_host_mean_std.png` | `bar_host_mean_std_<metric>.png`（可选） | 保留旧风格；主交付改为 `sorted_bar_<metric>.png` |
| `sustained_timeseries_*.png` | `timeseries_sustained_*.png`（可选） | 需 `record=gemm_sustained_sample` |
| `shape_tflops_vs_n.png` | `shape_tflops_vs_n.png`（可选） | 需 `record=gemm_shape_sample`；constitution128 默认关 shape |

旧报告 `card_screen_128.md` / `_figs/` **保留不覆盖**；新产物写入 `reports/rounds/card_constitution_<stamp>.md` + `_figs/`。

---

## 2. 图清单

### 2.1 必须（有字段即出）

| 图 | 文件名 | 输入字段 | 布局 |
|----|--------|----------|------|
| 直方图 | `hist_<metric>.png` | 各数值指标（见 §3） | 单轴 hist；红虚线 = 集群中位数 |
| host×device 热力图 | `heatmap_relmed_<metric>.png` | `host`, `device`, `<metric>`；着色 = relmed% | 行=host（短名），列=device；RdYlGn，vmin/vmax 自适应或 ±5 |
| 按 host 箱线 | `box_by_host_<metric>.png` | `host`, `<metric>` | 每 host 一箱；散点叠加 |
| 排序条形 | `sorted_bar_<metric>.png` | `host`, `device`, `<metric>` | 128 卡升序；x=短标签 `host:dN`；中位水平线 |
| 正交散点 | `scatter_<x>_vs_<y>.png` | 成对字段（见下） | 点=卡；可选按 host 着色 |

**正交散点对（缺任一轴则跳过）**:

| 文件 | X | Y | 说明 |
|------|---|---|------|
| `scatter_func_tflops_vs_vector_gflops.png` | `func_tflops` | `vector_gflops` | Cube × Vector |
| `scatter_hbm_gbps_vs_mte_gbps.png` | `hbm_gbps` | `mte_gbps` | HBM × MTE |
| `scatter_power_w_vs_func_tflops.png` | `power_w` 或 `health_power_w` | `func_tflops` | 功耗 × 算力 |
| `scatter_power_w_vs_hbm_gbps.png` | `power_w` 或 `health_power_w` | `hbm_gbps` | 功耗 × 带宽 |
| `scatter_launch_host_overhead_p50_us_vs_ctrlcpu_util_pct.png` | `launch_host_overhead_p50_us` | `ctrlcpu_util_pct` | Launch × CtrlCPU |

### 2.2 可选（有 round 行 / 足够点才出）

| 图 | 文件名 | 输入 | 条件 |
|----|--------|------|------|
| 三指标总览箱线 | `box_overview.png` | 当前有数据的核心指标（最多 6） | ≥2 指标各 ≥2 点 |
| host 均值±σ | `bar_host_mean_std_<metric>.png` | 同 heatmap | 兼容旧风格 |
| sustained 时序 | `timeseries_sustained_p05_p50.png` | `gemm_sustained_sample` | 取 sustained 分位卡各一条 |
| shape 曲线 | `shape_tflops_vs_n.png` | `gemm_shape_sample` | constitution 默认关，旧 perf128 可出 |

---

## 3. 指标集合与字段映射

脚本扫描下列 key（与 `jsonl.py` card 行对齐）；`n≥2` 才画分布图：

| key | 标签 | 旧 perf128 | constitution128 |
|-----|------|------------|----------------|
| `func_tflops` | Cube func TFLOPS | ✓ | ✓ |
| `hbm_gbps` | HBM GB/s | ✓ | ✓ |
| `sustained_tflops` | Sustained TFLOPS | ✓ | ✓ |
| `vector_gflops` | Vector GFLOPS | — | ✓ |
| `scalar_elems_per_s` | Scalar elems/s | — | ✓ |
| `mte_gbps` | MTE copy GB/s | — | ✓ |
| `cube_vector_tflops` | Cube+Vector TFLOPS | — | ✓ |
| `sfu_gflops` | SFU GFLOPS | — | ✓ |
| `launch_sync_p50_us` / `p99` | Launch sync | — | ✓ |
| `launch_host_overhead_p50_us` / `p99` | Host overhead | — | ✓ |
| `launch_burst_p50_us` / `launch_burst_per_kernel_p50_us` | Burst | — | ✓ |
| `health_temp_c` / `hbm_temp_c` / `board_temp_c` | 温度 | 毒值风险 | 修遥测后 |
| `health_power_w` / `power_w` / `power_limit_w` | 功耗 | — | 修遥测后 |
| `aicore_freq_mhz` / `*_util_pct` | 频/利用率 | — | 修遥测后 |
| `shape_sweep_peak_tflops` | Shape peak | ✓ | 默认关 |

`relmed% = (value - cluster_median) / cluster_median × 100`。

---

## 4. 报告与命名

```
reports/rounds/card_constitution_<YYYYMMDD_HHMMSS>.md
reports/rounds/card_constitution_<YYYYMMDD_HHMMSS>_figs/
  hist_func_tflops.png
  heatmap_relmed_func_tflops.png
  box_by_host_func_tflops.png
  sorted_bar_func_tflops.png
  scatter_func_tflops_vs_vector_gflops.png   # 有字段才有
  ...
  skipped.json   # 可选：跳过原因机器可读
```

Markdown 结构：

1. 元信息（卡数 / 源路径 / 生成时间）
2. **跳过说明**（缺字段 / n&lt;2）
3. 指标统计表（median / mean / std / CV / p5–p95 / min–max）
4. 相对中位数偏差摘要
5. 图表嵌入（相对 `_figs/` 路径）
6. 不写「换卡单 / slow 主键」结论

---

## 5. 实现入口

| 文件 | 职责 |
|------|------|
| `reports/plot_card_constitution.py` | 读 JSONL → 统计 + 全套图 + md |
| `reports/gen_card_constitution_report.py` | 薄封装，转发到 plot 脚本（兼容旧调用） |

```bash
# 仅旧字段冒烟（见 ROUNDS.md §体质可视化）
python3 reports/plot_card_constitution.py \
  --data-dir /Users/yinjinrun/random-thing/logs/card-screen-128-20260710_224218/results
```

---

## 6. 验收

- [ ] 旧 `card-screen-128-*` JSONL：至少出 hist / heatmap / box_by_host / sorted_bar（func/hbm/sustained）
- [ ] 缺 vector/mte/power 时报告注明跳过，进程 exit 0
- [ ] 新 constitution 全字段跑通后，散点与遥测图自动出现
- [ ] 不覆盖 `card_screen_128.md` / `_figs/`
