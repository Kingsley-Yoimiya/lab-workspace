# Ascend 910 卡体质筛查 · R2 合并方案（可发射）

**日期**: 2026-07-11  
**输入**:
- [Grok R2](research_card_constitution_r2_grok.md)（`grok-4.5-fast-xhigh`）
- [第二评审](research_card_constitution_r2_sonnet.md)（原定 Sonnet 5；Task 仅允许 composer，由 **composer-2.5-fast** 代评）
- parallel-cli 缓存：`tmp/research/*-r2.json`

**状态**: 调研合并完成；**下一步 = 修遥测阻塞项 → 16 卡冒烟 → 128 发射**

---

## 0. 一句话结论

本轮不要再堆同质 GEMM。**先修 `I2C→temp_c=2` 毒数据 + usages 真字段 + 健康红旗 + 节点残差**，再用已开的 Stage C（Vector / Scalar* / launch）做 128 卡分布；Scalar 只采集不进换卡主键。

---

## 1. 两家共识（必须做）

| # | 共识 | 动作 |
|---|------|------|
| 1 | `temp_c≡2` 是正则误匹配 `I2C` 中的 `2C`，不是占位 | 标签化解析 Temperature/HBM/Board；失败保持 null |
| 2 | usages 无 Vector%；真字段 Aicore/Aicpu/Ctrlcpu/MemBW | 改 `NpuSmiProvider` 字典 |
| 3 | HBM 慢尾已节点聚集 | 换卡单 ⊆ `within_host` 残差后的 intrinsic |
| 4 | R1 P0 过宽 | 砍掉本轮独立 power_ramp / 全散点 / msprof 扇出 |
| 5 | DCGM 精髓 = 相对容差 + 红旗 + 多部件正交 | Ascend：Stage A/C + health 快照 + slow_frac |

## 2. 第二评审补强（采纳）

| # | 补强 | 采纳 |
|---|------|------|
| A | sustained 窗内 power/freq **派生**应阻塞发射（不必新探针） | ✅ |
| B | `scalar_chain` 首轮不进换卡主键 | ✅ |
| C | `func` 用 0.20；`hbm/vector/launch` 用 0.15；sustained 不进主键 | ✅ |
| D | 报告禁止默认写「无 throttling」 | ✅ |
| E | `comparison_group`（驱动/固件分桶） | ✅ checklist |

## 3. 最小充分探针集（本轮发射）

| 层 | 探针 / 能力 | 进 slow 主键？ |
|----|-------------|----------------|
| 阻塞硬化 | 修温 · usages 四字段 · `telemetry_trust` · health 前后快照 · within_host 残差 · sustained 窗 power/freq 派生 | — |
| Stage A | `func_perf` / `hbm` / `sustained` | func(0.20), hbm(0.15)；sustained 仅看派生 |
| Stage C | `vector_fma_perf` / `scalar_chain_perf` / `launch_latency` | vector + launch_host_p99(0.15)；**scalar 仅采集** |
| SDC | 五类轻量 rounds=5 | 正确性红旗 |
| 关闭 | shape_sweep / bnmk_sweep | — |

**若发射后再加 3 个（P1）**：`cube_vector_pipeline` → `mte_copy_perf` → `launch burst`。

## 4. 发射流程

```
1 卡 npu-smi 标定（修正则后验证 temp≠2）
  → 16 卡冒烟（master-0，constitution 配置）
  → 确认 telemetry_trust=ok
  → 128 fanout（run_card_constitution_128.sh）
  → 残差分析 → intrinsic_slow 换卡候选
```

**冲突避免**：MFU 环保持 PAUSE；体质轮独占集群空闲窗。

## 5. 发射前代码阻塞清单（10）

1. `parse_npu_temp` 标签化，禁裸 `\d+C`
2. usages 四字段对齐，删无效 `vector_util_pct`
3. `telemetry_trust` 写入 JSONL/报告
4. `health_counters` + 压测前后 delta
5. `CONSTITUTION_PERF_KEYS` + 分指标 slow_frac
6. `constitution_residualize`（换卡 ⊆ intrinsic）
7. sustained 窗 power/freq 派生
8. scalar 排除 slow 主键
9. 报告删除「无 throttling」默认文案
10. `run_card_constitution_128.sh`：0.15 + 时间戳日志 + sync stage_c 提醒

## 6. 文档索引

| 文档 | 角色 |
|------|------|
| `research_card_constitution_r0.md` | 初版 Stage C 设计 |
| `research_card_constitution_r1_grok.md` | R1 宽 P0（已收窄） |
| `research_card_constitution_r2_grok.md` | R2 根因修正 + 收窄 |
| `research_card_constitution_r2_sonnet.md` | 第二评审（composer 代 Sonnet） |
| **本文** | 合并可发射方案 |

## 7. Sources（parallel-cli）

- [DCGM Diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html)
- [npu-smi 用法 / type 列表](https://blog.csdn.net/m0_37605642/article/details/137585875)
- [npu-smi usages 字段](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/24.1.RC3/re/npu/npusmi_020.html)
- [Parallel Scan on Ascend](https://arxiv.org/html/2505.15112v1)
- [MindSpore Ascend Profiling](https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling_ascend.html)
- [NVML clocks event reasons](https://docs.nvidia.com/deploy/archive/R535/nvml-api/group__nvmlClocksEventReasons.html)
