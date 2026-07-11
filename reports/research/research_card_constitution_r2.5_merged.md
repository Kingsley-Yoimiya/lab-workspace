# Ascend 910 卡体质筛查 · R2.5 合并方案（真 Sonnet 后）

**日期**: 2026-07-11  
**输入**:
- [Grok R2](research_card_constitution_r2_grok.md)
- [Sonnet 5 独立评审](research_card_constitution_r2_sonnet.md)（真 `claude-sonnet-5-thinking-high`，已覆盖旧 composer 代评）
- 旧 [R2 merged](research_card_constitution_r2_merged.md)（本文取代其「可发射」结论）

**状态**: 调研合并完成；**实现前需用户拍板 1 处分歧**（见 §3）

---

## 0. 一句话

`temp_c≡2` 必须修，但**只改正则不够**——要改 `npu-smi` 调用（加 `-t`）并标定 **Card/Chip 寻址**；首轮 slow 主键只用已验证的 **func + hbm**，Stage C 全观察。

---

## 1. 三方共识（必须做）

| # | 共识 |
|---|------|
| 1 | `temp_c≡2` 是毒数据，禁止热/降频结论 |
| 2 | usages 无 Vector%；对齐 Aicore/Aicpu/Ctrlcpu/MemBW |
| 3 | HBM 慢尾节点聚集 → 换卡 ⊆ within_host 残差 |
| 4 | scalar 不进 slow 主键 |
| 5 | 本轮不做：burst / 独立 power_ramp / msprof 128 / shape+bnmk |
| 6 | `slow_cause` 在遥测不可信时不得产出 `no_throttle` 等 hint |

## 2. Sonnet 新增（全部采纳进 P0）

| # | 发现 | 动作 |
|---|------|------|
| S1 | 根因更可能是 **`npu-smi info -i` 缺 `-t type`** → 帮助文本含 `i2c_check` → 裸正则吃到 2；真实 `npu-smi info` 表格式温 36–43℃ 且无 I2C | 改命令 + 标签化正则；验收 idle 温合理、满载上升 |
| S2 | **Card/Chip 寻址错位**：device 0–15 是芯粒级，`-i` 是卡级，从未传 `-c` | 机上标定 `device→(card,chip)` 后写死 |
| S3 | **`comparison_group` 空转**：字段在 aggregate 有，但从未写入 driver/firmware | 写入 JSONL |
| S4 | **launch `wall_sync` 静默豁免**：`host_overhead=None` 不进 slow 也不报警 | 聚合前检查 `timing_method` 一致性 |
| S5 | health_counters 是**新写代码**，不是勾选 | 排期按新功能估 |

## 3. 唯一需拍板的分歧

| | Grok / 旧 merged | **Sonnet（本文采纳为默认）** |
|--|------------------|------------------------------|
| Stage C 进 slow 主键？ | vector + launch_host_p99 @0.15 | **首轮全部仅观察**；只用 **func@0.20 + hbm@0.15** |
| 理由 | 尽快用正交维度 | Stage C 从未在本集群跑过，无 CV；launch 还有 timing 纯度问题 |

**默认按 Sonnet**：16 卡冒烟拿到 CV 后，再决定是否升级 vector（launch 更谨慎）。

若你坚持旧 merged「首轮就判 vector/launch」，请明确说一声。

## 4. 最小充分集（可发射）

| 层 | 内容 | 判 slow？ |
|----|------|-----------|
| 阻塞硬化 | 改 npu-smi 命令+正则 · Card/Chip 映射 · telemetry_trust→slow_cause · comparison_group 填充 · health_counters/delta · launch timing_method 检查 · within_host 残差 | — |
| Stage A | func / hbm / sustained | **func 0.20 · hbm 0.15**；sustained 仅辅证+派生 |
| Stage C | vector / scalar / launch | **仅观察** |
| SDC | 五类 rounds=5 | 正确性红旗 |
| 关 | shape / bnmk | — |

**发射后再加 3 探针（Sonnet 序）**：`mte_copy_perf` → `cube_vector_pipeline` → HBM 访问模式变体（非 launch burst）。

## 5. 发射流程

```
Phase 0  1 卡：修命令+正则+Card/Chip 映射；验收 temp 合理
Phase 1  16 卡冒烟：Stage C 有分布、timing_method 全 event
         → 决定 vector 是否升级主键
Phase 2  128 fanout：func+hbm 判 slow + 残差 + 健康红旗 + Stage C 观察
Phase 3  回流后补 mte_copy / cube_vector_pipeline / HBM 变体
```

MFU 环保持 PAUSE；体质轮独占空闲窗。

## 6. 发射前阻塞清单（可执行）

1. `NpuSmiProvider`：禁止裸 `info -i`；改用 `-t board|temp|usages|power|…` + `-i/-c`
2. 机上标定 `device → (card_id, chip_id)` 并写死
3. 标签化温度解析；失败保持 `null`；`telemetry_trust` 门禁进 `slow_cause.classify`
4. usages 四字段对齐；删除无效 `vector_util_pct`
5. 写入 `driver_version` / `firmware_version`
6. 新写 `health_counters` + 压测前后 `health_delta`
7. 聚合检查 `launch_latency.timing_method`；`wall_sync` 标不可比
8. within_host 残差；换卡 ⊆ intrinsic_slow
9. slow 主键仅 func+hbm（除非用户推翻 §3）
10. 报告禁止「无 throttling」默认文案；`run_card_constitution_128.sh` 时间戳日志

## 7. 文档索引

| 文档 | 角色 |
|------|------|
| `r2_grok.md` | I2C 正则假说 + P0 收窄 |
| `r2_sonnet.md` | **真 Sonnet**：命令根因、Card/Chip、comparison_group、launch 豁免、主键收紧 |
| `r2_merged.md` | 旧合并（已被本文取代决策） |
| **本文 `r2.5_merged.md`** | 当前可发射方案 |

## 8. Sources

- 本集群抓包：`logs/telemetry-20260710_224628/results/master0.jsonl`（真实 Temp 36–43℃）
- 本集群毒数据：`logs/card-screen-128-*/` 中 `temp_c` 全集 `{2.0}`
- [DCGM Diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html)
- [npu-smi 用法](https://blog.csdn.net/m0_37605642/article/details/137585875)
- [npu-smi usages](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/24.1.RC3/re/npu/npusmi_020.html)
