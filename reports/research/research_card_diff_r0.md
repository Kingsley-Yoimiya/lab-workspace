# Track A 调研门禁 R0：卡间 Diff-First 与 GEMM Shape 扩展

**状态**: 调研门禁（未宣称实现完成）  
**集群语境**: 128 卡 Ascend910 / 8 节点（见 `reports/card_screen_128.md`）  
**日期**: 2026-07-10

---

## 1. 当前 JSONL 字段与缺口

### 1.1 现有 record 类型（落库）

来源：`project/lab-workspace-main/projects/CARD_SCREEN/card_screen/io/jsonl.py`（`write_jsonl`）。

| record | 关键字段 | 说明 |
|--------|----------|------|
| `card` | `host`, `device`, `backend`, `device_name`, `verdict`, `func_tflops`, `hbm_gbps`, `sustained_tflops`, `shape_sweep_peak_tflops`, `func_ok`, `func_max_rel_err`, SDC/health 摘要 | 每卡一行汇总；聚合入口见 `card_screen/cluster/aggregate.py` 的 `PERF_KEYS` |
| `gemm_round` | `round`, `iter_ms`, `tflops`, `timing`, 遥测 | 来自 `func_perf` 多轮计时 |
| `hbm_round` | `round`, `iter_ms`, `gbps`, `timing`, 遥测 | 来自 `hbm_bandwidth` |
| `gemm_sustained_sample` | `t_s`, `iter`, `iter_ms`, `tflops`, self_check_*, 遥测 | 来自 `sustained_tflops` |
| `gemm_shape_sample` | **`n`**, `tflops`, `peak_tflops`, `windows`, `elapsed_s`, `dtype`, 遥测 | 来自 `gemm_shape_sweep` |

探针实现：`card_screen/probes/stage_a.py`（`func_perf` / `hbm_bandwidth` / `sustained_tflops` / `gemm_shape_sweep`）。  
配置：`config.yaml` 与集群扇出脚本 `scripts/cluster/run_card_screen_128.sh`（生成 `config.perf128.yaml`）。

### 1.2 已确认缺口（相对训练真实 GEMM）

当前 shape 扫描与主探针均为 **方阵边长 N**：

- `func_perf` / `sustained_tflops`：`C[N,N] = A[N,N] @ B[N,N]`（见 `stage_a.py` 中 `a @ b`，`flops = 2 * n**3`）。
- `gemm_shape_sweep`：对每个 `n` 同样构造方阵；`sweep_shapes()` 产出 2 的幂边长 + 可选端点（如 16880）。
- JSONL 的 `gemm_shape_sample` **只有 `n` + `dtype`**，没有 batch / 非方阵维度。

**缺失维度（本轮门禁明确记录，尚未定稿全量网格）**：

| 缺口 | 现状 | 影响 |
|------|------|------|
| **B**（batch） | 无 | 无法覆盖 batched GEMM / 多 token 批 |
| **M, K**（非方阵） | 仅方阵 N | 无法覆盖 Attention/FFN 的 tall-skinny / fat 等 |
| **layout**（NN/NT/TN/TT） | 隐式 NN（`a @ b`） | 无法区分转置路径与 Cube 调度差异 |
| **dtype 网格** | 主路径固定 bf16；报告侧仅有单卡 dtype 抽测（`card_screen_128.md` 拓展节） | 无法做卡间 dtype×shape 一致性对比 |
| **(B,M,N,K) 元数据** | 无独立 record | 报告/热力图无法按训练 shape 切片 |

128 卡基线报告（`reports/card_screen_128.md`，生成器 `reports/gen_card_screen_128_report.py`）已对 `func_tflops` / `hbm_gbps` / `sustained_tflops` 做了相对中位数偏差与 host×device 热力图；**shape 维仍是「方阵 N 扫频」**，不足以代表训练算子分布。

---

## 2. 训练 GEMM Shape 示例（非穷尽）

以下为 **示例**，用于锚定第一切片与后续调研；**不是**完整算子目录，也不声称覆盖全部模型族。

| 类别 | 典型形态（示意） | 备注 |
|------|------------------|------|
| **Attention QKᵀ / PV** | `(B·H, S, D) × (B·H, D, S)` 等 | 常出现中等 M/N、较小 K（head_dim）；layout 常含转置 |
| **Attention 投影** | `(B·S, H·D) × (H·D, H·D)` | 接近方阵或宽矩阵，随 hidden 变化 |
| **FFN / MLP** | `(B·S, H) × (H, 4H)` 或 `(B·S, 4H) × (4H, H)` | 经典 fat/tall；本仓库第一切片用 `(1,4096,4096,11008)` 作 FFN-like 代理 |
| **MoE 专家** | 专家维上的 `(tokens_e, H) × (H, I)` | token 数稀疏、专家间 shape 相近但负载不均 |
| **copy / 带宽型** | 大块 memcpy / `dst=src*α` | 已有 `hbm_bandwidth`；与 GEMM 正交，diff 时勿混为同一 metric |

FLOPS 约定（后续实现应对齐）：对 `C[B,M,N] = A[B,M,K] @ B[B,K,N]`，  
`flops ≈ 2 · B · M · N · K`（与当前方阵 `2·N³` 在 `B=1,M=N=K` 时一致）。

---

## 3. Diff-First 可视化目录

**原则**：先看卡间相对偏差，再看绝对值；与 `viz_diff_first_norm.md` 对齐。

### 3.1 强制项（每个新性能 metric）

| 图 / 表 | 定义 | 参考实现 |
|---------|------|----------|
| **host×device 相对中位数热力图** | `dev% = (x − med) / med × 100`，行=host、列=device | `gen_card_screen_128_report.py` → `plot_host_device_heatmap`；报告图 `card_screen_128_figs/heatmap_host_device_deviation.png` |
| **按值排序柱状图** | 128 卡升/降序条形，标出中位线 | 可扩展现有 `bar_host_mean_std` 思路到 per-card sorted bars |
| **CV + TopK** | CV%、最慢/最快 TopK 表 | 报告 §3–§5；`aggregate.py` 的 `slow_frac` 判定可复用阈值语义 |

### 3.2 建议目录（按 metric 复制一套）

对每个 metric（含未来 `gemm_bnmk` 各 shape 的 tflops）：

1. `heatmap_host_device_relmed_{metric}.png` — **强制**
2. `bar_sorted_cards_{metric}.png` — 强制
3. `stats_{metric}.json` — mean/median/std/min/max/cv_pct + TopK 列表
4. （可选）`boxplot_{metric}.png`、sustained 时序对比慢卡 vs 中位卡

现有三指标已在 `card_screen_128.md` §7 落地；**新增 BNMK 样本后，每个 (B,M,N,K,layout,dtype) 切片应视为独立 metric 或带 facet 的同一 metric**，仍须满足热力图强制项。

---

## 4. CARD_SCREEN 变更草案（保留旧探针）

### 4.1 目标

在 **不拆除** 现有方阵探针的前提下，增加 `(B,M,N,K)` GEMM 与新 JSONL record `gemm_bnmk_sample`。

### 4.2 建议改动面（草案，待实现切片确认）

| 组件 | 路径 | 草案动作 |
|------|------|----------|
| 探针 | `card_screen/probes/stage_a.py` | 新增 `gemm_bnmk(...)` / `gemm_bnmk_sweep(...)`；`flops = 2*B*M*N*K`；保留 `func_perf` / `sustained` / `gemm_shape_sweep` |
| 配置 | `config.yaml` + `run_card_screen_128.sh` 内嵌 YAML | 新增节如 `bnmk_sweep:`（shapes 列表、dtype、warmup/window/时间封顶）；`probes.bnmk_sweep.enabled` |
| 注册 | `card_screen/probes/builtin.py` / `registry.py` | 注册新探针；默认可先 `enabled: false`，第一切片再开 |
| 落库 | `card_screen/io/jsonl.py` | 写出 `record: gemm_bnmk_sample`，字段至少含 `B,M,N,K,layout,dtype,tflops,peak_tflops,...`；`card` 行可增加摘要字段（如 peak）但勿破坏旧字段 |
| 聚合 | `card_screen/cluster/aggregate.py` | **短期** `PERF_KEYS` 仍用旧三指标做 verdict；BNMK 先做观测/报告，是否进 slow 判定列为二次调研 |
| 绘图 | `card_screen/plot.py` | 可选：BNMK 曲线/分 facet；集群 diff 图优先放在 `reports/gen_*` 侧以复用 128 卡热力图逻辑 |
| 扇出 | `scripts/cluster/run_card_screen_128.sh` | 第一切片只注入最小 shape 列表，控制墙钟 |

### 4.3 兼容性

- 旧 `gemm_shape_sample.n` 继续存在；分析脚本（`gen_card_screen_128_report.py`）应 **并存读取**，勿假设只有一种 shape record。
- 方阵可视为 `B=1,M=N,K=N,layout=NN` 的特例，但 **不要** 在 R0 强行改写旧 record schema。

---

## 5. 最小第一切片示例

以下四条作为 **第一实现切片** 的候选清单（dtype/layout 先固定，避免组合爆炸）：

| # | (B, M, N, K) | dtype | layout | 意图 |
|---|--------------|-------|--------|------|
| 1 | `(1, 8192, 8192, 8192)` | bf16 | NN | 与现网 `gemm_n: 8192` 对齐的方阵锚点 |
| 2 | `(1, 4096, 4096, 11008)` | bf16 | NN | FFN-like 非方阵 |
| 3 | `(8, 2048, 2048, 2048)` | bf16 | NN | batched 方阵 |
| 4 | `(1, 16384, 1024, 1024)` | bf16 | NN | tall-skinny |

约束建议（实现时再钉死）：

- 每 shape：warmup + 短 sustained 窗口（可复用 `shape_sweep` 的 `min_seconds` / `max_seconds` 思路）。
- 显存：大 B 或大 M 需预检 OOM；失败应记 quality flag，而非静默跳过无记录。
- 判定：第一切片 **只出 JSONL + diff 图**，不改变 128 卡 `final_verdict` 逻辑（除非二次调研明确要求）。

---

## 6. 下一实现切片

按优先级（可并行准备，但建议串行合入）：

1. **Schema**：在 `jsonl.py` 增加 `gemm_bnmk_sample` 写入；文档化字段表。
2. **探针**：`stage_a.py` 实现 `(B,M,N,K)` GEMM + 上述 4 条最小列表；配置开关默认关。
3. **扇出配置**：扩展 `run_card_screen_128.sh` 生成含 `bnmk_sweep` 的轻量 YAML（短封顶）。
4. **报告**：扩展或新建 `gen_*` 脚本，对每个 BNMK shape 的 tflops 出 **host×device 相对中位数热力图 + sorted bars + CV/TopK**（见 `viz_diff_first_norm.md`）。
5. **回归**：单节点 smoke（1–2 卡）跑通后再 128 卡；对比方阵锚点与现有 `func_tflops` 中位数是否同量级。

---

## 7. 二次调研问题（未决）

以下问题 **本门禁不宣称已决策**：

1. **layout 网格**：NN 之外是否必须首轮覆盖 NT/TN？Ascend `torch_npu` 路径差异多大？
2. **dtype 网格**：fp16/fp32/bf16 是否进入集群扇出，还是保持单卡抽测？
3. **BNMK 是否进入 `PERF_KEYS` / slow 判定**？阈值是否与方阵共用 `slow_frac=0.2`？
4. **MoE / Attention 真实 shape** 从哪份训练配置采样？示例表如何升级为「可版本化的 shape 清单」？
5. **聚合中位数**：按 shape 分桶，还是只对「锚点方阵」做跨卡对比？
6. **墙钟预算**：4 shape × 128 卡 与现有 shape_sweep（多 N）叠加后的总时长上限？
7. **与 HCCL/链路筛查的边界**：卡间 compute diff 与通信 diff 的报告是否分轨？

---

## 8. 关键路径索引

| 用途 | 路径 |
|------|------|
| Stage A 探针 | `project/lab-workspace-main/projects/CARD_SCREEN/card_screen/probes/stage_a.py` |
| 默认配置 | `project/lab-workspace-main/projects/CARD_SCREEN/config.yaml` |
| JSONL 写出 | `project/lab-workspace-main/projects/CARD_SCREEN/card_screen/io/jsonl.py` |
| 集群聚合 | `project/lab-workspace-main/projects/CARD_SCREEN/card_screen/cluster/aggregate.py` |
| 单机绘图 | `project/lab-workspace-main/projects/CARD_SCREEN/card_screen/plot.py` |
| 128 卡扇出 | `project/lab-workspace/scripts/cluster/run_card_screen_128.sh` |
| 128 卡报告 | `project/lab-workspace/reports/card_screen_128.md` |
| 报告生成器 | `project/lab-workspace/reports/gen_card_screen_128_report.py` |
| Diff-first 规范 | `project/lab-workspace/reports/research/viz_diff_first_norm.md` |

---

*本文档为 Track A research gate R0：记录缺口与草案，不构成「全部 shape/判定已定稿」的声明。*
