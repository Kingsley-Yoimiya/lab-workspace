# Track B 调研门禁 R0：Ascend 128 卡 HCCL / P2P / 拓扑（对照 nccl-verify）

**状态**: 调研门禁（未宣称实现完成）  
**集群语境**: 128 卡 Ascend910 / 8 节点（见 `reports/hccl_128.md`）  
**对照仓库**: `project/nccl-verify/`  
**日期**: 2026-07-10

---

## 1. nccl-verify 做得好的地方（可迁移方法论）

> 目标不是把 NCCL algo×proto 全表搬到 Ascend，而是抽取「怎么测、怎么证、怎么出差异图」的方法。

### 1.1 参数矩阵（algo × proto × size × kernel）

| 能力 | 具体做法 | 文件锚点 |
|------|----------|----------|
| 标准 collective × algo×proto 全扫 | AllReduce/AllGather/ReduceScatter/Broadcast/Reduce；Ring/Tree/NVLS × LL/LL128/Simple；失败记 `FAILED` 而非静默跳过 | `benchmark/run_all_kernels.sh`（`run_bench`、Part 1–6） |
| 边界条件矩阵 | `NCCL_P2P_DISABLE`、`NCCL_NVLS_ENABLE=0`、二者叠加、禁 LL128、`NCCL_NTHREADS` 扫 | `benchmark/run_benchmark.sh` Phase 3（约 L101–144） |
| Symmetric / CE / DeviceImpl 扩展矩阵 | `-R 2` + `NCCL_SYM_KERNEL=<name>` 强制路径；含 alltoall/scatter/gather | `benchmark/run_sym_ce_kernels.sh` |
| GIN sync share 三维矩阵 | `TOPO × KERNEL × SIZE`；smoke/full 两档，避免一上来全量 | `kernel_breakdown/scripts/run_gin_sync_share_matrix.sh` |
| TUNING 参数抽取 | `NCCL_DEBUG_SUBSYS=TUNING` 落盘，供 model vs measured | `benchmark/run_benchmark.sh` Phase 0；`REPRODUCE.md` §5.3 |

**可迁移点**：矩阵要有 **smoke 档**、失败可观测、原始 log+CSV 成对落盘；**不要**在 R0 锁死 Ascend 侧完整 HCCL 参数表。

### 1.2 拓扑意识（单机 / 多机 / rail）

| 能力 | 具体做法 | 文件锚点 |
|------|----------|----------|
| 显式拓扑档位 | `TOPO=1n8\|2n16\|4n32`，kernel 列表随拓扑切换（单机 LSA vs 多机 Rail/GIN） | `run_gin_sync_share_matrix.sh` L28–85 |
| 协议 × 拓扑统一对比 | Step B 实验 C：Ring/Tree/NVLS/Sym 同图 | `kernel_breakdown/scripts/run_step_b_proto_topo.sh`；报告 `kernel_breakdown/step_b_report.md` |
| 机内拓扑快照 | 流水线开头 `nvidia-smi topo -m` | `kernel_breakdown/scripts/run_symmetric_breakdown_pipeline.sh` |
| 跨库拓扑复杂度对照 | 同步点按 O(N²)/O(N)/O(1) 标注会合点物理位置 | `compare/00_shared_brief.md` Q3；`compare/99_cross_lib_diff.md` §1 |

**可迁移点**：Ascend 侧至少区分 **机内 HCCS / 跨机 RoCE**；报告里写清「测的是哪一层拓扑」，避免把 16 卡机内峰值当成 128 卡跨机结论。

### 1.3 带宽与可复现流水线

| 能力 | 具体做法 | 文件锚点 |
|------|----------|----------|
| 标准 busbw / algbw | nccl-tests 输出解析为 CSV（含 inplace 列） | `run_all_kernels.sh` L51–53 |
| 复现文档 | 环境、编译顺序、冒烟 5 分钟、全量步骤 | `REPRODUCE.md` |
| 可视化报告链 | plot → markdown 报告；model vs measured 热力图 | `benchmark/plot_kernel_report.py`、`plot_model_vs_measured.py`；`nccl_kernel_visual_report.md` |
| 时间戳目录不覆盖 | `kernel_logs_${TIMESTAMP}`、`latest_kernel_logdir.txt` 指针 | `run_all_kernels.sh` L12–14、L163 |

**可迁移点**：与现有 `hccl_torch_bench.py` 的 `bus_bw_GBps` 公式对齐语义；报告侧继续用 `gen_hccl_128_report.py` 的 scale 曲线，但 **diff-first 热力图** 需 per-rank 数据（见 §3）。

### 1.4 正确性 / 路径可验证

| 能力 | 具体做法 | 文件锚点 |
|------|----------|----------|
| nccl-tests 正确性开关 | 多数 ktrace/breakdown 跑法带 `-c 0`（关闭 check 以提速时**显式声明**）；有回归时强调 check 通过 | 例：`run_sym_breakdown_matrix.sh` L177；`step_b_nvls_report.md` L726 |
| 「声称的路径是否真在跑」 | `NCCL_DEBUG` 查 multicast/NVLS；MC vs 非 MC 延迟/带宽对照 | `benchmark/verify_mc_active.sh` |
| 版本一致性门禁 | headers 与 library 版本必须一致，否则新能力表面可用实则 fallback | `REPRODUCE.md` §3.1 |

**可迁移点**：Ascend 第一切片至少要有 **可复现的数值校验**（all_reduce 结果 vs 期望、或 P2P echo 往返一致），不能只打 wall-clock。

### 1.5 故障 / 偏斜注入思路（研究向，非生产破坏）

| 能力 | 具体做法 | 文件锚点 |
|------|----------|----------|
| Sync 前 skew 注入 | `GIN_INJECT_SKEW_US` / `GIN_INJECT_SKEW_RANK`；扫 rank0 等 | `kernel_breakdown/microbench/README.md`「Skew 注入」；`run_sync_compare_1n8_skew_sweep.sh` |
| Kernel 侧 skew 参数 | `GinSyncSkewUs` / `GinSyncSkewRank` patch 进 NCCL | `kernel_breakdown/scripts/patch_gin_lsa_disable.py` L65–77 |
| 路径降级实验 | 关 P2P / 关 NVLS，观察带宽塌陷形态 | `run_benchmark.sh` Phase 3a–3c |
| 同步拓扑坍塌分析 | O(N²) P2P signal 在大 N 下的论点与代码对照 | `compare/00_shared_brief.md` 上层论点 2；`99_cross_lib_diff.md` |

**可迁移点**：R0 **不做**破坏性故障注入；但第一切片的 per-rank 延迟矩阵天然能暴露「慢边 / 慢卡」，为后续受控 skew 实验留接口。

---

## 2. Ascend 现状与相对 nccl-verify 的缺口

### 2.1 已有 ops 能力（基线）

| 组件 | 路径 | 现状 |
|------|------|------|
| HCCL collective 微基准 | `scripts/cluster/hccl_torch_bench.py` | torch.distributed HCCL；ops=`all_reduce,all_gather,reduce_scatter,broadcast`；sizes 可配；输出 `bus_bw_GBps` |
| 规模扇出 | `scripts/cluster/run_hccl_scale.sh` | 16/32/64/128；多节点 torchrun；结果 JSONL |
| 链路健康 | `scripts/cluster/run_link_health.sh` | 每节点 `npu-smi` + 尝试 `hccn_tool` |
| 128 卡报告 | `reports/hccl_128.md`（`gen_hccl_128_report.py`） | scale 曲线 + 扩展效率；链路节记录 hccn 缺失 |

### 2.2 关键缺口（相对 nccl-verify 方法论）

| 缺口 | 证据 | 影响 |
|------|------|------|
| **无 P2P / 点对点矩阵** | `hccl_torch_bench.py` 仅 collective；无 `send/recv`、无 rank 对延迟/带宽表 | 无法定位「哪条边慢」；跨机瓶颈只能从 scale 曲线间接猜 |
| **rank0-only 落盘** | `hccl_torch_bench.py` L136–142：仅 `rank == 0` 写 JSONL；每条 record 虽带 `rank` 字段，但实际只写 rank0 视角的计时 | 无法做 128 卡 host×device **通信 diff 热力图**（与 Track A compute 热力图不对齐） |
| **无 hccn_tool 实采** | `run_link_health.sh` 会找工具，但 `hccl_128.md` §6：8/8 节点 `hccn_tool not found` | 只有设备 Health=OK，无 HCCS/RoCE link/speed/stat |
| **无可验证正确性检查** | bench 只 `synchronize` + 计时；无结果校验、无 checksum、无 echo | 慢/错/静默错误不可分；与 nccl-tests `-c` / `verify_mc_active` 不对等 |
| **无拓扑档位显式建模** | scale 脚本按 world_size 扩，不区分机内/跨机子群、不扫 HCCL 算法环境变量 | 16 卡峰值与 128 卡跨机结论易混读 |
| **无参数/路径矩阵（有意延后）** | 对比 `run_all_kernels.sh` 的 algo×proto | R0 **不**锁全表；但需承认当前只有「默认路径单点」 |

### 2.3 与 `hccl_128.md` 结论的衔接

- 256M All-Reduce：16 卡 bus_bw ≈ 149.85 GB/s → 128 卡 ≈ 137.76 GB/s，弱扩展效率约 **11.5%**（报告定义）。
- 小消息 AG/RS 随规模急剧退化；Broadcast 相对更稳。
- **解释缺口**：现有数据是 **全局单点带宽**，不能回答「是少数坏边拖垮 ring，还是全网均匀变慢」。Track B 第一切片必须产出 **可差分的矩阵/向量**。

---

## 3. 多角度方案（≥3）与优先级

> **不**在此锁定完整 HCCL 环境变量 / 消息大小 / 算法全表；只排第一实现切片优先级。

| 优先级 | 方案 | 做法概要 | 产出 | 代价 / 风险 | 为何排此位 |
|:------:|------|----------|------|-------------|------------|
| **P0** | **最小 P2P 延迟/带宽矩阵（或等价可差分通信）** | 在现有 torchrun 128 进程上：固定小消息做 `isend/irecv`（或 `batch_isend_irecv`）采样；策略见 §4（全对全抽样 / 星型 / 环邻接） | `rank_i→rank_j` 延迟或 GB/s 矩阵；host×host 与 host×device 热力图；JSONL per-pair 或 per-rank | 全对全 128² 过重 → **必须抽样**；需确认 Ascend `torch.distributed` P2P 可用性 | 直接补「无 P2P 矩阵」最大缺口；与 nccl-verify 的边级/拓扑思维对齐；能解释 scale 退化 |
| **P1** | **Per-rank collective 计时（打破 rank0-only）** | 改 `hccl_torch_bench.py`：每 rank 记录本地 `avg_s`/`bus_bw`，**全部 rank append JSONL**（或写 `rank_{r}.jsonl` 再合并）；可选对 all_reduce 结果做期望校验 | 128 维向量 → 与 Track A 同构的 **host×device 相对中位数热力图** | 实现小；但不能定位单边；collective 掩盖单链路 | 改动最小、立刻可出 diff 图；可与 P0 并行准备，但信息量弱于边矩阵 |
| **P2** | **链路工具链补齐（hccn_tool / 驱动 env）** | 修镜像或 `setenv` 路径，使 `run_link_health.sh` 真正跑通 `-link/-speed/-stat`；解析进报告 | 每卡链路状态表；与 P0 慢边交叉验证 | 依赖镜像/驱动布局；当前 8 节点均未找到二进制 | 解释「物理链路 vs 软件路径」；不替代通信微基准 |
| **P3**（延后） | HCCL 算法/缓冲参数小矩阵 | 对照 `run_benchmark.sh` 边界思路，选 **极少** 环境变量做 A/B（非全表） | 路径降级曲线 | 易组合爆炸；需厂商文档 | 等 P0/P1 有 diff 基线后再做 |
| **P4**（研究） | 受控 skew / 故障注入 | 对照 `GIN_INJECT_SKEW_*`：指定 rank 在 collective 前 busy-wait | 验证「偏斜 → 全局带宽」敏感度 | 占集群、需隔离窗口 | 非 R0 实现范围 |

---

## 4. 推荐第一切片（实现范围）

### 4.1 目标一句话

在 **不锁全参数表** 的前提下，用 **最小可运行的 P2P（优先）或 per-rank collective（保底）**，产出 128 卡可差分的通信矩阵/向量，并画出 **差异热力图**。

### 4.2 推荐形态（P0 主路径）

**名称建议**：`hccl_p2p_probe`（新脚本）+ `run_hccl_p2p_128.sh`（扇出）+ `gen_hccl_p2p_report.py`（本机出图）。

| 项 | 建议（可在实现时微调，此处不定死全表） |
|----|----------------------------------------|
| 原语 | `dist.isend` / `irecv`（或文档确认后的 Ascend 等价 API）；单向测完再可选双向 |
| 规模 | world=128（8×16）；先单节点 16 卡 smoke |
| 消息 | **1–2 个锚点**（例：小延迟 4K–64K + 带宽 16M），禁止一上来扫满 size 轴 |
| 拓扑抽样（三选一，实现时钉一个） | **A.** 环邻接 + 跨机固定步长（O(N)）；**B.** 每 rank 测到 master 星型（O(N)）；**C.** 节点间代表卡全对全（8×8）再机内 16×16 | 推荐先 **A+B**：墙钟可控且能分机内/跨机 |
| 正确性 | payload 填 `rank` 图案；recv 端校验；失败写 `ok=false` |
| 落盘 | 每对一条 JSONL：`src,dst,nbytes,avg_s,bw_GBps,ok,...`；**禁止仅 rank0 写** |
| 图 | `heatmap_host_host_relmed_lat.png` 或 `heatmap_rank_rank_bw.png`；外加 sorted 慢边 TopK（对齐 `viz_diff_first_norm.md`） |
| 日志 | `logs/hccl-cluster-r0-<ts>/` + AFS 对等目录（见 `reports/ROUNDS.md`） |

### 4.3 保底形态（若 P2P API 不可用）

扩展 `hccl_torch_bench.py`：

1. **每 rank 写自己的计时**（修 rank0-only）。
2. 增加轻量 **all_reduce 正确性**：已知输入 → 校验输出。
3. 报告侧对 `avg_s` 做 host×device 热力图。

仍建议尽快回到 P0：per-rank collective **不能**替代边矩阵。

### 4.4 明确不在第一切片

- 完整 algo×proto / HCCL 环境变量全扫  
- hccn_tool 镜像大改（可并行记 issue，不挡 P0）  
- skew/故障注入  
- 覆盖历史 `hccl_128.md` 的 scale 曲线（保留；新切片另开 `hccl_cluster_r0.md`）

---

## 5. 第一切片之后的二次调研问题

1. Ascend `torch.distributed` 上 **P2P 是否走 HCCS/RoCE 预期路径**？有无强制绑网/HCCL 缓冲相关环境变量需记录？
2. 128² 全矩阵是否必要？节点代表卡 + 机内矩阵是否足够定位 95% 慢边？
3. `hccn_tool` 缺失是镜像裁剪还是 `PATH`/`setenv.bash` 问题？补齐后与 P0 慢边的相关性能否量化？
4. Per-rank collective 的「本地计时」是否受 tail latency 污染？是否需要独立 CUDA/NPU event 排除 host 调度噪声？
5. 与 Track A 卡间 compute diff 如何同屏对照（慢算力卡 vs 慢通信边）？
6. HCCL 是否提供类似 nccl-tests 的官方 perf + `-c`？若有，是否应逐步替换纯 PyTorch 路径？
7. 弱扩展效率公式与 bus_bw 定义是否与厂商文档一致，避免跨报告误读？

---

## 6. 关键路径索引

| 用途 | 路径 |
|------|------|
| NCCL 复现与矩阵入口 | `project/nccl-verify/REPRODUCE.md` |
| 标准 kernel 矩阵 | `project/nccl-verify/benchmark/run_all_kernels.sh` |
| 边界 / P2P disable | `project/nccl-verify/benchmark/run_benchmark.sh` |
| Sym/CE 矩阵 | `project/nccl-verify/benchmark/run_sym_ce_kernels.sh` |
| MC 路径验证 | `project/nccl-verify/benchmark/verify_mc_active.sh` |
| 拓扑×kernel×size | `project/nccl-verify/kernel_breakdown/scripts/run_gin_sync_share_matrix.sh` |
| Skew 注入思路 | `project/nccl-verify/kernel_breakdown/microbench/README.md` |
| 同步/拓扑对照 | `project/nccl-verify/compare/00_shared_brief.md`、`99_cross_lib_diff.md` |
| Ascend HCCL bench | `project/lab-workspace/scripts/cluster/hccl_torch_bench.py` |
| Scale 扇出 | `project/lab-workspace/scripts/cluster/run_hccl_scale.sh` |
| 链路健康 | `project/lab-workspace/scripts/cluster/run_link_health.sh` |
| 128 卡 HCCL 报告 | `project/lab-workspace/reports/hccl_128.md` |
| 报告生成器 | `project/lab-workspace/reports/gen_hccl_128_report.py` |
| Diff-first 规范 | `project/lab-workspace/reports/research/viz_diff_first_norm.md` |
| 轮次约定 | `project/lab-workspace/reports/ROUNDS.md` |
| 集群脚本说明 | `project/lab-workspace/scripts/cluster/README.md` |

### 建议新增（实现切片时）

| 用途 | 建议路径 |
|------|----------|
| P2P 探针 | `scripts/cluster/hccl_p2p_bench.py` |
| 128 卡扇出 | `scripts/cluster/run_hccl_p2p_128.sh` |
| 本轮报告 | `reports/rounds/hccl_cluster_r0.md` + `*_figs/` |
| 原始日志 | `logs/hccl-cluster-r0-<ts>/` |

---

*本文档为 Track B research gate R0：对照 nccl-verify 记录可迁移方法与 Ascend 缺口，给出优先级与第一切片范围；不构成「HCCL 全参数表 / 全拓扑矩阵已定稿」的声明。*
