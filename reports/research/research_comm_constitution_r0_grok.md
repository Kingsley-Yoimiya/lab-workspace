# 卡间通信体质报告 · 建模与测法升级（R0 / Grok）

**Track**: E1（`research_constitution_r4_tracks.md`）  
**日期**: 2026-07-11  
**作者**: Cursor Grok 4.5  
**输入**:
- `reports/hccl_128.md`（collective × scale 单点）
- `reports/rounds/hccl_cluster_r0.md`（P2P 16 卡边画像；128 SIGSEGV）
- `scripts/cluster/{hccl_torch_bench,hccl_p2p_bench,run_hccl_scale,run_hccl_p2p_128,run_link_health}.*`
- `reports/research/research_nccl_verify_r0.md`（方法论对照）
- 外部：NCCL busbw 定义、Ascend `hccn_tool` / `hccl_test` 用法

**状态**: 调研完成；可执行下一轮实验清单见 §6。

---

## 0. 一句话

算力体质看「单卡硅片/HBM 是否偏慢」；通信体质看「机内 HCCS + 跨机 RoCE 的边与 collective 模式是否正常」。现有 `hccl_128` 只有全局单点 bus_bw，且弱扩展效率公式对 bus_bw **误用**；`hccl_cluster_r0` 首次给出边级图，但 128 卡 P2P 仍崩。下一轮必须：**修正效率解读 + 稳住 128 P2P（ring 串行）+ per-rank collective 热力 + 拓扑分层**，再谈与烤机合并。

---

## 1. 通信体质定义（相对算力体质）

### 1.1 对照表

| 维度 | 算力体质（CARD_SCREEN / constitution） | 通信体质（本 Track） |
|------|----------------------------------------|----------------------|
| 对象 | 单卡 Cube / Vector / HBM / launch | 卡↔卡边、节点↔节点、collective 路径 |
| 本征差异 | 硅片、封装、供电、散热导致的可重复偏移 | 光模块、RoCE 口、交换机、HCCS 拓扑、驱动/HCCL 路径 |
| 非本征噪声 | 争用、驱动混部、遥测毒数据 | 并发 P2P 过多、rank0 热点、计时含 host 调度 |
| 主指标 | `func_tflops`、`hbm_gbps`、分布 CV / 相对中位数 | `bus_bw`、边延迟/带宽、慢边 TopK、节点聚集 |
| 判定哲学 | 分布优先，非绝对阈值（见 R2.5） | **模式是否正常**：机内≫跨机、大消息 bus_bw 近平台、无孤立慢边/慢节点 |
| 正确性 | SDC 五类 | payload 校验 / collective checksum（与性能分轨记录） |

### 1.2 操作定义

在固定机型（本集群：8×16 Ascend 910）与固定软件栈（CANN + torch_npu + HCCL）下，**通信体质**指：

1. **边级**：抽样 P2P（ring / star / 节点代表卡）的延迟与带宽分布是否健康（无异常慢边、无单节点聚集）。
2. **集合通信级**：关键 collective 在「消息尺寸 × world_size」矩阵上的 `bus_bw` / 延迟是否符合拓扑预期（机内饱和、跨机受 RoCE 约束、随 N 退化形态可解释）。
3. **拓扑分层**：显式区分 **机内 HCCS** vs **跨机 RoCE**，禁止把 16 卡机内峰值直接当 128 卡跨机结论。

参考 NCCL tests：`algbw = S/t` 描述算法吞吐；`busbw` 把 collective 的多跳传输折算成「可与硬件峰值对比的总线带宽」，使不同 rank 数下的结果可比（[nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)）。本仓库 `hccl_torch_bench.py` 已采用同构公式（AllReduce × `2(n-1)/n`，AG/RS × `(n-1)/n`）。

### 1.3 「模式正常」判定框架（非简单阈值）

| 层 | 正常模式（期望） | 异常信号 |
|----|------------------|----------|
| 机内 16 卡大消息 AR | bus_bw 接近平台峰值且 CV 低 | 单 host 内多卡集体偏低 → 机内 HCCS/驱动 |
| 跨机扩展 | bus_bw **近似持平或缓降**（非线性增长） | 某 world_size 断崖；或仅某 op 崩 |
| 小消息 | 延迟主导，随 N 上升可接受 | AG/RS 小消息极端塌陷且与大消息形态不一致 |
| P2P 边 | 机内边 ≫ 跨机边；同层边分布紧 | TopK 慢边集中在同一 host 或同一跨机 pair |
| 链路工具 | `hccn_tool` link/speed/stat 与慢边相关 | 仅 npu-smi Health=OK 无法证伪坏口 |

**禁止**：用「bus_bw 未随卡数线性增长」判定扩展失败（见 §3.3）。

---

## 2. 该测的矩阵

### 2.1 Collective × 消息尺寸 × world_size

| 轴 | Smoke | Full（通信体质报告） | 说明 |
|----|-------|----------------------|------|
| **ops** | `all_reduce` | + `all_gather`, `reduce_scatter`, `broadcast` | 训练梯度/激活常用；Broadcast 作对照 |
| **ops（P1）** | — | `all_to_all`（若 API 稳定） | MoE / 重排敏感；现有脚本未覆盖 |
| **sizes（延迟）** | `64K` | `4K, 64K` | 小消息看开销；现有 scale 从 1M 起偏粗 |
| **sizes（带宽）** | `16M, 256M` | `1M, 16M, 64M, 256M`（可加 `512M` 若内存允许） | 与 `hccl_128.md` 对齐，便于回归 |
| **world_size** | `16` | `16, 32, 64, 128` | 16=纯机内；≥32 引入跨机 |
| **dtype** | fp32 | fp32（主）+ 可选 bf16 对照 | 先固定一种，避免矩阵爆炸 |
| **落盘** | per-rank JSONL | 同左 + 合并 `scale_N.jsonl` | 已具备；报告必须用全 rank |

**拓扑档位（显式标注，写入每条 record）**：

| 档位 | world | 物理含义 |
|------|------:|----------|
| `1n16` | 16 | 单节点 HCCS |
| `2n32` | 32 | 2 节点 RoCE + 机内 |
| `4n64` | 64 | 4 节点 |
| `8n128` | 128 | 全集群 |

### 2.2 P2P 边矩阵

全对全 128² 不可取。采用 **O(N) 抽样 + 可选节点代表**：

| 策略 | 边数（N=128） | 用途 | 现状 |
|------|--------------:|------|------|
| **ring** | 128 | 环邻接；含机内+跨机步 | 代码默认；大 world 仅 ring |
| **star→rank0** | 2×127 | 到 master 的跨机/机内星型 | world≥64 默认关闭（防 SIGSEGV） |
| **host-rep（P1）** | ~8×7 双向 | 节点代表卡全对全，定位跨机慢对 | 未实现 |
| **intra-full（P1）** | 16×15 / 节点 | 单节点全对全，标定 HCCS 基线 | 未实现 |

消息锚点：`64K`（延迟）+ `16M`（带宽）；正确性：payload 图案 + recv 抽查（已有）。

**执行约束（硬）**：全局 **一次一对** 串行；size 间 barrier；边后释放 tensor（`hccl_p2p_bench.py` 已实现）。禁止 128 卡并发多 pair。

### 2.3 拓扑 / 链路层（与微基准正交）

| 层 | 工具 | 产出 |
|----|------|------|
| 设备健康 | `npu-smi info` / `-t health` | 已有；仅卡级 OK |
| 网口/链路 | `hccn_tool -i $i -link/-speed/-stat/-net_health/-lldp` | **当前全缺** |
| RoCE 带宽 | `hccn_tool -roce_test ib_send_bw`（厂商文档路径） | 未跑；可与跨机 P2P 交叉验证 |
| 官方 collective | CANN `tools/hccl_test`（需 MPI 编译） | 未用；PyTorch 路径为替代 |

参考：[vLLM Ascend 多机检查清单](https://vllm-ascend.readthedocs.io/en/latest/tutorials/multi_node_pd_disaggregation_llmdatadist.html)、[ModelArts Lite Server HCCL/RoCE 验证](https://support.huaweicloud.com/intl/en-us/usermanual-server-modelarts/usermanual-server-0011.html)、[910B 上 hccn_tool 用法笔记](https://arthurchiao.art/blog/gpu-advanced-notes-2-zh/)。

---

## 3. 现有缺口

### 3.1 128 卡 P2P SIGSEGV

- **现象**：`hccl_cluster_r0` — world=16 ring+star 成功（176 边，`ok=true`）；world=128 ring+star **SIGSEGV**（exit -11，local_rank=10）。
- **已采取措施**：`world>=64` 默认仅 ring；边严格串行；size/边后 barrier + 释 tensor。
- **仍缺**：128 ring-only 稳定跑通的正式结果与图；star 在大 world 的安全子集（例如仅测到各节点 rank0，而非全 rank→global0）。
- **风险假设**：HCCL P2P 在多节点 + 多并发/多连接场景下资源上限；star 在 rank0 堆积 O(N) 连接是触发器之一。

### 3.2 无 hccn_tool

- `run_link_health.sh` 会 `find` 二进制，但 8/8 节点均为 `hccn_tool not found`（`hccl_128.md` §6）。
- 后果：无法做 link UP/DOWN、speed、stat、LLDP、RoCE ib_send_bw；慢边只能停留在「软件路径慢」，无法交叉到光模块/交换机。
- 根因候选：镜像裁剪、PATH/`setenv.bash` 缺失（同报告已记 driver env 警告）。

### 3.3 弱扩展效率解读偏弱（方法论缺陷）

`hccl_128.md` 定义：

```text
效率 = (bus_bw_N / bus_bw_16) / (N / 16) × 100%
```

并声称「100% = 带宽随卡数线性增长」。

这对 **`bus_bw` 是错误的**：

- [NCCL busbw](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) 的设计目标是：在不同 `n` 下得到**可与硬件峰值对比、且理想情况下近似常数**的数。
- 若 16→128 时 bus_bw 完全不变，按上式效率 = `1/8 = 12.5%`，会把「完美持平」误判为「极差扩展」。
- 本集群 256M All-Reduce：149.85 → 137.76 GB/s（仅降 ~8%），报告却写 **11.5% 弱扩展效率**——数字被公式人为压低一个数量级量级的「卡数因子」。

**应改用的主指标**：

| 名称 | 公式 | 含义 |
|------|------|------|
| **bus_bw 保持率** | `bus_bw_N / bus_bw_16 × 100%` | 弱扩展主叙事（理想 ≈100%） |
| **alg_bw 对照** | 原样报告 | 随 N 下降是预期；勿与 bus_bw 混读 |
| （可选）相对理想模型 | 与 ring/tree 理论曲线比 | P2 再做 |

小消息仍应用 **延迟（µs）** 而非强行 bus_bw 扩展叙事。

### 3.4 其它缺口（相对「通信体质报告」）

| 缺口 | 证据 | 影响 |
|------|------|------|
| 仅全局单点（早期）/ 边级仅 16 卡 | `hccl_128` vs `hccl_cluster_r0` | 128 无法回答「少数坏边 vs 全网均匀变慢」 |
| 无 alltoall | `hccl_torch_bench.py` ops 列表 | MoE 类负载盲区 |
| size 轴无亚毫秒档 | scale 从 1M 起 | 延迟体制覆盖不足 |
| 拓扑未写入 record | 仅 world_size | 报告易混读机内/跨机 |
| 官方 `hccl_test` 未编译 | 用 PyTorch 替代 | 与厂商基线难对齐 |
| 与算力 slow 卡未同屏 | 分属不同报告 | 无法区分「慢卡」vs「慢边」 |

---

## 4. 升级测法与报告图

### 4.1 测法升级（相对现状）

```
Phase A  解读修正：重算 hccl_128 的 bus_bw 保持率；废弃「/ (N/16)」主叙事
Phase B  128 P2P 稳住：STRATEGIES=ring，串行；产出边 JSONL + 图
Phase C  Collective per-rank 热力：已有落盘 → gen 报告（host×device / 慢 rank TopK）
Phase D  拓扑分层标注 + 机内基线 vs 跨机曲线同图
Phase E  hccn_tool 补齐（镜像/PATH）→ 与慢边交叉表
Phase F  （P1）host-rep P2P、alltoall、size 细扫、可选 hccl_test 对照
```

**墙钟预算（经验）**：

| 实验 | 粗估 | 备注 |
|------|------|------|
| collective 16–128 × 4 size × 4 op | ~数十分钟级（已跑通） | 换 port 防 TIME_WAIT |
| P2P ring@128 × 2 size × 串行 128 边 | 墙钟 ≈ 边数 × (warmup+iters) × RTT | 先 smoke 16 再 128；超时则减 iters |
| link_health + hccn | 分钟级 | 依赖工具存在 |

### 4.2 报告图清单（通信体质交付物）

对齐 `viz_diff_first_norm.md` 的 diff-first 精神，通信侧扩展为：

| ID | 图 | 数据源 | 解读 |
|----|-----|--------|------|
| G1 | **bus_bw 热力**（op × size，facet=world 或 world×op） | collective JSONL | 绝对值平台；找断崖格子 |
| G2 | **弱扩展曲线**（bus_bw vs world；双轴可加保持率%） | 同左 | 主叙事：保持率，非旧效率 |
| G3 | **慢边 TopK**（lat / bw 各一） | P2P JSONL | 点名 src→dst、host 对 |
| G4 | **host×host 热力**（lat / bw） | P2P 聚合 | 跨机斑块 |
| G5 | **rank×rank 热力**（抽样边） | P2P | 细粒度；16 卡已有 |
| G6 | **节点聚集**：慢边按 src_host / dst_host 计数柱状 | P2P TopK | 坏节点 vs 坏链路 |
| G7 | **per-rank collective 热力**（host×device，相对中位数） | per-rank `avg_s`/`bus_bw` | P2P 未稳时的保底 diff |
| G8 | （有 hccn 后）链路状态 × 慢边交叉表 | link + P2P | 物理 vs 软件 |

已有资产：`hccl_128_figs/*`、`hccl_cluster_r0_figs/*`（16 卡 G3–G5）。缺口主要在 **G2 公式修正、G1/G7@128、G3–G6@128**。

### 4.3 报告结构建议（下一轮 `reports/rounds/hccl_comm_constitution_r1.md`）

1. 拓扑与工具链状态（含 hccn 有无）  
2. Collective：bus_bw 表 + **保持率**曲线 + per-rank 热力  
3. P2P：边分布 + TopK + 节点聚集  
4. 「模式是否正常」结论（分机内/跨机）  
5. 与算力 slow 卡对照入口（仅索引，不强制同跑）

---

## 5. 与烤机 constitution 的关系

| 方案 | 内容 | 建议 |
|------|------|------|
| **A. 分轨（推荐默认）** | 算力：`CARD_SCREEN` / `run_card_constitution_*`；通信：`run_hccl_*` + 独立 rounds 报告 | **采纳**。空闲窗、失败域、指标语义均不同；通信 SIGSEGV 不应拖垮算力轮 |
| **B. 报告层合并** | 总览页同时链「算力 diff」与「通信 diff」；slow 卡 × 慢边同屏 | **P1 做索引合并**，不合并作业脚本 |
| **C. 作业合并** | 一次 job 先烤机再 HCCL | **不建议**。拉长墙钟、故障耦合；仅在用户明确要「一键体检」时再包一层 wrapper |

**判定键**：算力 slow 主键（func/hbm）与通信慢边 **不要**塞进同一 `slow_frac` 公式；可在分析层输出关联表（「慢算力卡是否落在慢边端点」）。

MFU / 训练环与体质轮继续分时独占空闲窗（与 R2.5 一致）。

---

## 6. 可执行的下一轮实验清单

### P0（必须，构成「通信体质 r1」最小充分集）

| ID | 实验 | 命令/动作要点 | 成功标准 | 粗估时长 |
|----|------|---------------|----------|----------|
| **P0.1** | 修正扩展叙事 | 用现有 `hccl_128` 数据重算 **bus_bw 保持率**；更新报告文案/图例 | 256M AR 128/16 保持率 ≈ **92%** 成为主结论；旧 11.5% 降为脚注「误用公式」 | 本机分钟级 |
| **P0.2** | 128 P2P ring-only | `SCALES=128 STRATEGIES=ring SIZES=64K,16M ./run_hccl_p2p_128.sh`；确认串行 | 无 SIGSEGV；每 size 128 边；`ok=true`；出 G3–G6 | 先设 30–60min 观察日志；异常早停 |
| **P0.3** | Collective per-rank 热力 | 确认 `run_hccl_scale` 已 merge `scale_*.rank*.jsonl`；写/跑 `gen_*` 出 G7（至少 AR@256M@128） | host×device 相对中位图 + CV + TopK | 有数据则本机出图 |
| **P0.4** | 拓扑标签 | bench record 增加 `topo_tier`（`1n16`…`8n128`）与 `intra_or_cross`（P2P 边可标） | JSONL 可过滤机内/跨机 | 小改代码 + 重跑或后处理 |
| **P0.5** | hccn_tool 定位 | 各节点查二进制与 `setenv`；修 PATH 或记「镜像缺件」issue | 要么跑通 `-link -g`，要么书面关闭并列替代 | 一次运维窗口 |

### P1（增强，不挡 r1 初稿）

| ID | 实验 | 要点 |
|----|------|------|
| **P1.1** | 安全 star 子集 | 仅 `node_rank0 → master` 或每节点 1 代表，避免 O(N) 打满 rank0 |
| **P1.2** | host-rep 跨机矩阵 | 8 节点代表卡全对全；专打跨机慢对 |
| **P1.3** | 延迟 size 补点 | collective/P2P 加 `4K`（或 `16K`） |
| **P1.4** | alltoall 微基准 | 若 `dist.all_to_all` 稳定则加入 ops |
| **P1.5** | 官方 `hccl_test` 对照 | 编译 CANN `tools/hccl_test`，单节点 AR 与 PyTorch 路径对齐 |
| **P1.6** | RoCE `ib_send_bw` | hccn 可用后做节点对带宽，与跨机 P2P 相关 |
| **P1.7** | 算力×通信同屏索引 | 总览 markdown 链两套 figs；关联 slow 端点 |

### 明确不做（本轮）

- HCCL algo×proto 全环境变量扫  
- 破坏性 skew / 故障注入  
- 128² 全 P2P  
- 与烤机同一 slow 公式强行合并  

---

## 7. Sources

- 本集群：`reports/hccl_128.md`、`reports/rounds/hccl_cluster_r0.md`、`reports/research/research_nccl_verify_r0.md`
- [nccl-tests PERFORMANCE.md（algbw/busbw）](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- [What is the busBW in nccl-tests?（NVIDIA Forums）](https://forums.developer.nvidia.com/t/what-is-the-busbw-in-nccl-tests/256858)
- [vLLM Ascend：hccn_tool link/net_health 检查](https://vllm-ascend.readthedocs.io/en/latest/tutorials/multi_node_pd_disaggregation_llmdatadist.html)
- [ModelArts：hccl_test 与 hccn_tool RoCE 验证](https://support.huaweicloud.com/intl/en-us/usermanual-server-modelarts/usermanual-server-0011.html)
- [GPU 进阶笔记：910B / hccn_tool / HCCS](https://arthurchiao.art/blog/gpu-advanced-notes-2-zh/)
- 检索落盘：`tmp/research/hccl-comm-constitution-r0.json`、`tmp/research/hccl-comm-constitution-r0-topo.json`

---

## 8. P0 实验清单摘要（交付回传）

1. **P0.1** 重算并改写弱扩展：**bus_bw 保持率**替代「/(N/16)」误公式。  
2. **P0.2** 128 卡 **ring-only 串行 P2P** 跑通 → 慢边 TopK + 节点聚集图。  
3. **P0.3** Collective **per-rank 热力**（host×device）作为边矩阵未全覆盖时的保底 diff。  
4. **P0.4** 记录写入 **拓扑档位**（机内/跨机）。  
5. **P0.5** 定位/修复 **hccn_tool**，否则书面关闭链路交叉验证。

**与烤机关系**：作业分轨；报告层可索引合并。
