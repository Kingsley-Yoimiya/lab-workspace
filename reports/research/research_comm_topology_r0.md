# 卡间拓扑探测方案（R0）

**日期**: 2026-07-11  
**前置**: `reports/hccl_128.md` §6、`research_comm_constitution_r0_{grok,sonnet}.md`  
**脚本**: `scripts/cluster/probe_hccl_topology.sh`  
**状态**: 方案 + 骨架；链路级字段需机上验收（本集群曾 8/8 缺 `hccn_tool`）

---

## 0. 一句话

通信刻画前必须先拿到 **卡间拓扑图 + 档位标签**：机内走 HCCS、跨机走 RoCE。没有拓扑，慢边只能报「慢」，无法解释「该不该慢」。本轮用 **best-effort 探测**（有啥采啥、缺工具不崩），把拓扑插到通信 list **最前**（烤机后、collective 前）。

---

## 1. 拓扑该采什么

### 1.1 两层语义：物理 vs 逻辑

| 层 | 含义 | 典型来源 | 用途 |
|----|------|----------|------|
| **物理拓扑** | 硬件真实互联：机内 HCCS 全连接/分片、RoCE 口 UP/速率、光模块/LLDP | `npu-smi -t topo`、`hccn_tool -link/-speed`、`/etc/hccn.conf` | 解释「机内应 ≫ 跨机」；交叉验证慢边是否坏口 |
| **逻辑拓扑** | 软件视角的 rank↔device↔host↔IP 映射；HCCL 建链用的集群描述 | ranktable JSON、torchrun `nnodes/node_rank`、env | 给每条 P2P/collective 记录打 `topo_tier` / `intra_or_cross` |

两者都要采。物理缺了（如无 `hccn_tool`）仍可用逻辑层 + 探测性带宽推断；逻辑缺了则无法把边映射到节点对。

### 1.2 机内 vs 跨机（本集群硬事实）

本集群：**8 节点 × 16 NPU = 128 卡**（Ascend 910，双 Chip）。

| 域 | 物理介质 | 期望量级（经验） | 判定 |
|----|----------|------------------|------|
| **机内** | HCCS（`npu-smi -t topo` 矩阵里标 `HCCS`） | 大消息 P2P / 16 卡 AR bus_bw 近平台（~150 GB/s 量级，见 `hccl_128`） | 机内边若接近跨机 → 该节点 HCCS/驱动异常 |
| **跨机** | 每卡 RoCE（`hccn.conf` 的 `address_*`）经交换机 | 明显低于机内；128 卡大消息 AR 保持率可仍高（~92%），但边延迟/带宽分层清晰 | 慢边聚集在同一 host 对 → 坏节点/坏口/坏交换机端口 |

### 1.3 档位标签（写入每条 bench record）

| `topo_tier` | world | 物理含义 |
|-------------|------:|----------|
| `1n16` | 16 | 纯机内 HCCS |
| `2n32` | 32 | 首次跨机（2 节点 RoCE + 机内） |
| `4n64` | 64 | 4 节点 |
| `8n128` | 128 | 全集群 |

P2P 边额外字段：

| 字段 | 取值 | 规则 |
|------|------|------|
| `intra_or_cross` | `intra` / `cross` | `src_host == dst_host` → intra，否则 cross |
| `link_hint` | `HCCS` / `RoCE` / `unknown` | 有物理 topo 时：同 host 且 topo 矩阵为 HCCS → HCCS；跨 host → RoCE |

---

## 2. 可用工具优先级

按「信息密度 × 本集群可用性」排序。缺工具时降级，**禁止因缺工具让流水线崩**。

| 优先级 | 工具 / 源 | 采什么 | 本集群现状 | 缺了怎么办 |
|:------:|-----------|--------|------------|------------|
| **P0** | `npu-smi info` / `-t health` | 卡数、型号、Health、温功耗 | 已通（8/8 OK） | 阻塞：无卡则不跑通信 |
| **P0** | `npu-smi info -t topo`（或 `-l`） | **机内** NPU×NPU 矩阵（HCCS/SYS/PIX…）+ CPU affinity | **需机上验证**（文档普遍支持；本集群未系统落盘） | 用双 Chip 结构 + 16 卡假设标 `assumed_hccs_fullmesh` |
| **P1** | `/etc/hccn.conf` | 每 device 的 RoCE IP（`address_0`…） | **需机上验证** | 记 `hccn_conf=missing`；跨机边只能靠 host 名推断 |
| **P1** | `hccn_tool -i $i -link/-speed/-stat/-net_health/-lldp/-ip -g` | 链路 UP/DOWN、速率、错包、交换机口 | **8/8 not found**（`hccl_128.md`） | 书面关闭链路交叉；慢边只标软件路径 |
| **P1** | ranktable JSON / `RANK_TABLE_FILE` | server_id、device_id、device_ip、rank_id | torchrun 路径常无显式文件 | 从 `HOSTNAME` + `LOCAL_RANK` + `WORLD_SIZE` 合成逻辑图 |
| **P2** | HCCL 相关 env | `HCCL_*`、`ASCEND_*`、`RANK_TABLE_FILE` 等 | 随镜像变 | 只做快照，不改值 |
| **P2** | `/usr/local/Ascend/driver/topo/`（若存在） | 出厂物理拓扑描述（部分机型） | 不确定 | 有则整目录拷贝 raw |
| **P3** | 探测性 allreduce / 机内 vs 跨机 P2P | 用带宽分层 **反推** 拓扑档位是否符合预期 | 有 bench 脚本 | 仅作校验，不替代物理探测 |
| **P3** | `hccn_tool -roce_test ib_send_bw` | 跨机 RoCE 裸带宽 | 依赖 hccn | 与跨机 P2P 相关；非本轮必须 |

**原则**：P0 必须出图；P1 尽力；P2/P3 增强。`Health=OK` **不等于** 链路健康（Sonnet R0 已强调）。

---

## 3. 「HCCL 搜索拓扑」具体指什么

社区口语里的「让 HCCL 搜/读拓扑」**不是**一条像 `nvidia-smi topo -m` 的单一公开 CLI，而是下面三件事的合称。不确定处标 **【需机上验证】**。

### 3.1 读配置（静态，优先做）

1. **机内物理矩阵**：`npu-smi info -t topo`  
   - 产物形态：文本矩阵，单元格为 `HCCS` / `SYS` / `PIX` / `NA` 等 + CPU Affinity 列。  
   - 例（8 卡节点文献）：对角 `X`，其余多为 `HCCS`。  
   - **【需机上验证】** 本集群 16 卡（8×双 Chip）矩阵是否全 HCCS、是否分 NUMA 岛。

2. **RoCE 地址表**：`cat /etc/hccn.conf`  
   - 产物：`address_0=x.x.x.x` … 每逻辑 device 一张口。  
   - 用途：填 ranktable 的 `device_ip`；跨机可达性前提。

3. **ranktable JSON**（若存在）：  
   - 路径常见：`$RANK_TABLE_FILE`、作业目录 `rank_table*.json`、MindSpore/ModelArts 生成物。  
   - 产物字段（模板一）：`server_id`、`device_id`、`device_ip`、`rank_id`、`pod_name`。  
   - 部分机型还有 `level_list` / `net_type: TOPO_FILE_DESC|CLOS`（逻辑分层）。  
   - 本仓库 torchrun+HCCL 路径 **通常不显式传 ranktable**（用 RootInfo 建域），故探测脚本以「搜索落盘」为主。

4. **驱动 topo 目录**：`/usr/local/Ascend/driver/topo/`（出厂配置，**【需机上验证】** 是否挂载进容器）。

### 3.2 HCCL 运行时「发现」（动态，CommInit 阶段）

官方语义（CANN HCCL Overview）：通信域初始化时，HCCL 根据 **用户提供的集群信息（ranktable 或 RootInfo）+ 网络拓扑** 与其它 NPU 建链并交换参数；超时则报建链错误（本集群已见 `Getting socket times out` / `hcclCommInitRootInfoConfig error code 4`）。

触发方式：

| 方式 | 怎么触发 | 期望产物 | 风险 |
|------|----------|----------|------|
| A. 正常 `dist.init_process_group("hccl")` | 跑任意 16/32 卡 smoke | 成功=拓扑至少可建链；失败日志含首个报错 rank | 大 world 可能挂（见 Sonnet R0） |
| B. 提高日志级别 | `ASCEND_GLOBAL_LOG_LEVEL=0`（DEBUG）或 `1`（INFO）后 init | `~/ascend/log` 或容器内 Ascend 日志目录中可能出现拓扑/链路相关行 | **【需机上验证】** 具体关键字与路径；日志量大 |
| C. 官方 `hccl_test` | 编译 CANN `tools/hccl_test` 后 mpirun | 厂商基线带宽，间接证明拓扑可用 | 未编译；非本轮必须 |

**结论**：所谓「HCCL 搜索拓扑」= **读静态配置（npu-smi/hccn/ranktable）+ CommInit 时按配置建链**；没有稳定的「导出完整拓扑 JSON」公开 API。本方案以静态探测为主，动态仅作可选 smoke。

### 3.3 探测性推断（无物理工具时的保底）

固定消息（如 16M）测：

- 同 host 任意两卡 P2P 带宽 → 标定 **机内基线**  
- 不同 host 代表卡 P2P → 标定 **跨机基线**  
- 若 `intra_bw / cross_bw` 接近 1 → 拓扑标签或全走慢路径（异常）

产物：写入 summary JSON 的 `inferred_tiers`，**不替代** `npu-smi -t topo`。

---

## 4. 与现有 P2P / collective 测法挂接

### 4.1 流水线位置（强制）

```
烤机 constitution
  → [本方案] 拓扑探测 probe_hccl_topology   ← 通信 list 最前
  → HCCL collective scale（16→128）
  → HCCL P2P（ring …）
```

入口：`run_constitution_then_comm.sh` 在 `hccl_scale` 前插入拓扑步；`SKIP_TOPO=1` 可跳过。

### 4.2 拓扑图 → 解释慢边

| 慢边现象 | 拓扑如何解释 |
|----------|--------------|
| TopK 全是 `cross` 且量级符合 RoCE | **模式正常**（跨机本就慢） |
| TopK 出现 `intra` 且接近 cross 量级 | **机内 HCCS/驱动异常**；对照该 host 的 topo 矩阵与 health |
| TopK 聚集同一 `src_host` 或 `dst_host` | **坏节点**；有 hccn 则查该节点全部 `-link/-stat` |
| TopK 聚集同一跨机 host 对 | **坏链路/交换机口**；有 LLDP 则对端口 |
| collective@16 正常、@32+ 断崖 | **跨机引入点**；对照 2n32 档与 hccn_conf IP 可达性 |
| 建链超时但 Health=OK | **拓扑/网络层问题**；设备健康无法证伪（已知缺口） |

### 4.3 记录字段约定（后续改 bench 时对齐）

每条 JSONL 建议带：

```json
{
  "topo_tier": "8n128",
  "intra_or_cross": "cross",
  "link_hint": "RoCE",
  "src_host": "...",
  "dst_host": "...",
  "topo_probe_id": "topo-20260711_..."
}
```

`topo_probe_id` 指向同一次 `probe_hccl_topology` 的 summary，便于报告交叉引用。

---

## 5. 脚本行为（`probe_hccl_topology.sh`）

- **best-effort**：每条命令 `|| true`；缺 `hccn_tool` / 缺 conf 只记 `available=false`。  
- **每节点 raw**：`npu-smi info`、`-t health`、`-t topo`、`-l`、`-m`；找 `hccn_tool` 后按 device 采 link/speed/stat/ip；拷 `hccn.conf`；搜 ranktable；dump 相关 env。  
- **汇总 JSON**：`tools_available`、每节点 `npu_count`/`health_ok`、topo 是否非空、hccn 路径、conf 是否存在、发现的 ranktable 路径列表。  
- **不跑** 破坏性 roce_test（可选后续加 `ENABLE_ROCE_TEST=1`）。

---

## 6. 机上验收 checklist

见文末交付回传；通过标准：summary JSON 可解析 + 至少 P0 字段齐全。

---

## 7. Sources

- 本集群：`reports/hccl_128.md` §6、`research_comm_constitution_r0_{grok,sonnet}.md`、`scripts/cluster/run_link_health.sh`
- [npu-smi `-t topo` / HCCS 矩阵（SWIFT NPU Support）](https://swift.readthedocs.io/en/latest/BestPractices/NPU-support.html)
- [vLLM Ascend：hccn_tool link / optical / ping](https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/kv_pool.html)
- [ModelArts：hccl_test + hccn RoCE ib_send_bw](https://support.huaweicloud.com/intl/en-us/usermanual-server-modelarts/usermanual-server-0011.html)
- [CANN：HCCL Overview（建链阶段读拓扑）](https://www.hiascend.com/document/detail/en/canncommercial/800/hcclug/hcclug/hcclug_000001.html)
- [ranktable 字段 / hccn.conf → device_ip](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/developmentguide/hccl/hcclug/hcclug_000014.html)
- [GPU 进阶笔记：910B hccn_tool / HCCS](https://arthurchiao.art/blog/gpu-advanced-notes-2-zh/)
- 检索落盘：`tmp/research/ascend-hccl-topo.json`、`tmp/research/hccl-comm-constitution-r0-topo.json`

---

## 8. 交付摘要

1. 拓扑 = 物理（HCCS/RoCE）+ 逻辑（rank↔host↔IP）+ 档位标签。  
2. 工具优先级：`npu-smi` → `hccn.conf` / `hccn_tool` → ranktable/env → 带宽推断。  
3. 「HCCL 搜拓扑」= 读配置 + CommInit 建链发现；无单一 dump API。  
4. 流水线：烤机后、collective 前跑 `probe_hccl_topology`；用拓扑解释慢边而非只报 TopK。
