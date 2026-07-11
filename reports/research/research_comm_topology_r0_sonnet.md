# 通信前必须先摸清拓扑 · 独立评审（R0 / Sonnet）

> **文首声明**：本文实际生成模型为 **Claude Sonnet 5**（非代评、非 Composer 转述）。

**日期**: 2026-07-11
**性质**: 独立评审（未发现 `research_comm_topology_r0.md` 既有版本，按任务说明独立撰写；结论与 `research_comm_constitution_r0_grok.md`、`research_comm_constitution_r0_sonnet.md` 中已提出但未展开的"拓扑分层"点交叉对照）
**必读输入（已逐一读取原文）**：
- `reports/hccl_128.md`、`reports/rounds/hccl_cluster_r0.md`
- `reports/research/research_comm_constitution_r0_grok.md`、`research_comm_constitution_r0_sonnet.md`
- `scripts/cluster/{hccl_torch_bench,hccl_p2p_bench,run_hccl_scale,run_hccl_p2p_128,run_link_health,run_constitution_then_comm}.{py,sh}`
- 原始日志：`logs/link-health-20260710_224719/results/*.txt`（8 节点 `npu-smi info` 全量输出，逐行核查 bus-id / chip 结构）
- `logs/train-manual-dense16/run.log`（确认集群 CPU 架构为 `aarch64`，见 §3.1）
- `tmp/research/hccl-comm-constitution-r0-topo.json`（Grok R0 遗留的检索落盘）
- 外部：华为官方 `npu-smi`/`hccn_tool`/rank table 文档（见 §8）

**状态**：调研完成，未改代码、未上机器跑新实验。

---

## 0. 一句话

现有两份通信体质报告（Grok R0、Sonnet R0）都提到"应该区分机内 HCCS / 跨机 RoCE"，但**都把"机内=一个匀质的快域，跨机=一个匀质的慢域"当作不需要验证的常识**——而这正是全篇最大的未核实假设。翻华为官方 `npu-smi info -t topo` 的公开示例后发现：**16 个 NPU 的节点内部，HCCS 矩阵完全可能不是一整块，而是两个 8-way HCCS 子团用 PCIe/`PIX`/`PHB`/`SYS` 桥接**——如果我们的 8×16 集群恰好也是这种结构（每台机器实际是"2×8 卡板 + 板间 PCIe 桥"而不是"16 卡全对全 HCCS"），那么当前所有"机内边应该整体快、跨机边应该整体慢"的判定逻辑，在节点内部就已经可能踩坑：把"跨半板"的正常慢边误判为坏卡，或者把"同半板"的正常快边当基线去比较跨半板边，两种误判方向都成立。**而验证这件事只需要跑一条从未被调用过的命令——`npu-smi info -t topo`**（集群是 `aarch64`，已在 `logs/train-manual-dense16/run.log` 核实满足该命令的架构前提），且这条命令已经比 `hccn_tool`（8/8 节点缺失）便宜得多。这是本文与两份 R0 报告的核心分歧点：**拓扑不是"报告里加一列 tag"就完事的元数据，它是决定"哪些边可比、哪些边不可比"的判定前提，必须在测通信性能之前先拿到，而不是测完之后再补标签。**

---

## 1. 为什么"通信前必须先摸清拓扑"是第一性问题（不是第四性）

两份 R0 报告的建模顺序都是：先定义通信体质的几个轴（可靠性/性能/边级/正确性），再在"边级拓扑健康"这一条里补一句"应显式区分机内/跨机"。这个顺序看似合理，但从第一性原理看反了：

- **"哪两条边可以互相比较"这件事，本身就依赖拓扑**。没有拓扑标签之前，"host×host 热力图""慢边 TopK""bus_bw 保持率"这些统计量的**分母（谁跟谁比）**都是猜的。
- 通信体质报告的所有下游判定——"这条边慢吗""这个 world_size 断崖正常吗""这台机器是不是坏了"——本质上都是**"实测值 vs 拓扑期望值"的差**。如果拓扑期望值本身是错的（比如误把 2×8 结构当 1×16），那么下游所有"正常/异常"判定都会系统性偏移，而且**偏移的方向不可预测**（可能把正常的结构性差异当坏卡，也可能把真坏的边淹没在"反正跨机就该慢"的宽容里）。
- 所以拓扑摸清必须是**流水线里排在通信性能测试之前的独立步骤**，产出一份"拓扑基线"作为后续所有 diff 判定的分母，而不是在报告写作阶段"顺便"打个 `topo_tier` 标签。这是本文相对两份 R0 报告的核心补充。

---

## 2. Q1：没有拓扑标签时，通信「体质」报告最大的假阳性是什么

按风险大小排序，给出具体机制（不是泛泛的"应该分层"）：

### 2.1 【最大假阳性来源】把"结构性带宽阶梯"误读成"性能退化"

`hccl_128.md` 的核心图（bus_bw vs world_size）把 16→32→64→128 画成一条连续曲线，配一个"扩展效率"。但这条曲线上至少跨了两个**物理上不连续**的边界：

1. 16→32：从"全部机内"跨到"引入第一次跨机 RoCE 跳"；
2. 32→64、64→128：跨机 host 数量继续增加，但**跨机部分本身的拓扑形状**（是否所有 host 对等价、是否有 leaf/spine 分层）完全未知。

把这条曲线当一条"扩展效率"曲线来解读，等价于假设"网络是匀质的，性能只随 world_size 单调变化"。两份 R0 报告已经指出"效率公式除以 N/16 是误用"，但**即便换成保持率，只要曲线还是按 world_size 一维画的，就依然隐含"这几个 world_size 档位在拓扑意义上是可比的"这个假设**——这个假设从未被验证。真正的风险是：如果 32 卡这一档恰好选中的 2 个节点物理上离得特别近（同一 leaf 交换机）、而 64 卡这一档凑巧覆盖了跨 spine 的节点对，那么"32→64 保持率下降"里有多少是"world_size 变大导致的固有开销"、有多少是"这次选中的节点对恰好网络更远"，**在没有节点间物理拓扑标签之前，这两种解释无法区分**——报告会把"选样运气差"错判成"扩展性差"，或者反过来把真实的网络分层问题错判成"符合预期的扩展开销"。

### 2.2 【次大假阳性】P2P 慢边 TopK 在"混合分布"上被稀释或被夸大

`hccl_cluster_r0.md` 的慢边 TopK 是在**全体边**（机内+跨机混合）上排序的。这里有两个相反方向都成立的假阳性/假阴性：

- **假阳性方向**：如果全局统计量（如全体边的中位数/标准差）被机内快边拉低了基线，那么"跨机边比机内边慢"这个**完全符合物理预期**的现象，会在"相对全局中位数偏离"这类通用异常检测里，把**所有**跨机边都标成"异常慢边"——TopK 表会被跨机边占满，淹没了其中真正应该被关注的、跨机边里又格外慢的那一小撮。
- **假阴性方向**：反过来，如果分析者已经"知道"跨机边该慢、不去深究，那么一条**跨机边里真正坏掉的链路**（比如某个光模块信号质量差、比同类跨机边慢 3 倍）会被"反正跨机就该慢"这个先验直接放过，因为它可能仍然在"跨机边"这个大类的正常范围边缘，不会被绝对阈值或全局分布捕捉到。

**两者的共同根因是同一个**：没有把"同层比较"（跨机边只跟跨机边比，机内边只跟机内边比）作为异常检测的前提。这正是本文与 §1 呼应的地方：拓扑分层不是锦上添花的维度，是让 TopK/异常检测本身**有意义**的前提条件。

### 2.3 【结构性假阳性，本文新增，两份 R0 未覆盖】"机内=一个匀质快域"本身未经验证

这是本文核后发现、两份 R0 报告都没深挖的点。证据链：

1. `hccl_128.md` §6 原始日志（`logs/link-health-20260710_224719/results/*.txt`，已逐节点核查）显示：npu-smi 把每个物理 NPU 卡拆成两行——"NPU X / Chip 0"与"NPU X / Chip 1"，对应两个不同的 PCIe Bus-Id（如 `0000:9D:00.0` 和 `0000:9F:00.0`）。`hccl_128.md` 末尾也写了"所有 NPU 型号为 Ascend910，**双 Chip 结构**"。这意味着 16 个逻辑 NPU 实际是 **8 张物理卡 × 每卡 2 die**，不是 16 张独立卡。
2. 华为官方 `npu-smi info -t topo` 文档给出的 16-NPU 单节点真实示例（见 §8 引用）显示：**NPU0–NPU7 互相全部是 `HCCS`，NPU8–NPU15 互相全部是 `HCCS`，但 NPU0 与 NPU8 之间是 `PIX`（更慢的 PCIe 路径），不是 `HCCS`**。也就是说，一个 16-NPU 节点完全可能是"**两个 8-way HCCS 全互联子团 + 子团间靠 PCIe 桥接**"的结构，而不是我们隐含假设的"16-way 全互联"。
3. 我们的 8×16 集群目前**从未跑过 `npu-smi info -t topo`**——`run_link_health.sh` 只跑了 `npu-smi info` 和 `npu-smi info -t health`，两者都不包含 NPU 间拓扑矩阵。也就是说，**"机内是不是一个匀质快域"这个当前所有报告都隐含依赖的前提，在本集群上是零证据的**。

如果本集群的真实结构也是"2×8 HCCS 子团"（考虑到都是 Ascend910 双 Chip、8 卡/节点，这是完全合理的先验，见 §3），那么：
- 现有"机内边应该整体快、且分布紧"的检查逻辑，会把**跨子团**的机内边（本该走 PCIe，天然比同子团 HCCS 边慢）误判为"这台机器有问题"；
- 反过来，如果只抽样了同子团内的边（比如 P2P ring 策略正好只连了子团内相邻 rank），会把"机内应该整体快"的结论建立在**只覆盖了一半真相**的样本上，一旦later 用这份"机内基线"去对比跨机边，基线本身就是偏高的，会让跨机退化显得比实际更严重。

**这是本文认为排名第一的假阳性风险**：它不是某条边测错了，而是**分析框架的分母（"机内基线"）本身可能建立在错误的匀质性假设上**，会系统性污染所有下游判定，且传播路径隐蔽（报告读者只会看到"某几条机内边莫名慢"或"跨机退化比预期严重"，很难反推到"是子团划分的问题"）。

### 2.4 【调度层假阳性，本文新增】rank 序号 = 物理位置的假设未经验证

`hccl_cluster_r0.md` 和 `hccl_128.md` 的 host×host 热力图，隐含把 `rank // 16` 当作"节点索引"、`rank % 16` 当作"该节点内的物理槎位/device_id"。但 `run_hccl_scale.sh`/`run_hccl_p2p_128.sh` 用的是 `torchrun --node_rank=$r` 里 `pod_for_rank()` 函数手动指定的顺序（`master-0`→rank0，`worker-0..6`→rank1..7），这个顺序是**运维脚本里硬编码的字符串顺序**，**不代表**这些 pod 在物理机架/交换机上的排布顺序。也就是说：

- host×host 热力图里"host0 vs host3"这类标签，只反映"哪个 pod 被 torchrun 赋了哪个 node_rank"，**不代表**"host0 和 host3 在物理网络拓扑上离得有多远"；
- 如果调度器（`vcctl`/K8s）本身在物理机架上打乱了 `master-0`/`worker-0..6` 的排布（这在容器化集群里是常态，pod 名字与物理位置解耦是设计目标之一），那么"host×host 热力图"上任何"看起来像规律"的模式（比如"host0 到所有 worker 都慢"）**既可能是 host0 真的网络差，也可能只是碰巧 host0 在物理上离其余节点都远**——这两种解释在当前证据下无法区分，需要额外读取调度层的物理位置元数据（若存在）或做穷举的 host-pair P2P 矩阵（§3 已提及但尚未在 128 卡跑通）来交叉验证。

---

## 3. Q2：Ascend 8×16=128 上合理的拓扑假设 vs 必须实测的部分

### 3.1 集群基本事实（已核实，非假设）

| 事实 | 证据 | 状态 |
|---|---|---|
| CPU 架构为 `aarch64` | `logs/train-manual-dense16/run.log` 内 `uname`/环境输出含 `aarch64` | **已核实** |
| 每节点 8 张物理 Ascend910 卡 × 每卡 2 die = 16 逻辑 NPU | `hccl_128.md`§6 + `link-health` 原始 `npu-smi info` 输出（NPU X 分两行 Chip 0/1，两个独立 PCIe Bus-Id） | **已核实** |
| 8 节点（master-0 + worker-0..6） | `run_link_health.sh` POD 列表、`hccl_128.md`§6 | **已核实** |
| `hccn_tool` 全部 8 节点缺失 | `hccl_128.md`§6、`run_link_health.sh` 输出 | **已核实** |
| `npu-smi info -t topo` 从未被调用 | 通读 `run_link_health.sh`（仅 `info` / `info -t health`） | **已核实（缺口，非事实性结论）** |

### 3.2 分层拓扑假设表（哪些能合理假设，哪些必须实测）

| 层级 | 合理假设（依据公开资料，非本集群实测） | 置信度 | 必须实测的原因 | 验证方式 |
|---|---|:---:|---|---|
| **L0：die 内 / die 间（同一物理卡的 2 个 Chip）** | 同卡两个 die 间有专用高速互联（HCCS die-to-die 或封装内互联），带宽应显著高于卡间 HCCS | 中 | 华为文档专门列了"Querying the SIO Status Between Dies of a Chip"这个独立子命令，说明厂商自己都把它当作**独立于卡间 HCCS 的一层**，具体带宽/是否对称需要实测，不能直接套用卡间 HCCS 的公开带宽数字 | `npu-smi info -t topo`（若支持 die 级）+ die-pair 内 P2P 微基准（同卡两 device_id 之间） |
| **L1：节点内卡间（8 卡 / 16 die）** | 8 张卡之间用 HCCS 互联；**但不能假设是 16-way 全互联**——§2.3 已用官方示例证明存在"2×8 HCCS 子团 + PCIe 桥接"的真实案例 | **低**（这正是本文要打掉的默认假设） | 唯一能证伪/证实的方法是拿到本节点的 `npu-smi info -t topo` 真实矩阵；不实测就无法排除"我们也是 2×8 结构"的可能性 | `npu-smi info -t topo`（逐节点跑一次即可，见 §4） |
| **L2：跨节点（8 节点，RoCE）** | 每节点通过若干 RoCE NIC（`device_ip`，见 `/etc/hccn.conf`/rank table 文档）接入网络；带宽受 NIC 线速与是否多口聚合限制，公开资料给出 910B 常见配置为 200Gbps 级 NIC，但**本集群实际 NIC 数量、线速、是否所有 16 个 device 共享同一组 NIC 或存在亲和性分组，完全未知** | 低 | 直接决定"跨机 collective 的理论峰值"这个报告里反复用来对照的分母；`hccl_128.md` 的 256M All-Reduce 128 卡 137.76 GB/s 到底是"接近峰值"还是"远低于峰值"，没有这个数字就无法判断 | `cat /etc/hccn.conf`（若存在）+ `hccn_tool -i <id> -ip -g`（`hccn_tool` 缺失时降级为向平台侧要网络规格） |
| **L3：跨节点物理网络形状（leaf/spine、是否分组、是否有 oversubscription）** | 完全没有公开信息可假设——这是数据中心私有网络设计，任何"应该是 fat-tree/non-blocking"的假设都是纯猜测 | **无**（不应假设） | 决定"8 个节点两两之间是否等价"，直接影响 §2.4 提到的 host×host 热力图能否被信任 | `hccn_tool -i <id> -lldp -g`（读直连交换机端口）或向平台/网络侧要机架-交换机拓扑图；纯软件测量方式是穷举 8 选 2=28 对 host-pair 做 P2P 矩阵，看是否有非均匀分组（成本可控，见 §5） |
| **L4：rank↔物理位置映射** | 当前用 `torchrun --node_rank` 硬编码顺序，**不能假设**这个顺序与任何物理排布一致 | 无 | 直接决定 host×host 热力图的标签是否有意义（§2.4） | 每次实验落盘时额外记录 `HOSTNAME`、`vcctl pod exec` 返回的物理节点标识（若调度层暴露），把"host_label"和"pod_name"分开记录，不要把 pod 名字直接当物理位置 |

### 3.3 一句话结论

**唯一可以直接沿用公开资料、不需要实测的层级几乎没有**——L0/L1 的具体连接形状因封装/主板设计而异，L2 的实际网卡规格因采购配置而异，L3/L4 完全是私有信息。可以合理假设的只是"存在分层"这个**结构性事实**（HCCS 通常比 PCIe 快、PCIe 通常比跨机 RoCE 快），但**分层的具体位置（哪些设备在哪一层）必须实测**，不能假设"16 个逻辑 NPU = 1 个匀质机内域"。

---

## 4. Q3：HCCL/hccn/npu-smi 各自能给出什么；搜拓扑/读 config 的可行路径

### 4.1 三个工具的能力边界（对照表）

| 工具/命令 | 能给出什么 | 当前集群状态 | 覆盖层级（对照 §3.2） |
|---|---|---|---|
| `npu-smi info` | 设备级健康、功耗、温度、HBM 用量、PCIe Bus-Id、进程占用 | **已用**（`run_link_health.sh`），8/8 节点 Health=OK | 仅设备自身状态，**不含任何拓扑信息** |
| `npu-smi info -t health -i <n>` | 单卡健康细节（MCU/ECC 等） | **已用** | 同上 |
| **`npu-smi info -t topo`** | NPU×NPU 亲和矩阵（`HCCS`/`PIX`/`PHB`/`SYS`）+ NPU-CPU NUMA 亲和 | **从未调用**（本文最大发现） | **直接回答 L1**（节点内卡间是否全互联），间接约束 L0（若同卡两 chip 在矩阵里可辨识） |
| `npu-smi info -t topo`（更深子命令：SuperPod / HCCS Lane / SIO between dies，见官方 24.1.0 命令参考） | die 级互联状态、HCCS lane 带宽、SuperPod 归属 | 未调用；**需先确认本集群 npu-smi 版本（`25.3.rc1.2`）是否支持这些子命令** | 直接回答 **L0**（die 间是否独立于卡间 HCCS） |
| `hccn_tool -i <id> -link -g` | 单个 RoCE 网口 link up/down、速率协商状态 | **8/8 节点缺失**（`hccl_128.md`§6） | L2（网口物理层状态） |
| `hccn_tool -i <id> -ip -g` | 该 NPU 绑定的 RoCE NIC IP（`device_ip`） | 缺失 | L2（跨机寻址映射，是 rank table 的数据源） |
| `hccn_tool -i <id> -lldp -g` | 直连交换机的 LLDP 信息（对端设备/端口/系统描述） | 缺失 | **唯二能直接回答 L3**（是否同交换机）的手段之一 |
| `hccn_tool -i <id> -net_health -g` / `-roce_test ib_send_bw` | 网络连通性检测 / RoCE 带宽压测 | 缺失 | L2/L3 交叉验证 |
| `cat /etc/hccn.conf` | 静态配置文件，含每个 device 的 `address_x`（RoCE NIC IP） | **未检查过**（比 `hccn_tool` 更底层，二进制缺失不代表配置文件也缺失） | L2（不需要 `hccn_tool` 二进制就能读，是当前被忽略的最便宜路径之一） |
| `RANK_TABLE_FILE`（rank table json，若使用） | 显式的 `server_id`/`device_id`/`device_ip`/`rank_id` 映射，是 HCCL 官方"拓扑地图" | **本集群未使用**（当前用 `torchrun` + TCP store 初始化，不走 rank table 路径） | 若存在，直接给 L2/L4 的权威映射；本集群大概率没有现成文件，需要判断是否值得为了拿这份映射额外跑 `hccl_tools.py` 生成一次 |
| HCCL/CANN 运行时日志（`ASCEND_SLOG_PRINT_TO_STDOUT` 等 CANN 日志环境变量，具体开关名**未在本集群验证**） | 理论上 comm init 阶段可能打印 rank↔device 映射或建链细节 | **未验证，本文不下结论**，仅作为低优先级候选 | 需先在 16 卡小规模开一次日志确认输出格式，再决定是否纳入正式流程 |
| 调度层元数据（`vcctl`/K8s node 标签，若暴露机架/交换机信息） | 物理位置标签（如 rack/switch id） | **未检查**——这是本文认为在"软件测量"之外，成本最低、最该先问一句的路径 | 直接回答 L3/L4，且不需要登录到 pod 内部执行任何命令 |

### 4.2 可行路径排序（成本从低到高）

1. **问平台/运维一句话**：这 8 个 pod 当前物理机架/交换机分组是什么（若调度系统本身就记录）。零代码成本，直接部分回答 L3/L4，但依赖对方是否愿意/能够提供。
2. **`cat /etc/hccn.conf`**（8 节点各一次）：不需要 `hccn_tool` 二进制，只要文件存在就能拿到 `device_ip` 映射，直接回答 L2 的寻址结构。当前完全没试过，成本几乎为零。
3. **`npu-smi info -t topo`**（8 节点各一次）：本文最推荐的下一步，直接证实/证伪 §2.3/§3.2 的"2×8 HCCS 子团"假设，成本极低（几秒钟一条命令），且架构前提（`aarch64`）已核实满足。
4. **`npu-smi info -t topo` 的深层子命令**（SuperPod/HCCS lane/die 间 SIO）：先确认版本支持，若支持则直接回答 L0。
5. **`hccn_tool` 修复**（镜像补件或 PATH 修复）：两份 R0 报告都已列为 P0/P1 项，本文认同优先级，但强调**即使修不好，1-4 也能拿到大部分 L1/L2 拓扑信息**——不应把"摸清拓扑"整体阻塞在"hccn_tool 何时修好"上。
6. **穷举 host-pair P2P 矩阵**（8 选 2 = 28 对代表卡，`host-rep` 策略，两份 R0 报告 P1 已提议）：软件层面交叉验证 L3，成本可控但需要 128 卡通信先稳定（依赖 Sonnet R0 §3.1 指出的可靠性问题先解决）。
7. **RANK_TABLE_FILE 生成 + HCCL/CANN 日志开关验证**：优先级最低，仅在 1-6 都做完仍有疑点时再考虑。

---

## 5. Q4：最小充分拓扑交付物（一张图 + 一张表就够吗？）

**不够。** 理由和推荐的最小集合如下：

### 5.1 为什么"一图一表"不充分

- 一张图撑不住两个不同尺度的矩阵：**节点内**是 16×16（或考虑 die 结构后更细）的高密度矩阵，**节点间**是 8×8 的稀疏矩阵，两者物理意义（HCCS 亲和类型 vs RoCE 实测带宽/延迟）完全不同，硬塞进一张图会互相稀释视觉信息（这也是 `hccl_cluster_r0.md` 已经分了 host×host 和 rank×rank 两套图的原因，本文认为拓扑基线部分同理需要拆开）。
- 一张表撑不住"哪些是实测、哪些是假设"的置信度区分——如果只给一张扁平的"设备→层级"表，读者无法分辨表里每一行是"npu-smi topo 实测出来的"还是"我们假设的"，这正是 §2.3 说的假阳性风险的根源，**表本身必须自带置信度列**，否则等于什么都没说清楚。

### 5.2 推荐的最小充分集合

| 编号 | 产物 | 内容 | 更新频率 |
|---|---|---|---|
| **T1（表，机器可读优先）** | 拓扑基线表 | 每个 `(CLUSTER_JOB, pod, device_id)` → `host_label`、`chip_pair_id`（同卡 2 die 归组）、`hccs_group_id`（§4 §3.2 L1 实测结果）、`device_ip`（若拿到）、置信度列（`measured` / `assumed` / `unknown`） | **每个 CLUSTER_JOB 生命周期一次**（除非重新调度/换 job） |
| **G1（图）** | 单节点 NPU×NPU 拓扑矩阵（`npu-smi info -t topo` 原始矩阵可视化，或至少贴一份代表节点的原始文本） | 直接证实/证伪 L0/L1 的匀质性假设 | 每种物理节点型号跑一次即可（8 节点若同型号，抽 1-2 个节点验证一致性，不需要全跑 8 次图） |
| **G2（图）** | host×host 拓扑感知矩阵（不是性能矩阵，是"距离层级"矩阵：同节点/跨节点，若能拿到 L3 则细分是否同交换机） | 让后续所有 host×host **性能**热力图有一份"预期结构"可以对照 | 与 T1 同频率；若 L3 无法测得，明确标注"未知，按跨节点统一处理"而不是留空 |
| **附：一段 3-5 句的文字说明** | 用自然语言把 T1/G1/G2 的核心结论写清楚（比如"本集群 8 节点均为 8 卡×2die=16 NPU；npu-smi topo 显示节点内为 [全 16-way HCCS / 2×8 HCCS 子团]（实测后二选一填写）；跨节点拓扑分组未知，按均匀处理"） | 防止 T1/G1/G2 被后续报告引用时脱离上下文误读 | 与基线同步更新 |

**核心原则**：拓扑基线是**环境指纹**，不是**实验产物**。它应该像"CANN/驱动版本号"一样被当作一次性（或低频）采集、随后被所有性能测试引用的元数据，而不是每轮通信体质报告都要重新画的图。这也直接回答 §1 提出的问题——它必须在性能测试**之前**产出并冻结，而不是性能测试之后再补。

---

## 6. Q5：插入流水线的位置与失败降级策略

### 6.1 插入位置

在 `run_constitution_then_comm.sh` 现有的 4 步（constitution → plot → hccl_scale → hccl_p2p）之前，新增 **Step 0：拓扑探针**（建议新脚本 `run_topo_probe.sh`，复用 `run_link_health.sh` 的 pod 遍历骨架）：

```
Step 0（新增）  拓扑探针：npu-smi info -t topo + cat /etc/hccn.conf（+ hccn_tool 若存在）
Step 1          烤机体质（现有）
Step 2          本地出图（现有）
Step 3          HCCL collective（现有）
Step 4          HCCL P2P（现有）
```

放在最前面而不是并行/事后补充，原因见 §1：后面所有步骤的判定逻辑都依赖这份基线。

### 6.2 缓存与失效策略（对应 §5.2「环境指纹」原则）

- 探针结果按 `CLUSTER_JOB` 落盘缓存（如 `logs/topo-probe-<CLUSTER_JOB>/`）。
- `run_constitution_then_comm.sh` 每次运行时，先做一次**极低成本的指纹检查**（比对当前 8 个 pod 名称 + `HOSTNAME` 的哈希是否与缓存一致），一致则跳过完整探针，只打一条"复用拓扑基线 @ `<stamp>`"的日志；不一致（job 重建、pod 漂移）则强制重新跑完整探针。
- 提供 `FORCE_TOPO_PROBE=1` 手动强制刷新的开关，与现有 `SKIP_CONSTITUTION`/`SKIP_COMM` 风格一致。
- 每条 collective/P2P 的 JSONL record 增加一个 `topo_fingerprint` 字段（指向本次探针的 stamp），使后续分析可以严格核对"这批性能数据用的是哪份拓扑基线"，避免拓扑在两轮实验之间悄悄漂移却无人发现。

### 6.3 失败降级策略（每一层单独降级，不整体阻塞）

| 故障场景 | 降级动作 | 报告侧标注 |
|---|---|---|
| `npu-smi info -t topo` 在某个/某几个节点报错（版本不支持、非 Arm 等，虽然本集群已核实 `aarch64`） | 该节点跳过，其余节点正常收集；不中止 Step 0 | 明确列出"缺失节点名单"，而不是笼统写"部分数据不全" |
| `npu-smi info -t topo` 深层子命令（SuperPod/HCCS lane/die 间 SIO）不支持 | 退回只用粗粒度 `-t topo` 矩阵，L0（die 间）标记为 `unknown`，不假装已验证 | T1 表里 `chip_pair_id` 置信度列标 `assumed`，不是 `measured` |
| `/etc/hccn.conf` 不存在或为空 | 跳过 `device_ip` 采集，L2 寻址映射标记为 `unknown` | 与 `hccn_tool` 缺失合并成同一条"跨机链路层不可验证"限制说明，**提升为阻塞项级别的措辞**（呼应 Sonnet R0 §3.4 对"仅小字脚注"的批评，不能重复同样的问题） |
| `hccn_tool` 全部缺失（当前已知状态） | Step 0 不因此失败；探针脚本本身不依赖 `hccn_tool`，只是"如果存在就多采集一些" | 报告需要写清楚"L3（跨机物理网络形状）完全不可验证"，并说明这会限制哪些结论（host×host 热力图上的"聚集"判断不能排除是拓扑本身分组导致，而非坏节点） |
| Step 0 整体运行时间异常长（比如某节点 ssh 连接卡住） | 遵循现有 `run_step` 的 `CONTINUE_ON_FAIL` 模式，单节点超时不阻塞其余节点；Step 0 本身失败也不应阻断 Step 1（烤机）——拓扑探针失败 ≠ 不能烤机，只是通信部分的判定要打折扣 | Step 0 失败时，Step 3/4（通信性能测试）仍可运行，但生成的通信报告必须在开头醒目位置写"本轮拓扑基线缺失/不完整，以下 host×host/rank×rank 判断的可信度受限"，不能悄悄跳过这句话直接出图 |
| 拓扑基线与实际性能测试之间存在时间窗口，期间发生了 pod 重建/漂移 | §6.2 的指纹检查机制兜底；若指纹检查本身因为某种原因跳过了（比如手动改了脚本），至少在合并报告时交叉核对 `topo_fingerprint` 与实际测试时间戳是否落在同一个 `CLUSTER_JOB` 生命周期内 | 报告生成脚本（`gen_*.py`）里加一条断言：若某条 record 的 `topo_fingerprint` 缺失或找不到对应基线文件，直接在图上用醒目颜色/文字标出"该批数据无拓扑基线可对照"，而不是静默当作"机内"处理 |

**降级设计的核心原则**：拓扑探针的失败应该是**可见的降级**（报告里明确写清楚哪一层不可验证、因此哪些结论不可信），而不是**静默的降级**（报告照常出图，只在文末小字提一句限制说明）——这正是 Sonnet R0 §3.4 已经点名批评过 `hccl_128.md` 的问题（"链路健康结论建立在会产生假阳性的信号上"），本文把同样的原则系统化应用到"拓扑"这一整个前置步骤上。

---

## 7. 与两份 R0 报告的关系

| 结论 | Grok R0 / Sonnet R0 | 本文 | 关系 |
|---|---|---|---|
| 应显式区分机内 HCCS / 跨机 RoCE | 两份均已提出，作为"体质"定义里的一条 | 认同，但指出这只是**必要不充分**——"机内"本身可能不是匀质的（§2.3） | **一致 + 关键修正** |
| `topo_tier`（`1n16`/`2n32`/`4n64`/`8n128`）写入 record | Grok R0 P0.4 已提出 | 认同该标签有用，但指出它只标注了"跨了几个节点"，**没有回答节点内部/节点间是否匀质**这个更底层的问题 | **一致 + 不充分** |
| `hccn_tool` 缺失是硬缺口 | 两份均指出 | 认同，但补充：**`npu-smi info -t topo` 和 `/etc/hccn.conf` 是完全独立于 `hccn_tool` 二进制的两条路径，此前从未被尝试**，不应把"摸清拓扑"整体阻塞在 `hccn_tool` 修复上 | **一致 + 新增可行路径** |
| host×host / rank×rank 热力图 | 两份均已给出图表方案（G3-G8） | 认同图表设计，但指出：**这些图的"host"标签本身可能不代表物理位置**（§2.4），需要先有拓扑基线交叉验证标签含义 | **一致 + 前置依赖** |
| 128 卡通信可靠性问题（SIGSEGV / socket timeout 等） | Sonnet R0 §3.1 深入分析 | 本文不重复该分析，但指出：**如果这些失败与特定的物理拓扑位置（如特定 host pair、特定跨子团边）相关，当前完全没有能力做这个交叉分析**——因为拓扑基线不存在 | **正交补充**（可靠性维度 × 拓扑维度的交叉分析，目前两个维度都还没准备好互相印证） |

---

## 8. Top 5（交付摘要）

1. **"机内=一个匀质快域"是当前所有通信体质报告共享的最大未核实假设**：华为官方 `npu-smi info -t topo` 的公开示例证明，16-NPU 节点完全可能是"2×8 HCCS 子团 + PCIe 桥接"结构，而本集群恰好是 8 卡×2die=16 NPU 的配置，与该示例结构高度吻合但从未验证。这个假设一旦错误，会系统性污染所有下游"机内边应该整体快"的判定，且传播路径隐蔽。
2. **`npu-smi info -t topo` 从未被调用，是当前最便宜、零新依赖就能填的最大拓扑缺口**：不需要修 `hccn_tool`（8/8 节点缺失），不需要新工具，集群架构（`aarch64`，已核实）满足该命令的前提，几秒钟一条命令就能直接证实/证伪 §1 的核心假设。
3. **拓扑标签不是"报告里加一列 tag"，而是决定"哪些边可比"的判定前提**，必须在性能测试**之前**产出并冻结（环境指纹式的一次性/低频基线），不能等性能测出来之后再补标签——补标签解决不了"基线本身建立在错误匀质性假设上"的问题。
4. **rank 序号 = 物理位置这个隐含假设同样未经验证**：`pod_for_rank()` 是运维脚本硬编码的字符串顺序，不代表 pod 在物理机架/网络上的真实排布；host×host 热力图上的任何"规律"在拿到调度层物理位置元数据或做穷举 host-pair 交叉验证之前，都存在"标签对但物理含义错"的风险。
5. **最小充分拓扑交付物不是"一图一表"，而是"一份带置信度列的机器可读基线表 + 至少两张图（节点内 HCCS 矩阵 + 节点间距离层级矩阵）+ 一段防脱离上下文误读的文字说明"**，且必须配套失败时的**可见降级**（明确写清楚哪层不可验证、因此哪些结论打折扣），不能像 `hccl_128.md` §6 那样把关键限制压缩成文末小字。

## 最小充分交付物（复述，供快速引用）

1. **T1**：拓扑基线表（`CLUSTER_JOB × pod × device_id` → `host_label`/`chip_pair_id`/`hccs_group_id`/`device_ip`/置信度），按 job 生命周期缓存。
2. **G1**：单节点 NPU×NPU 拓扑矩阵图（`npu-smi info -t topo` 可视化），验证 L0/L1 匀质性假设。
3. **G2**：host×host 拓扑感知（距离层级）矩阵图，作为后续性能热力图的"预期结构"对照。
4. 三者必须在通信性能测试（`run_hccl_scale.sh`/`run_hccl_p2p_128.sh`）**之前**产出，并通过 `topo_fingerprint` 字段与后续每条性能 record 关联。

---

## 9. Sources

**本集群一次证据（均已逐行核查，非转述）**：
- `reports/hccl_128.md`（§6 链路健康，双 Chip 结构提示）
- `reports/rounds/hccl_cluster_r0.md`（16 卡 P2P 边画像方法）
- `logs/link-health-20260710_224719/results/huawei-8node-copy-master-0.txt`（原始 `npu-smi info` 输出，确认 NPU/Chip 双行结构与独立 PCIe Bus-Id）
- `logs/train-manual-dense16/run.log`（确认集群 `aarch64` 架构）
- `scripts/cluster/run_link_health.sh`（确认当前探针仅覆盖 `npu-smi info`/`-t health`，未覆盖 `-t topo`）
- `scripts/cluster/run_hccl_scale.sh`（`pod_for_rank()` 硬编码顺序，§2.4 证据）
- `scripts/cluster/hccl_p2p_bench.py`（ring/star 边生成逻辑）
- `scripts/cluster/run_constitution_then_comm.sh`（现有流水线结构，Step 0 插入点参照）
- `reports/research/research_comm_constitution_r0_grok.md`、`research_comm_constitution_r0_sonnet.md`（对照文本）
- `tmp/research/hccl-comm-constitution-r0-topo.json`（Grok R0 遗留检索，本文独立复核）

**外部引用**：
- [华为文档：查询多 NPU 的拓扑结构（`npu-smi info -t topo`，HCCS/PIX/PHB/SYS 矩阵示例）](https://support.huawei.com/enterprise/en/doc/EDOC1100442268/da99ae64/Querying%20the%20Topology%20of%20Multiple%20NPUs.htm)
- [npu-smi 基本使用（含 16-NPU 节点 `-t topo` 真实输出示例，显示 NPU0-7/NPU8-15 两个 HCCS 子团）](https://www.chenshaowen.com/blog/basic-usage-of-npu-smi.html)
- [华为文档：`npu-smi info -t topo` 子命令列表（SuperPod 信息、HCCS Lane 拓扑/带宽、Die 间 SIO 状态）](https://support.huawei.com/enterprise/en/doc/EDOC1100442265/f1eb778f/Querying%20the%20Topology%20of%20Multiple%20NPUs.htm)
- [昇腾社区：rank table 文件配置资源信息（`server_id`/`device_id`/`device_ip`/`rank_id` 定义）](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/hccl/hcclug/hcclug_000014.html)
- [昇腾分布式训练实战：`hccn_tool` 配置到 Rank table 生成全流程](https://blog.csdn.net/ii567/article/details/152885564)
- [MindSpore：rank table 与 `/etc/hccn.conf` 的 `address_x` 对应关系](https://www.mindspore.cn/docs/en/r2.4.10/model_train/parallel/rank_table.html)
- [华为文档：HCCN Tool 接口参考（`-link`/`-lldp`/`-ip`/`-net_health`/`-roce_test`）](https://support.huawei.com/enterprise/zh/doc/EDOC1100523657/426cffd9)
- [GPU 进阶笔记（二）：华为昇腾 910B（HCCS/`hccn_tool` 基础用法，此前已被 Grok R0 引用，本文独立复核）](https://arthurchiao.art/blog/gpu-advanced-notes-2-zh/)
