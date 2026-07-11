# 卡间通信体质报告 · 独立调研（R0 / Sonnet）

> **文首声明**：本文实际生成模型为 **Claude Sonnet 5**（非代评、非 Composer 转述）。

**Track**: E2（`research_constitution_r4_tracks.md`，与 E1/Grok 并行独立调研）
**日期**: 2026-07-11
**输入**（必读，已逐一读取原文/原始日志，非转述）：
- `reports/hccl_128.md`
- `reports/rounds/hccl_cluster_r0.md`
- `scripts/cluster/hccl_torch_bench.py`、`hccl_p2p_bench.py`、`run_hccl_scale.sh`、`run_hccl_p2p_128.sh`、`run_link_health.sh`
- `reports/research/research_nccl_verify_r0.md`（方法论对照，Track B 早期调研）
- `reports/research/research_comm_constitution_r0_grok.md`（**已对照，但本文结论独立复核；不一致处见 §7）
- **原始日志逐行核查**（`logs/hccl-*`、`logs/hccl-cluster-r0-*`）— 这是本文与 Grok R0 最大的方法差异：我没有停在“文档说 SIGSEGV”，而是把 8 次相关跑批的日志全部翻出来对时间线。

**状态**：调研完成，未改代码、未上机器跑新实验（遵循 E2 任务边界：独立判断 + 交付 markdown）。

---

## 0. 一句话

现有测法把“通信体质”简化成了“稳态 bus_bw 曲线好不好看”，但翻遍 `logs/` 下 8 次 128 卡相关尝试后发现：**同一套脚本、同一个集群、同一小时窗口内，128 卡跨机通信用三种完全不同的方式失败过（SIGSEGV / 建链 socket 超时 / `hcclCommInitRootInfoConfig` 内部错误），而“成功”的那次报告只是幸存者**——64 卡那次尝试其实也失败过一回，报告只留了干净的一份。所以本轮最该建的不是更漂亮的热力图，而是**把“重复调用同一规模，多少次能成功”本身当成通信体质的第一指标**，其次才是 bus_bw 口径修正、拓扑分层和边级画像。

---

## 1. 建模：128 卡（8×16）通信体质该怎么刻画

### 1.1 先对齐一个前提：算力体质 vs 通信体质不是同一把尺子

算力体质（`CARD_SCREEN`/`research_card_constitution_*`）问的是“这张卡是不是偏慢”，本质是**单点确定性测量 + 分布定位坏个体**。通信体质问的是完全不同的问题：**这条边/这次建链/这个 collective 能不能稳定跑完，跑完之后快不快**。两者共享“分布优先、不用单点阈值”的哲学，但通信体质多了一个算力体质没有的维度——**过程本身会失败**（进程 SIGSEGV、建链超时、comm 初始化报错），而算力探针几乎不会在测量过程中把整机拖挂。这一点在 Grok R0 的对照表里没有单独列出来，我认为它应该是第一条。

### 1.2 通信体质的四个正交轴

| 轴 | 问题 | 现状覆盖 | 关键证据 |
|---|---|---|---|
| **A. 可靠性（建链/存活率）** | 同一规模重复跑 N 次，能跑完的概率是多少？失败长什么样？ | **几乎零覆盖**——所有脚本 `run_scale \|\| true` 吞掉失败继续下一档，失败样本不落盘、不进报告 | 见 §3.1 |
| **B. 稳态性能（bus_bw / 延迟）** | 跑完之后，collective 在 op×size×world 矩阵上快不快，退化形态是否符合拓扑预期 | 有（`hccl_128.md`），但口径公式有误用 | 见 §3.2 |
| **C. 边级拓扑健康** | 具体哪条边/哪个节点对慢，是孤立坏边还是全网退化 | 仅 16 卡有（`hccl_cluster_r0.md`），128 卡从未产出过一份完整边矩阵 | 见 §3.1 |
| **D. 正确性** | collective/P2P 的结果本身对不对，还是“跑得快但值错了” | P2P 有抽样校验；**collective 完全没有** | 见 §3.3 |

`128 = 8×16` 这个具体拓扑还决定了 A/B/C 三轴都必须**显式区分机内 HCCS（16 卡内）与跨机 RoCE（跨节点）**，因为失败模式、带宽量级、正常退化曲线在两层上完全不同——这一点与 Grok R0 §1.2 的判断一致，我独立复核后认同，不重复展开。

### 1.3 我与 Grok R0 建模上的分歧点

Grok R0 把“通信体质”定义收窄在 B+C（性能模式是否正常），A（可靠性）被处理成“128 P2P SIGSEGV 是个待修的 bug”，是**实现缺陷**而不是**体质的一个维度**。我不同意这个归类：翻日志后发现，A 轴的失败不是 P2P 脚本独有的 bug，是**任何跨节点 HCCL 初始化/建链路径**在这个集群上都会偶发触发的现象（下面 §2、§3.1 用原始日志证明）。如果只把它当 bug 修一次，下次别的脚本/别的负载在同样规模下大概率还会踩到同一类问题——所以它必须被**建模为体质的一个可测量维度**（成功率 + 失败签名分布），而不是修完就消失的一次性缺陷。

---

## 2. 模式是否正常：怎么用分布/拓扑/消息尺寸曲线判断（而非单一阈值）

### 2.1 稳态性能：用“保持率”而不是“效率”

`hccl_128.md` §4 的公式：

```
效率 = (bus_bw_N / bus_bw_16) / (N / 16) × 100%
```

我独立重算了一遍 §3 的原始表：256M All-Reduce，16 卡 149.85 GB/s → 128 卡 137.76 GB/s，**只降了 8.1%**，但报告套公式后写成“11.5% 弱扩展效率”。`bus_bw`（NCCL/HCCL 语境下的 bus bandwidth）本身的设计目标就是在不同 rank 数下给出一个**理论上应接近常数**的数字（多跳传输已经折算掉了），所以拿它再除以 `N/16` 相当于把“几乎没退化”硬算成“退化了 8 倍”——这是公式误用，不是集群真的差。**这一点上我与 Grok R0 §3.3 结论一致**（两人独立复算，数字吻合），建议直接采纳：主指标改成 `保持率 = bus_bw_N / bus_bw_16 × 100%`（256M AR ≈ 92%），旧“效率”公式降级为脚注说明。

判断“模式是否正常”不能只看这一个保持率数字，要看**曲线形状**：

| 观察 | 正常模式 | 异常信号 |
|---|---|---|
| bus_bw vs message size（固定 world） | 随消息增大单调上升后趋于平台（带宽受限区），小消息由延迟主导 | 大消息仍剧烈波动，或平台值远低于同规模历史基线 |
| bus_bw vs world_size（固定大消息） | 机内→跨机跨界处有一次台阶式下降，之后缓降或持平（保持率） | 某个 world_size 断崖式下跌（如 `hccl_128.md` 里 16M All-Gather 从 16 卡→32 卡直接掉到 31% 保持率，值得单独标注复测，而不是被平均掉） |
| 同规模重复跑的方差 | 多次跑的 bus_bw 应该落在小范围内（CV 低） | **本集群目前无法回答这个问题**——因为从没在同一规模上跑过≥2次“都成功”的重复实验来算方差（见 §3.1，唯一的两次全量 sweep 里一次还失败了） |

### 2.2 边级拓扑：host×host 聚集 + TopK，而非“平均延迟”

16 卡的 `hccl_cluster_r0.md` 已经示范了正确姿势（host×host 相对中位数偏差热力图 + 慢边 TopK），判断逻辑应该是：

- **机内边应显著快于跨机边**（HCCS vs RoCE 物理带宽/跳数不同），如果某台机器的机内边慢到接近跨机边，这台机器本身有问题；
- **慢边应该是孤立点，不是某个 host 的所有边都慢**——如果 TopK 慢边全部聚集在同一个 host（无论是 src 还是 dst），说明这是“坏节点”问题，不是“坏边”问题，需要用 `npu-smi`/`hccn_tool` 交叉定位到具体网口/光模块，而不是简单说“这条边慢”。

### 2.3 可靠性模式（本文新增维度，见 §3.1 证据）：失败率 + 失败签名分布，而非“跑通了就算过”

这是本文与 Grok R0 判断框架上最大的补充。当前流程对“正常”的隐含定义是“这次跑成功了”，但对“128 卡通信体质是否健康”这个问题，真正该问的是：

1. **同一规模连续跑 5 次，成功几次？**（如果 <100%，成功率本身就是一个需要长期跟踪的体质指标，类似芯片体质里的良率）
2. **失败的时候，签名是单一的还是多样的？** 单一签名（比如永远是同一个 rank 报错）指向真实坏链路/坏节点；签名随机分布在不同 rank、不同错误类型之间，指向环境/资源竞争问题（残留进程、端口冲突、超时参数不合适），这是**完全不同的处理路径**，但现有报告从未区分过。
3. **失败是否有级联特征？** HCCL 官方文档明确写了“建链超时存在级联传递现象，一个 rank 卡住会导致所有邻居 rank 也报建链超时”（见 §7 引用）。也就是说，日志里“38 个 rank 都报 socket timeout”**不代表 38 个节点都坏了**，可能只有 1 个 rank 真正卡死/崩溃，其余是被级联拖累的受害者。当前所有失败日志都是把所有报错 rank 平铺列出，从未去找“第一个报错的时间戳最早、且错误类型与其他 rank 不同”的那个 rank 作为真正 root cause——这是判定框架里缺失的一环，必须补。

---

## 3. 现有测法哪里初级、哪里危险（原始日志证据）

### 3.1 【危险】128 卡跨机通信在 ~100 分钟内、无并发争用的情况下，用三种不同方式失败过——而报告只留了成功的样本

这是本文独立翻查 `logs/` 目录后发现的核心问题，Grok R0 没有覆盖到（它只提到了一次 P2P SIGSEGV）。完整时间线（均为 2026-07-10 晚 ~ 2026-07-11 凌晨，同一个 8 节点集群 `huawei-8node-copy`，前后没有其它烤机/MFU 作业占用同一批 pod，已核实无并发争用）：

| 时间 | 跑批 | 内容 | 结果 | 失败签名 |
|---|---|---|---|---|
| 22:47:56 | `hccl-20260710_224756` | collective all_reduce/all_gather/reduce_scatter/broadcast，16→32→64→128 | scale=64 **FAIL**，其余 OK | `logs/hccl-20260710_224756/scale64_rank2.log:15`：`hcclCommInitRootInfoConfig(...) error code is 4`（HCCL 内部错误，多与网络/建链异常相关） |
| 22:49:02 | `hccl-20260710_224902` | 同上，重跑一遍 | 16/32/64/128 **全部 OK** | — （**这一份被 `hccl_128.md` 采用为唯一数据来源**） |
| 23:53:58 | `hccl-cluster-r0-20260710_235358` | P2P（ring+star），world=16 | 成功，176 边全 `ok=true` | — |
| 23:57:41 | `hccl-cluster-r0-128-20260710_235741` | P2P，world=128 | 未真正启动（8 个 rank 日志各仅 4 行 warning，无 bench 输出） | 疑似 launcher/rendezvous 未及时就位 |
| 00:03:23 | `hccl-cluster-r0-128-20260711_000323` | P2P（ring+star），world=128，380 边 | **SIGSEGV** | `scale128_rank2.log:54`：`exitcode: -11`，`local_rank: 10` |
| 00:08:33 | `hccl-cluster-r0-128ring-20260711_000833` | P2P（**仅 ring**，128 边，已按“修复”改为串行 ring-only） | **仍失败**，非 SIGSEGV | `scale128_rank0.log:48`：`dist.barrier()` 触发的 `HcclAllreduce` 内部错误，`Communication_Error_Get_Socket(EI0006): Getting socket times out` |
| 00:15:20 | `hccl-cluster-r0-perrank-20260711_001520` | **collective**（`run_hccl_scale.sh`，验证 per-rank 落盘新功能），world=128 | **FAIL** | 同样是 `HcclAllreduce` + `Getting socket times out`（`scale128_rank0.log:38-62`），rank0/1/2/3/10 等均报同类错误 |

这张表本身就是关键结论，展开三点：

**(1) “ring-only 是安全缓解措施”这个假设从未被验证通过**。`research_comm_constitution_r0_grok.md` §3.1 和 `hccl_p2p_bench.py` 代码注释都把“world≥64 默认仅 ring”当作对 SIGSEGV 的缓解，但 00:08:33 那次跑批**已经是 ring-only**，依然失败——只是换了个失败方式（不是段错误，是 `barrier()` 内部触发的建链超时）。也就是说，`hccl_cluster_r0.md` 里“ring-only 重试进行中”这句话，实测的结果是“重试了，还是没过”，但这个事实没有被写回任何一份现有报告。

**(2) 被当作“保底”的 collective per-rank 热力方案，自己也没扛过 128 卡**。`research_nccl_verify_r0.md` §2.2 和 Grok R0 §6（P0.3）都把“per-rank collective 计时”当成 P2P 未修好前的安全保底（“已有落盘 → gen 报告”）。但 00:15:20 这次跑批，**就是专门为了验证这个新功能（`hccl_torch_bench.py` 改成每 rank 落盘）而跑的 128 卡验证**，结果它自己先挂了，而且挂的方式和 P2P ring-only 那次一模一样（`HcclAllreduce` + `Getting socket times out`）。这说明“P1 保底”目前只是**代码写完了，从未在 128 规模上被证明真的能保底**——这是一个被隐藏的风险，如果不点破，下一个人会拿着“已有落盘可以直接出图”的假设去用一份可能不完整/带错误 rank 的数据出报告。

**(3) 短短 100 分钟内出现三种不同的失败签名**（SIGSEGV / `Getting socket times out` 建链超时 / `hcclCommInitRootInfoConfig error code 4`），且都发生在跨节点场景，与具体测的是 P2P 还是 collective无关。三种签名指向的官方文档原因也不同（段错误是进程崩溃；`Getting socket times out` 是建链阶段对端未及时响应，可能因为对端进程异常退出/卡死/网络不通；`error code 4` 官方定义是“内部错误”，社区案例多与网络通信异常相关）。**签名多样 + 无固定复现 rank + 无并发争用**这三个特征组合在一起，最像的解释是：**多节点 torchrun 反复复用同一批 pod，每次只是端口 +1，但没有在两次尝试之间检查/清理残留进程或连接状态**（脚本里 `run_hccl_p2p_128.sh`、`run_hccl_scale.sh` 均无 pod 侧 `pkill`/端口探活步骤）。这个混淆变量（环境脏 vs 真实链路坏）**在下结论“HCCL P2P 在多节点+多并发下有资源上限”之前，从未被排除**——而 Grok R0 §3.1 的“风险假设”正是直接下了这个结论，我认为证据还不足以支撑，见 §4.1 P0.0。

**(4) 幸存者偏差**：`hccl_128.md`（也是 `research_comm_constitution_r0_grok.md` 大量分析所依赖的底层数据）明确写“数据来源：`hccl-20260710_224902`”——但同一晚更早的 `224756` 那次完整 sweep 里，**64 卡这一档确确实实失败过一次**，只是脚本 `run_scale() || true` 把这次失败悄悄跳过、继续测下一档，报告作者随后选了跑得干净的第二次当“结论”。这意味着现在展示的稳态曲线，运维体感上其实是“某规模有一定概率会失败”，但这个失败概率从未被计算或写进任何报告——**报告的“干净”是挑出来的，不是集群真实状态的完整描述**。

### 3.2 【初级】弱扩展效率公式误用（与 Grok R0 独立同一结论）

已在 §2.1 展开，独立复算数字一致（149.85→137.76，-8.1%，非报告所称 -88.5%/"11.5%效率"）。这是一处纯方法论 bug，修复成本极低（改公式 + 改文案），不需要重新采数据。

### 3.3 【危险】collective 微基准完全没有正确性校验——性能测了，值对不对没人知道

读 `hccl_torch_bench.py` 全文（243 行）可以确认：`all_reduce`/`all_gather`/`reduce_scatter`/`broadcast` 四个 op 全程只做 `torch.npu.synchronize()` 计时，**没有任何地方检查结果张量的值**。对比同目录的 `hccl_p2p_bench.py`，P2P 至少有 `_pattern_ok()` 做首尾抽样校验（§64-82 行）。这意味着：

- 如果 HCCL 在大规模下发生**静默数据错误**（SDC，与算力体质报告反复强调的“正确性红旗”是同一类风险，只是发生在通信路径而非计算路径），当前 collective 基准**测不出来**——它只会告诉你“跑得挺快”，哪怕 all_reduce 的结果是错的。
- 这是一个**假阴性（false negative）**风险，比 §3.1 的假阳性/崩溃更隐蔽：崩溃至少会被看见，静默错误不会。给定 128 卡规模下已经反复出现建链层面的不稳定（§3.1），我认为“通信层在压力下会不会偷偷传错数据”这个问题的优先级不该低于“跑得多快”。

### 3.4 【初级】链路健康结论建立在会产生假阳性的信号上

`hccl_128.md` §6 的结论是“8/8 节点 npu-smi Health=OK”，`run_link_health.sh` 试图跑 `hccn_tool` 但 8 个节点全部 `not found`。问题不是“没跑成”本身，而是**报告仍然用这份不完整的检查给出了“链路健康”的措辞**（虽然报告写了限制说明，但只在文末小字注明）。鉴于 §3.1 里三次跨机建链失败，官方排障文档的第一步就是“用 `hccn_tool -i $devid -tls -g` 检查 TLS 状态”——**唯一能验证网络层的工具恰好缺失，而设备级 Health=OK 完全无法回答“建链为什么超时”这个问题**。这是典型的“看起来测了，但测的东西回答不了要问的问题”，风险等级应该从“限制说明”提升到“阻塞项”。

### 3.5 【初级】P2P 128 卡从未产出过一份完整边矩阵

`hccl_cluster_r0.md` 的边级画像（host×host / rank×rank 热力图、慢边 TopK）**只在 16 卡跑通过**。128 卡三次尝试（§3.1 表中三行）全部失败，没有一份完整的 128 卡边级数据落盘。当前唯一的“128 卡通信数据”是 collective 全局单点 bus_bw（`hccl_128.md`），**无法回答“128 卡里是少数坏边拖垮平均值，还是全网均匀退化”**——这正是 `research_nccl_verify_r0.md` §2.3 一年前就点名的缺口，一轮调研过去了依然没填上。

---

## 4. 推荐升级：测项、扇出、报告图、与算力烤机分轨

### 4.1 测法升级（按依赖顺序）

```
P0.0  环境卫生检查（新增，优先级最高，成本最低）：
      每次 128 卡实验前，在 8 个 pod 上检查/清理残留 python/torchrun 进程与端口占用，
      记录检查结果。目的：把 §3.1 的"环境脏 vs 真实链路坏"这个混淆变量摘出去，
      再谈"P2P/collective 在128规模有资源上限"是否成立。
P0.1  可靠性 instrumentation（新增）：
      run_hccl_scale.sh / run_hccl_p2p_128.sh 去掉 `|| true` 的静默丢弃，
      改为：失败也落盘（错误签名 + exit code + 首个报错 rank 的时间戳），
      同一规模重跑 N=3~5 次，报告里必须出现"成功率"这一列，不能只挑干净的一次。
P0.2  bus_bw 口径修正（与 Grok R0 一致）：
      重算 hccl_128 的"保持率"；旧效率公式降级为脚注。
P0.3  collective 正确性校验：
      对 all_reduce（已知期望和）、broadcast（已知源值）加最小校验，
      对齐 P2P 脚本 `_pattern_ok` 的思路。
P0.4  128 P2P 与 per-rank collective 的"稳定复现"：
      在 P0.0 排除环境混淆变量后重跑；若仍失败，才能下"多节点+多连接确有限制"的结论；
      若通过，回填 hccl_cluster_r0.md 的 128 卡边矩阵（G3-G6）。
P1.1  拓扑分层标注：record 增加 topo_tier（1n16/2n32/4n64/8n128）与 intra_or_cross。
P1.2  级联根因定位：失败时自动找"首个报错时间戳 + 与其他 rank 错误类型不同"的 rank，
      标记为 root cause candidate，其余标 cascade victim（依据 HCCL 官方建链级联语义）。
P1.3  hccn_tool 补齐（镜像/PATH）：与 Grok R0 P0.5 一致，跑通则做慢边×链路交叉表。
P1.4  host-rep 跨机 P2P 矩阵、alltoall、延迟 size 补点（4K/16K）：与 Grok R0 §2.2/2.3 一致，采纳。
```

（P1.1/P1.3/P1.4 与 Grok R0 结论一致，不重复展开设计细节，直接采纳其方案；P0.0/P0.1/P0.3/P1.2 是本文独立新增。）

### 4.2 扇出策略

- **规模阶梯**：16（纯机内）→ 32（2 节点，首次跨机）→ 64（4 节点）→ 128（全集群），每档**先做 P0.0 环境检查，再跑，且每档重复 ≥3 次**才能计入“稳态”结论；单次成功不构成结论。
- **P2P 边采样**：维持现有 ring（O(N)）+ star→rank0（仅小 world）策略，128 规模按 §4.1 P0.4 顺序先解决可靠性问题再谈边矩阵；不做 128² 全对全（与 Grok R0 一致）。
- **时间窗口**：可靠性重跑（P0.1）会显著拉长墙钟（每档 ×3～5 次），必须单独申请空闲窗口，不能塞进现有“一次性冒烟”的时间预算里——这是本文与 Grok R0 时间预算表（§4.1 表）的主要修正：Grok R0 假设“smoke 16 → 128”是线性时间，但没有为“重复验证稳定性”单独留出预算。

### 4.3 报告图清单（在 Grok R0 G1-G8 基础上新增两项）

| ID | 图 | 数据源 | 解读 |
|---|---|---|---|
| G1-G8 | 同 `research_comm_constitution_r0_grok.md` §4.2（bus_bw 热力/保持率曲线/慢边 TopK/host×host/rank×rank/节点聚集/per-rank collective 热力/链路交叉表） | 同上 | 采纳，不重复设计 |
| **G9（新增）** | **规模×成功率柱状图**：x 轴 world_size，y 轴“N 次尝试中成功次数” | §4.1 P0.1 的重跑记录 | 这是本文认为**最该有但现在完全没有**的一张图——直接回答“128 卡通信到底稳不稳”，比任何 bus_bw 曲线都更贴近“体质”这个词的本意 |
| **G10（新增）** | **失败签名时间线**：按时间顺序标注每次失败的错误类型（SIGSEGV / socket timeout / comm init error / 其它） | §3.1 时间线表 | 用于判断失败是随机分布（环境问题）还是集中在特定条件（真实硬件/链路问题） |

### 4.4 与算力烤机的分轨策略

同意 Grok R0 §5 的结论（方案 A：作业分轨，报告层可索引合并），补充一条本文特有的理由：**通信可靠性测试（P0.1）本质上是“故意反复触发一个已知会偶发崩溃/挂起的路径”**，如果与算力烤机共享同一批 pod 且时间窗口有重叠，一次 HCCL 挂起/SIGSEGV 有可能连带拖死同节点上正在跑的算力烤机进程（两者共享物理机器与网络栈，不是逻辑隔离）。所以分轨不仅是“指标语义不同”（Grok R0 的理由），更是**故障隔离**的硬需求：通信可靠性重跑必须申请独占窗口，不能与算力烤机并行，哪怕两者用的是不同脚本。

---

## 5. 若只能先做三件事

按“成本极低 + 直接回答‘稳不稳’这个最根本问题 + 不需要新硬件/新工具”排序：

1. **环境卫生检查（P0.0）+ 可靠性 instrumentation（P0.1）**：在每次 128 卡实验前加一步进程/端口清理检查，并把 `run_hccl_scale.sh`/`run_hccl_p2p_128.sh` 里的 `|| true` 静默失败改成“失败也落盘 + 记录首个报错 rank/时间戳/签名”，同规模重跑 3～5 次。这一步几乎零代码改动成本，但能立刻回答两个悬而未决的问题：128 卡通信到底稳不稳定？现在的失败是环境脏还是真链路坏？—— 这是当前最大的空白，也是本文与 Grok R0 判断框架上最本质的分歧点。
2. **bus_bw 保持率口径修正 + collective 正确性校验**：一次改代码/改报告文案，同时解决“退化被夸大 8 倍”的误判风险和“SDC 完全不可见”的假阴性风险，成本低、收益直接。
3. **128 卡 P2P 边矩阵补全（在 1 完成后再做）**：先用第 1 步排除环境混淆变量，确认失败是否复现；如果 P0.0 之后 ring-only 能稳定跑完，直接产出 128 卡版本的 host×host/rank×rank 热力图 + 慢边 TopK，把现有唯一的“128 卡通信数据”从“全局单点 bus_bw”升级为“可差分的边矩阵”，这是 `research_nccl_verify_r0.md` 一年前就点名、至今没填的最大缺口。

---

## 6. Top 5（交付摘要）

1. **把“重复调用同一规模的成功率”当成通信体质的第一指标**，而不是只报“跑成功那次”的性能——现有流程用 `\|\| true` 悄悄丢弃失败样本，造成幸存者偏差（`hccl_128.md` 的底层数据 `224902` 是同晚第二次尝试，第一次 `224756` 在 64 卡这一档实测失败过）。
2. **128 卡跨机通信在约 100 分钟窗口内、无并发争用条件下，用三种不同方式失败过**（P2P SIGSEGV / collective `barrier` 触发的建链 socket 超时 / `hcclCommInitRootInfoConfig error code 4`），且被当作“安全缓解”（ring-only）和“安全保底”（per-rank collective 落盘）的两个方案都在各自唯一一次 128 卡验证中失败——这两个假设目前都未经证实，不该被当成已解决问题继续沿用。
3. **在下结论“HCCL 多节点+多连接有资源上限”之前，先排除环境混淆变量**：所有 128 卡脚本反复复用同一批 pod、仅端口自增，从未在两次尝试间检查/清理残留进程或连接状态，签名多样+无固定复现 rank 更像环境不干净，而不是坐实的硬件/协议限制。
4. **collective 微基准（`hccl_torch_bench.py`）完全没有正确性校验**，只测时间不测值，是一处比崩溃更隐蔽的假阴性风险（静默数据错误测不出来）；`hccn_tool` 全缺导致的“链路健康=OK”结论同样是假阳性来源（设备健康 ≠ 建链/链路健康，而建链恰恰是当前唯一在崩的地方）。
5. **`hccl_128.md` 的弱扩展效率公式误用**（`/(N/16)`），把“256M All-Reduce 只降 8.1%”硬算成“11.5%效率”；应改用“保持率”作主叙事——此结论与 Grok R0 独立复算一致，可直接采纳。

## 最小三件事

1. 环境卫生检查（清残留进程/端口）+ 可靠性 instrumentation（失败不吞、记签名、同规模重跑 3～5 次）。
2. bus_bw 保持率口径修正 + collective 加最小正确性校验（一次改动，两个风险一起解决）。
3. 在 1 完成后，补全 128 卡 P2P 边矩阵（host×host / rank×rank 热力图 + 慢边 TopK），把“128 卡通信数据”从单点 bus_bw 升级为可差分矩阵。

---

## 7. 与 Grok R0 的对照：一致 / 独立新增

| 结论 | Grok R0 | 本文（Sonnet） | 关系 |
|---|---|---|---|
| bus_bw 弱扩展效率公式误用 | 已指出，给出保持率修正 | 独立复算，数字一致 | **一致，采纳** |
| 拓扑分层（intra/cross）应显式标注 | 已提出 | 认同 | **一致，采纳** |
| hccn_tool 缺失是硬缺口 | 已指出 | 认同，补充“唯一能验证网络层的工具恰好缺失”的严重性判断 | **一致，加强表述** |
| P2P G1-G8 报告图清单 | 已给出完整方案 | 直接采纳，新增 G9/G10 | **一致 + 增量** |
| 128 P2P SIGSEGV | 归类为“P2P 在多节点多连接下的资源限制”待验证 | 归类为**通信体质的可靠性维度本身**，且指出 ring-only/per-rank 保底两个“缓解方案”都未被证实有效，怀疑环境混淆变量优先于硬件限制假说 | **独立新增/修正** |
| 64 卡曾失败但报告未采用 | 未提及 | 通过原始日志时间线发现并展开为“幸存者偏差”论点 | **独立新增** |
| collective 无正确性校验 | 未提及 | 独立新增，作为假阴性/SDC 风险单列 | **独立新增** |
| 建链失败的级联语义（一个 rank 卡死→全员超时） | 未提及 | 独立新增，引用 HCCL 官方排障文档，提出 root cause vs cascade victim 的定位方法 | **独立新增** |
| 与算力烤机分轨 | 方案 A（分轨），理由是指标语义不同 | 认同方案 A，补充“故障隔离”作为第二条理由（通信可靠性重跑可能拖死同 pod 的算力烤机） | **一致 + 补充理由** |

---

## 8. Sources

**本集群一次证据（均已逐行核查，非转述）**：
- `reports/hccl_128.md`、`reports/rounds/hccl_cluster_r0.md`
- `logs/hccl-20260710_224756/hccl.log`、`scale64_rank2.log:15`（`hcclCommInitRootInfoConfig error code 4`）
- `logs/hccl-20260710_224902/hccl.log`（全量 OK，`hccl_128.md` 唯一数据来源）
- `logs/hccl-cluster-r0-128-20260711_000323/scale128_rank2.log:54`（SIGSEGV, local_rank=10）
- `logs/hccl-cluster-r0-128ring-20260711_000833/scale128_rank0.log:38-48`（ring-only 仍失败，`Getting socket times out`）
- `logs/hccl-cluster-r0-perrank-20260711_001520/hccl.log`、`scale128_rank0.log:38-62`（per-rank collective 保底方案首次 128 卡验证即失败）
- `scripts/cluster/hccl_torch_bench.py`（无正确性校验，全文核查）
- `scripts/cluster/hccl_p2p_bench.py`（`_pattern_ok` 抽样校验，§64-82 行）
- `scripts/cluster/run_hccl_scale.sh`、`run_hccl_p2p_128.sh`（`\|\| true` 静默丢弃失败，无 pod 侧进程/端口清理步骤）
- `reports/research/research_nccl_verify_r0.md`（P1 per-rank 保底方案的最初提出处）
- `reports/research/research_comm_constitution_r0_grok.md`（对照文本，独立复核）

**外部引用**：
- [nccl-tests PERFORMANCE.md（algbw/busbw 定义）](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- [CANN 社区版：EI0006 Communication_Error_Get_Socket 错误码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/maintenref/troubleshooting/atlaserrorcode_15_0246.html)
- [CANN 社区版：建链超时 (EI0006) 故障诊断（含级联传递现象说明）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/hccl/hcclug/hcclug_000050.html)
- [昇腾社区：HCCL 集合通信常见问题定位思路（root cause vs 级联受害 rank 的判定方法）](https://www.hiascend.com/developer/techArticles/20240930-1)
- [CANN：HCCL_CONNECT_TIMEOUT 环境变量说明（默认 120s，范围 120–7200）](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/envvar/envref_07_0077.html)
- [Ascend/pytorch：HCCL 错误码映射表（error code 1–20 含义，`349fc0d` commit）](https://github.com/Ascend/pytorch/commit/349fc0d22c2d7caec5f0653e31de6b98f9c37ff3)
- 社区案例：`hcclCommInitRootInfoConfig error code 9`/`error code 4` 复现（Gitee Ascend/ModelZoo-PyTorch Issue #IC8OEW；Ascend/pytorch Issue #IADMZI）
