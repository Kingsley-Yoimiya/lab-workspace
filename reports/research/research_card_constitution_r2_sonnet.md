# Ascend 910 卡体质筛查增强方案（R2.5 / Sonnet 独立评审）

**实际模型**: Claude Sonnet 5 (claude-sonnet-5-thinking-high)
**日期**: 2026-07-11
**版本**: R2.5-sonnet（独立评审，非代评；对 `research_card_constitution_r2_sonnet.md` 旧版 composer-2.5-fast 代评内容全文推翻重写）
**前置**: [`research_card_constitution_r0.md`](research_card_constitution_r0.md) · [`research_card_constitution_r2_grok.md`](research_card_constitution_r2_grok.md) · [`research_card_constitution_r2_merged.md`](research_card_constitution_r2_merged.md)
**评审依据**: 实际代码通读（`stage_c.py` / `telemetry.py` / `_smi.py` / `health.py` / `slow_cause.py` / `screening.py` / `aggregate.py` / `jsonl.py` / `gates.py` / `builtin.py`）+ 128 卡真实基线（`reports/card_screen_128.md`）+ 128 卡真实 npu-smi 原始遥测抓包（`logs/telemetry-20260710_224628/results/master0.jsonl`）+ npu-smi 官方命令语法核查

---

## 0. 结论先行（给决策者的四句话）

1. **`temp_c≡2` 确实是垫脏数据，不能用于任何热/降频结论** —— 这一点我同意 Grok/merged。但我用本项目自己抓到的真实 `npu-smi info` 原始输出核实了根因链条，发现 Grok「`I2C` 出现在正常输出里被误吃」的叙述**站不住**：真实输出里根本没有 `I2C` 字符串，温度也是正常的 36–43°C。真正根因更可能是**调用命令本身非法**（`npu-smi info -i <device>` 缺少必需的主命令 `-t type`），导致 npu-smi 回显自己的用法/帮助文本，而该文本的 `-t` 类型枚举里恰好含有 `i2c_check`——`"2c"` 撞上裸正则 `\d+\s*C`，全卡零方差返回 2.0。**这是修复方式的分水岭**：只改正则不改调用命令，问题不会解决（拿不到任何数据，`temp_c` 会变成全 `None`，不是变好）。
2. 我在通读代码后发现了**三个此前四份文档都没有识别的新风险**（详见 §2）：卡/芯粒（Card/Chip）ID 寻址错位、`comparison_group` 分桶字段已声明但从未被赋值（空转）、`launch_latency` 在 `event_timer()` 退化到 `wall_sync` 时会静默产生「未测等于测过且合格」的假阳性豁免。这三项我认为都应升级为 **P0 阻塞项**。
3. 对 Grok 的 P0 收窄方向、对 composer 代评（旧版本文）「scalar 不进主键 / sustained 派生升阻塞」的判断，我**总体同意**，但我认为 `vector_fma_perf` 和 `launch_host_overhead_p99` 目前同样**没有任何真实集群数据支撑其 0.15 阈值**——这两个探针在这套硬件上一次都没有真正跑过 128 卡，把从未验证过的信号直接定为「可判 slow 的主键」，比 scalar 的假阳性风险还危险（scalar 至少已知问题所在；vector/launch 是「未知的未知」）。我的建议是**首轮全部只观察，不判 slow**，等 16 卡冒烟拿到第一批真实分布后再决定阈值——这是本文与 merged 方案**唯一的正面分歧**。
4. **最小可发射体质集 = 修遥测（含命令本身）+ 已开 Stage C（仅采集）+ 健康红旗（需新写代码，非勾选框）+ within_host 残差 + 只用 func/hbm 两个已验证维度判 slow**。SDC、shape_sweep/bnmk 维持现状。

---

## 1. 对 Grok R2「temp_c≡2 = I2C 误匹配」与「P0 收窄」的意见

### 1.1 温度 bug：同意方向，推翻具体机制

**我做了什么核查**：项目里恰好留有一份用于对照的真实 npu-smi 抓包脚本产出——`scripts/cluster/npu_telemetry_bench.py` 生成的 `logs/telemetry-20260710_224628/results/master0.jsonl`，里面完整保留了 idle/load 两阶段的 `npu-smi info`（无 `-i`，查询全部设备）原始文本。这是**这个集群这次跑出来的真实回显**，不是文档里手造的示例。摘录（idle 阶段，8 卡 16 chip）：

```
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip  Phy-ID              | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
| 0     Ascend910           | OK            | 160.2       37                0    / 0             |
| 0     Ascend910           | OK            | -           39                0    / 0             |
...
```

满载阶段 Temp(C) 升到 40–43。**通篇没有 `I2C` 字符串，温度读数完全正常**。这直接证伪了 Grok/composer 旧文「`I2C check: pass` 之类文本会出现在正常 `npu-smi info` 输出里」的叙述——那只是他们为演示正则漏洞手造的示例字符串，**不是在这台机器上实际复现的结果**，旧文用「本地复现」这个措辞是不准确的。

**那真正的 2.0 从哪来？** 我查了 npu-smi 官方命令语法（`npu-smi info --help`）：

```
Usage: npu-smi info <watch|proc|-h|-m|-l|-t type> [Options...]
Commands: watch | proc | -h/--help | -m | -l | -t type
Options:  -i %d  Card ID   -c %d  Chip ID   -p %d  Chip Physical ID
```

关键点：**`-i` 属于 `Options`，不是 `Commands`**。`npu-smi info` 必须先给出 `watch|proc|-h|-m|-l|-t type` 之一，`-i` 只是附加在其后的修饰符。而生产代码 `telemetry.py::NpuSmiProvider.sample()` 的调用是：

```python
info = _run(["npu-smi", "info", "-i", str(device)])   # 没有 -t / -m / -l / watch / proc
```

这极可能是**语法不完整的调用**——npu-smi 大概率会拒绝执行并回显自己的用法帮助文本，而该帮助文本里 `-t type` 的类型枚举原文是：

```
type: board, flash, memory, usages, sensors, temp, power, volt, mac-addr,
      common, health, product, ecc, ip, sys-time, i2c_check, work-mode, ...
```

`i2c_check` 中的 `"2c"`（数字 2 紧跟字母 c，大小写不敏感）恰好撞上裸正则 `(\d+(?:\.\d+)?)\s*C`，命中并返回 `2.0`。**这个假说完美解释「128 卡全样本 temp_c 集合恰好是 `{2.0}`，方差为零」**——如果是「读到了某张卡真实但偶然是 2℃ 的数据」，128 卡不可能零方差；但「128 次调用都触发同一条固定的用法帮助文本」在数学上必然零方差，这与观测完全吻合，比「巧合都读到 I2C 状态」的概率论证更强。

**结论与更正**：
- 同意：`temp_c≡2` 是解析垫脏数据，绝不能用于热/降频结论——这个大方向不变。
- 推翻：根因不是「`I2C` 作为正常字段出现在数据行里」，而**很可能是命令本身缺 `-t type` 参数，触发用法帮助文本回显**。
- 工程含义不同：Grok/merged 的修复方案（「标签化解析 Temperature/HBM/Board，禁裸 `\d+C`」）**只修了正则，没修命令**——如果调用本身仍是非法语法，换了正则之后 `_first_float` 找不到任何 `Temperature` 标签行，会返回 `None`（这是好事，不会再污染），但**依然拿不到真实温度**，体质报告里 `temp_c` 会变成全空，问题被「隐藏」而不是被「解决」。真正修复必须同时改调用本身，例如 `npu-smi info -t board -i <card_id> -c <chip_id>` 或退回到解析裸 `npu-smi info`（无 `-i`，已验证真实可用，见上面抓包）按行定位目标设备。

### 1.2 P0 收窄：同意方向，但补三项遗漏（详见 §2），且反对把未经验证的信号直接列为可判 slow 的主键（详见 §3.2）

同意 Grok 收窄「burst / 独立 power_ramp / 全套散点 / msprof 128 扇出」为可延后。但收窄之余，我在读代码时发现三个新的、体量不大但足以让「收窄后的 P0」本身站不住的 gap，必须一并塞进 P0（不是新增探针，是修 bug）：卡/芯粒寻址错位、`comparison_group` 空转、`launch_latency` 静默豁免。这三项任何一个不修，后续「残差」「换卡单」「集群相对判定」这些收窄后保留的核心机制都会在错误的输入上运行。

---

## 2. 新发现：四份前置文档都没有识别的问题

### 2.1 卡（Card）/芯粒（Chip）ID 寻址错位 —— 新发现，未见于 R0/R1/R2-grok/R2-sonnet(旧)

- 事实核查：`npu_telemetry_bench` 的抓包显示这台 Ascend910_9392 是 **8 张物理卡 × 每卡 2 个 chip = 16 个逻辑 NPU**（`NPU 0..7` 每个又有 `Chip 0`/`Chip 1` 两个子行）；`torch.npu.device_count()` 在同一次抓包里返回 16，`card_screen_128.md` 的 `device` 列范围也确实是 0–15（如 `master-0 device 13`、`worker-5 device 14`）。也就是说 **CARD_SCREEN 的 `device` 是「芯粒级」索引（0–15）**。
- npu-smi 官方语法里 `-i %d` 明确标注为 **Card ID**（卡级，应为 0–7），`-c %d` 才是 **Chip ID**（芯粒级，0–1）。
- 但 `telemetry.py` 里所有 npu-smi 调用——`sample()`、`_parse_extended()`、`health_probe()`——无一例外把 `str(device)`（0–15 的芯粒索引）直接塞进 `-i`，**从未传 `-c`**。这意味着：即便按 §1.1 修好了命令与正则，device=8–15 时 `-i 8`..`-i 15` 对应的 Card ID 在 8 卡机器上根本不存在（或被 npu-smi 静默钳到某张邻卡），device=0–7 时又只能拿到「整卡」粒度、混合了卡上两个 chip 的数据，无法区分两个 chip 各自的真实温度/功耗。
- **动作**：必须先在机上做一次一次性映射标定（例如逐 chip 跑一个短时高负载，观察哪个 `-i/-c` 组合的温度先升），确认 `device -> (card_id, chip_id)` 的真实公式（大概率是 `card_id = device // 2, chip_id = device % 2`，但**不能假设**，需要机上验证），然后把这个映射写死进 `NpuSmiProvider`，所有 `-i` 调用都要配对 `-c`。这个映射标定应该和 §1.1 的命令修复**同一次上机验证一起做完**，不要分两次改。

### 2.2 `comparison_group`（driver/firmware 分桶）已声明但从未被赋值 —— 新发现

- `cluster/aggregate.py` 里 `GROUP_FIELDS = ("backend", "device_name", "driver_version", "firmware_version")`，分桶逻辑、`_group_label`、按组算中位数都已经写好，**这部分代码本身没问题，架构是对的**。
- 但我 grep 了整个 `card_screen` 包：`driver_version` / `firmware_version` **在任何地方都没有被写入过**——`screening.py::_new_record()` 不写，`health.py::check_health()` 不写，`telemetry.py` 也不写。`aggregate.py::_field_value()` 在字段缺失时 fallback 成字符串 `"unknown"`，所以**今天所有卡的 comparison_group 都会折叠成同一个 `(backend, device_name, "unknown", "unknown")`**——分桶字段形同虚设，跨驱动/固件混部完全没有被隔离，只是看起来"有防护"。
- 这与旧版（composer 代评）checklist 里「补充 comparison_group」的描述不准确：**不是缺字段，是有字段没有写入逻辑**。修复量很小（在 `check_health()` 或 `_new_record()` 里调一次 `npu-smi info -t common`/驱动版本文件，塞进 `rec["driver_version"]`），但必须动手写代码，不是打勾就能过。

### 2.3 `launch_latency` 的 `timing_method` 分裂会制造「测了等于没测但看起来合格」的静默假阳性 —— 新发现

- `stage_c.py::launch_latency()` 里：`events = adapter.event_timer(device)`；如果某张卡 `events is None`（驱动/环境原因导致事件计时不可用），代码走 else 分支，**只采 `tiny_wall_us`，`host_overhead_us` 永远是空列表**。
- `_percentiles([])` 返回 `{"p50": None, "p99": None}`；这张卡的 JSONL 里 `launch_host_overhead_p99_us = None`。
- `aggregate.py` 判 slow 的逻辑是 `isinstance(x, (int, float)) and x < med*(1-slow_frac)`——`None` 直接被这个 `isinstance` 挡掉，**不会被判 slow，也不会被判任何异常**。也就是说：一张卡如果因为某种环境原因掉进了 `wall_sync` 兜底分支，它在这个维度上会**悄无声息地「合格通过」，而实际上根本没有被这个维度测过**。128 卡换卡单如果直接读 `final_verdict==good`，会把「未测」和「测了且好」混为一谈。
- **动作**：checklist 必须新增一条——聚合前检查所有卡 `perf.launch_latency.timing_method` 是否全部为 `"event"`；出现 `"wall_sync"` 的卡要单独标记「launch 维度不可比」，而不是让它静默进入「good」桶。这是此前所有文档都没提到的 aggregate 层盲点。

---

## 3. 对 composer 代评（旧版本文）/ merged 方案的意见

### 3.1 同意

- `scalar_chain_perf` 首轮不进 slow 主键：同意。`cumsum` 是否真的压制 Vector 全宽并行在 torch_npu 上完全没验证过，R0 自己也承认这个风险。
- sustained 窗内 power/freq **派生**（不新增探针，从既有窗口样本算）升级为发射阻塞：方向同意，但要**追加一句前提**——这个派生的输入是 `temp_c`/`sm_clock_mhz`，如果 §1.1/§2.1 的遥测命令 bug 不先修，派生出来的还是同一份垫脏数据的加工品，「派生」不能替代「修复数据源」，两者必须按顺序做（先 §1.1/§2.1，再谈派生），不能并列成两条独立的 checklist 项。
- `health.py` 目前完全没有 pcie-err / err-count / health delta 的实现（只有 `ecc_uncorrected` 单点读取）——这一点composer代评/merged 都把它列成一条 checklist 打勾项，但实际上这是**从零开发一个新函数（`health_counters` + `health_delta`），工作量不是「补丁」量级**。我同意应该做，但建议明确标注这是「新增代码，非配置改动」，评估排期时不要低估。
- 报告模板删除默认「无 throttling」文案：同意，且我核实了触发路径——`slow_cause.py::classify()` 的 `has_temp` 判断是 `any(temp_c is not None)`；垫脏数据 `temp_c=2.0` 不是 `None`，所以 `has_temp=True`，分类器会**自信地**给出 `hint="no_throttle"`，这正是 `card_screen_128.md` 敢写「无 thermal/power throttling 标记」的直接原因。修复不能只改报告文案，`telemetry_trust=false` 时必须让 `slow_cause.classify()` 本身拒绝产出 hint（返回 `None` 或专门的 `untrusted` 标记），否则下游任何读 `slow_cause` 字段的代码都会被骗。

### 3.2 明确不同意（本文与 merged 方案的核心分歧）

- **不同意**把 `vector_fma_perf` 的 `vector_gflops` 和 `launch_latency` 的 `launch_host_overhead_p99_us` 在首轮就定为可判 slow 的主键（`slow_frac=0.15`）。理由：
  1. 这两个探针在这套硬件、这个规模的集群上**从未真正跑过一次**——128 卡基线报告（`card_screen_128.md`）用的是旧 `config.perf128`，根本没开 Stage C，没有任何真实 CV 数据支撑 0.15 这个数字。相比之下 `func_tflops`（CV 2.4%）、`hbm_gbps`（CV 4.0%）都有 128 卡实测支撑。拿一个从没见过分布的信号直接定阈值判 slow，比 scalar 的已知风险更危险——scalar 是「已知有毒」，vector/launch 是「毒性未知」。
  2. `launch_latency` 还叠加 §2.3 的 `timing_method` 分裂风险和 NPU event 计时本身含 host enqueue（`gates.py` 里 CUDA 有 `StreamValueGate` 去 enqueue，NPU 侧的 `AclValueGate` 还只是 TODO，注释写着「需要在真机验证」）——在计时纯度本身没有自检通过之前把它的尾部当作换卡判据，风险明显偏高。
  3. 我的替代方案：**首轮 Stage C 三探针全部保持 `verdict_neutral=True`（代码现状本来就是这样，不要改）**，只用 `func_tflops`（0.20）+ `hbm_gbps`（0.15，已通过 128 卡节点聚集现象验证有信号）两个维度判 slow；`vector_gflops` / `scalar_elems_per_s` / `launch_host_overhead_p99_us` 三者都只观察、写入报告供人工复核，**等 16 卡冒烟拿到第一批真实 CV 之后**，再决定是否/以什么阈值把 vector（大概率可以，launch 需更谨慎）升级为主键。这是本文与「merged 方案」§3 表格（`vector + launch_host_p99(0.15)` 直接进主键）唯一的正面分歧，也是我认为最需要父 agent 决策的一点。
- **不同意**「comparison_group 只是 checklist 里补充一条即可」的轻描淡写——如 §2.2，这需要写代码（填充 driver_version/firmware_version），不是配置或文档层面能解决的。
- **不同意**（澄清）根因表述本身，见 §1.1，不再重复。

---

## 4. 最小充分探针集（可发射）与「若只能再加 3 个」

### 4.1 首轮发射的探针/判定角色（本文版本，与 merged 的差异见 §6）

| 层 | 探针 | 首轮判定角色 |
|----|------|-------------|
| Stage A | `func_perf` | 判 slow（0.20，已验证） |
| Stage A | `hbm` | 判 slow（0.15，已验证，128 卡已见节点聚集信号） |
| Stage A | `sustained` | 仅辅证（CV 1.34% 太窄，不进主键——同意 R2-sonnet 旧文原判断） |
| Stage C | `vector_fma_perf` | **仅观察**，不进主键（本文改动点，见 §3.2） |
| Stage C | `scalar_chain_perf` | 仅采集，不进主键（各方一致） |
| Stage C | `launch_latency` | **仅观察**，不进主键，且需检查 `timing_method` 一致性（本文改动点） |
| SDC | 五类轻量 rounds=5 | 正确性红旗，独立判据 |
| 关闭 | `shape_sweep` / `bnmk_sweep` | 不开（已有全量覆盖） |

### 4.2 若只能再加 3 个探针，我的选择

先声明前提（同意 R2-sonnet 旧文的框架）：`health_counters`（ECC/PCIe 红旗前后快照）不占「3 个探针」名额——它不是体质探针，是发射阻塞的基础设施，优先级在这 3 个之上。在此前提下：

1. **`mte_copy_perf`**（纯 MTE/DMA 拷贝吞吐，与 `hbm` 正交）——**我把它排第一，与 merged 方案排序不同**。理由：128 卡目前唯一被真实数据证实、且有强信号（节点聚集、-18% 尾部）的体质维度是 HBM，而现有 `hbm` 探针测的是「计算发起的读写」，无法区分「HBM 介质本身弱」还是「发起访存的 MTE/DMA 通路弱」。既然 HBM 是当前唯一已知的真问题，第一个新探针应该直接服务于拆解这个已知问题，而不是先去开一个全新的正交维度。
2. **`cube_vector_pipeline`**（Cube+Vector 流水线，解释「Cube 齐、step 不齐」）——同意 R2-sonnet 旧文排序，这是 910B 分核架构下最独特、覆盖面最广的缺口。
3. **HBM 访问模式扫描**（不同 elem size / stride / 读写比例的 `hbm` 变体）——**这里是我与 merged 方案「launch burst」不同的选择**。理由：`launch_latency` 单发三桶目前一次都没有在真实集群跑过、且有 §2.3 的静默假阳性风险，在没有第一批真实数据验证它「有没有信号」之前，投入 burst 扩展的性价比存疑；而 HBM 已经是唯一确认有信号的维度，多花一个探针交叉验证「本征介质弱」还是「特定访问模式弱」，边际价值更高、风险更低。`launch burst` 建议保留在 R1 待办，但不进「首批 3 个」。

---

## 5. 128 卡发射 Checklist（阻塞项要可执行）

### 5.1 阻塞项（不满足不允许 128 fanout；全部要求可验证，不是打勾）

- [ ] **遥测命令修复**：`NpuSmiProvider.sample()` / `_parse_extended()` / `health_probe()` 不再使用裸 `npu-smi info -i <device>`；改为带 `-t <type>` 的合法调用，或退回解析裸 `npu-smi info`（无 `-i`）多行表并按行定位设备。**验收标准**：1 卡上手动跑一次改好的命令，用 `idle` 和 `gemm` 两种负载各采样，确认 `temp_c` 在 idle≈30–45℃ 量级、gemm 满载后有可观测上升（不是恒定值）。
- [ ] **Card/Chip ID 映射标定并写死**：在机上确认 `device(0-15) -> (card_id, chip_id)` 的真实对应关系（不要假设 `//2, %2` 一定对，需要实测验证，比如逐 chip 打满载观察哪组 `-i/-c` 组合温度先变化），所有 npu-smi 调用同步传 `-c`。
- [ ] **`telemetry_trust` 门禁落地到分类器**：不只是报告模板加个开关，`slow_cause.classify()` 本身在 `temp_c` 全集群零方差或 `<10℃` 时要拒绝产出 `hw_thermal`/`sw_thermal`/`power_cap`/`no_throttle` 任何 hint（返回 `None` 或新增 `untrusted` 值），从根上掐断「无 throttling」类误导性结论。
- [ ] **`comparison_group` 填充**：`driver_version`/`firmware_version` 需要真正写入 JSONL（例如从 `npu-smi info -t common` 或驱动版本文件读取），不是保留现有「字段存在但恒为 unknown」的状态。
- [ ] **`health_counters` + `health_delta` 新写代码**：ecc + pcie-err + err-count，压测前后各一次快照，增量触发 `red_flag`，与体质 `slow` 正交上报，不等 aggregate 完成才暴露。
- [ ] **`launch_latency.timing_method` 一致性检查**：聚合前统计所有卡该字段是否全为 `"event"`；出现 `"wall_sync"` 的卡单独列出「该维度不可比」，不允许静默计入 good。
- [ ] **within_host 残差**：换卡单 ⊆ `intrinsic_slow`（同 host ≥50% 同向慢 → 强制 `node_env`，不进换卡单）——沿用 R2-grok/merged 已有设计，本文无异议。

### 5.2 判定与配置

- [ ] `constitution_slow_frac`：仅 `func_tflops`(0.20) / `hbm_gbps`(0.15) 进主键；`vector_gflops` / `scalar_elems_per_s` / `launch_host_overhead_p99_us` 首轮仅观察写报告，**不**触发 slow（见 §3.2，与 merged 方案的核心分歧点，需要父 agent / 用户拍板）。
- [ ] `sustained` 仅辅证，不进主键（各方一致）。
- [ ] Stage C 三探针 `enabled: true` 维持现状；`shape_sweep`/`bnmk_sweep` 保持 `false`。
- [ ] `require_idle` + `max_memory_used_mib` 生效（配置已就位，无需改动）。

### 5.3 分阶段发射流程

```
Phase 0 — 1 卡硬化（遥测命令 + Card/Chip 映射 + telemetry_trust）
Phase 1 — 1 节点 16 卡冒烟：确认 Stage C 三指标有分布、timing_method 全 event、无大面积失败
           -> 用这批数据决定 vector_gflops 是否可以升级为主键（launch 暂缓）
Phase 2 — 128 卡 fanout：func+hbm 判 slow，其余观察 + within_host 残差 + 健康红旗
Phase 3 — 数据回流后：mte_copy_perf / cube_vector_pipeline / HBM 变体扫描按优先级补测
```

### 5.4 明确本轮不做（同意各方一致意见）

- 不实现 burst launch / 独立 power_ramp / SFU 吞吐 / msprof 128 扇出。
- `telemetry_trust=false`（或未修复前）时，报告不得出现任何「无热节流」类文字结论。

---

## 6. 假阳性与 torch 代理 Cube/Vector/Scalar 可信度评估

| 维度 | 代理方式 | 可信度 | 依据 |
|------|---------|--------|------|
| **Cube** | `func_perf`（golden fp64 校验）+ `sustained` | **高** | 128 卡真实数据、CV 2.4%/1.34%、有独立正确性校验 |
| **HBM** | `src*2` 读写 | **高（信号已确认，机制待拆解）** | 128 卡实测尾部 -18.45%、节点聚集清晰，但「本征介质」vs「MTE/DMA 通路」未区分（见 §4.2 建议新探针） |
| **Vector** | `a*b+c` fp32 FMA | **中，未验证** | 设计合理，但从未在此集群跑过；且 `warmup=20` 没有类似 `sustained` 的 flatline 检测，DVFS 是否真正到达稳态未知，可能引入环境温度相关的方差 |
| **Scalar** | `torch.cumsum` 依赖链 | **低** | 各方一致：不排除向量化快路径，代理有效性未证实，首轮仅采集是正确决策 |
| **Launch** | wall/event 三桶 | **中低** | NPU event 计时含 host enqueue（无 `AclValueGate` 等价物）；`timing_method` 可能分裂导致跨卡不可比（§2.3 新发现）；建议探针运行期间**直接暂停** `TelemetrySampler` 后台线程（而不只是「降频率」），因为 Python GIL 下后台线程的 `subprocess`/`threading` 调度本身就可能在 µs 级测量窗口里制造尖峰，这是比「OS jitter」更具体、更可控的一个干扰源，此前文档未提及。 |
| **SFU** | 仅 SDC 正确性 | N/A（非吞吐） | 优先级低，维持现状 |

---

## 7. Top 5（本文独立结论，供父 agent 参考）

1. **`temp_c≡2` 的根因链条被前四份文档写错了一半**：不是「`I2C` 出现在正常数据里」，而是（大概率）`npu-smi info -i <device>` 本身语法不完整触发用法帮助文本回显，帮助文本里的 `i2c_check` 撞上裸正则。用本集群真实抓包（`logs/telemetry-20260710_224628`）验证：真实输出无 `I2C`、温度正常 36–43℃。**只改正则不改命令，问题不会解决**。
2. **卡/芯粒 ID 寻址错位**是一个全新发现：这台机器是 8 卡×2 chip=16 逻辑设备，npu-smi 用 `-i`(Card)+`-c`(Chip) 寻址，代码只传 `-i str(device 0-15)` 从未传 `-c`——必须机上标定映射后一起修，不能只修正则/命令。
3. **`comparison_group`（driver/firmware 分桶）是空转的**：字段和分桶逻辑都在代码里，但 `driver_version`/`firmware_version` 从未被赋值，所有卡实际上共享同一个「unknown」桶。这不是配置项，是需要新写的赋值逻辑。
4. **`launch_latency` 存在「测了等于没测但看起来合格」的静默假阳性**：`event_timer()` 退化到 `wall_sync` 时 `host_overhead_us` 为空列表 → `None` → aggregate 里被 `isinstance` 挡掉 → 既不判 slow 也不报警，该卡在这个维度上「悄悄免检」。聚合前必须核对 `timing_method` 全集群一致。
5. **`vector_fma_perf` / `launch_latency` 不应该在首轮就定为 slow 判定主键**——这两个信号在这个规模的集群上一次都没跑过，没有 CV 数据支撑阈值。建议首轮沿用「只观察」，仅用已验证的 `func_tflops`/`hbm_gbps` 判 slow，16 卡冒烟后再决定是否升级 vector（launch 需更谨慎，原因见 §2.3/§6）。这是本文与 merged 方案唯一的实质性分歧，需要人工拍板。

---

## 附录 A：与 merged 方案的逐条裁决

| merged 方案条目 | 本文裁决 |
|-----------------|----------|
| `temp_c≡2` = I2C 误匹配 | **部分推翻**：现象判断（垫脏数据）成立，机制判断（I2C 出现在正常输出里）被本集群真实抓包证伪，改为「非法命令触发帮助文本」假说 |
| usages 无 Vector%，真字段 Aicore/Aicpu/Ctrlcpu/MemBW | 同意，无异议 |
| HBM 慢尾节点聚集 → within_host 残差为换卡前置 | 同意，无异议 |
| R1 P0 过宽，收窄方向 | 同意方向；补充三项新发现（§2）为额外 P0 |
| scalar 首轮不进主键 | 同意 |
| sustained 窗 power/freq 派生升阻塞 | 同意，但要求先修数据源（§3.1） |
| `func`(0.20)/`hbm`(0.15) 进主键；`vector`+`launch_host_p99`(0.15) 进主键；sustained 辅证 | **不同意 vector/launch 进主键部分**（§3.2）；func/hbm/sustained 部分同意 |
| comparison_group 列入 checklist 补充项 | **裁决更正**：不是「补充缺失字段」，是「填充已存在但空转的字段」（§2.2） |
| health_counters 作为 checklist 一行 | 同意需要，但标注为「新写代码」而非配置改动 |
| 报告删除「无 throttling」默认文案 | 同意，且要求 `slow_cause.classify()` 本身也要有 `telemetry_trust` 门禁，不只是报告模板 |
| 若加 3 个探针：pipeline → mte_copy → launch burst | **部分不同意排序**：本文排序为 mte_copy_perf → cube_vector_pipeline → HBM 访问模式扫描（不选 launch burst，理由见 §4.2） |

## 附录 B：核查证据清单

| 证据 | 路径 |
|------|------|
| 真实 npu-smi 原始抓包（无 I2C，温度正常） | `logs/telemetry-20260710_224628/results/master0.jsonl` |
| npu-smi 官方命令语法（`-i`=Card ID 是 Option 不是 Command） | 陈少文博客 `npu-smi 基本使用`（Grok 附录 A 已引用同一来源，本文重新解读） |
| 128 卡真实基线（func/hbm/sustained CV） | `reports/card_screen_128.md` |
| `telemetry.py` 裸 `-i` 调用、无 `-c` | `projects/CARD_SCREEN/card_screen/telemetry.py` |
| `aggregate.py` GROUP_FIELDS 声明但从未被赋值 | `projects/CARD_SCREEN/card_screen/cluster/aggregate.py` + 全包 grep 无 writer |
| `launch_latency` event/wall_sync 分裂 | `projects/CARD_SCREEN/card_screen/probes/stage_c.py` |
| `slow_cause.classify()` has_temp 逻辑 | `projects/CARD_SCREEN/card_screen/slow_cause.py` |
| `report.py` 「无(满载未降频)」默认文案 | `projects/CARD_SCREEN/card_screen/report.py` |
| CUDA 有 `StreamValueGate`，NPU 侧 `AclValueGate` 仍是 TODO | `projects/CARD_SCREEN/card_screen/timing/gates.py` |
