# Muxi Fabric Probe 最终交付

- 执行周期：2026-07-18～2026-07-19
- 集群：`vc-c550-h3c-test`
- 作业：`yinjinrun-cs512-20260716-221823`
- 身份：`yinjinrun.p`
- 本机结果根：`myportal/results/muxi-h3c/`
- AFS根：`/afs-a3-weight-share/yinjinrun.p/results/`
- 最终交付包：`myportal/results/muxi-h3c/20260719-final-delivery/`

## 1. 最终阶段判定

| 阶段 | 判定 | 含义 |
|---|---|---|
| G0 | 通过 | torch脚本与厂商独立基准分别在自身口径复现规模增大后吞吐下降 |
| G1 | 通过（行为分类层） | 可区分rank/world、节点覆盖、每节点注入、尾部/高基线/间歇stall表型 |
| G2 | 未通过稳定拓扑聚类 | 完成2016 pair矩阵，但最差pair复测恢复，无稳定双向慢或方向性慢 |
| G3 | 未通过QoS/ECMP强结论 | 仅证实MCCL读取TC128；缺QP/wire/switch/counter闭环 |
| W4 | 无可重复缓解措施 | channel、rail、GPU mapping干预均未满足预注册扩大门槛 |

## 2. W3/G3证据链

当前可证实链：

`MCCL_IB_TC=128被MCCL读取`

证据：3个代表pod近期MCCL INFO共30次记录
`MCCL_IB_TC set by environment to 128`。

后续链条：

`QP traffic class → wire DSCP/ECN/PCP → switch trust → priority →
PG/TC/queue → PFC/ECN/CNP → RNIC rate`

均未闭环：

1. QP tclass：libmccl有`MCCL_IB_TC`、`tclass`、`ibv_modify_qp`线索，但没有
   源码调用链或运行日志打印QP值。
2. 线上DSCP/ECN：无抓包或等价遥测；禁止按标准ToS推断实际值。
3. switch trust/priority/PG/PFC/ECN：无网络侧配置快照。
4. RNIC counter：xscale无标准counters/hw_counters目录；perfquery、rdma、
   ethtool不可用。
5. 通用netdev counter：16MiB×13次P2P只增加520 bytes，证实不反映RoCE
   payload。

软件控制能力：

- `MCCL_ALGO=Ring/Tree`、`MCCL_PROTO=Simple`：被接受/读取，但实际
  collective算法/protocol kernel未独立闭环。
- `MCCL_MIN/MAX_NCHANNELS=4/8`：INFO明确报告4/8 coll channels，实际生效。
- `MCCL_IB_QPS_PER_CONNECTION`：只有库字符串，合法值域和实际QP数未知，
  未运行。

G3终判：只能形成“端到端多信号与受控软件A/B”的有限结论，不能对交换机
拥塞、PFC、ECN、CNP或ECMP机制作强结论。

## 3. W4干预终判

### 3.1 Channel

固定world256=64×4：

| 配置 | VALID数 | bus_bw中位/范围(GB/s) | p50/p95/p99中位(ms) |
|---|---:|---:|---:|
| default（32 channels） | 2 | 13.491 / 5.854–21.129 | 21.012 / 305.597 / 397.879 |
| fixed8 | 3 | 5.064 / 4.182–11.900 | 16.847 / 575.925 / 588.736 |
| fixed4 | 1 | 3.302 | 22.202 / 539.175 / 903.206 |

fixed8首轮曾触发补跑，但三轮收益未复现，最终相对default中位低62.5%。
default 5次尝试中3次post-destroy SIGSEGV，只获得2个VALID。

判定：fixed4/8均不作为修复；回滚unset MIN/MAX。

### 3.2 Rail

| 配置 | bus_bw中位/范围(GB/s) | p50中位(ms) | 极端stall |
|---|---:|---:|---:|
| dual01 | 16.033 / 6.406–28.367 | 16.789 | 2/3 |
| dual23 | 14.515 / 5.328–14.613 | 36.368 | 1/3 |
| quad | 28.236 / 2.723–31.875 | 17.882 | 1/3 |

dual01相对quad中位低43.2%，只有quad发生极端stall的block3为正，未达到
“>=20%且3块一致”。dual23的中心确实更慢，但不能反推dual01优于quad。

判定：dual01不是可重复缓解；回滚quad。

### 3.3 GPU物理集合/顺序

能力门禁由JSON `physical_gpu`、rank_mapping和MCCL PCI busId三层确认。

| 配置 | bus_bw中位/范围(GB/s) | p50中位(ms) | 极端stall |
|---|---:|---:|---:|
| lower 0,1,2,3 | 29.801 / 7.437–32.233 | 17.327 | 1/3 |
| upper 4,5,6,7 | 24.257 / 4.435–25.231 | 17.336 | 1/3 |
| spread 0,2,4,6 | 33.683 / 8.100–42.092 | 11.048 | 1/3 |

spread常态p50较低，但相对lower中位仅+13%，且block3为-72.8%；upper为-18.6%。
无人达到“>=20%且3块一致”。

判定：不进入w512；回滚unset CUDA_VISIBLE_DEVICES。

### 3.4 Rank block permutation背景

默认quad三轮bus_bw中位37.543、范围29.645–38.951 GB/s；3个固定permutation
中位35.896、范围31.393–39.907。范围高度重叠，不能建立rank顺序敏感性或
多峰。

## 4. 最终候选根因排序

### 1. 大规模同步尾部与间歇stall

- 支持：多个world、nproc、rail、channel、GPU配置均出现数百毫秒到秒级尖峰；
  w512的p95/p99达871/1069ms；post-destroy SIGSEGV跨节点/rank漂移。
- 反证：p50常保持稳定，异常不是每轮持续。
- 替代解释：MCCL运行时、启动/销毁、背景扰动、共享fabric瞬时事件。
- 置信度：高（现象），低～中（具体机制）。
- 下一证据：同一时间窗MCCL阶段日志+RNIC/交换机queue/PFC/ECN/counter。

### 2. 稳定rail/GPU-NIC路径行为差异

- 支持：single xscale0/1中位约20.146 GB/s，xscale2/3约12.241 GB/s；
  world256 dual23 p50约36ms，dual01/quad约17ms。
- 反证：quad仍通常优于dual01；无逐rail运行时bytes，无法证明利用均衡。
- 替代解释：GPU↔NIC亲和、PCI/NUMA、MCCL channel到HCA映射。
- 置信度：中高（行为分组），低（物理原因）。
- 下一证据：逐rail bytes/QP/channel映射和GPU-HCA亲和ground truth。

### 3. 节点覆盖与每节点注入耦合

- 支持：matched-world呈非单调布局效应；nproc8稳定尾部型，nproc1/2整体
  高基线型，nproc4低基线但间歇stall；world512出现额外断崖。
- 反证：同节点覆盖变化方向不总一致。
- 替代解释：MCCL规模阈值、拓扑选择、GPU mapping、fabric覆盖共同变化。
- 置信度：中。
- 下一证据：能打印实际collective算法/channel/QP的固定变量实验。

### 4. MCCL/本地软件映射

- 支持：channel数和GPU集合能显著改变p50/尾部；变量透传和物理映射已证实。
- 反证：channel、rail、GPU三类W4干预都没有3块一致的>=20%收益；rank
  permutation无明显敏感性。
- 替代解释：参数只改变暴露stall的时序，并非根因。
- 置信度：中（参与因素），低（单一根因/可修复性）。
- 下一证据：厂商支持的算法/protocol/QP值域与逐collective选择日志。

### 5. 固定坏节点或固定坏pair

- 支持：原始矩阵有极慢pair。
- 反证：64节点静态inventory一致；节点发送/接收边际范围窄；原始最差10
  pair正反重复全部恢复；稳定双向慢=0、方向性慢=0、节点边际异常=0。
- 替代解释：原始单次慢边是间歇stall。
- 置信度：高置信反证。
- 下一证据：无需优先换卡/换线；只对未来重复慢点做定点复核。

### 未证实机制

交换机拥塞、PFC、ECN、CNP、共享buffer和ECMP碰撞仍是可能解释，但当前没有
counter、配置或抓包闭环，**不得写成已证实根因**。

## 5. 网络侧最小请求

可直接转发：

`reports/rounds/MUXI_NETWORK_GROUND_TRUTH_REQUEST_20260719.md`

核心字段：

1. TC128的QP字段、实际DSCP/ECN/PCP和重写规则；
2. host xscale0..3到leaf/物理端口/rail映射；
3. trust mode与DSCP/PCP→priority→PG/TC/queue映射；
4. PFC enable/threshold/headroom/pause；
5. ECN threshold/CE与CNP priority/counter；
6. w512和W4异常窗口的端口bytes、queue occupancy、drop、retry、CRC/FEC。

## 6. 可复跑入口

完整runbook：

`docs/MUXI_FABRIC_PROBE_RUNBOOK.md`

组件索引：

`fabric-probe/README.md`

关键规则：

- 跳板home kubeconfig，身份yinjinrun.p
- 集群节点操作从跳板并发发射
- 每run新端口、新目录、manifest/run.log/raw/SUMMARY
- 关键里程碑立即回拉 `myportal/results/muxi-h3c/<run_id>/`
- INVALID永久排除，不因后续成功改写
- 默认回滚态：quad、默认channel、unset CUDA_VISIBLE_DEVICES

## 7. 计划§14交付完成度

| # | 交付项 | 状态 | 说明 |
|---:|---|---|---|
| 1 | 可复现benchmark与指标口径 | 完成 | w0.1/global-max、厂商独立基准、schema/runbook |
| 2 | rank/节点/注入/rail/算法因子分解 | 完成（行为层） | W1 matched-world与W4三类A/B |
| 3 | 节点×rail inventory | 完成 | 64节点×4rail，静态字段全量一致 |
| 4 | 端点亲和与异常pair | 部分完成 | 2016矩阵+50次复测；无稳定拓扑聚类 |
| 5 | QoS映射和可见性清单 | 部分完成 | 仅TC环境变量读取；QP/wire/switch链阻塞 |
| 6 | 候选根因排序 | 完成 | 本报告§4 |
| 7 | 修复/缓解A/B | 完成（均否决） | channel、rail、GPU mapping均未达门槛 |
| 8 | 网络组最小ground truth | 完成请求/外部阻塞 | 已生成可转发清单，尚未收到数据 |
| 9 | 已验证probe、JSONL与自动报告 | 完成（最小组件） | inventory/pair/qos组件和测试 |
| 10 | 匿名化数据与决策日志 | 部分完成 | 决策日志完整；匿名化数据未生成 |

总体：工程/行为归因主线完成；QoS/ECMP物理机制和匿名化交付仍阻塞。

## 8. 关键报告与结果根

报告：

- `reports/rounds/muxi_fabric_w0_20260718_174428.md`
- `reports/rounds/muxi_fabric_w1_20260718_185610.md`
- `reports/rounds/muxi_fabric_w2_20260718_233922.md`
- `reports/rounds/muxi_fabric_w3_20260719_080732.md`
- `reports/rounds/muxi_fabric_w4_20260719_083011.md`

关键结果：

- W2 inventory：`20260718_233922-w21-muxi-inventory`
- W2完整矩阵：`20260719_003513-w22-full-pair-matrix`
- W2异常复测：`20260719_013841-w22-full-pair-analysis-retest`
- W3可见性：`20260719_080732-w31-qos-visibility`
- W3 counter差分：`20260719_081238-w32-netdev-counter-delta`
- W4 channel：`20260719_083011-w4-world256-channel-ab`
- W4 rail：`20260719_084825-w4-world256-rail-ab`
- W4 GPU：`20260719_090611-w4-world256-gpu-affinity`

## 9. 最终默认态

- 无实验进程残留
- HCA回滚quad `xscale_0..3`
- channel回滚unset MIN/MAX
- GPU可见性回滚unset CUDA_VISIBLE_DEVICES
- TC/QoS/MTU/交换机从未修改
- 未提交git
