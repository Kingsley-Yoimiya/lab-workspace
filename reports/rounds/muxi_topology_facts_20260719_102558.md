# 沐曦物理 / 逻辑 / 行为三层拓扑报告

- 时间：2026-07-19
- 作业：`yinjinrun-cs512-20260716-221823`
- 身份：`yinjinrun.p`（跳板 `~/.kube/config-vc-c550-h3c-test.yaml`，kubectl=`/root/.cache/volcano/kubectl/kubectl`）
- 本机结果：`myportal/results/muxi-h3c/20260719_102558-topology-facts/`
- AFS：`/afs-a3-weight-share/yinjinrun.p/results/muxi-topology-facts/20260719_102558-topology-facts/`

## 证据等级约定

| 等级 | 含义 |
|---|---|
| 直接配置 | Kubernetes/CNI/注解或厂商配置快照原文 |
| LLDP | pod 内 AF_PACKET 实际收到并解析的邻居 TLV |
| 主机静态 | sysfs /proc/net/GID/ARP 等只读主机状态 |
| 行为推断 | pair/collective 性能聚类等行为证据 |
| 未知 | 当前权限或观测窗口下无法证实 |

## 一句话结论

已恢复 **64 节点 × 4 逻辑 rail/子网/宿主机网卡名/逻辑交换机 UUID**，并完成真正的 AF_PACKET LLDP 实测：
**eth0 可见的是宿主机自通告（非 RoCE 交换机）**；**net1..4 在 CAP_NET_RAW+70s 窗口下 0 帧**。
RoCE 直连 switch chassis/port、leaf/spine 角色与交换机 QoS **仍未知**。

---

## A. 物理直连事实

### A1. RoCE 数据面（net1..4 / xscale_0..3）

| 观测 | 结果 | 证据等级 |
|---|---|---|
| AF_PACKET 打开 | 64/64 成功，`CAP_NET_RAW` 有效 | 主机静态 |
| net1..4 LLDP 帧 | **0 / 64 pods × 4 ifaces × ≥70s** | LLDP（负结果） |
| 直连 switch chassis/port | **unknown** | 未知 |

负结果含义：在 pod 网络命名空间、给定接口与时长下 **LLDP 不可见**；
**不能**写成「网络没有 LLDP」。可能原因包括：交换机未对 PF/容器 netns 发 LLDP、
被硬件卸载旁路、仅在宿主机 netns/管理口可见等——需宿主机或交换机侧验证。

抽样证据（master，完整见 `lldp/pods/*/net*/result.json`）：

- 每接口 `duration_s≈70.2`，`raw_frames_seen=0`，`lldp_frames=0`，`af_packet_open=true`

被动旁证（主机静态，**不是**物理端口命名）：

- 每 rail ARP 主导邻居 MAC OUI 主要为 `90:25:f2`（计数 `{'90:25:f2': 256}`）
- 每 rail 存在多个主导邻居 MAC（约按节点分散），符合「网关/代理 ARP / ToR 侧 MAC」而非 peer NIC 直连
- 端侧 NIC MAC OUI：`{'5c:6a:ec': 256}`

### A2. 控制面 eth0 LLDP（可见，但不是交换机拓扑）

| 观测 | 结果 | 证据等级 |
|---|---|---|
| eth0 LLDP 总帧 | 162（64 pods） | LLDP |
| `system_name` == k8s `nodeName` | 64/64 | LLDP+直接配置 |
| `system_description` | Ubuntu 22.04 主机内核串 | LLDP |
| 解释 | **宿主机（或主机桥）LLDP 自通告**，不是 RoCE leaf/spine | 解释边界 |

示例：master-0 → node `host-10-12-144-137`，LLDP chassis `f0:fb:7f:89:a3:44`，
port_description 形如 `975fdb8bb0ef_h`，mgmt IPv4 在 `172.23.20x.x`（与 rail0 业务子网不同段）。

完整映射：`report/eth0_lldp_host_map.jsonl`。

**禁止**把 eth0 LLDP chassis 当成 xscale/rail 的直连交换机。

---

## B. 逻辑 rail / IP 子网 / CNI（可证实）

### B1. 四逻辑子网（主机静态）

| rail | netdev | 逻辑子网 | 覆盖 |
|---|---|---|---|
| xscale_0 | net1 | 172.23.168.0/22 | 64/64 |
| xscale_1 | net2 | 172.24.168.0/22 | 64/64 |
| xscale_2 | net3 | 172.25.168.0/22 | 64/64 |
| xscale_3 | net4 | 172.26.168.0/22 | 64/64 |

- 四逻辑子网：`true`；同子网跨 rail：`false`
- eth0 默认网关：全部 `10.120.16.1`（64/64）
- rail 接口无默认路由（仅连接路由）；控制面与数据面路由角色分离

解释：这是 **四个逻辑 IPv4 子网 / 逻辑 rail 隔离** 的直接证实；
**不能**据此声称四个物理独立交换平面。

### B2. 控制面直接配置（Kubernetes 注解/标签）

| 字段 | 值 | 证据等级 |
|---|---|---|
| `hostNetwork` | false（64/64） | 直接配置 |
| `boson.sensecore.cn/nic-device` | `{"net1": "xscale_0", "net2": "xscale_1", "net3": "xscale_2", "net4": "xscale_3"}` | 直接配置 |
| `boson.sensecore.cn/nic-host` | `{"net1": "eth10", "net2": "eth11", "net3": "eth12", "net4": "eth13"}` | 直接配置 |
| `k8s.v1.cni.cncf.io/network-status` | eth0=`kube-ovn`；net1..4=`prod-boson/<uuid>` | 直接配置 |
| 逻辑交换机 UUID | 恰好 4 个（与四 rail 对齐） | 直接配置 |
| `topology.sensecore.cn/zone` | `cn-pj-01a`（64/64） | 直接配置 |
| `topology.sensecore.cn/region` | `cn-pj`（64/64） | 直接配置 |
| `topology.sensecore.cn/boson-rdma-network-cluster` | `roce-cluster-01`（64/64） | 直接配置 |
| `resource.sensecore.cn/boson-rdma-network` | `RoCE` | 直接配置 |
| `resource.sensecore.cn/boson-nic-access-mode` | `PF`（64/64） | 直接配置 |
| `resource.sensecore.cn/boson-nic-training-port` | `4` | 直接配置 |
| `metax-tech.com/gpu.sriov` | `forbidden`（64/64） | 直接配置 |
| `topology.kubernetes.io/*` rack 级 | **未出现** | 未知 |

逻辑交换机 UUID 与接口对应（来自 network-status name）：

- `019beb44-0063-7b24-9f67-e4ff27ec57e4` ← 出现 64 次
- `019beb44-007e-7981-9e4f-2b5a47de1d18` ← 出现 64 次
- `019beb44-009a-70fb-9068-363cf28ee575` ← 出现 64 次
- `019beb44-00b3-7259-b650-de125cbb3be6` ← 出现 64 次

注意：部分注解同时带 `network_type=geneve` / `pod_nic_type=veth-pair`（OVN 风格），
与 `boson-nic-access-mode=PF`、`nic-host=eth10..13` 并存。
前者更像控制面/OVN 元数据，**不能单独否定** PF RoCE 数据面。

### B3. 控制面 vs 数据面角色

| 平面 | 接口 | 角色 |
|---|---|---|
| 控制面 | eth0 / kube-ovn | Pod 网络、默认路由、调度/管理 |
| 数据面 | net1..4 ↔ xscale_0..3 ↔ 宿主机 eth10..13 | RoCE 训练流量逻辑 rail |

---

## C. 行为 rail 分组（单独层，禁止命名 leaf/spine）

| 来源 | 行为观察 | 证据等级 |
|---|---|---|
| W2 2016 pair 矩阵 | 无稳定双向慢链路；最差 pair 复测恢复 | 行为推断 |
| W4 rail 掩码 | dual01/dual23/quad 差异未形成可重复物理拓扑结论 | 行为推断 |
| W1/W4 collective | 规模增大吞吐下降可复现，但机制未闭环到交换机 | 行为推断 |

行为聚类可画「慢 pair / rail 掩码」图，**不得**命名 leaf/spine。

---

## D. 为何 2016 pair 矩阵在对称 Clos+ECMP 下无法唯一反演物理拓扑

1. **ECMP 多路径**：同一逻辑 pair 报文可哈希到不同上行；重复测量不保证同路径。
2. **对称性**：无拥塞时同构路径的带宽/时延重叠，缺少独特指纹。
3. **四 rail 并行**：矩阵单元是端到端行为，不是单条物理边。
4. **缺直连锚点**：net1..4 无 LLDP/无交换机 port counter 时，行为边无法对齐 chassis/port。
5. **复测恢复**：最差 pair 不稳定，说明噪声/间歇拥塞大于稳定拓扑割。

因此 pair 矩阵最多支持行为分组假设，不能唯一反演物理拓扑。

---

## E. host×rail 总表

见：

- `report/host_rail_topology_table.csv`（256 行）
- `report/eth0_lldp_host_map.jsonl`（控制面 LLDP）

缺项（RoCE switch chassis/port）明确为 `unknown`。

每 rail 主导邻居 MAC 多样性（主机静态）：

- xscale_0: unique_macs=10, top3=[('90:25:f2:bd:20:d3', 10), ('90:25:f2:bd:1f:53', 9), ('90:25:f2:11:21:13', 9)]
- xscale_1: unique_macs=10, top3=[('90:25:f2:bd:22:03', 10), ('90:25:f2:bd:20:63', 9), ('90:25:f2:bd:20:43', 9)]
- xscale_2: unique_macs=10, top3=[('90:25:f2:11:21:63', 10), ('90:25:f2:bd:21:d3', 9), ('90:25:f2:11:20:d3', 9)]
- xscale_3: unique_macs=10, top3=[('90:25:f2:bd:1e:c3', 10), ('90:25:f2:bd:20:c3', 9), ('90:25:f2:bd:1f:23', 9)]

---

## F. 已有报告检索

见 `report/prior_evidence_search.md`。个人 AFS/本地报告中 **没有** 具体 leaf/spine 设备名或配线表；
此前 leaf/spine 字样均为待请求项或禁止命名边界。

---

## G. 仍未知 / 下一步需要的访问

| 未知项 | 需要的访问 |
|---|---|
| host×rail → 交换机 chassis/port | **宿主机 netns** 对 eth10..13 做 LLDP，或交换机 LLDP neighbor |
| leaf/spine 角色与配线 | 网络组正式拓扑 / 配线表（直接配置） |
| VLAN/LAG/MLAG/ECMP hash | 交换机只读配置 |
| DSCP/PCP trust、PFC/ETS/ECN | 交换机 QoS 与 counter |
| xscale RoCE 硬件 counter | 驱动/宿主机/厂商 API |
| rack 级 K8s topology label | 平台是否写入；当前 node 无 `topology.kubernetes.io/rack` |

---

## H. 可直接让网络组执行的厂商中立字段清单

1. LLDP neighbor：local if ↔ remote chassis/port/system/mgmt  
2. interface description / admin+oper status / speed / FEC  
3. VLAN / port-channel / MLAG 成员与角色  
4. L3/BGP/ECMP：neighbor、前缀、ECMP 成员、hash 字段  
5. DSCP/PCP trust map → priority / PG / ETS  
6. PFC enable/headroom/xon/xoff threshold  
7. ECN/WRED min/max/probability  
8. queue/buffer counters（bytes/packets/occupancy/drop/PFC/ECN/CNP）  
9. host 端口映射：`hostname / PCI BDF / host_netdev(eth10..13) / switch / port / rail`

对齐：`nic-host` 已给出宿主机侧名 eth10..13；请网络组按 **hostname=`host-10-12-14x-y` + eth10..13** 导出邻居。

独立短文：`report/网络组交换机取证请求.md`
