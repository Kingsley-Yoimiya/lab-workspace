# fabric-probe 已验证组件索引

本目录只保留已在沐曦集群实机验证的最小probe，不是常驻平台。

## inventory

路径：`probes/inventory/`

- `muxi_inventory_probe.sh`：pod内只读采集xscale、GID、PCI、netdev、topo
- `run_muxi_inventory_jump.sh`：跳板64并发
- `parse_muxi_inventory.py`：JSONL/CSV/schema与中文摘要
- schema：`muxi.inventory.v1`

实机结果：64 pod、256 rail完整；ACTIVE/LinkUp/200Gb/s/MTU4096/GID5一致。

## pair

路径：`probes/pair/`

- `generate_round_robin.py`：63轮perfect matching，覆盖2016条无向pair
- `muxi_pair_bench.py`：16MiB `torch.distributed.isend/irecv`
- `run_muxi_pair_rounds_jump.sh`：跳板node-disjoint并发、pair级重试/清理
- `parse_pair_rounds.py`：结果schema、分布与明显离群
- `analyze_full_pair_matrix.py`、`run_muxi_pair_retest_jump.sh`、
  `classify_pair_retest.py`：完整矩阵独立复检和异常复测

实机结果：2016/2016 pair；原始最差10条正反复测全部恢复，无稳定坏pair。

## topology（配置与拓扑事实优先）

路径：`probes/topology/`

- `collect_control_plane.py`：pod→node、topology/CNI/boson 注解（脱敏）
- `muxi_logical_rail_probe.py` / `parse_muxi_logical_rail.py`：64×4 子网/MAC/BDF/GID
- `muxi_lldp_listen.py` / `parse_muxi_lldp.py`：AF_PACKET 只读 LLDP（不依赖 lldpctl）
- `run_muxi_topology_facts_jump.sh`：跳板编排；`aggregate_topology_report.py`：三层中文报告

实机结果（`20260719_102558-topology-facts`）：四逻辑子网与 `eth10..13`/`prod-boson` UUID 已恢复；
eth0 LLDP 为宿主机自通告；net1..4 LLDP 0 帧（pod netns 不可见，非“网络无 LLDP”）。

## qos

路径：`probes/qos/`

- `muxi_visibility_probe.py` / `parse_muxi_visibility.py`
- `muxi_netdev_counter_snapshot.py` / `parse_netdev_counter_delta.py`

实机结果：

- 可证实 `MCCL_IB_TC=128` 被MCCL读取
- QP tclass与线上DSCP/ECN未证实
- xscale专用counter不可得
- 通用netdev统计不反映RoCE payload

## 共同规则

- 集群操作从 `ais-cf3e61a5` 发起
- kubeconfig：`~/.kube/config-vc-c550-h3c-test.yaml`
- 身份：`yinjinrun.p`
- AFS仅写 `/afs-a3-weight-share/yinjinrun.p/{lab-workspace,results}`
- INVALID永久排除，不能因后续成功改写
- raw、manifest、run.log、SUMMARY和schema必须一起保留

运行入口和停止条件见 `docs/MUXI_FABRIC_PROBE_RUNBOOK.md`。
