# 沐曦 Fabric 最小网络侧 ground truth 请求

可直接转发给网络/沐曦平台同事。

## 背景

集群：`vc-c550-h3c-test`  
作业：`yinjinrun-cs512-20260716-221823`  
身份：`yinjinrun.p`  
端侧HCA：每节点 `xscale_0..3`，GID index5，业务设置 `MCCL_IB_TC=128`

端侧已确认MCCL读取TC128，但无法确认QP traffic class、线上DSCP/ECN和交换机
队列映射；容器内无xscale CE/CNP/PFC/retry/FEC counter。

## 请求1：TC128准确语义

请提供当前驱动/MCCL版本下：

1. `MCCL_IB_TC=128`写入哪个QP字段；
2. 最终QP `traffic_class` 的8-bit值；
3. RoCEv2线上IPv4/IPv6 DSCP和ECN值；
4. 是否设置VLAN PCP；如设置，其值；
5. 是否存在驱动、固件或交换机重写。

请勿只回复“按标准ToS换算”，需要本版本实现或实测依据。

## 请求2：端口与物理映射

请提供64节点：

`pod/host → xscale_0..3 → NIC BDF → 交换机设备 → 物理端口 → rail/平面`

至少包含：

- leaf/ToR设备和端口
- leaf-spine上行组
- 是否四rail物理独立
- 端口速率与超分关系
- LAG/ECMP成员

## 请求3：trust与队列配置

对上述业务端口和上行，请提供：

1. trust模式：L2 PCP或L3 DSCP
2. DSCP/PCP → switch priority映射
3. priority → PG/TC/queue映射
4. PFC enable bitmap
5. PFC xon/xoff/headroom阈值
6. ECN min/max/probability阈值
7. CNP报文priority/queue
8. MTU与buffer配置

## 请求4：指定时间窗counter

请按设备/端口/priority/PG/queue导出：

- bytes / packets
- queue occupancy / max occupancy
- shared buffer occupancy
- PFC pause tx/rx帧数与duration
- ECN marks / CE packets
- CNP sent/received
- queue drop / discard
- retry / timeout
- CRC、symbol error、FEC corrected/uncorrected

重点时间窗：

1. w512终验：run `20260718_222641-w11-world512-64n8r-final-validation`
   - 约2026-07-18 22:26:53～22:27:38 CST
2. world256 channel A/B：
   - campaign `20260719_083011-w4-world256-channel-ab`
3. world256 rail A/B：
   - campaign `20260719_084825-w4-world256-rail-ab`
4. world256 GPU mapping A/B：
   - campaign `20260719_090611-w4-world256-gpu-affinity`

后3组请以各run `manifest.yaml` 中的 `fire_begin/fire_end` 为准确窗口。

## 请求5：xscale端侧counter方法

请给出当前xscale驱动可读取以下counter的正式命令/API和权限要求：

- data bytes/packets
- retry
- CE/CNP
- PFC pause
- drop/discard
- CRC/FEC/symbol error
- 当前发送rate/DCQCN状态

容器内当前不存在标准`counters`/`hw_counters`目录，perfquery、rdma、ethtool
也不可用。若只能在宿主机或管理面读取，请给出按pod/PCI BDF关联的方法。

## 返回格式建议

- 配置快照：文本/YAML
- counter：CSV，包含timestamp、device、port、priority/PG/queue、counter name/value
- 拓扑：CSV，包含host、HCA、switch、port、rail
- 时间使用CST并带时区

这些字段用于端到端行为证据对齐，不会据此直接修改生产QoS。
