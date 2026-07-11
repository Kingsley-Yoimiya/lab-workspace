# Muxi 链路健康（G6）

时间: 2026-07-11  
AFS: `/afs-a3-weight-share/montyyin/results/link-health-muxi-20260711_154648`  
日志: `logs/link-health-muxi-20260711_154648/`

## 结论

- **16/16 节点**均采到 `mx-smi` 温度/内存/ECC/功耗/PCIe/MetaXLink/topo 与 `ibv_devinfo`
- master-0 样例：8× MXC550-PL 均为 **Available**
- 脚本: `scripts/cluster/run_link_health_muxi.sh`

## 说明

跨节点通信瓶颈仍在 eth0 socket（见 G4/G5）；本步确认设备侧健康与 IB 设备可见（mlx5/xscale），为后续切 IB 提供基线。
