# Muxi IB 门禁报告 · 2026-07-12

## 判定：**跨节点门禁未通过**（保持率 0.13% ≪ 50%）

| world | 配置 | AR@256M bus_bw 中位 | 相对 w8 |
|------:|------|-------------------:|--------:|
| 8 | xscale HCA（机内） | **202.5 GB/s** | 100% |
| 16 | eth0 socket + `IB_DISABLE=1` | **0.259 GB/s** | **0.13%** |
| 16 | `IB_HCA=xscale`（无 DISABLE） | **失败**（Proxy Connect / no transport） | — |

## 根因（更新）

1. verbs 可用设备是 **xscale_***（RoCE），不是 mlx5（ibv 找不到 mlx5_*）。
2. 容器内 **net1–4 无 IP**；RoCE GID index 4/5 映射到 172.23–26.x，但 pod netns 未挂这些地址 → 选 xscale 后无法建链，硬失败。
3. `IB_DISABLE=1` + eth0 可跑通跨节点，但带宽与历史战役一致（~0.2 GB/s）。

## 决策（执行）

- **多机 MFU 矩阵暂停**（避免假象 0.2% MFU）。
- **全力跑单机 8 卡** Dense/MoE 并行策略扫参（多 worker 并行占满）。
- IB 继续后台试：hostNetwork / RDMA device plugin / GID+IP 配置（需平台侧配合）。

## 产物

- AFS: `/afs-a3-weight-share/montyyin/results/muxi-ib-gate-20260712_083113/`
- 本地: `logs/muxi-ib-gate-20260712_083113/`
- 跳板: `ais-cf3e61a5`（见 `docs/AIS_JUMP_CLUSTER.md`）
