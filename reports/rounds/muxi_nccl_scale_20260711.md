# Muxi NCCL scale 结果（G4）

时间: 2026-07-11  
AFS: `/afs-a3-weight-share/montyyin/results/nccl-20260711_142129`  
日志: `logs/muxi-nccl-20260711_142129/`

## 结论

- **scale 8/16/32/64/128 全完成**（每档 4 op × 4 size，全 rank 落盘）
- **单机 8 卡**（scale8）带宽正常：all_reduce 64M ≈125 GB/s，256M ≈191 GB/s
- **跨节点**（16+）走 `NCCL/MCCL_SOCKET_IFNAME=eth0`，bus_bw 约 **0.1–0.3 GB/s**（功能通、性能差）；后续应用 IB/`net*` 重测

## all_reduce bus_bw (GB/s, rank0)

| world | 1M | 16M | 64M | 256M |
|------:|---:|----:|----:|-----:|
| 8 | 4.50 | 55.78 | 125.08 | 190.68 |
| 16 | 0.29 | 0.35 | 0.25 | 0.26 |
| 32 | 0.21 | 0.23 | 0.25 | 0.27 |
| 64 | 0.08 | 0.18 | 0.23 | 0.26 |
| 128 | 0.13 | 0.15 | 0.17 | 0.24 |

## 运维要点

- 本机长 SSH/`nohup` 易被 Cursor 杀掉 → pod 内 `setsid nohup` + 5min tick `poll`/`fire`
- 多机必须设 `eth0`，否则 MCCL Proxy Connect 失败
