# Muxi NCCL P2P（G5）

时间: 2026-07-11  
AFS: `/afs-a3-weight-share/montyyin/results/nccl-p2p-20260711_150700`

## 结论

- scale **16 / 128** ring 均完成，全部 send 边 `ok=256/256`（128 档）
- **机内** 16M 中位约 **30–33 GB/s**（MetaXLink）
- **跨节点** 16M 中位约 **0.35 GB/s**（eth0 socket；与 G4 一致）

## scale=16 ring

| 类型 | 64K med | 16M med |
|------|--------:|--------:|
| 机内 | 0.49 GB/s | **33.3 GB/s** |
| 跨节点 | 0.22 GB/s | **0.35 GB/s** |

## scale=128 ring

| 类型 | 64K med | 16M med |
|------|--------:|--------:|
| 机内 (224 edges) | 0.38 GB/s | **30.4 GB/s** |
| 跨节点 (32 edges) | 0.19 GB/s | **0.35 GB/s** |

## 脚本

`nccl_p2p_bench.py` · `fire_nccl_p2p_muxi.sh` · `poll_nccl_p2p_muxi.sh`
