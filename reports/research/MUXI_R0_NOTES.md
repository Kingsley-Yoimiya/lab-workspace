# MUXI R0 调研笔记 · 2026-07-12

## 决策摘要

| 议题 | 结论 |
|------|------|
| IB 未生效根因 | `*_IB_HCA=mlx5` 与 verbs 错配；`ibv_devinfo` 仅有 Active 的 `xscale_*`（RoCE） |
| 下一刀 | 保留 `*_SOCKET_IFNAME=eth0`；改 `NCCL/MCCL_IB_HCA=xscale`；DEBUG=INFO |
| Dense | tiny 已通 → 升配 20L/H4096 + `--log-throughput` + local/unfused + nvcc shim |
| MoE | 真 Megatron MoE 未落地；先 proxy（G8）→ 缩小版 EP wrapper（experts 8–16） |
| 慢卡 | bad=`worker-12:0`；整节点剔 worker-7/12/14 → clean=13×8=104 |

## IB 试探顺序

1. D0 只读：`ibv_devinfo` / `mx-smi topo -n` / `ls /sys/class/infiniband`
2. T1：`IB_HCA=xscale`（双写 NCCL/MCCL）
3. T2：显式 `xscale_0,xscale_1,xscale_2,xscale_3`
4. T3：+ `IB_GID_INDEX`
5. 门禁：w16 AR@256MB 相对 w8 保持率 ≥ 50%

## Dense / MoE 矩阵初稿

见计划 Phase 2a/2b；parser 需 `--log-throughput`，peak=279.9。

## 证据锚点

- fire 脚本写死 mlx5：`fire_nccl_scale_muxi.sh` L49–54
- topo raw：verbs 仅 xscale
- 冒烟：`muxi_smoke_20260711.md`
