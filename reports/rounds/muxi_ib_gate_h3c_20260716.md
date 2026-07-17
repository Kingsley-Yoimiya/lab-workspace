# Muxi IB 门禁 · h3c-test · 2026-07-16

## 判定：**跨节点门禁通过**；全尺度 8→512 功能跑通

| world | 配置 | AR@256M bus_bw 中位 | 相对 w8 |
|------:|------|-------------------:|--------:|
| 8 | `xscale_0..3` + `GID=5` + `VSWITCH=1` | **127.0 GB/s** | 100% |
| 16 | 同上 | **73.5 GB/s** | **57.9% PASS** |
| 32 | 同上 | 52.0 GB/s | 41.0% |
| 64 | 同上 | 38.8 GB/s | 30.5% |
| 128 | 同上 | 28.3 GB/s | 22.3% |
| 256 | 同上 | 26.8 GB/s | 21.1% |
| 512 | 同上 | 4.47 GB/s | 3.5%（64/64 done） |

## Env（对齐线上 `muxi-128node`）

```
MCCL/NCCL_SOCKET_IFNAME=eth0
MCCL/NCCL_IB_HCA=xscale_0,xscale_1,xscale_2,xscale_3
MCCL/NCCL_IB_GID_INDEX=5
MCCL_IB_TC=128
MCCL_ENABLE_VSWITCH=1
MCCL_PCIE_BUFFER_MODE=0
rdma-training/roce: 1
```

## 相对历史

- 2026-07-12 mohe：w16 RoCE FAIL / eth0 keep 0.13%
- 本轮 h3c-test + 生产 env：**跨机 RoCE 可用，并扩到 512 卡**

## 产物

- AFS: `/afs-a3-weight-share/yinjinrun.p/results/muxi-ib-gate-20260716_222055/`
- Job: `yinjinrun-cs512-20260716-221823`
- 战役总报: [`muxi_overnight512_h3c_20260716.md`](muxi_overnight512_h3c_20260716.md)
