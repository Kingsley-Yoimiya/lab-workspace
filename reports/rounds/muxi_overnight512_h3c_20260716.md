# Muxi h3c-test · 512 卡夜间战役 · 2026-07-16（收口）

## 作业

- Job: `yinjinrun-cs512-20260716-221823`（1+63=64 节点 × 8 = **512** MXC550）
- 集群: `vc-c550-h3c-test` / API `10.140.158.130:3231`
- AFS: `/afs-a3-weight-share/yinjinrun.p/results/`
- CARD_SCREEN: `muxi-overnight`（montyyin_develop @ d8c8311）
- 日志: `~/random-thing/logs/muxi-overnight512-20260716_222055/`
- 本机: `myportal/results/muxi-h3c/20260716_222055-overnight512/`

## 任务 B：多卡通信 — **全尺度跑通**

生产 MCCL（对齐线上 `muxi-128node`）：

```
SOCKET_IFNAME=eth0
IB_HCA=xscale_0,xscale_1,xscale_2,xscale_3
IB_GID_INDEX=5
MCCL_ENABLE_VSWITCH=1
MCCL_IB_TC=128
rdma-training/roce: 1
```

| world | AR@256M bus_bw 中位 | keep vs w8 | 节点 done |
|------:|--------------------:|-----------:|----------:|
| 8 | **127.0 GB/s** | 100% | 1/1 |
| 16 | **73.5 GB/s** | **57.9%**（门禁 PASS） | 2/2 |
| 32 | 52.0 GB/s | 41.0% | 4/4 |
| 64 | 38.8 GB/s | 30.5% | 8/8 |
| 128 | 28.3 GB/s | 22.3% | 16/16 |
| 256 | 26.8 GB/s | 21.1% | 32/32 |
| **512** | **4.47 GB/s** | 3.5% | **64/64** |

相对 2026-07-12 mohe（跨机 RoCE FAIL / eth0 keep 0.13%）：本轮 **跨机 RoCE 全尺度功能可用**；带宽随 world 下降属预期，但不再是假通路径。  
专报：[`muxi_ib_gate_h3c_20260716.md`](muxi_ib_gate_h3c_20260716.md)。

## 任务 A：筛卡

### V1 smoke（64/64）

- CLI：`--gemm-n 4096 --sustained-s 10`；默认 config 含 shape_sweep → 记录翻倍口径
- Aggregate：good=976 / slow=48（物理 512 卡）
- 中位：func≈168（N=4096）、hbm≈1475、sustained≈260

### V2 sentinel（60/64）

- 缺 worker-0/1/20/28
- Aggregate：good=461 / slow=19（480 卡）
- 中位：func≈169、hbm≈1445、sustained≈260

### V3 constitution128（63–64/64）

- `config.constitution128.yaml`，N=8192，sustained 30s
- Aggregate：**n_cards=512**；good=319 / contended=192 / slow=1
- 中位：**func≈276.5**、**hbm≈1475**、**sustained≈277**（对齐 0711≈280）
- contended 主因：部分节点 preflight 仍见残留 compute 进程（与通信战役交错）；slow=1（worker-43/dev1，intrinsic）

## 对照 0711（128 卡 constitution）

| 指标 | 0711 | 本轮 constitution512 |
|------|-----:|------------------:|
| 卡数 | 128 | 512 |
| func 中位 | ≈280 | **276.5** |
| HBM 中位 | ≈1469 | **1475** |
| sustained 中位 | ≈280 | **277** |

## 固化改动

- `muxi.env`：默认 h3c kube + kubectl；生产 MCCL
- `job_helpers.sh`：`CLUSTER_EXEC_MODE=kubectl`
- `fire_nccl_scale_muxi.sh`：GID=5 / xscale_0..3 / VSWITCH
- `fire_screen_durable_muxi.sh` / `finish_watch.py`（LaunchAgent 可选）
