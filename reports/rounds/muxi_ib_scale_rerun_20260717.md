# Muxi H3C AllReduce@256M 规模复现（2026-07-17）

## 结论

掉带宽是**系统性跨机扩展问题**，不是偶发：干净复现与昨夜曲线形状一致，w512 相对机内 w8 的 keep% 都落在约 **3%**。

| world | 昨夜 med (GB/s) | 复现 med (GB/s) | 复现 keep% |
|------:|----------------:|----------------:|-----------:|
| 8 | 127.0 | **258.6** | 100 |
| 16 | 73.5 | 78.0 | 30.2 |
| 32 | 52.0 | 56.0 | 21.7 |
| 64 | 38.8 | 38.0 | 14.7 |
| 128 | 28.3 | 33.0 | 12.8 |
| 256 | 26.8 | 13.0 | 5.0 |
| 512 | **4.47** | **7.64** | **3.0** |

说明：

- 复现 w8 显著更高（单机内 MetaXLink / 无跨机争用），说明昨夜 w8 可能仍受脏场/残留影响；**跨机阶梯掉崖仍复现**。
- 同 world 内 min≈max，rank 间几乎无离群 → 不是少数病卡拖后腿，而是**集合通信整体被拓扑/拥塞卡住**。
- 最陡跌落仍在 **机内→跨机（8→16）** 与 **256→512**。

## 条件

- 作业：`yinjinrun-cs512-20260716-221823`（64×8）
- 算子：`all_reduce`，消息 `256M`，MCCL RoCE：`xscale_0..3`，`GID_INDEX=5`，`VSWITCH=1`，`IB_TC=128`，socket=`eth0`
- AFS：`/afs-a3-weight-share/yinjinrun.p/results/muxi-ib-rerun-20260717_103913/`
- 本机备份：`myportal/results/muxi-h3c/20260717_103913-ib-rerun/`

## 链路层（D0 抽样）

`ibv_devinfo`：四张 `xscale_*` 均为 `PORT_ACTIVE`，`link_layer=Ethernet`（RoCE）。  
sysfs counters 采集当时为空/不完整，不足以判定 FEC/符号错；建议后续在跑 AR 前后对各节点 `xscale_*/ports/*/counters` 做差分。

## 点火方式改进

本机串行 `kubectl exec` 起 64 节点过慢且易被本机进程打断。已验证：

- **本机并行**：`PARALLEL=16` + `xargs -P`（`fire512_parallel.sh`）可在约 1–2 分钟补齐缺失 rank。
- **ais 跳板**：有 `vcctl` + `/tmp/config-vc-c550-h3c-test.yaml`，**无 kubectl**；若要把 fanout 放到跳板，应用 `vcctl pod exec` 并行，而不是照搬本机 kubectl 脚本。

建议把 `fire_nccl_scale_muxi.sh` 改为默认 `CLUSTER_FANOUT_PARALLEL` 并行点火，避免再串行 64 次。

## 判断

| 假说 | 证据 |
|------|------|
| 偶发抖动 | ❌ 两次战役同形态；同 world 方差极小 |
| 单卡/单链路坏 | ❌ min≈max |
| 跨机 RoCE/交换机拥塞或树形算法扩展差 | ✅ 8→16 断崖 + 规模单调掉 |
| 环境变量配错导致走以太网 | ❌ D0 显示 xscale ACTIVE + RoCE env 固化 |

下一步优先：交换机侧队列/PFC/ECN、MCCL 算法/通道数；不必再为「是否偶发」重复整条 scale。

## 链路故障排查追加（2026-07-17 下午）

顾问结论（[Muxi 链路排查](29a9011b-2d67-4f93-a72e-552177d3334d)）+ 本机跟进：

| 假说 | 可能性 | 新证据 |
|------|--------|--------|
| 物理口/线缆硬故障 | **低** | 16 节点×4 口：`ACTIVE` / phys LinkUp / `rate=200`；GID5=RoCEv2；8 节点 P2P 全 ok |
| 误走 eth0 数据面 | **低** | MCCL INFO：`Using network IB`；`NET/IB : Using xscale_0..3/RoCE`；通道 `via NET/IB/*/GDRDMA` |
| 交换机拥塞 / PFC/ECN | **中** | 容器无 xscale sysfs counters；缺交换机侧 pause/ECN/drop |
| MCCL 树形扩展 / fabric 并发瓶颈 | **中高** | AR 随 world 系统性掉；8 节点 AR@256M med≈36.6 与 scale 曲线一致；INFO 显示 Trees + 32 channel |

### 机间摸底数字

- 静态（16 节点）：64 口全部 `rate=200`，无 DOWN/降速。
- 8 节点 P2P ring@16M：cross med≈**23.6 GB/s**（16 边），intra med≈**33.6 GB/s**，`ok` 全过。
- 同 8 节点 AR@256M：med≈**36.55 GB/s**（与复现 w64≈38 同量级）。
- **xscale 无 `/sys/.../counters`**，宿主机侧计数需另开权限。

产物：

- 链路采样：`.../muxi-link-sample-20260717_135116/`、`myportal/results/muxi-h3c/20260717_135116-link-sample/`
- P2P64：`.../muxi-p2p64-link-20260717_135927/`
- AR64+INFO：`.../muxi-ar64-info-20260717_140105/`
