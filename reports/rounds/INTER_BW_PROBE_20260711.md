# 机间带宽探针简报（2026-07-11）

## 结论

在 **8×16=128 卡 A3 job**（`whj4stu-copy-copy-copy`）上，用严格串行 HCCL P2P（流水线单向，`inflight=4`）实测：

- **机间单链路有效带宽（256MB，recv 中位）≈ 119.3 GB/s**
- **机内单链路有效带宽（256MB，recv 中位）≈ 122.4 GB/s**
- **inter/intra ≈ 0.975**（机间几乎打满机内量级）

无 `hccn.conf` / `hccn_tool` 时，这是对机间带宽的直接反推。结合 A3 常见 **UB/UBoE scale-up** 拓扑，机间≈机内是合理现象（8 节点很可能仍在同一 UB 域内），**不能按传统 RoCE「机间远低于机内」的经验去判异常**。

## 方法

- 脚本：`scripts/cluster/hccl_inter_bw_probe.py` + `launch_inter_bw_kubectl.sh`
- 全员 barrier；每轮只测一对；非端点空等
- **intra**：同节点 pairs `(0,1)(0,8)(7,8)(1,9)` 双向
- **inter**：相邻节点环 + 少量跨跳；同 `local_rank∈{0,5,10,15}` 对齐，双向
- sizes：`1M,16M,64M,256M`；warmup=8；iters=30；`HCCL_BUFFSIZE=2048`
- 原始数据：`logs/inter-bw-20260711_141922/`（远端 `/data/montyyin/results/inter-bw-20260711_141922`）

## 带宽表（recv 侧中位）

| kind | 1M | 16M | 64M | 256M |
|---|---:|---:|---:|---:|
| intra | 7.70 | 91.27 | 114.04 | 122.35 |
| inter | 8.64 | 89.39 | 110.52 | 119.32 |

## 交叉对照

- send 与 recv 带宽几乎一致 → 不像「本地入队假完成」（假完成通常 send ≫ recv）。
- 此前 AllReduce@128 卡 / 256MB 的 `bus_bw` 中位约 **138 GB/s**，与单链路 P2P ~120 GB/s 同量级，互相支撑。
- 机内 topo 为 `SIO` / `HCCS_SW`；机间无 hccn 可读，只能靠本探针。

## 解读注意

1. 测的是**空载单对上限**，不是 128 卡全打满时的可用份额。
2. 「host 不同」≠「跨了 UB SuperPod 边界」；若需确认是否跨 scale-out，需问集群方 UB 域划分。
3. 脚本已支持 `--pingpong` / `--bidir` 做进一步交叉验证。

## 图

默认 `plot_style` SVG 版本见 [`inter_bw_20260711.md`](inter_bw_20260711.md) 及其图目录 [`inter_bw_20260711_figs/inter_vs_intra_bw.svg`](inter_bw_20260711_figs/inter_vs_intra_bw.svg)；原始 PNG 见 `logs/inter-bw-20260711_141922/summary/inter_vs_intra_bw.png`。

## Ping-pong 交叉验证（已完成）

- 模式：`A→B→A`，带宽按 RTT/2 换算单程；sizes=`16M,256M`；`local_samples=0,8`
- **inter@256M 中位 ≈ 117 GB/s**，与单向流水线 **119 GB/s** 一致（偏差 <2%）
- **intra@256M** 亦同量级（~120 GB/s，SIO 对可达 ~144）
- 日志：`logs/inter-bw-20260711_142537/`

→ **采信：本批 8 节点机间单链路有效带宽约 115–120 GB/s（大包饱和区）。**
