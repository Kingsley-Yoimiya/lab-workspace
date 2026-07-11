# 华为全能力 → 沐曦适配地图

> 冒烟/快慢卡只是 L0。华为侧真正要的是：**体质分布** + **通信阶梯**（+ 可选 MFU）。

## 华为已有能力全景

```text
run_constitution_then_comm.sh
├── ① run_card_constitution_128.sh     # 体质烤机（Stage A+C+SDC+BNMK）
├── ② plot_card_constitution.py        # 分布图 / 报告
├── ③ probe_hccl_topology.sh           # 机内拓扑 + RoCE/hccn
├── ④ run_hccl_scale.sh                # collective：AR/AG/RS/Bcast × size × world
└── ⑤ run_hccl_p2p_128.sh              # 边级 P2P 慢边

旁路（不在流水线内）:
├── run_card_screen*.sh                # 快慢卡筛查（已有 muxi 冒烟）
├── run_link_health.sh                 # 轻量链路健康
├── run_mfu_bench_scale.sh             # 微基准 MFU
├── run_train_mfu_scale.sh             # 真 MindSpeed 训练 MFU
└── loop_mfu_*.sh                      # MFU 超参优化环
```

## 1. 体质（constitution）— 核心

**目的**：不是「好/坏」二分类，而是每张卡在多部件上的**分布画像**。

| 层 | 探针 | 看什么 |
|----|------|--------|
| Stage A | func_perf / hbm / sustained | Cube 算力、HBM、稳态 |
| Stage C | vector / scalar / sfu / launch | 向量、标量、SFU、启动开销 |
| Stage C | hbm_modes / mte / cube_vector | 访存模式、搬运、流水 |
| Stage C | health_counters | ECC 等快照 |
| Shape | bnmk_sweep（10 shape） | 训练代理 GEMM |
| SDC | 五类静默错误 | 正确性红旗 |

- 入口：`CASE_NAME=constitution128 ./scripts/cluster/run_card_constitution_128.sh`
- 配置：`config.constitution128.yaml`
- 出图：`python3 reports/plot_card_constitution.py --data-dir … --out-dir reports/rounds`
- 华为今日已有产物：`reports/rounds/card_constitution_20260711_134639.md`

**沐曦**：探针数学大多可复用（Metax 已接线）；扇出改 16×8 + `muxi.env` 即可。**P0**。

## 2. 通信 — 核心

| 能力 | 入口 | Ascend 依赖 | 沐曦对应 |
|------|------|-------------|----------|
| 拓扑 | `probe_hccl_topology.sh` | npu-smi topo / hccn | mx-smi / 厂商 topo（P1） |
| Collective scale | `run_hccl_scale.sh` + `hccl_torch_bench.py` | HCCL + torch_npu | **NCCL** + cuda（P0） |
| P2P 慢边 | `run_hccl_p2p_128.sh` + `hccl_p2p_bench.py` | HCCL | NCCL P2P（P1） |
| 链路健康 | `run_link_health.sh` | hccn_tool | 厂商链路工具（P2） |

指标语义可复用：`bus_bw` 保持率、边延迟 TopK、同规模重跑成功率。

## 3. MFU（次优先）

| 能力 | 入口 | 沐曦难度 |
|------|------|----------|
| 微基准 dense/moe | `run_mfu_bench_scale.sh` | 中（改 NCCL） |
| 真训练 MindSpeed | `run_train_mfu_scale.sh` | 高（需沐曦训练栈） |
| 优化环 | `loop_mfu_*.sh` | 依赖真训练 |

## 沐曦适配优先级（修订）

| 优先级 | 项 | 状态 |
|--------|-----|------|
| P0 | 快慢卡冒烟 128 | **已完成**（2026-07-11） |
| P0 | 双集群 KUBECONFIG 隔离 | **已完成** |
| P0 | **体质 constitution 扇出 + 出图** | 待做 |
| P0 | **NCCL collective scale**（对标 hccl_scale） | 待做 |
| P1 | 拓扑探测（mx-smi / 网卡） | 待做 |
| P1 | NCCL P2P ring | 待做 |
| P1 | 流水线 `run_constitution_then_comm` muxi 版 | 待做 |
| P1 | MFU 微基准 | 待做 |
| P2 | 真训练 MFU / 链路工具 / MFU 环 | 按需 |

## 建议下一步

1. 写 `run_card_constitution_muxi.sh`（复用 constitution128 配置 + 有界扇出）
2. 移植 `hccl_torch_bench.py` → `nccl_torch_bench.py`，先跑 world=8（单机）再 16/32/64/128
3. 出图直接吃 JSONL，与华为报告格式对齐，便于两边对照
