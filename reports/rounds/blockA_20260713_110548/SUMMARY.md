# Block A：Ascend 双轨分解（stamp `20260713_110548`）

对应计划：`ASCEND-8x16-CLUSTER-EXPERIMENT-PLAN.md` §2。

## 主结果

**gap** = 各 iter 上 `max(rank_ms) − median(rank_ms)`，再对稳态 iter 取中位数。  
**network_contrib** = `gap_real − gap_indep`。

| N | gap_indep (ms) | gap_real (ms) | network_contrib (ms) | 跨节点 |
|---|---:|---:|---:|---|
| 8 | 0.29 | — | — | 否（半节点） |
| 16 | 0.53 | **1.56** | **1.02** | 否（整节点） |
| 32 | 0.58 | **1.97** | **1.40** | 是 |
| 64 | 1.15 | **2.43** | **1.28** | 是 |
| 96 | 1.26 | **2.67** | **1.40** | 是 |

图：`blockA_gap_decomp.svg`

## 关键判据：N=16→32

| 量 | 16（节点内） | 32（跨节点） | Δ |
|---|---:|---:|---:|
| gap_indep | 0.53 | 0.58 | +0.04（几乎平坦） |
| gap_real | 1.56 | 1.97 | **+0.42** |
| network_contrib | 1.02 | 1.40 | **+0.37** |

结论（一阶）：**跨节点抬升主要由 network_contrib 贡献，而非纯计算抖动**——与计划里「16→32 台阶」判读一致。跨节点之后 32→64 的 network_contrib 基本持平（1.40→1.28），gap_real 继续升更多来自 indep 抬升与步长变化。

## 采集链路

- indep：`virtual_sync_bench_npu.py --mode independent`（纯 MLP，无 HCCL）
- real：Dense Qwen3-8B，TP=4 PP=2，GBS=DP×160；`failslow_step_timer.py`
- 分解：`parse_network_contrib.py`

## 路径

- AFS indep：`/afs-a3-241ceshi-shared/montyyin/results/blockA_indep/20260713_110548/`
- AFS real-16：`.../blockA_real/20260713_110548/`
- real 32/64/96：`.../dense_failslow_gbsprop/20260713_071316/`
