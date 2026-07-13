# 华为 A3 离线收口简报

> 生成时间：2026-07-13 10:11 +0800
> 路径：`/afs-a3-241ceshi-shared/montyyin/results/reports/offline_20260713`

## MoE 弱扩展 MFU（Phase1 汇总）

| N | stamp | TFLOP/s/GPU (median) | MFU% |
|---|-------|----------------------|------|
| 32 | 20260712_181247 | 46.4 | 15.8 |
| 64 | 20260712_181247 | 40.55 | 13.8 |
| 96 | 20260712_221912 | 41.75 | 14.3 |

### Dense FailSlow 固定 GBS

| N | gap_med_ms | median_step_ms |
|---|------------|----------------|
| 16 | 6.572500000002037 | 119755.8475 |
| 32 | 3.963999999999942 | 57668.16525 |
| 64 | 2.700749999999971 | 31111.907000000003 |
| 96 | 2.2647500000020955 | 20139.4955 |

### Dense FailSlow GBS∝DP

| N | gap_med_ms | median_step_ms |
|---|------------|----------------|
| 32 | 1.9735000000000582 | 19270.74075 |
| 64 | 2.430499999998574 | 19327.9385 |
| 96 | 2.668999999999869 | 20837.698750000003 |

### MoE FailSlow (20260713_003312)

| N | gap_med_ms | median_step_ms |
|---|------------|----------------|
| 32 | 25.353750000002037 | 129206.833 |
| 64 |  |  |


## 实验三 lite：PP stage 注入 AB (Dense 16, TP4PP2 GBS=320)

Stamp: 20260713_101626

\`\`\`json
{
  "baseline": {
    "global_median_ms": 19412.1645,
    "stage_median_ms": {
      "0": 19411.361,
      "1": 19413.0605
    },
    "stage_spread_ms": 1.6994999999988067,
    "delayed_frac": 0.0,
    "n_ranks": 16
  },
  "inject": {
    "global_median_ms": 20876.190499999997,
    "stage_median_ms": {
      "0": 21087.2745,
      "1": 20648.4895
    },
    "stage_spread_ms": 438.78499999999985,
    "delayed_frac": 0.24193548387096775,
    "n_ranks": 16
  },
  "delta_global_median_ms": 1464.025999999998,
  "delta_stage_spread_ms": 437.08550000000105,
  "verdict": "WEAK: stage spread rose but modest"
}
\`\`\`

## 实验二 lite：npu-smi 采样

- baseline: /afs-a3-241ceshi-shared/montyyin/results/dense_pp_inject/20260713_101626/baseline/npu_smi_sample.log
- inject: /afs-a3-241ceshi-shared/montyyin/results/dense_pp_inject/20260713_101626/inject/npu_smi_sample.log

## 实验四 lite：NPU busy preempt (cards 14/15)
Stamp path: /afs-a3-241ceshi-shared/montyyin/results/exp45_parallel/20260713_104631/exp4_preempt

## 实验三 rematch：DELAY_RANKS=12-15

{
  "injected_ranks_median_ms": 19807.7915,
  "other_ranks_median_ms": 20190.338,
  "delta_ms": -382.5465000000004,
  "per_rank_median_ms": {
    "0": 20189.558,
    "1": 20189.797,
    "2": 20189.816,
    "3": 20189.785,
    "4": 20190.295,
    "5": 20190.382,
    "6": 20190.066,
    "7": 20190.381,
    "8": 20191.551,
    "9": 20191.644,
    "10": 20191.605,
    "11": 20191.668,
    "12": 19807.641,
    "13": 19807.454,
    "14": 19808.117,
    "15": 19807.942
  }
}
## Block A final network_contrib

world_npu,gap_real_ms,gap_indep_ms,network_contrib_ms,median_step_real_ms,median_step_indep_ms,cross_node
8,,0.2915000000000063,,,81.434,False
16,1.5562500000014552,0.534000000000006,1.0222500000014492,20177.701500000003,81.498,False
32,1.9735000000000582,0.5782500000000041,1.395250000000054,19270.74075,81.45475,True
64,2.430499999998574,1.1455000000000055,1.2849999999985684,19327.9385,81.44800000000001,True
96,2.668999999999869,1.264000000000003,1.4049999999998661,20837.698750000003,81.43924999999999,True

## 实验三 lite：PP stage 注入 AB (Dense 16, TP4PP2 GBS=320)

Stamp: 20260713_113423_ab

\`\`\`json
{
  "baseline": {
    "global_median_ms": 18791.8255,
    "stage_median_ms": {
      "0": 18791.023999999998,
      "1": 18792.886
    },
    "stage_spread_ms": 1.8620000000009895,
    "delayed_frac": 0.0,
    "n_ranks": 16
  },
  "inject": {
    "global_median_ms": 18836.606,
    "stage_median_ms": {
      "0": 18835.3795,
      "1": 18837.4905
    },
    "stage_spread_ms": 2.111000000000786,
    "delayed_frac": 0.24193548387096775,
    "n_ranks": 16
  },
  "delta_global_median_ms": 44.78050000000076,
  "delta_stage_spread_ms": 0.24899999999979627,
  "verdict": "WEAK: stage spread rose but modest"
}
\`\`\`

## 实验二 lite：npu-smi 采样

- baseline: /afs-a3-241ceshi-shared/montyyin/results/dense_pp_inject/20260713_113423_ab/baseline/npu_smi_sample.log
- inject: /afs-a3-241ceshi-shared/montyyin/results/dense_pp_inject/20260713_113423_ab/inject/npu_smi_sample.log

## Ascend 战役终稿 20260713

见 /afs-a3-241ceshi-shared/montyyin/results/reports/ascend_campaign_20260713/CAMPAIGN_FINAL.md

- Block A network_contrib: 16→32 抬升印证主假设
- Block E dual_signal: PASS
- Block D delayed-iter step_lift: PASS_SYNC_EFFECT (+1974ms)

## Block C full96
stamp: 20260713_134848_train
train: /afs-a3-241ceshi-shared/montyyin/results/blockC_full96/20260713_134848/train
smi: /afs-a3-241ceshi-shared/montyyin/results/blockC_full96/20260713_134848/npu_smi
world_npu,gap_median_ms,gap_mean_ms,gap_p90_ms,median_step_ms,p99_over_p50_median,tflops_median,mfu,n_iters,n_ranks,step_files
96,2.183499999999185,2.4918124999995825,3.500350000000981,21827.95025,1.000098415598964,108.8,0.37159739062126435,32,96,96
