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
