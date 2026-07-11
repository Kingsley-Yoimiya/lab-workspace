# Card Constitution 分布报告

- 生成时间: 2026-07-11 14:11:24
- 卡数: 256
- 数据源: /Users/yinjinrun/random-thing/logs/card-fillgap-20260711_140301/results/constitution128.merged.jsonl, /Users/yinjinrun/random-thing/logs/card-fillgap-20260711_140301/results/master-0/master-0.whj4stu-copy-copy-copy-master-0.jsonl, /Users/yinjinrun/random-thing/logs/card-fillgap-20260711_140301/results/worker-0/worker-0.whj4stu-copy-copy-copy-worker-0.jsonl, /Users/yinjinrun/random-thing/logs/card-fillgap-20260711_140301/results/worker-1/worker-1.whj4stu-copy-copy-copy-worker-1.jsonl, /Users/yinjinrun/random-thing/logs/card-fillgap-20260711_140301/results/worker-2/worker-2.whj4stu-copy-copy-copy-worker-2.jsonl, /Users/yinjinrun/random-thing/logs/card-fillgap-20260711_140301/results/worker-3/worker-3.whj4stu-copy-copy-copy-worker-3.jsonl, /Users/yinjinrun/random-thing/logs/card-fillgap-20260711_140301/results/worker-4/worker-4.whj4stu-copy-copy-copy-worker-4.jsonl, /Users/yinjinrun/random-thing/logs/card-fillgap-20260711_140301/results/worker-5/worker-5.whj4stu-copy-copy-copy-worker-5.jsonl …

> 本报告只做分布统计与可视化，不强调 slow / 坏卡判定。

## 跳过说明

- `aicore_freq_mhz`（AICore freq (MHz)）：字段缺失或全空，跳过
- `hbm_temp_c`（HBM temp (C)）：字段缺失或全空，跳过
- `power_limit_w`（Power limit (W)）：字段缺失或全空，跳过
- `gemm_shape_sample`：无可用曲线，跳过 shape

## 指标分布

| 指标 | n | median | mean | std | CV% | min | max | p5 | p50 | p95 |
|------|---|--------|------|-----|-----|-----|-----|----|----|-----|
| Cube func TFLOPS | 256 | 292.4 | 291.5 | 5.536 | 1.899 | 273.3 | 302.8 | 281.4 | 292.4 | 299.4 |
| HBM GB/s | 256 | 1240.7 | 1214.7 | 52.77 | 4.344 | 1012.4 | 1269.0 | 1111.7 | 1240.7 | 1257.3 |
| Sustained TFLOPS | 256 | 306.9 | 306.2 | 4.247 | 1.387 | 294.8 | 313.7 | 299.0 | 306.9 | 312.0 |
| Vector GFLOPS | 256 | 98.82 | 98.83 | 0.3073 | 0.311 | 98.05 | 99.47 | 98.17 | 98.82 | 99.32 |
| Scalar elems/s | 256 | 279916705.0 | 279587799.1 | 2162177.7 | 0.7733 | 262411084.3 | 280053220.4 | 279744944.5 | 279916705.0 | 280024441.1 |
| MTE copy GB/s | 256 | 1267.9 | 1266.8 | 3.238 | 0.2556 | 1255.3 | 1271.1 | 1258.1 | 1267.9 | 1270.1 |
| Cube+Vector TFLOPS | 256 | 240.2 | 241.0 | 6.13 | 2.544 | 225.3 | 254.6 | 231.0 | 240.2 | 252.8 |
| SFU GFLOPS | 256 | 156.5 | 157.0 | 1.4 | 0.8917 | 152.3 | 159.2 | 155.0 | 156.5 | 159.0 |
| Launch sync p50 (us) | 256 | 6.069 | 6.238 | 0.745 | 11.94 | 5.009 | 9.481 | 5.409 | 6.069 | 8.24 |
| Launch sync p99 (us) | 256 | 6.78 | 8.02 | 2.96 | 36.9 | 5.81 | 26.28 | 6.109 | 6.78 | 11.46 |
| Host overhead p50 (us) | 256 | 240.1 | 244.2 | 16.45 | 6.738 | 216.2 | 307.8 | 223.4 | 240.1 | 272.8 |
| Host overhead p99 (us) | 256 | 628.7 | 640.7 | 53.49 | 8.35 | 567.5 | 898.4 | 579.5 | 628.7 | 731.2 |
| Burst total p50 (us) | 256 | 472.5 | 495.7 | 82.93 | 16.73 | 364.7 | 794.5 | 389.7 | 472.5 | 634.7 |
| Burst/kernel p50 (us) | 256 | 7.383 | 7.745 | 1.296 | 16.73 | 5.698 | 12.41 | 6.089 | 7.383 | 9.917 |
| Health temp (C) | 256 | 40 | 40.19 | 0.7881 | 1.961 | 38 | 43 | 39 | 40 | 41 |
| Health power (W) | 256 | 167.9 | 184.0 | 59.45 | 32.31 | 155.4 | 473.7 | 159.1 | 167.9 | 255.6 |
| Board temp (C) | 256 | 66 | 64.95 | 3.946 | 6.075 | 48 | 70 | 56 | 66 | 68 |
| AICore util % | 256 | 92 | 67.08 | 36.27 | 54.07 | 0 | 93 | 0 | 92 | 92 |
| AICPU util % | 256 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CtrlCPU util % | 256 | 7 | 7.438 | 3.225 | 43.36 | 1 | 18 | 3 | 7 | 14 |
| MemBW util % | 256 | 18 | 17.16 | 5.337 | 31.09 | 0 | 24 | 0 | 18 | 22 |
| Power (W) | 256 | 871.5 | 772.2 | 214.1 | 27.73 | 180.1 | 958.9 | 207.1 | 871.5 | 937.0 |
| Shape sweep peak TFLOPS | 256 | 310.7 | 311.1 | 1.101 | 0.354 | 310.1 | 315.5 | 310.4 | 310.7 | 313.6 |

## 相对中位数偏差

偏差 = `(值 - 集群中位数) / 集群中位数 × 100%`。

- **Cube func TFLOPS** (`func_tflops`): [-6.52%, +3.54%]，|偏差|均值 1.48%
- **HBM GB/s** (`hbm_gbps`): [-18.40%, +2.29%]，|偏差|均值 3.00%
- **Sustained TFLOPS** (`sustained_tflops`): [-3.94%, +2.22%]，|偏差|均值 1.16%
- **Vector GFLOPS** (`vector_gflops`): [-0.78%, +0.65%]，|偏差|均值 0.24%
- **Scalar elems/s** (`scalar_elems_per_s`): [-6.25%, +0.05%]，|偏差|均值 0.14%
- **MTE copy GB/s** (`mte_gbps`): [-1.00%, +0.25%]，|偏差|均值 0.16%
- **Cube+Vector TFLOPS** (`cube_vector_tflops`): [-6.21%, +5.98%]，|偏差|均值 1.96%
- **SFU GFLOPS** (`sfu_gflops`): [-2.66%, +1.74%]，|偏差|均值 0.73%
- **Launch sync p50 (us)** (`launch_sync_p50_us`): [-17.48%, +56.21%]，|偏差|均值 6.69%
- **Launch sync p99 (us)** (`launch_sync_p99_us`): [-14.31%, +287.61%]，|偏差|均值 22.63%
- **Host overhead p50 (us)** (`launch_host_overhead_p50_us`): [-9.95%, +28.18%]，|偏差|均值 5.47%
- **Host overhead p99 (us)** (`launch_host_overhead_p99_us`): [-9.73%, +42.90%]，|偏差|均值 5.66%
- **Burst total p50 (us)** (`launch_burst_p50_us`): [-22.83%, +68.14%]，|偏差|均值 13.46%
- **Burst/kernel p50 (us)** (`launch_burst_per_kernel_p50_us`): [-22.83%, +68.14%]，|偏差|均值 13.46%
- **Health temp (C)** (`health_temp_c`): [-5.00%, +7.50%]，|偏差|均值 1.25%
- **Health power (W)** (`health_power_w`): [-7.47%, +182.05%]，|偏差|均值 12.57%
- **Board temp (C)** (`board_temp_c`): [-27.27%, +6.06%]，|偏差|均值 3.71%
- **AICore util %** (`aicore_util_pct`): [-100.00%, +1.09%]，|偏差|均值 27.16%
- **AICPU util %** (`aicpu_util_pct`): [+0.00%, +0.00%]，|偏差|均值 0.00%
- **CtrlCPU util %** (`ctrlcpu_util_pct`): [-85.71%, +157.14%]，|偏差|均值 35.71%
- **MemBW util %** (`mem_bw_util_pct`): [-100.00%, +33.33%]，|偏差|均值 14.37%
- **Power (W)** (`power_w`): [-79.34%, +10.02%]，|偏差|均值 15.74%
- **Shape sweep peak TFLOPS** (`shape_sweep_peak_tflops`): [-0.17%, +1.55%]，|偏差|均值 0.21%

## 元数据

- hosts (8): whj4stu-copy-copy-copy-master-0, whj4stu-copy-copy-copy-worker-0, whj4stu-copy-copy-copy-worker-1, whj4stu-copy-copy-copy-worker-2, whj4stu-copy-copy-copy-worker-3, whj4stu-copy-copy-copy-worker-4, whj4stu-copy-copy-copy-worker-5, whj4stu-copy-copy-copy-worker-6
- backends: npu
- launch_timing_method: event

## 图表

### box overview

![box overview](card_constitution_20260711_141113_figs/box_overview.png)

### hist func tflops

![hist func tflops](card_constitution_20260711_141113_figs/hist_func_tflops.png)

### hist hbm gbps

![hist hbm gbps](card_constitution_20260711_141113_figs/hist_hbm_gbps.png)

### hist sustained tflops

![hist sustained tflops](card_constitution_20260711_141113_figs/hist_sustained_tflops.png)

### hist vector gflops

![hist vector gflops](card_constitution_20260711_141113_figs/hist_vector_gflops.png)

### hist scalar elems per s

![hist scalar elems per s](card_constitution_20260711_141113_figs/hist_scalar_elems_per_s.png)

### hist mte gbps

![hist mte gbps](card_constitution_20260711_141113_figs/hist_mte_gbps.png)

### hist cube vector tflops

![hist cube vector tflops](card_constitution_20260711_141113_figs/hist_cube_vector_tflops.png)

### hist sfu gflops

![hist sfu gflops](card_constitution_20260711_141113_figs/hist_sfu_gflops.png)

### hist launch sync p50 us

![hist launch sync p50 us](card_constitution_20260711_141113_figs/hist_launch_sync_p50_us.png)

### hist launch sync p99 us

![hist launch sync p99 us](card_constitution_20260711_141113_figs/hist_launch_sync_p99_us.png)

### hist launch host overhead p50 us

![hist launch host overhead p50 us](card_constitution_20260711_141113_figs/hist_launch_host_overhead_p50_us.png)

### hist launch host overhead p99 us

![hist launch host overhead p99 us](card_constitution_20260711_141113_figs/hist_launch_host_overhead_p99_us.png)

### hist launch burst p50 us

![hist launch burst p50 us](card_constitution_20260711_141113_figs/hist_launch_burst_p50_us.png)

### hist launch burst per kernel p50 us

![hist launch burst per kernel p50 us](card_constitution_20260711_141113_figs/hist_launch_burst_per_kernel_p50_us.png)

### hist health temp c

![hist health temp c](card_constitution_20260711_141113_figs/hist_health_temp_c.png)

### hist health power w

![hist health power w](card_constitution_20260711_141113_figs/hist_health_power_w.png)

### hist board temp c

![hist board temp c](card_constitution_20260711_141113_figs/hist_board_temp_c.png)

### hist aicore util pct

![hist aicore util pct](card_constitution_20260711_141113_figs/hist_aicore_util_pct.png)

### hist aicpu util pct

![hist aicpu util pct](card_constitution_20260711_141113_figs/hist_aicpu_util_pct.png)

### hist ctrlcpu util pct

![hist ctrlcpu util pct](card_constitution_20260711_141113_figs/hist_ctrlcpu_util_pct.png)

### hist mem bw util pct

![hist mem bw util pct](card_constitution_20260711_141113_figs/hist_mem_bw_util_pct.png)

### hist power w

![hist power w](card_constitution_20260711_141113_figs/hist_power_w.png)

### hist shape sweep peak tflops

![hist shape sweep peak tflops](card_constitution_20260711_141113_figs/hist_shape_sweep_peak_tflops.png)

### heatmap relmed func tflops

![heatmap relmed func tflops](card_constitution_20260711_141113_figs/heatmap_relmed_func_tflops.png)

### box by host func tflops

![box by host func tflops](card_constitution_20260711_141113_figs/box_by_host_func_tflops.png)

### sorted bar func tflops

![sorted bar func tflops](card_constitution_20260711_141113_figs/sorted_bar_func_tflops.png)

### bar host mean std func tflops

![bar host mean std func tflops](card_constitution_20260711_141113_figs/bar_host_mean_std_func_tflops.png)

### heatmap relmed hbm gbps

![heatmap relmed hbm gbps](card_constitution_20260711_141113_figs/heatmap_relmed_hbm_gbps.png)

### box by host hbm gbps

![box by host hbm gbps](card_constitution_20260711_141113_figs/box_by_host_hbm_gbps.png)

### sorted bar hbm gbps

![sorted bar hbm gbps](card_constitution_20260711_141113_figs/sorted_bar_hbm_gbps.png)

### bar host mean std hbm gbps

![bar host mean std hbm gbps](card_constitution_20260711_141113_figs/bar_host_mean_std_hbm_gbps.png)

### heatmap relmed sustained tflops

![heatmap relmed sustained tflops](card_constitution_20260711_141113_figs/heatmap_relmed_sustained_tflops.png)

### box by host sustained tflops

![box by host sustained tflops](card_constitution_20260711_141113_figs/box_by_host_sustained_tflops.png)

### sorted bar sustained tflops

![sorted bar sustained tflops](card_constitution_20260711_141113_figs/sorted_bar_sustained_tflops.png)

### bar host mean std sustained tflops

![bar host mean std sustained tflops](card_constitution_20260711_141113_figs/bar_host_mean_std_sustained_tflops.png)

### heatmap relmed vector gflops

![heatmap relmed vector gflops](card_constitution_20260711_141113_figs/heatmap_relmed_vector_gflops.png)

### box by host vector gflops

![box by host vector gflops](card_constitution_20260711_141113_figs/box_by_host_vector_gflops.png)

### sorted bar vector gflops

![sorted bar vector gflops](card_constitution_20260711_141113_figs/sorted_bar_vector_gflops.png)

### bar host mean std vector gflops

![bar host mean std vector gflops](card_constitution_20260711_141113_figs/bar_host_mean_std_vector_gflops.png)

### heatmap relmed scalar elems per s

![heatmap relmed scalar elems per s](card_constitution_20260711_141113_figs/heatmap_relmed_scalar_elems_per_s.png)

### box by host scalar elems per s

![box by host scalar elems per s](card_constitution_20260711_141113_figs/box_by_host_scalar_elems_per_s.png)

### sorted bar scalar elems per s

![sorted bar scalar elems per s](card_constitution_20260711_141113_figs/sorted_bar_scalar_elems_per_s.png)

### bar host mean std scalar elems per s

![bar host mean std scalar elems per s](card_constitution_20260711_141113_figs/bar_host_mean_std_scalar_elems_per_s.png)

### heatmap relmed mte gbps

![heatmap relmed mte gbps](card_constitution_20260711_141113_figs/heatmap_relmed_mte_gbps.png)

### box by host mte gbps

![box by host mte gbps](card_constitution_20260711_141113_figs/box_by_host_mte_gbps.png)

### sorted bar mte gbps

![sorted bar mte gbps](card_constitution_20260711_141113_figs/sorted_bar_mte_gbps.png)

### bar host mean std mte gbps

![bar host mean std mte gbps](card_constitution_20260711_141113_figs/bar_host_mean_std_mte_gbps.png)

### heatmap relmed cube vector tflops

![heatmap relmed cube vector tflops](card_constitution_20260711_141113_figs/heatmap_relmed_cube_vector_tflops.png)

### box by host cube vector tflops

![box by host cube vector tflops](card_constitution_20260711_141113_figs/box_by_host_cube_vector_tflops.png)

### sorted bar cube vector tflops

![sorted bar cube vector tflops](card_constitution_20260711_141113_figs/sorted_bar_cube_vector_tflops.png)

### bar host mean std cube vector tflops

![bar host mean std cube vector tflops](card_constitution_20260711_141113_figs/bar_host_mean_std_cube_vector_tflops.png)

### heatmap relmed sfu gflops

![heatmap relmed sfu gflops](card_constitution_20260711_141113_figs/heatmap_relmed_sfu_gflops.png)

### box by host sfu gflops

![box by host sfu gflops](card_constitution_20260711_141113_figs/box_by_host_sfu_gflops.png)

### sorted bar sfu gflops

![sorted bar sfu gflops](card_constitution_20260711_141113_figs/sorted_bar_sfu_gflops.png)

### bar host mean std sfu gflops

![bar host mean std sfu gflops](card_constitution_20260711_141113_figs/bar_host_mean_std_sfu_gflops.png)

### heatmap relmed launch sync p50 us

![heatmap relmed launch sync p50 us](card_constitution_20260711_141113_figs/heatmap_relmed_launch_sync_p50_us.png)

### box by host launch sync p50 us

![box by host launch sync p50 us](card_constitution_20260711_141113_figs/box_by_host_launch_sync_p50_us.png)

### sorted bar launch sync p50 us

![sorted bar launch sync p50 us](card_constitution_20260711_141113_figs/sorted_bar_launch_sync_p50_us.png)

### heatmap relmed launch sync p99 us

![heatmap relmed launch sync p99 us](card_constitution_20260711_141113_figs/heatmap_relmed_launch_sync_p99_us.png)

### box by host launch sync p99 us

![box by host launch sync p99 us](card_constitution_20260711_141113_figs/box_by_host_launch_sync_p99_us.png)

### sorted bar launch sync p99 us

![sorted bar launch sync p99 us](card_constitution_20260711_141113_figs/sorted_bar_launch_sync_p99_us.png)

### heatmap relmed launch host overhead p50 us

![heatmap relmed launch host overhead p50 us](card_constitution_20260711_141113_figs/heatmap_relmed_launch_host_overhead_p50_us.png)

### box by host launch host overhead p50 us

![box by host launch host overhead p50 us](card_constitution_20260711_141113_figs/box_by_host_launch_host_overhead_p50_us.png)

### sorted bar launch host overhead p50 us

![sorted bar launch host overhead p50 us](card_constitution_20260711_141113_figs/sorted_bar_launch_host_overhead_p50_us.png)

### heatmap relmed launch host overhead p99 us

![heatmap relmed launch host overhead p99 us](card_constitution_20260711_141113_figs/heatmap_relmed_launch_host_overhead_p99_us.png)

### box by host launch host overhead p99 us

![box by host launch host overhead p99 us](card_constitution_20260711_141113_figs/box_by_host_launch_host_overhead_p99_us.png)

### sorted bar launch host overhead p99 us

![sorted bar launch host overhead p99 us](card_constitution_20260711_141113_figs/sorted_bar_launch_host_overhead_p99_us.png)

### heatmap relmed launch burst p50 us

![heatmap relmed launch burst p50 us](card_constitution_20260711_141113_figs/heatmap_relmed_launch_burst_p50_us.png)

### box by host launch burst p50 us

![box by host launch burst p50 us](card_constitution_20260711_141113_figs/box_by_host_launch_burst_p50_us.png)

### sorted bar launch burst p50 us

![sorted bar launch burst p50 us](card_constitution_20260711_141113_figs/sorted_bar_launch_burst_p50_us.png)

### heatmap relmed launch burst per kernel p50 us

![heatmap relmed launch burst per kernel p50 us](card_constitution_20260711_141113_figs/heatmap_relmed_launch_burst_per_kernel_p50_us.png)

### box by host launch burst per kernel p50 us

![box by host launch burst per kernel p50 us](card_constitution_20260711_141113_figs/box_by_host_launch_burst_per_kernel_p50_us.png)

### sorted bar launch burst per kernel p50 us

![sorted bar launch burst per kernel p50 us](card_constitution_20260711_141113_figs/sorted_bar_launch_burst_per_kernel_p50_us.png)

### heatmap relmed health temp c

![heatmap relmed health temp c](card_constitution_20260711_141113_figs/heatmap_relmed_health_temp_c.png)

### box by host health temp c

![box by host health temp c](card_constitution_20260711_141113_figs/box_by_host_health_temp_c.png)

### sorted bar health temp c

![sorted bar health temp c](card_constitution_20260711_141113_figs/sorted_bar_health_temp_c.png)

### bar host mean std health temp c

![bar host mean std health temp c](card_constitution_20260711_141113_figs/bar_host_mean_std_health_temp_c.png)

### heatmap relmed health power w

![heatmap relmed health power w](card_constitution_20260711_141113_figs/heatmap_relmed_health_power_w.png)

### box by host health power w

![box by host health power w](card_constitution_20260711_141113_figs/box_by_host_health_power_w.png)

### sorted bar health power w

![sorted bar health power w](card_constitution_20260711_141113_figs/sorted_bar_health_power_w.png)

### bar host mean std health power w

![bar host mean std health power w](card_constitution_20260711_141113_figs/bar_host_mean_std_health_power_w.png)

### heatmap relmed board temp c

![heatmap relmed board temp c](card_constitution_20260711_141113_figs/heatmap_relmed_board_temp_c.png)

### box by host board temp c

![box by host board temp c](card_constitution_20260711_141113_figs/box_by_host_board_temp_c.png)

### sorted bar board temp c

![sorted bar board temp c](card_constitution_20260711_141113_figs/sorted_bar_board_temp_c.png)

### heatmap relmed aicore util pct

![heatmap relmed aicore util pct](card_constitution_20260711_141113_figs/heatmap_relmed_aicore_util_pct.png)

### box by host aicore util pct

![box by host aicore util pct](card_constitution_20260711_141113_figs/box_by_host_aicore_util_pct.png)

### sorted bar aicore util pct

![sorted bar aicore util pct](card_constitution_20260711_141113_figs/sorted_bar_aicore_util_pct.png)

### heatmap relmed aicpu util pct

![heatmap relmed aicpu util pct](card_constitution_20260711_141113_figs/heatmap_relmed_aicpu_util_pct.png)

### box by host aicpu util pct

![box by host aicpu util pct](card_constitution_20260711_141113_figs/box_by_host_aicpu_util_pct.png)

### sorted bar aicpu util pct

![sorted bar aicpu util pct](card_constitution_20260711_141113_figs/sorted_bar_aicpu_util_pct.png)

### heatmap relmed ctrlcpu util pct

![heatmap relmed ctrlcpu util pct](card_constitution_20260711_141113_figs/heatmap_relmed_ctrlcpu_util_pct.png)

### box by host ctrlcpu util pct

![box by host ctrlcpu util pct](card_constitution_20260711_141113_figs/box_by_host_ctrlcpu_util_pct.png)

### sorted bar ctrlcpu util pct

![sorted bar ctrlcpu util pct](card_constitution_20260711_141113_figs/sorted_bar_ctrlcpu_util_pct.png)

### heatmap relmed mem bw util pct

![heatmap relmed mem bw util pct](card_constitution_20260711_141113_figs/heatmap_relmed_mem_bw_util_pct.png)

### box by host mem bw util pct

![box by host mem bw util pct](card_constitution_20260711_141113_figs/box_by_host_mem_bw_util_pct.png)

### sorted bar mem bw util pct

![sorted bar mem bw util pct](card_constitution_20260711_141113_figs/sorted_bar_mem_bw_util_pct.png)

### heatmap relmed power w

![heatmap relmed power w](card_constitution_20260711_141113_figs/heatmap_relmed_power_w.png)

### box by host power w

![box by host power w](card_constitution_20260711_141113_figs/box_by_host_power_w.png)

### sorted bar power w

![sorted bar power w](card_constitution_20260711_141113_figs/sorted_bar_power_w.png)

### bar host mean std power w

![bar host mean std power w](card_constitution_20260711_141113_figs/bar_host_mean_std_power_w.png)

### heatmap relmed shape sweep peak tflops

![heatmap relmed shape sweep peak tflops](card_constitution_20260711_141113_figs/heatmap_relmed_shape_sweep_peak_tflops.png)

### box by host shape sweep peak tflops

![box by host shape sweep peak tflops](card_constitution_20260711_141113_figs/box_by_host_shape_sweep_peak_tflops.png)

### sorted bar shape sweep peak tflops

![sorted bar shape sweep peak tflops](card_constitution_20260711_141113_figs/sorted_bar_shape_sweep_peak_tflops.png)

### scatter func tflops vs vector gflops

![scatter func tflops vs vector gflops](card_constitution_20260711_141113_figs/scatter_func_tflops_vs_vector_gflops.png)

### scatter hbm gbps vs mte gbps

![scatter hbm gbps vs mte gbps](card_constitution_20260711_141113_figs/scatter_hbm_gbps_vs_mte_gbps.png)

### scatter power w vs func tflops

![scatter power w vs func tflops](card_constitution_20260711_141113_figs/scatter_power_w_vs_func_tflops.png)

### scatter health power w vs func tflops

![scatter health power w vs func tflops](card_constitution_20260711_141113_figs/scatter_health_power_w_vs_func_tflops.png)

### scatter power w vs hbm gbps

![scatter power w vs hbm gbps](card_constitution_20260711_141113_figs/scatter_power_w_vs_hbm_gbps.png)

### scatter health power w vs hbm gbps

![scatter health power w vs hbm gbps](card_constitution_20260711_141113_figs/scatter_health_power_w_vs_hbm_gbps.png)

### scatter launch host overhead p50 us vs ctrlcpu util pct

![scatter launch host overhead p50 us vs ctrlcpu util pct](card_constitution_20260711_141113_figs/scatter_launch_host_overhead_p50_us_vs_ctrlcpu_util_pct.png)

### timeseries sustained p05 p50

![timeseries sustained p05 p50](card_constitution_20260711_141113_figs/timeseries_sustained_p05_p50.png)

