# Card Constitution 分布报告

- 生成时间: 2026-07-11 13:46:46
- 卡数: 256
- 数据源: /Users/yinjinrun/random-thing/logs/card-constitution-128-20260711_134309-constitution128/results/constitution128.merged.jsonl, /Users/yinjinrun/random-thing/logs/card-constitution-128-20260711_134309-constitution128/results/master-0/master-0.whj4stu-copy-copy-copy-master-0.jsonl, /Users/yinjinrun/random-thing/logs/card-constitution-128-20260711_134309-constitution128/results/worker-0/worker-0.whj4stu-copy-copy-copy-worker-0.jsonl, /Users/yinjinrun/random-thing/logs/card-constitution-128-20260711_134309-constitution128/results/worker-1/worker-1.whj4stu-copy-copy-copy-worker-1.jsonl, /Users/yinjinrun/random-thing/logs/card-constitution-128-20260711_134309-constitution128/results/worker-2/worker-2.whj4stu-copy-copy-copy-worker-2.jsonl, /Users/yinjinrun/random-thing/logs/card-constitution-128-20260711_134309-constitution128/results/worker-3/worker-3.whj4stu-copy-copy-copy-worker-3.jsonl, /Users/yinjinrun/random-thing/logs/card-constitution-128-20260711_134309-constitution128/results/worker-4/worker-4.whj4stu-copy-copy-copy-worker-4.jsonl, /Users/yinjinrun/random-thing/logs/card-constitution-128-20260711_134309-constitution128/results/worker-5/worker-5.whj4stu-copy-copy-copy-worker-5.jsonl …

> 本报告只做分布统计与可视化，不强调 slow / 坏卡判定。

## 跳过说明

- `health_temp_c`（Health temp (C)）：字段缺失或全空，跳过
- `health_power_w`（Health power (W)）：字段缺失或全空，跳过
- `aicore_freq_mhz`（AICore freq (MHz)）：字段缺失或全空，跳过
- `hbm_temp_c`（HBM temp (C)）：字段缺失或全空，跳过
- `board_temp_c`（Board temp (C)）：字段缺失或全空，跳过
- `aicore_util_pct`（AICore util %）：字段缺失或全空，跳过
- `aicpu_util_pct`（AICPU util %）：字段缺失或全空，跳过
- `ctrlcpu_util_pct`（CtrlCPU util %）：字段缺失或全空，跳过
- `mem_bw_util_pct`（MemBW util %）：字段缺失或全空，跳过
- `power_w`（Power (W)）：字段缺失或全空，跳过
- `power_limit_w`（Power limit (W)）：字段缺失或全空，跳过
- `shape_sweep_peak_tflops`（Shape sweep peak TFLOPS）：字段缺失或全空，跳过
- 散点 `power_w` × `func_tflops`（Power × Cube）：缺轴字段，跳过
- 散点 `health_power_w` × `func_tflops`（Health power × Cube）：缺轴字段，跳过
- 散点 `power_w` × `hbm_gbps`（Power × HBM）：缺轴字段，跳过
- 散点 `health_power_w` × `hbm_gbps`（Health power × HBM）：缺轴字段，跳过
- 散点 `launch_host_overhead_p50_us` × `ctrlcpu_util_pct`（Launch overhead × CtrlCPU）：缺轴字段，跳过
- `gemm_shape_sample`：无可用曲线，跳过 shape

## 指标分布

| 指标 | n | median | mean | std | CV% | min | max | p5 | p50 | p95 |
|------|---|--------|------|-----|-----|-----|-----|----|----|-----|
| Cube func TFLOPS | 256 | 291.7 | 290.6 | 6.523 | 2.245 | 268.5 | 302.7 | 279.7 | 291.7 | 299.3 |
| HBM GB/s | 256 | 1241.7 | 1220.5 | 48.45 | 3.969 | 1051.4 | 1264.0 | 1096.4 | 1241.7 | 1256.7 |
| Sustained TFLOPS | 256 | 306.4 | 305.9 | 4.301 | 1.406 | 294.7 | 314.2 | 298.2 | 306.4 | 311.4 |
| Vector GFLOPS | 256 | 98.81 | 98.82 | 0.3602 | 0.3645 | 97.81 | 99.45 | 98.14 | 98.81 | 99.33 |
| Scalar elems/s | 256 | 279923564.2 | 279785523.3 | 1540601.3 | 0.5506 | 262441712.7 | 280053505.8 | 279811010.0 | 279923564.2 | 280034818.1 |
| MTE copy GB/s | 256 | 1267.4 | 1266.1 | 3.837 | 0.303 | 1255.4 | 1270.8 | 1257.3 | 1267.4 | 1270.3 |
| Cube+Vector TFLOPS | 256 | 239.5 | 240.2 | 5.851 | 2.436 | 227.3 | 254.6 | 229.9 | 239.5 | 250.4 |
| SFU GFLOPS | 256 | 156.6 | 156.9 | 1.331 | 0.8479 | 153.4 | 159.0 | 154.5 | 156.6 | 158.8 |
| Launch sync p50 (us) | 256 | 6.08 | 6.315 | 0.8188 | 12.97 | 5.331 | 10.24 | 5.54 | 6.08 | 8.339 |
| Launch sync p99 (us) | 256 | 6.971 | 8.603 | 4.51 | 52.43 | 5.852 | 35.13 | 6.279 | 6.971 | 15.13 |
| Host overhead p50 (us) | 256 | 241.5 | 245.6 | 15.54 | 6.329 | 215.3 | 292.7 | 226.0 | 241.5 | 273.8 |
| Host overhead p99 (us) | 256 | 630.3 | 640.4 | 46.44 | 7.251 | 558.7 | 905.0 | 588.5 | 630.3 | 728.6 |
| Burst total p50 (us) | 256 | 467.6 | 489.4 | 81.26 | 16.6 | 367.3 | 674.3 | 389.0 | 467.6 | 644.7 |
| Burst/kernel p50 (us) | 256 | 7.306 | 7.647 | 1.27 | 16.6 | 5.739 | 10.54 | 6.078 | 7.306 | 10.07 |

## 相对中位数偏差

偏差 = `(值 - 集群中位数) / 集群中位数 × 100%`。

- **Cube func TFLOPS** (`func_tflops`): [-7.93%, +3.77%]，|偏差|均值 1.84%
- **HBM GB/s** (`hbm_gbps`): [-15.33%, +1.80%]，|偏差|均值 2.52%
- **Sustained TFLOPS** (`sustained_tflops`): [-3.82%, +2.56%]，|偏差|均值 1.13%
- **Vector GFLOPS** (`vector_gflops`): [-1.01%, +0.65%]，|偏差|均值 0.30%
- **Scalar elems/s** (`scalar_elems_per_s`): [-6.25%, +0.05%]，|偏差|均值 0.07%
- **MTE copy GB/s** (`mte_gbps`): [-0.94%, +0.27%]，|偏差|均值 0.21%
- **Cube+Vector TFLOPS** (`cube_vector_tflops`): [-5.09%, +6.32%]，|偏差|均值 1.88%
- **SFU GFLOPS** (`sfu_gflops`): [-2.04%, +1.56%]，|偏差|均值 0.69%
- **Launch sync p50 (us)** (`launch_sync_p50_us`): [-12.32%, +68.41%]，|偏差|均值 6.84%
- **Launch sync p99 (us)** (`launch_sync_p99_us`): [-16.05%, +403.97%]，|偏差|均值 29.60%
- **Host overhead p50 (us)** (`launch_host_overhead_p50_us`): [-10.85%, +21.16%]，|偏差|均值 5.07%
- **Host overhead p99 (us)** (`launch_host_overhead_p99_us`): [-11.36%, +43.58%]，|偏差|均值 5.15%
- **Burst total p50 (us)** (`launch_burst_p50_us`): [-21.45%, +44.22%]，|偏差|均值 14.28%
- **Burst/kernel p50 (us)** (`launch_burst_per_kernel_p50_us`): [-21.45%, +44.22%]，|偏差|均值 14.28%

## 元数据

- hosts (8): whj4stu-copy-copy-copy-master-0, whj4stu-copy-copy-copy-worker-0, whj4stu-copy-copy-copy-worker-1, whj4stu-copy-copy-copy-worker-2, whj4stu-copy-copy-copy-worker-3, whj4stu-copy-copy-copy-worker-4, whj4stu-copy-copy-copy-worker-5, whj4stu-copy-copy-copy-worker-6
- backends: npu
- launch_timing_method: event

## 图表

### box overview

![box overview](card_constitution_20260711_134639_figs/box_overview.png)

### hist func tflops

![hist func tflops](card_constitution_20260711_134639_figs/hist_func_tflops.png)

### hist hbm gbps

![hist hbm gbps](card_constitution_20260711_134639_figs/hist_hbm_gbps.png)

### hist sustained tflops

![hist sustained tflops](card_constitution_20260711_134639_figs/hist_sustained_tflops.png)

### hist vector gflops

![hist vector gflops](card_constitution_20260711_134639_figs/hist_vector_gflops.png)

### hist scalar elems per s

![hist scalar elems per s](card_constitution_20260711_134639_figs/hist_scalar_elems_per_s.png)

### hist mte gbps

![hist mte gbps](card_constitution_20260711_134639_figs/hist_mte_gbps.png)

### hist cube vector tflops

![hist cube vector tflops](card_constitution_20260711_134639_figs/hist_cube_vector_tflops.png)

### hist sfu gflops

![hist sfu gflops](card_constitution_20260711_134639_figs/hist_sfu_gflops.png)

### hist launch sync p50 us

![hist launch sync p50 us](card_constitution_20260711_134639_figs/hist_launch_sync_p50_us.png)

### hist launch sync p99 us

![hist launch sync p99 us](card_constitution_20260711_134639_figs/hist_launch_sync_p99_us.png)

### hist launch host overhead p50 us

![hist launch host overhead p50 us](card_constitution_20260711_134639_figs/hist_launch_host_overhead_p50_us.png)

### hist launch host overhead p99 us

![hist launch host overhead p99 us](card_constitution_20260711_134639_figs/hist_launch_host_overhead_p99_us.png)

### hist launch burst p50 us

![hist launch burst p50 us](card_constitution_20260711_134639_figs/hist_launch_burst_p50_us.png)

### hist launch burst per kernel p50 us

![hist launch burst per kernel p50 us](card_constitution_20260711_134639_figs/hist_launch_burst_per_kernel_p50_us.png)

### heatmap relmed func tflops

![heatmap relmed func tflops](card_constitution_20260711_134639_figs/heatmap_relmed_func_tflops.png)

### box by host func tflops

![box by host func tflops](card_constitution_20260711_134639_figs/box_by_host_func_tflops.png)

### sorted bar func tflops

![sorted bar func tflops](card_constitution_20260711_134639_figs/sorted_bar_func_tflops.png)

### bar host mean std func tflops

![bar host mean std func tflops](card_constitution_20260711_134639_figs/bar_host_mean_std_func_tflops.png)

### heatmap relmed hbm gbps

![heatmap relmed hbm gbps](card_constitution_20260711_134639_figs/heatmap_relmed_hbm_gbps.png)

### box by host hbm gbps

![box by host hbm gbps](card_constitution_20260711_134639_figs/box_by_host_hbm_gbps.png)

### sorted bar hbm gbps

![sorted bar hbm gbps](card_constitution_20260711_134639_figs/sorted_bar_hbm_gbps.png)

### bar host mean std hbm gbps

![bar host mean std hbm gbps](card_constitution_20260711_134639_figs/bar_host_mean_std_hbm_gbps.png)

### heatmap relmed sustained tflops

![heatmap relmed sustained tflops](card_constitution_20260711_134639_figs/heatmap_relmed_sustained_tflops.png)

### box by host sustained tflops

![box by host sustained tflops](card_constitution_20260711_134639_figs/box_by_host_sustained_tflops.png)

### sorted bar sustained tflops

![sorted bar sustained tflops](card_constitution_20260711_134639_figs/sorted_bar_sustained_tflops.png)

### bar host mean std sustained tflops

![bar host mean std sustained tflops](card_constitution_20260711_134639_figs/bar_host_mean_std_sustained_tflops.png)

### heatmap relmed vector gflops

![heatmap relmed vector gflops](card_constitution_20260711_134639_figs/heatmap_relmed_vector_gflops.png)

### box by host vector gflops

![box by host vector gflops](card_constitution_20260711_134639_figs/box_by_host_vector_gflops.png)

### sorted bar vector gflops

![sorted bar vector gflops](card_constitution_20260711_134639_figs/sorted_bar_vector_gflops.png)

### bar host mean std vector gflops

![bar host mean std vector gflops](card_constitution_20260711_134639_figs/bar_host_mean_std_vector_gflops.png)

### heatmap relmed scalar elems per s

![heatmap relmed scalar elems per s](card_constitution_20260711_134639_figs/heatmap_relmed_scalar_elems_per_s.png)

### box by host scalar elems per s

![box by host scalar elems per s](card_constitution_20260711_134639_figs/box_by_host_scalar_elems_per_s.png)

### sorted bar scalar elems per s

![sorted bar scalar elems per s](card_constitution_20260711_134639_figs/sorted_bar_scalar_elems_per_s.png)

### bar host mean std scalar elems per s

![bar host mean std scalar elems per s](card_constitution_20260711_134639_figs/bar_host_mean_std_scalar_elems_per_s.png)

### heatmap relmed mte gbps

![heatmap relmed mte gbps](card_constitution_20260711_134639_figs/heatmap_relmed_mte_gbps.png)

### box by host mte gbps

![box by host mte gbps](card_constitution_20260711_134639_figs/box_by_host_mte_gbps.png)

### sorted bar mte gbps

![sorted bar mte gbps](card_constitution_20260711_134639_figs/sorted_bar_mte_gbps.png)

### bar host mean std mte gbps

![bar host mean std mte gbps](card_constitution_20260711_134639_figs/bar_host_mean_std_mte_gbps.png)

### heatmap relmed cube vector tflops

![heatmap relmed cube vector tflops](card_constitution_20260711_134639_figs/heatmap_relmed_cube_vector_tflops.png)

### box by host cube vector tflops

![box by host cube vector tflops](card_constitution_20260711_134639_figs/box_by_host_cube_vector_tflops.png)

### sorted bar cube vector tflops

![sorted bar cube vector tflops](card_constitution_20260711_134639_figs/sorted_bar_cube_vector_tflops.png)

### bar host mean std cube vector tflops

![bar host mean std cube vector tflops](card_constitution_20260711_134639_figs/bar_host_mean_std_cube_vector_tflops.png)

### heatmap relmed sfu gflops

![heatmap relmed sfu gflops](card_constitution_20260711_134639_figs/heatmap_relmed_sfu_gflops.png)

### box by host sfu gflops

![box by host sfu gflops](card_constitution_20260711_134639_figs/box_by_host_sfu_gflops.png)

### sorted bar sfu gflops

![sorted bar sfu gflops](card_constitution_20260711_134639_figs/sorted_bar_sfu_gflops.png)

### bar host mean std sfu gflops

![bar host mean std sfu gflops](card_constitution_20260711_134639_figs/bar_host_mean_std_sfu_gflops.png)

### heatmap relmed launch sync p50 us

![heatmap relmed launch sync p50 us](card_constitution_20260711_134639_figs/heatmap_relmed_launch_sync_p50_us.png)

### box by host launch sync p50 us

![box by host launch sync p50 us](card_constitution_20260711_134639_figs/box_by_host_launch_sync_p50_us.png)

### sorted bar launch sync p50 us

![sorted bar launch sync p50 us](card_constitution_20260711_134639_figs/sorted_bar_launch_sync_p50_us.png)

### heatmap relmed launch sync p99 us

![heatmap relmed launch sync p99 us](card_constitution_20260711_134639_figs/heatmap_relmed_launch_sync_p99_us.png)

### box by host launch sync p99 us

![box by host launch sync p99 us](card_constitution_20260711_134639_figs/box_by_host_launch_sync_p99_us.png)

### sorted bar launch sync p99 us

![sorted bar launch sync p99 us](card_constitution_20260711_134639_figs/sorted_bar_launch_sync_p99_us.png)

### heatmap relmed launch host overhead p50 us

![heatmap relmed launch host overhead p50 us](card_constitution_20260711_134639_figs/heatmap_relmed_launch_host_overhead_p50_us.png)

### box by host launch host overhead p50 us

![box by host launch host overhead p50 us](card_constitution_20260711_134639_figs/box_by_host_launch_host_overhead_p50_us.png)

### sorted bar launch host overhead p50 us

![sorted bar launch host overhead p50 us](card_constitution_20260711_134639_figs/sorted_bar_launch_host_overhead_p50_us.png)

### heatmap relmed launch host overhead p99 us

![heatmap relmed launch host overhead p99 us](card_constitution_20260711_134639_figs/heatmap_relmed_launch_host_overhead_p99_us.png)

### box by host launch host overhead p99 us

![box by host launch host overhead p99 us](card_constitution_20260711_134639_figs/box_by_host_launch_host_overhead_p99_us.png)

### sorted bar launch host overhead p99 us

![sorted bar launch host overhead p99 us](card_constitution_20260711_134639_figs/sorted_bar_launch_host_overhead_p99_us.png)

### heatmap relmed launch burst p50 us

![heatmap relmed launch burst p50 us](card_constitution_20260711_134639_figs/heatmap_relmed_launch_burst_p50_us.png)

### box by host launch burst p50 us

![box by host launch burst p50 us](card_constitution_20260711_134639_figs/box_by_host_launch_burst_p50_us.png)

### sorted bar launch burst p50 us

![sorted bar launch burst p50 us](card_constitution_20260711_134639_figs/sorted_bar_launch_burst_p50_us.png)

### heatmap relmed launch burst per kernel p50 us

![heatmap relmed launch burst per kernel p50 us](card_constitution_20260711_134639_figs/heatmap_relmed_launch_burst_per_kernel_p50_us.png)

### box by host launch burst per kernel p50 us

![box by host launch burst per kernel p50 us](card_constitution_20260711_134639_figs/box_by_host_launch_burst_per_kernel_p50_us.png)

### sorted bar launch burst per kernel p50 us

![sorted bar launch burst per kernel p50 us](card_constitution_20260711_134639_figs/sorted_bar_launch_burst_per_kernel_p50_us.png)

### scatter func tflops vs vector gflops

![scatter func tflops vs vector gflops](card_constitution_20260711_134639_figs/scatter_func_tflops_vs_vector_gflops.png)

### scatter hbm gbps vs mte gbps

![scatter hbm gbps vs mte gbps](card_constitution_20260711_134639_figs/scatter_hbm_gbps_vs_mte_gbps.png)

### timeseries sustained p05 p50

![timeseries sustained p05 p50](card_constitution_20260711_134639_figs/timeseries_sustained_p05_p50.png)

