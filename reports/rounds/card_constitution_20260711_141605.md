# Card Constitution 分布报告

- 生成时间: 2026-07-11 14:16:13
- 卡数: 128
- 数据源: /Users/yinjinrun/random-thing/logs/muxi-constitution-20260711_140024-muxi-constitution128/results/constitution128.yushan-muxi-card-screen-128-cp-copy-master-0.jsonl, /Users/yinjinrun/random-thing/logs/muxi-constitution-20260711_140024-muxi-constitution128/results/constitution128.yushan-muxi-card-screen-128-cp-copy-worker-0.jsonl, /Users/yinjinrun/random-thing/logs/muxi-constitution-20260711_140024-muxi-constitution128/results/constitution128.yushan-muxi-card-screen-128-cp-copy-worker-1.jsonl, /Users/yinjinrun/random-thing/logs/muxi-constitution-20260711_140024-muxi-constitution128/results/constitution128.yushan-muxi-card-screen-128-cp-copy-worker-10.jsonl, /Users/yinjinrun/random-thing/logs/muxi-constitution-20260711_140024-muxi-constitution128/results/constitution128.yushan-muxi-card-screen-128-cp-copy-worker-11.jsonl, /Users/yinjinrun/random-thing/logs/muxi-constitution-20260711_140024-muxi-constitution128/results/constitution128.yushan-muxi-card-screen-128-cp-copy-worker-12.jsonl, /Users/yinjinrun/random-thing/logs/muxi-constitution-20260711_140024-muxi-constitution128/results/constitution128.yushan-muxi-card-screen-128-cp-copy-worker-13.jsonl, /Users/yinjinrun/random-thing/logs/muxi-constitution-20260711_140024-muxi-constitution128/results/constitution128.yushan-muxi-card-screen-128-cp-copy-worker-14.jsonl …

> 本报告只做分布统计与可视化，不强调 slow / 坏卡判定。

## 跳过说明

- `aicore_freq_mhz`（AICore freq (MHz)）：字段缺失或全空，跳过
- `hbm_temp_c`（HBM temp (C)）：字段缺失或全空，跳过
- `board_temp_c`（Board temp (C)）：字段缺失或全空，跳过
- `aicore_util_pct`（AICore util %）：字段缺失或全空，跳过
- `aicpu_util_pct`（AICPU util %）：字段缺失或全空，跳过
- `ctrlcpu_util_pct`（CtrlCPU util %）：字段缺失或全空，跳过
- `mem_bw_util_pct`（MemBW util %）：字段缺失或全空，跳过
- `shape_sweep_peak_tflops`（Shape sweep peak TFLOPS）：字段缺失或全空，跳过
- 散点 `launch_host_overhead_p50_us` × `ctrlcpu_util_pct`（Launch overhead × CtrlCPU）：缺轴字段，跳过
- `gemm_shape_sample`：无可用曲线，跳过 shape

## 指标分布

| 指标 | n | median | mean | std | CV% | min | max | p5 | p50 | p95 |
|------|---|--------|------|-----|-----|-----|-----|----|----|-----|
| Cube func TFLOPS | 127 | 279.3 | 279.7 | 2.898 | 1.036 | 273.3 | 288.7 | 275.3 | 279.3 | 285.0 |
| HBM GB/s | 127 | 1470.1 | 1387.8 | 156.7 | 11.29 | 977.6 | 1483.1 | 1033.9 | 1470.1 | 1478.4 |
| Sustained TFLOPS | 127 | 279.5 | 280.5 | 7.882 | 2.81 | 258.0 | 299.7 | 271.8 | 279.5 | 297.0 |
| Vector GFLOPS | 127 | 122.1 | 119.3 | 5.807 | 4.869 | 98.64 | 122.7 | 108.5 | 122.1 | 122.5 |
| Scalar elems/s | 127 | 120470586122.5 | 114082042585.0 | 20041026248.3 | 17.57 | 42118252637.5 | 135404961691.2 | 42597429893.3 | 120470586122.5 | 126688679327.4 |
| MTE copy GB/s | 127 | 1386.5 | 1367.4 | 56.73 | 4.149 | 1195.0 | 1406.1 | 1202.5 | 1386.5 | 1397.6 |
| Cube+Vector TFLOPS | 127 | 195.1 | 190.4 | 11.08 | 5.818 | 157.9 | 201.7 | 165.7 | 195.1 | 200.0 |
| SFU GFLOPS | 127 | 176.8 | 169.4 | 18.26 | 10.78 | 124.4 | 179.8 | 125.6 | 176.8 | 179.1 |
| Launch sync p50 (us) | 127 | 2.71 | 3.393 | 1.633 | 48.13 | 2.501 | 9.917 | 2.583 | 2.71 | 8.814 |
| Launch sync p99 (us) | 127 | 3.78 | 6.03 | 5.592 | 92.74 | 2.84 | 32.19 | 2.973 | 3.78 | 21.13 |
| Host overhead p50 (us) | 127 | 184.8 | 196.0 | 28.96 | 14.78 | 151.9 | 297.3 | 164.6 | 184.8 | 253.3 |
| Host overhead p99 (us) | 127 | 574.7 | 694.3 | 434.8 | 62.63 | 455.9 | 3872.2 | 479.6 | 574.7 | 1223.5 |
| Burst total p50 (us) | 127 | 1460.4 | 1333.5 | 348.9 | 26.16 | 609.1 | 1905.6 | 652.7 | 1460.4 | 1842.6 |
| Burst/kernel p50 (us) | 127 | 22.82 | 20.84 | 5.451 | 26.16 | 9.517 | 29.77 | 10.2 | 22.82 | 28.79 |
| Health temp (C) | 128 | 38.75 | 38.87 | 0.7335 | 1.887 | 37.25 | 41.25 | 37.84 | 38.75 | 40.25 |
| Health power (W) | 128 | 94.83 | 94.73 | 1.825 | 1.927 | 90.67 | 98.84 | 91.47 | 94.83 | 97.82 |
| Power (W) | 127 | 467.0 | 464.7 | 26.4 | 5.681 | 316.0 | 513.0 | 422.3 | 467.0 | 499.1 |
| Power limit (W) | 127 | 550.0 | 550.0 | 0 | 0 | 550.0 | 550.0 | 550.0 | 550.0 | 550.0 |

## 相对中位数偏差

偏差 = `(值 - 集群中位数) / 集群中位数 × 100%`。

- **Cube func TFLOPS** (`func_tflops`): [-2.17%, +3.37%]，|偏差|均值 0.81%
- **HBM GB/s** (`hbm_gbps`): [-33.50%, +0.88%]，|偏差|均值 5.95%
- **Sustained TFLOPS** (`sustained_tflops`): [-7.68%, +7.21%]，|偏差|均值 2.19%
- **Vector GFLOPS** (`vector_gflops`): [-19.23%, +0.49%]，|偏差|均值 2.50%
- **Scalar elems/s** (`scalar_elems_per_s`): [-65.04%, +12.40%]，|偏差|均值 8.19%
- **MTE copy GB/s** (`mte_gbps`): [-13.82%, +1.41%]，|偏差|均值 1.78%
- **Cube+Vector TFLOPS** (`cube_vector_tflops`): [-19.08%, +3.38%]，|偏差|均值 3.37%
- **SFU GFLOPS** (`sfu_gflops`): [-29.65%, +1.71%]，|偏差|均值 4.85%
- **Launch sync p50 (us)** (`launch_sync_p50_us`): [-7.73%, +265.91%]，|偏差|均值 27.84%
- **Launch sync p99 (us)** (`launch_sync_p99_us`): [-24.88%, +751.66%]，|偏差|均值 73.13%
- **Host overhead p50 (us)** (`launch_host_overhead_p50_us`): [-17.81%, +60.90%]，|偏差|均值 11.31%
- **Host overhead p99 (us)** (`launch_host_overhead_p99_us`): [-20.68%, +573.78%]，|偏差|均值 32.53%
- **Burst total p50 (us)** (`launch_burst_p50_us`): [-58.29%, +30.49%]，|偏差|均值 19.65%
- **Burst/kernel p50 (us)** (`launch_burst_per_kernel_p50_us`): [-58.29%, +30.49%]，|偏差|均值 19.65%
- **Health temp (C)** (`health_temp_c`): [-3.87%, +6.45%]，|偏差|均值 1.41%
- **Health power (W)** (`health_power_w`): [-4.38%, +4.23%]，|偏差|均值 1.54%
- **Power (W)** (`power_w`): [-32.33%, +9.85%]，|偏差|均值 4.06%
- **Power limit (W)** (`power_limit_w`): [+0.00%, +0.00%]，|偏差|均值 0.00%

## 元数据

- hosts (16): yushan-muxi-card-screen-128-cp-copy-master-0, yushan-muxi-card-screen-128-cp-copy-worker-0, yushan-muxi-card-screen-128-cp-copy-worker-1, yushan-muxi-card-screen-128-cp-copy-worker-10, yushan-muxi-card-screen-128-cp-copy-worker-11, yushan-muxi-card-screen-128-cp-copy-worker-12, yushan-muxi-card-screen-128-cp-copy-worker-13, yushan-muxi-card-screen-128-cp-copy-worker-14, yushan-muxi-card-screen-128-cp-copy-worker-2, yushan-muxi-card-screen-128-cp-copy-worker-3, yushan-muxi-card-screen-128-cp-copy-worker-4, yushan-muxi-card-screen-128-cp-copy-worker-5, yushan-muxi-card-screen-128-cp-copy-worker-6, yushan-muxi-card-screen-128-cp-copy-worker-7, yushan-muxi-card-screen-128-cp-copy-worker-8, yushan-muxi-card-screen-128-cp-copy-worker-9
- backends: metax
- launch_timing_method: event

## 图表

### box overview

![box overview](card_constitution_20260711_141605_figs/box_overview.png)

### hist func tflops

![hist func tflops](card_constitution_20260711_141605_figs/hist_func_tflops.png)

### hist hbm gbps

![hist hbm gbps](card_constitution_20260711_141605_figs/hist_hbm_gbps.png)

### hist sustained tflops

![hist sustained tflops](card_constitution_20260711_141605_figs/hist_sustained_tflops.png)

### hist vector gflops

![hist vector gflops](card_constitution_20260711_141605_figs/hist_vector_gflops.png)

### hist scalar elems per s

![hist scalar elems per s](card_constitution_20260711_141605_figs/hist_scalar_elems_per_s.png)

### hist mte gbps

![hist mte gbps](card_constitution_20260711_141605_figs/hist_mte_gbps.png)

### hist cube vector tflops

![hist cube vector tflops](card_constitution_20260711_141605_figs/hist_cube_vector_tflops.png)

### hist sfu gflops

![hist sfu gflops](card_constitution_20260711_141605_figs/hist_sfu_gflops.png)

### hist launch sync p50 us

![hist launch sync p50 us](card_constitution_20260711_141605_figs/hist_launch_sync_p50_us.png)

### hist launch sync p99 us

![hist launch sync p99 us](card_constitution_20260711_141605_figs/hist_launch_sync_p99_us.png)

### hist launch host overhead p50 us

![hist launch host overhead p50 us](card_constitution_20260711_141605_figs/hist_launch_host_overhead_p50_us.png)

### hist launch host overhead p99 us

![hist launch host overhead p99 us](card_constitution_20260711_141605_figs/hist_launch_host_overhead_p99_us.png)

### hist launch burst p50 us

![hist launch burst p50 us](card_constitution_20260711_141605_figs/hist_launch_burst_p50_us.png)

### hist launch burst per kernel p50 us

![hist launch burst per kernel p50 us](card_constitution_20260711_141605_figs/hist_launch_burst_per_kernel_p50_us.png)

### hist health temp c

![hist health temp c](card_constitution_20260711_141605_figs/hist_health_temp_c.png)

### hist health power w

![hist health power w](card_constitution_20260711_141605_figs/hist_health_power_w.png)

### hist power w

![hist power w](card_constitution_20260711_141605_figs/hist_power_w.png)

### hist power limit w

![hist power limit w](card_constitution_20260711_141605_figs/hist_power_limit_w.png)

### heatmap relmed func tflops

![heatmap relmed func tflops](card_constitution_20260711_141605_figs/heatmap_relmed_func_tflops.png)

### box by host func tflops

![box by host func tflops](card_constitution_20260711_141605_figs/box_by_host_func_tflops.png)

### sorted bar func tflops

![sorted bar func tflops](card_constitution_20260711_141605_figs/sorted_bar_func_tflops.png)

### bar host mean std func tflops

![bar host mean std func tflops](card_constitution_20260711_141605_figs/bar_host_mean_std_func_tflops.png)

### heatmap relmed hbm gbps

![heatmap relmed hbm gbps](card_constitution_20260711_141605_figs/heatmap_relmed_hbm_gbps.png)

### box by host hbm gbps

![box by host hbm gbps](card_constitution_20260711_141605_figs/box_by_host_hbm_gbps.png)

### sorted bar hbm gbps

![sorted bar hbm gbps](card_constitution_20260711_141605_figs/sorted_bar_hbm_gbps.png)

### bar host mean std hbm gbps

![bar host mean std hbm gbps](card_constitution_20260711_141605_figs/bar_host_mean_std_hbm_gbps.png)

### heatmap relmed sustained tflops

![heatmap relmed sustained tflops](card_constitution_20260711_141605_figs/heatmap_relmed_sustained_tflops.png)

### box by host sustained tflops

![box by host sustained tflops](card_constitution_20260711_141605_figs/box_by_host_sustained_tflops.png)

### sorted bar sustained tflops

![sorted bar sustained tflops](card_constitution_20260711_141605_figs/sorted_bar_sustained_tflops.png)

### bar host mean std sustained tflops

![bar host mean std sustained tflops](card_constitution_20260711_141605_figs/bar_host_mean_std_sustained_tflops.png)

### heatmap relmed vector gflops

![heatmap relmed vector gflops](card_constitution_20260711_141605_figs/heatmap_relmed_vector_gflops.png)

### box by host vector gflops

![box by host vector gflops](card_constitution_20260711_141605_figs/box_by_host_vector_gflops.png)

### sorted bar vector gflops

![sorted bar vector gflops](card_constitution_20260711_141605_figs/sorted_bar_vector_gflops.png)

### bar host mean std vector gflops

![bar host mean std vector gflops](card_constitution_20260711_141605_figs/bar_host_mean_std_vector_gflops.png)

### heatmap relmed scalar elems per s

![heatmap relmed scalar elems per s](card_constitution_20260711_141605_figs/heatmap_relmed_scalar_elems_per_s.png)

### box by host scalar elems per s

![box by host scalar elems per s](card_constitution_20260711_141605_figs/box_by_host_scalar_elems_per_s.png)

### sorted bar scalar elems per s

![sorted bar scalar elems per s](card_constitution_20260711_141605_figs/sorted_bar_scalar_elems_per_s.png)

### bar host mean std scalar elems per s

![bar host mean std scalar elems per s](card_constitution_20260711_141605_figs/bar_host_mean_std_scalar_elems_per_s.png)

### heatmap relmed mte gbps

![heatmap relmed mte gbps](card_constitution_20260711_141605_figs/heatmap_relmed_mte_gbps.png)

### box by host mte gbps

![box by host mte gbps](card_constitution_20260711_141605_figs/box_by_host_mte_gbps.png)

### sorted bar mte gbps

![sorted bar mte gbps](card_constitution_20260711_141605_figs/sorted_bar_mte_gbps.png)

### bar host mean std mte gbps

![bar host mean std mte gbps](card_constitution_20260711_141605_figs/bar_host_mean_std_mte_gbps.png)

### heatmap relmed cube vector tflops

![heatmap relmed cube vector tflops](card_constitution_20260711_141605_figs/heatmap_relmed_cube_vector_tflops.png)

### box by host cube vector tflops

![box by host cube vector tflops](card_constitution_20260711_141605_figs/box_by_host_cube_vector_tflops.png)

### sorted bar cube vector tflops

![sorted bar cube vector tflops](card_constitution_20260711_141605_figs/sorted_bar_cube_vector_tflops.png)

### bar host mean std cube vector tflops

![bar host mean std cube vector tflops](card_constitution_20260711_141605_figs/bar_host_mean_std_cube_vector_tflops.png)

### heatmap relmed sfu gflops

![heatmap relmed sfu gflops](card_constitution_20260711_141605_figs/heatmap_relmed_sfu_gflops.png)

### box by host sfu gflops

![box by host sfu gflops](card_constitution_20260711_141605_figs/box_by_host_sfu_gflops.png)

### sorted bar sfu gflops

![sorted bar sfu gflops](card_constitution_20260711_141605_figs/sorted_bar_sfu_gflops.png)

### bar host mean std sfu gflops

![bar host mean std sfu gflops](card_constitution_20260711_141605_figs/bar_host_mean_std_sfu_gflops.png)

### heatmap relmed launch sync p50 us

![heatmap relmed launch sync p50 us](card_constitution_20260711_141605_figs/heatmap_relmed_launch_sync_p50_us.png)

### box by host launch sync p50 us

![box by host launch sync p50 us](card_constitution_20260711_141605_figs/box_by_host_launch_sync_p50_us.png)

### sorted bar launch sync p50 us

![sorted bar launch sync p50 us](card_constitution_20260711_141605_figs/sorted_bar_launch_sync_p50_us.png)

### heatmap relmed launch sync p99 us

![heatmap relmed launch sync p99 us](card_constitution_20260711_141605_figs/heatmap_relmed_launch_sync_p99_us.png)

### box by host launch sync p99 us

![box by host launch sync p99 us](card_constitution_20260711_141605_figs/box_by_host_launch_sync_p99_us.png)

### sorted bar launch sync p99 us

![sorted bar launch sync p99 us](card_constitution_20260711_141605_figs/sorted_bar_launch_sync_p99_us.png)

### heatmap relmed launch host overhead p50 us

![heatmap relmed launch host overhead p50 us](card_constitution_20260711_141605_figs/heatmap_relmed_launch_host_overhead_p50_us.png)

### box by host launch host overhead p50 us

![box by host launch host overhead p50 us](card_constitution_20260711_141605_figs/box_by_host_launch_host_overhead_p50_us.png)

### sorted bar launch host overhead p50 us

![sorted bar launch host overhead p50 us](card_constitution_20260711_141605_figs/sorted_bar_launch_host_overhead_p50_us.png)

### heatmap relmed launch host overhead p99 us

![heatmap relmed launch host overhead p99 us](card_constitution_20260711_141605_figs/heatmap_relmed_launch_host_overhead_p99_us.png)

### box by host launch host overhead p99 us

![box by host launch host overhead p99 us](card_constitution_20260711_141605_figs/box_by_host_launch_host_overhead_p99_us.png)

### sorted bar launch host overhead p99 us

![sorted bar launch host overhead p99 us](card_constitution_20260711_141605_figs/sorted_bar_launch_host_overhead_p99_us.png)

### heatmap relmed launch burst p50 us

![heatmap relmed launch burst p50 us](card_constitution_20260711_141605_figs/heatmap_relmed_launch_burst_p50_us.png)

### box by host launch burst p50 us

![box by host launch burst p50 us](card_constitution_20260711_141605_figs/box_by_host_launch_burst_p50_us.png)

### sorted bar launch burst p50 us

![sorted bar launch burst p50 us](card_constitution_20260711_141605_figs/sorted_bar_launch_burst_p50_us.png)

### heatmap relmed launch burst per kernel p50 us

![heatmap relmed launch burst per kernel p50 us](card_constitution_20260711_141605_figs/heatmap_relmed_launch_burst_per_kernel_p50_us.png)

### box by host launch burst per kernel p50 us

![box by host launch burst per kernel p50 us](card_constitution_20260711_141605_figs/box_by_host_launch_burst_per_kernel_p50_us.png)

### sorted bar launch burst per kernel p50 us

![sorted bar launch burst per kernel p50 us](card_constitution_20260711_141605_figs/sorted_bar_launch_burst_per_kernel_p50_us.png)

### heatmap relmed health temp c

![heatmap relmed health temp c](card_constitution_20260711_141605_figs/heatmap_relmed_health_temp_c.png)

### box by host health temp c

![box by host health temp c](card_constitution_20260711_141605_figs/box_by_host_health_temp_c.png)

### sorted bar health temp c

![sorted bar health temp c](card_constitution_20260711_141605_figs/sorted_bar_health_temp_c.png)

### bar host mean std health temp c

![bar host mean std health temp c](card_constitution_20260711_141605_figs/bar_host_mean_std_health_temp_c.png)

### heatmap relmed health power w

![heatmap relmed health power w](card_constitution_20260711_141605_figs/heatmap_relmed_health_power_w.png)

### box by host health power w

![box by host health power w](card_constitution_20260711_141605_figs/box_by_host_health_power_w.png)

### sorted bar health power w

![sorted bar health power w](card_constitution_20260711_141605_figs/sorted_bar_health_power_w.png)

### bar host mean std health power w

![bar host mean std health power w](card_constitution_20260711_141605_figs/bar_host_mean_std_health_power_w.png)

### heatmap relmed power w

![heatmap relmed power w](card_constitution_20260711_141605_figs/heatmap_relmed_power_w.png)

### box by host power w

![box by host power w](card_constitution_20260711_141605_figs/box_by_host_power_w.png)

### sorted bar power w

![sorted bar power w](card_constitution_20260711_141605_figs/sorted_bar_power_w.png)

### bar host mean std power w

![bar host mean std power w](card_constitution_20260711_141605_figs/bar_host_mean_std_power_w.png)

### heatmap relmed power limit w

![heatmap relmed power limit w](card_constitution_20260711_141605_figs/heatmap_relmed_power_limit_w.png)

### box by host power limit w

![box by host power limit w](card_constitution_20260711_141605_figs/box_by_host_power_limit_w.png)

### sorted bar power limit w

![sorted bar power limit w](card_constitution_20260711_141605_figs/sorted_bar_power_limit_w.png)

### scatter func tflops vs vector gflops

![scatter func tflops vs vector gflops](card_constitution_20260711_141605_figs/scatter_func_tflops_vs_vector_gflops.png)

### scatter hbm gbps vs mte gbps

![scatter hbm gbps vs mte gbps](card_constitution_20260711_141605_figs/scatter_hbm_gbps_vs_mte_gbps.png)

### scatter power w vs func tflops

![scatter power w vs func tflops](card_constitution_20260711_141605_figs/scatter_power_w_vs_func_tflops.png)

### scatter health power w vs func tflops

![scatter health power w vs func tflops](card_constitution_20260711_141605_figs/scatter_health_power_w_vs_func_tflops.png)

### scatter power w vs hbm gbps

![scatter power w vs hbm gbps](card_constitution_20260711_141605_figs/scatter_power_w_vs_hbm_gbps.png)

### scatter health power w vs hbm gbps

![scatter health power w vs hbm gbps](card_constitution_20260711_141605_figs/scatter_health_power_w_vs_hbm_gbps.png)

### timeseries sustained p05 p50

![timeseries sustained p05 p50](card_constitution_20260711_141605_figs/timeseries_sustained_p05_p50.png)

