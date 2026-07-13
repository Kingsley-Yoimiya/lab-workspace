# Card Constitution 分布报告

- 生成时间: 2026-07-11 23:31:20
- 卡数: 116
- 数据源: /Users/yinjinrun/random-thing/logs/muxi-constitution-20260711_232400-muxi-constitution128/results/constitution128.merged.jsonl

> 本报告只做分布统计与可视化，不强调 slow / 坏卡判定。

## 跳过说明

- `hbm_temp_c`（HBM temp (C)）：字段缺失或全空，跳过
- `aicpu_util_pct`（AICPU util %）：字段缺失或全空，跳过
- `ctrlcpu_util_pct`（CtrlCPU util %）：字段缺失或全空，跳过
- `mem_bw_util_pct`（MemBW util %）：字段缺失或全空，跳过
- 散点 `launch_host_overhead_p50_us` × `ctrlcpu_util_pct`（Launch overhead × CtrlCPU）：缺轴字段，跳过
- `gemm_shape_sample`：无可用曲线，跳过 shape

## 指标分布

| 指标 | n | median | mean | std | CV% | min | max | p5 | p50 | p95 |
|------|---|--------|------|-----|-----|-----|-----|----|----|-----|
| GEMM func TFLOPS | 115 | 279.9 | 279.3 | 3.103 | 1.111 | 271.4 | 288.0 | 273.9 | 279.9 | 284.0 |
| HBM GB/s | 115 | 1469.4 | 1397.7 | 147.1 | 10.52 | 871.0 | 1487.1 | 1037.0 | 1469.4 | 1478.8 |
| Sustained TFLOPS | 115 | 279.3 | 280.9 | 7.91 | 2.816 | 258.1 | 299.6 | 274.4 | 279.3 | 299.5 |
| Vector FMA GFLOPS | 115 | 122.2 | 120.5 | 4.451 | 3.695 | 108.3 | 122.7 | 109.0 | 122.2 | 122.6 |
| Scalar elems/s | 115 | 121138635141.4 | 118423211497.6 | 12041443420.5 | 10.17 | 42335916511.7 | 132129034045.3 | 102737581023.3 | 121138635141.4 | 128000000323.2 |
| DMA copy GB/s | 115 | 1387.9 | 1374.2 | 51.88 | 3.776 | 1186.8 | 1409.4 | 1201.4 | 1387.9 | 1401.8 |
| GEMM+epilogue TFLOPS | 115 | 195.5 | 189.1 | 16.1 | 8.511 | 104.9 | 202.1 | 162.9 | 195.5 | 198.3 |
| SFU GFLOPS | 115 | 177.4 | 172.2 | 15.55 | 9.033 | 125.1 | 179.4 | 126.7 | 177.4 | 179.0 |
| Launch sync p50 (us) | 115 | 2.68 | 3.057 | 1.012 | 33.1 | 2.471 | 10.03 | 2.57 | 2.68 | 3.973 |
| Launch sync p99 (us) | 115 | 4.289 | 5.998 | 5.807 | 96.81 | 2.939 | 32.99 | 3.028 | 4.289 | 20.22 |
| Host overhead p50 (us) | 115 | 183.9 | 197.6 | 36.69 | 18.56 | 154.3 | 345.5 | 165.3 | 183.9 | 249.0 |
| Host overhead p99 (us) | 115 | 550.8 | 651.4 | 329.0 | 50.51 | 465.5 | 3430.0 | 482.9 | 550.8 | 965.3 |
| Burst total p50 (us) | 115 | 1317.8 | 1288.1 | 354.7 | 27.54 | 600.6 | 1918.7 | 641.8 | 1317.8 | 1826.1 |
| Burst/kernel p50 (us) | 115 | 20.59 | 20.13 | 5.543 | 27.54 | 9.384 | 29.98 | 10.03 | 20.59 | 28.53 |
| Health temp (C) | 116 | 38.5 | 39.88 | 5.675 | 14.23 | 37 | 66.75 | 37.5 | 38.5 | 48.06 |
| Health power (W) | 116 | 94.89 | 124.1 | 108.7 | 87.54 | 90.14 | 549.7 | 91.61 | 94.89 | 517.0 |
| XCORE clk (MHz) | 115 | 1371.0 | 1431.2 | 66.81 | 4.668 | 1285.0 | 1500.0 | 1371.0 | 1371.0 | 1500.0 |
| Board temp (C) | 115 | 54 | 54.28 | 2.599 | 4.789 | 47 | 62 | 50.7 | 54 | 59 |
| GPU util % | 115 | 98 | 81.47 | 30.89 | 37.92 | 0 | 98 | 2.4 | 98 | 98 |
| Power (W) | 115 | 472.0 | 470.3 | 20.55 | 4.369 | 417.0 | 517.0 | 439.4 | 472.0 | 500.4 |
| Power limit (W) | 115 | 550.0 | 550.0 | 0 | 0 | 550.0 | 550.0 | 550.0 | 550.0 | 550.0 |
| Shape sweep peak TFLOPS | 115 | 286.0 | 285.2 | 5.755 | 2.018 | 271.8 | 299.5 | 274.8 | 286.0 | 299.1 |

## 相对中位数偏差

偏差 = `(值 - 集群中位数) / 集群中位数 × 100%`。

- **GEMM func TFLOPS** (`func_tflops`): [-3.03%, +2.90%]，|偏差|均值 0.84%
- **HBM GB/s** (`hbm_gbps`): [-40.72%, +1.21%]，|偏差|均值 5.22%
- **Sustained TFLOPS** (`sustained_tflops`): [-7.61%, +7.27%]，|偏差|均值 2.22%
- **Vector FMA GFLOPS** (`vector_gflops`): [-11.35%, +0.42%]，|偏差|均值 1.57%
- **Scalar elems/s** (`scalar_elems_per_s`): [-65.05%, +9.07%]，|偏差|均值 4.63%
- **DMA copy GB/s** (`mte_gbps`): [-14.49%, +1.55%]，|偏差|均值 1.40%
- **GEMM+epilogue TFLOPS** (`cube_vector_tflops`): [-46.33%, +3.39%]，|偏差|均值 4.22%
- **SFU GFLOPS** (`sfu_gflops`): [-29.45%, +1.16%]，|偏差|均值 3.47%
- **Launch sync p50 (us)** (`launch_sync_p50_us`): [-7.82%, +274.18%]，|偏差|均值 16.08%
- **Launch sync p99 (us)** (`launch_sync_p99_us`): [-31.47%, +669.16%]，|偏差|均值 60.26%
- **Host overhead p50 (us)** (`launch_host_overhead_p50_us`): [-16.08%, +87.94%]，|偏差|均值 12.92%
- **Host overhead p99 (us)** (`launch_host_overhead_p99_us`): [-15.48%, +522.68%]，|偏差|均值 27.40%
- **Burst total p50 (us)** (`launch_burst_p50_us`): [-54.43%, +45.59%]，|偏差|均值 22.84%
- **Burst/kernel p50 (us)** (`launch_burst_per_kernel_p50_us`): [-54.43%, +45.59%]，|偏差|均值 22.84%
- **Health temp (C)** (`health_temp_c`): [-3.90%, +73.38%]，|偏差|均值 4.94%
- **Health power (W)** (`health_power_w`): [-5.01%, +479.34%]，|偏差|均值 32.53%
- **XCORE clk (MHz)** (`aicore_freq_mhz`): [-6.27%, +9.41%]，|偏差|均值 4.61%
- **Board temp (C)** (`board_temp_c`): [-12.96%, +14.81%]，|偏差|均值 3.77%
- **GPU util %** (`aicore_util_pct`): [-100.00%, +0.00%]，|偏差|均值 16.87%
- **Power (W)** (`power_w`): [-11.65%, +9.53%]，|偏差|均值 3.50%
- **Power limit (W)** (`power_limit_w`): [+0.00%, +0.00%]，|偏差|均值 0.00%
- **Shape sweep peak TFLOPS** (`shape_sweep_peak_tflops`): [-4.97%, +4.74%]，|偏差|均值 1.45%

## 元数据

- hosts (16): m0, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14
- backends: metax
- launch_timing_method: event

## 图表

### box overview

![box overview](card_constitution_muxi_20260711_rerun_figs/box_overview.svg)

### hist func tflops

![hist func tflops](card_constitution_muxi_20260711_rerun_figs/hist_func_tflops.svg)

### hist hbm gbps

![hist hbm gbps](card_constitution_muxi_20260711_rerun_figs/hist_hbm_gbps.svg)

### hist sustained tflops

![hist sustained tflops](card_constitution_muxi_20260711_rerun_figs/hist_sustained_tflops.svg)

### hist vector gflops

![hist vector gflops](card_constitution_muxi_20260711_rerun_figs/hist_vector_gflops.svg)

### hist scalar elems per s

![hist scalar elems per s](card_constitution_muxi_20260711_rerun_figs/hist_scalar_elems_per_s.svg)

### hist mte gbps

![hist mte gbps](card_constitution_muxi_20260711_rerun_figs/hist_mte_gbps.svg)

### hist cube vector tflops

![hist cube vector tflops](card_constitution_muxi_20260711_rerun_figs/hist_cube_vector_tflops.svg)

### hist sfu gflops

![hist sfu gflops](card_constitution_muxi_20260711_rerun_figs/hist_sfu_gflops.svg)

### hist launch sync p50 us

![hist launch sync p50 us](card_constitution_muxi_20260711_rerun_figs/hist_launch_sync_p50_us.svg)

### hist launch sync p99 us

![hist launch sync p99 us](card_constitution_muxi_20260711_rerun_figs/hist_launch_sync_p99_us.svg)

### hist launch host overhead p50 us

![hist launch host overhead p50 us](card_constitution_muxi_20260711_rerun_figs/hist_launch_host_overhead_p50_us.svg)

### hist launch host overhead p99 us

![hist launch host overhead p99 us](card_constitution_muxi_20260711_rerun_figs/hist_launch_host_overhead_p99_us.svg)

### hist launch burst p50 us

![hist launch burst p50 us](card_constitution_muxi_20260711_rerun_figs/hist_launch_burst_p50_us.svg)

### hist launch burst per kernel p50 us

![hist launch burst per kernel p50 us](card_constitution_muxi_20260711_rerun_figs/hist_launch_burst_per_kernel_p50_us.svg)

### hist health temp c

![hist health temp c](card_constitution_muxi_20260711_rerun_figs/hist_health_temp_c.svg)

### hist health power w

![hist health power w](card_constitution_muxi_20260711_rerun_figs/hist_health_power_w.svg)

### hist aicore freq mhz

![hist aicore freq mhz](card_constitution_muxi_20260711_rerun_figs/hist_aicore_freq_mhz.svg)

### hist board temp c

![hist board temp c](card_constitution_muxi_20260711_rerun_figs/hist_board_temp_c.svg)

### hist aicore util pct

![hist aicore util pct](card_constitution_muxi_20260711_rerun_figs/hist_aicore_util_pct.svg)

### hist power w

![hist power w](card_constitution_muxi_20260711_rerun_figs/hist_power_w.svg)

### hist power limit w

![hist power limit w](card_constitution_muxi_20260711_rerun_figs/hist_power_limit_w.svg)

### hist shape sweep peak tflops

![hist shape sweep peak tflops](card_constitution_muxi_20260711_rerun_figs/hist_shape_sweep_peak_tflops.svg)

### heatmap relmed func tflops

![heatmap relmed func tflops](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_func_tflops.svg)

### box by host func tflops

![box by host func tflops](card_constitution_muxi_20260711_rerun_figs/box_by_host_func_tflops.svg)

### sorted bar func tflops

![sorted bar func tflops](card_constitution_muxi_20260711_rerun_figs/sorted_bar_func_tflops.svg)

### bar host mean std func tflops

![bar host mean std func tflops](card_constitution_muxi_20260711_rerun_figs/bar_host_mean_std_func_tflops.svg)

### heatmap relmed hbm gbps

![heatmap relmed hbm gbps](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_hbm_gbps.svg)

### box by host hbm gbps

![box by host hbm gbps](card_constitution_muxi_20260711_rerun_figs/box_by_host_hbm_gbps.svg)

### sorted bar hbm gbps

![sorted bar hbm gbps](card_constitution_muxi_20260711_rerun_figs/sorted_bar_hbm_gbps.svg)

### bar host mean std hbm gbps

![bar host mean std hbm gbps](card_constitution_muxi_20260711_rerun_figs/bar_host_mean_std_hbm_gbps.svg)

### heatmap relmed sustained tflops

![heatmap relmed sustained tflops](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_sustained_tflops.svg)

### box by host sustained tflops

![box by host sustained tflops](card_constitution_muxi_20260711_rerun_figs/box_by_host_sustained_tflops.svg)

### sorted bar sustained tflops

![sorted bar sustained tflops](card_constitution_muxi_20260711_rerun_figs/sorted_bar_sustained_tflops.svg)

### bar host mean std sustained tflops

![bar host mean std sustained tflops](card_constitution_muxi_20260711_rerun_figs/bar_host_mean_std_sustained_tflops.svg)

### heatmap relmed vector gflops

![heatmap relmed vector gflops](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_vector_gflops.svg)

### box by host vector gflops

![box by host vector gflops](card_constitution_muxi_20260711_rerun_figs/box_by_host_vector_gflops.svg)

### sorted bar vector gflops

![sorted bar vector gflops](card_constitution_muxi_20260711_rerun_figs/sorted_bar_vector_gflops.svg)

### bar host mean std vector gflops

![bar host mean std vector gflops](card_constitution_muxi_20260711_rerun_figs/bar_host_mean_std_vector_gflops.svg)

### heatmap relmed scalar elems per s

![heatmap relmed scalar elems per s](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_scalar_elems_per_s.svg)

### box by host scalar elems per s

![box by host scalar elems per s](card_constitution_muxi_20260711_rerun_figs/box_by_host_scalar_elems_per_s.svg)

### sorted bar scalar elems per s

![sorted bar scalar elems per s](card_constitution_muxi_20260711_rerun_figs/sorted_bar_scalar_elems_per_s.svg)

### bar host mean std scalar elems per s

![bar host mean std scalar elems per s](card_constitution_muxi_20260711_rerun_figs/bar_host_mean_std_scalar_elems_per_s.svg)

### heatmap relmed mte gbps

![heatmap relmed mte gbps](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_mte_gbps.svg)

### box by host mte gbps

![box by host mte gbps](card_constitution_muxi_20260711_rerun_figs/box_by_host_mte_gbps.svg)

### sorted bar mte gbps

![sorted bar mte gbps](card_constitution_muxi_20260711_rerun_figs/sorted_bar_mte_gbps.svg)

### bar host mean std mte gbps

![bar host mean std mte gbps](card_constitution_muxi_20260711_rerun_figs/bar_host_mean_std_mte_gbps.svg)

### heatmap relmed cube vector tflops

![heatmap relmed cube vector tflops](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_cube_vector_tflops.svg)

### box by host cube vector tflops

![box by host cube vector tflops](card_constitution_muxi_20260711_rerun_figs/box_by_host_cube_vector_tflops.svg)

### sorted bar cube vector tflops

![sorted bar cube vector tflops](card_constitution_muxi_20260711_rerun_figs/sorted_bar_cube_vector_tflops.svg)

### bar host mean std cube vector tflops

![bar host mean std cube vector tflops](card_constitution_muxi_20260711_rerun_figs/bar_host_mean_std_cube_vector_tflops.svg)

### heatmap relmed sfu gflops

![heatmap relmed sfu gflops](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_sfu_gflops.svg)

### box by host sfu gflops

![box by host sfu gflops](card_constitution_muxi_20260711_rerun_figs/box_by_host_sfu_gflops.svg)

### sorted bar sfu gflops

![sorted bar sfu gflops](card_constitution_muxi_20260711_rerun_figs/sorted_bar_sfu_gflops.svg)

### bar host mean std sfu gflops

![bar host mean std sfu gflops](card_constitution_muxi_20260711_rerun_figs/bar_host_mean_std_sfu_gflops.svg)

### heatmap relmed launch sync p50 us

![heatmap relmed launch sync p50 us](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_launch_sync_p50_us.svg)

### box by host launch sync p50 us

![box by host launch sync p50 us](card_constitution_muxi_20260711_rerun_figs/box_by_host_launch_sync_p50_us.svg)

### sorted bar launch sync p50 us

![sorted bar launch sync p50 us](card_constitution_muxi_20260711_rerun_figs/sorted_bar_launch_sync_p50_us.svg)

### heatmap relmed launch sync p99 us

![heatmap relmed launch sync p99 us](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_launch_sync_p99_us.svg)

### box by host launch sync p99 us

![box by host launch sync p99 us](card_constitution_muxi_20260711_rerun_figs/box_by_host_launch_sync_p99_us.svg)

### sorted bar launch sync p99 us

![sorted bar launch sync p99 us](card_constitution_muxi_20260711_rerun_figs/sorted_bar_launch_sync_p99_us.svg)

### heatmap relmed launch host overhead p50 us

![heatmap relmed launch host overhead p50 us](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_launch_host_overhead_p50_us.svg)

### box by host launch host overhead p50 us

![box by host launch host overhead p50 us](card_constitution_muxi_20260711_rerun_figs/box_by_host_launch_host_overhead_p50_us.svg)

### sorted bar launch host overhead p50 us

![sorted bar launch host overhead p50 us](card_constitution_muxi_20260711_rerun_figs/sorted_bar_launch_host_overhead_p50_us.svg)

### heatmap relmed launch host overhead p99 us

![heatmap relmed launch host overhead p99 us](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_launch_host_overhead_p99_us.svg)

### box by host launch host overhead p99 us

![box by host launch host overhead p99 us](card_constitution_muxi_20260711_rerun_figs/box_by_host_launch_host_overhead_p99_us.svg)

### sorted bar launch host overhead p99 us

![sorted bar launch host overhead p99 us](card_constitution_muxi_20260711_rerun_figs/sorted_bar_launch_host_overhead_p99_us.svg)

### heatmap relmed launch burst p50 us

![heatmap relmed launch burst p50 us](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_launch_burst_p50_us.svg)

### box by host launch burst p50 us

![box by host launch burst p50 us](card_constitution_muxi_20260711_rerun_figs/box_by_host_launch_burst_p50_us.svg)

### sorted bar launch burst p50 us

![sorted bar launch burst p50 us](card_constitution_muxi_20260711_rerun_figs/sorted_bar_launch_burst_p50_us.svg)

### heatmap relmed launch burst per kernel p50 us

![heatmap relmed launch burst per kernel p50 us](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_launch_burst_per_kernel_p50_us.svg)

### box by host launch burst per kernel p50 us

![box by host launch burst per kernel p50 us](card_constitution_muxi_20260711_rerun_figs/box_by_host_launch_burst_per_kernel_p50_us.svg)

### sorted bar launch burst per kernel p50 us

![sorted bar launch burst per kernel p50 us](card_constitution_muxi_20260711_rerun_figs/sorted_bar_launch_burst_per_kernel_p50_us.svg)

### heatmap relmed health temp c

![heatmap relmed health temp c](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_health_temp_c.svg)

### box by host health temp c

![box by host health temp c](card_constitution_muxi_20260711_rerun_figs/box_by_host_health_temp_c.svg)

### sorted bar health temp c

![sorted bar health temp c](card_constitution_muxi_20260711_rerun_figs/sorted_bar_health_temp_c.svg)

### bar host mean std health temp c

![bar host mean std health temp c](card_constitution_muxi_20260711_rerun_figs/bar_host_mean_std_health_temp_c.svg)

### heatmap relmed health power w

![heatmap relmed health power w](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_health_power_w.svg)

### box by host health power w

![box by host health power w](card_constitution_muxi_20260711_rerun_figs/box_by_host_health_power_w.svg)

### sorted bar health power w

![sorted bar health power w](card_constitution_muxi_20260711_rerun_figs/sorted_bar_health_power_w.svg)

### bar host mean std health power w

![bar host mean std health power w](card_constitution_muxi_20260711_rerun_figs/bar_host_mean_std_health_power_w.svg)

### heatmap relmed aicore freq mhz

![heatmap relmed aicore freq mhz](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_aicore_freq_mhz.svg)

### box by host aicore freq mhz

![box by host aicore freq mhz](card_constitution_muxi_20260711_rerun_figs/box_by_host_aicore_freq_mhz.svg)

### sorted bar aicore freq mhz

![sorted bar aicore freq mhz](card_constitution_muxi_20260711_rerun_figs/sorted_bar_aicore_freq_mhz.svg)

### bar host mean std aicore freq mhz

![bar host mean std aicore freq mhz](card_constitution_muxi_20260711_rerun_figs/bar_host_mean_std_aicore_freq_mhz.svg)

### heatmap relmed board temp c

![heatmap relmed board temp c](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_board_temp_c.svg)

### box by host board temp c

![box by host board temp c](card_constitution_muxi_20260711_rerun_figs/box_by_host_board_temp_c.svg)

### sorted bar board temp c

![sorted bar board temp c](card_constitution_muxi_20260711_rerun_figs/sorted_bar_board_temp_c.svg)

### heatmap relmed aicore util pct

![heatmap relmed aicore util pct](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_aicore_util_pct.svg)

### box by host aicore util pct

![box by host aicore util pct](card_constitution_muxi_20260711_rerun_figs/box_by_host_aicore_util_pct.svg)

### sorted bar aicore util pct

![sorted bar aicore util pct](card_constitution_muxi_20260711_rerun_figs/sorted_bar_aicore_util_pct.svg)

### heatmap relmed power w

![heatmap relmed power w](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_power_w.svg)

### box by host power w

![box by host power w](card_constitution_muxi_20260711_rerun_figs/box_by_host_power_w.svg)

### sorted bar power w

![sorted bar power w](card_constitution_muxi_20260711_rerun_figs/sorted_bar_power_w.svg)

### bar host mean std power w

![bar host mean std power w](card_constitution_muxi_20260711_rerun_figs/bar_host_mean_std_power_w.svg)

### heatmap relmed power limit w

![heatmap relmed power limit w](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_power_limit_w.svg)

### box by host power limit w

![box by host power limit w](card_constitution_muxi_20260711_rerun_figs/box_by_host_power_limit_w.svg)

### sorted bar power limit w

![sorted bar power limit w](card_constitution_muxi_20260711_rerun_figs/sorted_bar_power_limit_w.svg)

### heatmap relmed shape sweep peak tflops

![heatmap relmed shape sweep peak tflops](card_constitution_muxi_20260711_rerun_figs/heatmap_relmed_shape_sweep_peak_tflops.svg)

### box by host shape sweep peak tflops

![box by host shape sweep peak tflops](card_constitution_muxi_20260711_rerun_figs/box_by_host_shape_sweep_peak_tflops.svg)

### sorted bar shape sweep peak tflops

![sorted bar shape sweep peak tflops](card_constitution_muxi_20260711_rerun_figs/sorted_bar_shape_sweep_peak_tflops.svg)

### scatter func tflops vs vector gflops

![scatter func tflops vs vector gflops](card_constitution_muxi_20260711_rerun_figs/scatter_func_tflops_vs_vector_gflops.svg)

### scatter hbm gbps vs mte gbps

![scatter hbm gbps vs mte gbps](card_constitution_muxi_20260711_rerun_figs/scatter_hbm_gbps_vs_mte_gbps.svg)

### scatter power w vs func tflops

![scatter power w vs func tflops](card_constitution_muxi_20260711_rerun_figs/scatter_power_w_vs_func_tflops.svg)

### scatter health power w vs func tflops

![scatter health power w vs func tflops](card_constitution_muxi_20260711_rerun_figs/scatter_health_power_w_vs_func_tflops.svg)

### scatter power w vs hbm gbps

![scatter power w vs hbm gbps](card_constitution_muxi_20260711_rerun_figs/scatter_power_w_vs_hbm_gbps.svg)

### scatter health power w vs hbm gbps

![scatter health power w vs hbm gbps](card_constitution_muxi_20260711_rerun_figs/scatter_health_power_w_vs_hbm_gbps.svg)

### timeseries sustained p05 p50

![timeseries sustained p05 p50](card_constitution_muxi_20260711_rerun_figs/timeseries_sustained_p05_p50.svg)

