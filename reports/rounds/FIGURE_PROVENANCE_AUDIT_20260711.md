# 图溯源审计笔记 · 20260711 战役

> 审计范围：5 个图目录，共 **151** 张 SVG（112+12+3+23+1）。  
> 审计日期：2026-07-11。  
> 集群：A3 job `whj4stu-copy-copy-copy`，**8×16=128 卡**（hosts: master-0 + worker-0..6）。  
> 出图规范：默认 `reports/plot_style.py`（不再单独加风格后缀）。

---

## 0. 总览：图目录 ↔ 出图脚本 ↔ 原始数据

| 图目录 | SVG 数 | 出图脚本 | 原始数据 |
|--------|--------|----------|----------|
| `card_constitution_20260711_figs/` | 112 | `reports/plot_card_constitution.py` | `logs/card-fillgap-20260711_140301/results/constitution128.merged.jsonl` |
| `constitution_extra_fillgap_20260711_figs/` | 12 | `reports/plot_constitution_extra.py` | 同上 merged JSONL |
| `bnmk_shapes_20260711_figs/` | 3 | `reports/plot_bnmk_shapes.py` | 同上 `results/**/*.jsonl`（见 §3 双计坑） |
| `hccl_campaign_20260711_figs/` | 23 | `reports/plot_hccl_campaign.py` | `logs/pipeline-comm-20260711_134811/{hccl-results,p2p-results,hccl-topo/raw}/` |
| `inter_bw_20260711_figs/` | 1 | `scripts/cluster/summarize_inter_bw.py`（`--plot`） | `logs/inter-bw-20260711_141922/` |

样式统一走 `reports/plot_style.py`（大字号、去顶右边框、y 点线网格、空心 hatch 柱、SVG 主交付）。

### 0.1 体质筛查采集公共条件（§1–§3 共用）

- **Launch**：`logs/card-fillgap-20260711_140301/start_*.log`  
  ```text
  python3 screen.py --device all --config config.constitution128.yaml \
    --sdc-rounds 5 --gemm-n 8192 --sustained-s 30 \
    --idle-max-memory-mib 1024 --out .../{pod}.jsonl --no-plot
  ```
  （由 `scripts/cluster/launch_constitution_kubectl.sh` 同型发射；本批 stamp=`20260711_140301-fillgap`）
- **配置**：`projects/CARD_SCREEN/config.constitution128.yaml`
- **落库**：`projects/CARD_SCREEN/card_screen/io/jsonl.py` → `record=card` + 各 round/sample 行
- **遥测**：`card_screen/telemetry.py` → `npu-smi info -t <typ> -i <card> -c <chip>`（禁止裸 `info -i`）
- **合流**：各 pod `*.jsonl` → `constitution128.merged.jsonl`

`record=card` 主字段与探针映射（出图直接读这些 key）：

| card 字段 | 探针 / 来源 | 配置要点 |
|-----------|-------------|----------|
| `func_tflops` | `func_perf` / GEMM | N=8192, bf16, warmup=20, iters=50（中位） |
| `hbm_gbps` | `hbm` | 1024 MB, warmup=20, iters=50 |
| `sustained_tflops` | `sustained` | 30s, window=50 |
| `vector_gflops` | `vector_fma_perf` | 64M elems, fp32, w20/i50 |
| `scalar_elems_per_s` | `scalar_chain_perf` | 16M elems, w10/i50 |
| `mte_gbps` | `mte_copy_perf` | 512 MB, w20/i50 |
| `cube_vector_tflops` | `cube_vector_pipeline` | n=4096, bf16, w20/i50 |
| `sfu_gflops` | `vector_sfu_perf` | 64M elems, op=exp, w20/i50 |
| `hbm_mode_*_gbps` | `hbm_modes_perf` | 512 MB, stride=16, w10/i30；四模式 seq_copy/strided/read_heavy/write_heavy |
| `launch_*` | `launch_latency` | samples=500, warmup=50, burst_count=64, timing_method=event |
| `health_temp_c` / `health_power_w` | `health` / `health_counters` | npu-smi（昇腾 NPU 系统管理命令行，可查功耗/温度/usages 等） |
| `board_temp_c` / `*_util_pct` / `power_w` | 探针 round 末次遥测 | 多为 vector_fma_round 末条回填到 card |
| `shape_sweep_peak_tflops` | **本批无 shape_sweep**；`jsonl.py` 用 `max(bnmk_tflops)` 回填 | 见 §1.9 / §3 |

明细行（部分图用）：

| record | 用途 |
|--------|------|
| `gemm_sustained_sample` | timeseries（iter, t_s, tflops, …） |
| `gemm_bnmk_sample` | BNMK 三图（B/M/N/K/label/tflops/dtype/windows…） |
| `gemm_shape_sample` | shape 曲线；本批 **0 行**（`shape_sweep.enabled=false`） |

merged 行数抽样：`card=128`, `gemm_sustained_sample=19898`, `gemm_bnmk_sample=1280`, `launch_latency=128`。

---

## 1. `card_constitution_20260711_figs/`（112 SVG）

**出图入口**：`plot_card_constitution.generate`（`--jsonl` 指向上述 merged；报告生成时间 2026-07-11 20:14）。  
**跳过**（`skipped.json`）：`aicore_freq_mhz` / `hbm_temp_c` / `power_limit_w` 全空；`gemm_shape_sample` 无 → 无 `shape_tflops_vs_n.svg`。

有数据指标共 23 个（各出 hist / heatmap_relmed / box_by_host / sorted_bar）；其中 CORE 11 个额外出 `bar_host_mean_std`；另 +7 scatter +1 box_overview +1 timeseries = **23×4 + 11 + 7 + 1 + 1 = 112**。

### 1.1 `hist_<metric>.svg`（23）

| 项 | 内容 |
|----|------|
| 函数 | `plot_card_constitution.plot_hist` |
| 数据 | `record=card` 的对应数值字段（见 §0.1） |
| 聚合 | 全集群 128 点直方图；红虚线 = **集群中位数**；bins=`min(30, max(5, n//4))` |
| 指标全集 | `func_tflops`, `hbm_gbps`, `sustained_tflops`, `vector_gflops`, `scalar_elems_per_s`, `mte_gbps`, `cube_vector_tflops`, `sfu_gflops`, `launch_sync_p{50,99}_us`, `launch_host_overhead_p{50,99}_us`, `launch_burst_p50_us`, `launch_burst_per_kernel_p50_us`, `health_temp_c`, `health_power_w`, `board_temp_c`, `aicore_util_pct`, `aicpu_util_pct`, `ctrlcpu_util_pct`, `mem_bw_util_pct`, `power_w`, `shape_sweep_peak_tflops` |

### 1.2 `heatmap_relmed_<metric>.svg`（23）

| 项 | 内容 |
|----|------|
| 函数 | `plot_card_constitution.plot_heatmap_relmed` |
| 数据 | `card.host` × `card.device` × 指标值 |
| 聚合 | 单元格 = **相对中位数偏差 %**：`(v - median) / median × 100`；median 为全集群该指标中位数 |
| 着色 | RdYlGn，`vmin/vmax = ±max(5, \|finite\| 的 p95)` |
| 标数规则 | **仅 \|Δ\| ≥ 1%** 才写 `+x.x` / `-x.x` |

### 1.3 `box_by_host_<metric>.svg`（23）

| 项 | 内容 |
|----|------|
| 函数 | `plot_card_constitution.plot_box_by_host` |
| 数据 | 按 `host` 分组的卡级指标 |
| 聚合 | 每 host 一箱（matplotlib boxplot）+ 半透明散点叠卡点；host 短名 `short_host_label` |

### 1.4 `sorted_bar_<metric>.svg`（23）

| 项 | 内容 |
|----|------|
| 函数 | `plot_card_constitution.plot_sorted_bar` |
| 数据 | 每卡一点；标签 `host:dN` |
| 聚合 | **按指标值升序**；橙虚线 = 集群中位数；≥median 与 <median 用不同 hatch |

### 1.5 `bar_host_mean_std_<metric>.svg`（11）

| 项 | 内容 |
|----|------|
| 函数 | `plot_card_constitution.plot_bar_host_mean_std` |
| 仅 CORE | `func_tflops`, `hbm_gbps`, `sustained_tflops`, `vector_gflops`, `mte_gbps`, `cube_vector_tflops`, `sfu_gflops`, `scalar_elems_per_s`, `power_w`, `health_power_w`, `health_temp_c`（见 `CORE_FOR_LAYOUT`；`aicore_freq_mhz` 本批空故无图） |
| 聚合 | 每 host：**均值 ± σ**（`np.mean` / `np.std`，总体标准差） |

### 1.6 `scatter_<x>_vs_<y>.svg`（7）

| 项 | 内容 |
|----|------|
| 函数 | `plot_card_constitution.plot_scatter` |
| 配对（`SCATTER_PAIRS`） | `func×vector`, `hbm×mte`, `power×func`, `health_power×func`, `power×hbm`, `health_power×hbm`, `launch_host_overhead_p50×ctrlcpu_util` |
| 聚合 | 每卡一点，按 host 着色（tab10）；缺任一轴则跳过 |

### 1.7 `box_overview.svg`（1）

| 项 | 内容 |
|----|------|
| 函数 | `plot_card_constitution.plot_box_overview` |
| 聚合 | CORE 中前 ≤6 个有数据指标，各画全集群单箱 |

### 1.8 `timeseries_sustained_p05_p50.svg`（1）— 新旧语义必读

| 项 | 内容 |
|----|------|
| 函数 | `plot_card_constitution.plot_sustained_timeseries` |
| 原始数据 | `record=gemm_sustained_sample`（字段 `iter`, `t_s`, `tflops`；本批 ~19898 行 / 128 卡） |
| 采集 | `sustained` 探针：`--sustained-s 30`，window=50，GEMM N 随 `--gemm-n 8192` |

**现行语义（20260711 本目录交付）**：

1. 按 `iter` 桶收集**全部卡**该步的 `tflops`。
2. 仅保留覆盖足够广的 iter：`len(vals) ≥ max(8, 0.9 × n_cards)`（128 卡 → ≥115）。
3. 对该 iter 的跨卡序列取 **p05**（排序后 index≈0.05）与 **p50（中位数）**。
4. 横轴 `t`：同 iter 上各卡 `t_s` 的**中位数**（不是单卡时钟）。
5. 图例明确为 `cross-card p05` / `cross-card p50`。

**旧语义（已废弃，曾写入 `research_constitution_viz_r0.md` §2.2）**：

> 「取 sustained 分位卡各一条」——即先按卡汇总 sustained，挑出一张「p05 代表卡」和一张「p50 代表卡」，再各自画单卡时序。

旧画法把**跨卡分位**误画成**两张固定卡的轨迹**，会把单卡噪声 / 相位差当成「集群 p05 曲线」。现行实现已在函数 docstring 写明纠正。

### 1.9 体质图已知坑

1. **`shape_sweep_peak_tflops` 名不副实**：`shape_sweep` 探针关闭，无 `gemm_shape_sample`；card 上该字段由 `jsonl.py` **回退为 `max(bnmk_tflops)`**，直方图/热力图画的是 BNMK 峰值，不是方阵 2 幂 sweep。
2. **`power_w` vs `health_power_w`**：前者多为负载探针末次遥测（可到 800–900W），后者偏 health 快照（常 ~160W）；两套 scatter 并存，勿混读。
3. **`aicpu_util_pct` 全 0**：直方图退化，但仍出图。
4. **`aicore_util_pct` / `mem_bw_util_pct` 含大量 0**：相对中位热力图会出现 −100% 格，属采样时刻问题，非「卡死」。

---

## 2. `constitution_extra_fillgap_20260711_figs/`（12 SVG）

**出图**：`plot_constitution_extra.generate`，同一 fillgap merged JSONL，stamp=`constitution_extra_fillgap_20260711`。

| 文件 | 函数 | 字段 / 规则 |
|------|------|-------------|
| `radar_host_median_norm.svg` | `plot_radar_and_parallel` | 每 host 对 `RADAR_METRICS` 取中位，再 ÷ 集群中位；极坐标；1.0=集群中位；ylim 0.85–1.15 |
| `parallel_host_median_norm.svg` | 同上 | 同一归一化矩阵的平行坐标 |
| `hbm_modes_grouped_bar.svg` | `plot_hbm_modes_grouped` | `hbm_mode_{seq_copy,strided,read_heavy,write_heavy}_gbps`；首组「全集群」中位 + 各 host 中位，四模式并排柱 |
| `corr_cube_vector_sfu_mte.svg` | `plot_corr_heatmap` | 四指标都非空的卡对齐后 Pearson `corrcoef`；Cube（矩阵计算单元：主计算核内专做大规模矩阵乘加、提供主算力的部件）/Vector（向量计算单元：做逐元素/向量运算与部分数学函数，灵活度高于矩阵单元、峰值算力通常更低）/SFU（特殊函数类吞吐代理；本探针默认 torch.exp，按 1 op/元素计，公开叙述常归在向量计算能力面）/MTE（Memory Transfer Engine，片上 Buffer 与 Global Memory 之间的数据搬运引擎；本字段多用 Tensor.copy_ 作纯搬运带宽代理，并非直接读该引擎指令计数器） |
| `box_launch_by_host.svg` | `plot_launch_boxplot` | `launch_sync_p99_us`, `launch_host_overhead_p99_us`, `launch_burst_p50_us` 按 host 箱线 |
| `cdf_core_metrics.svg` | `plot_cdf_panel` | `func/hbm/vector/mte/sfu/sustained` 六面板经验 CDF + 中位竖线 |
| `extreme10_small_multiples.svg` | `plot_extreme_cards` | 按 `sustained_tflops` 升序取最慢/最快各 10 卡；多指标画 **相对集群中位偏差 %** 水平条 |
| `heatmap_host_device_{vector_gflops,mte_gbps,sfu_gflops,scalar_elems_per_s}.svg` | `plot_host_device_heatmap` | **绝对值**热力图（非 relmed）；色标 YlOrRd，vmin/vmax=矩阵 p5/p95；**偏离中位 ≥0.5%** 才标数 |
| `scatter_sustained_vs_func.svg` | `plot_sustained_vs_func` | x=`func_tflops`, y=`sustained_tflops`；y=x 虚线 + 中位比 `sustained/func` 点线 |

**与 §1 差异**：extra 热力图画绝对值；constitution 热力图画 relmed%。雷达/平行坐标是 host 中位归一化，不是卡级。

---

## 3. `bnmk_shapes_20260711_figs/`（3 SVG）

| 项 | 内容 |
|----|------|
| 出图 | `plot_bnmk_shapes.py` → `plot_box_by_label` / `plot_bar_median_by_label` / `plot_host_heatmap` |
| 原始数据 | `record=gemm_bnmk_sample`；探针 `gemm_bnmk_sweep`（`stage_a.gemm_bnmk_sweep`） |
| 测量条件 | dtype=**bf16**, layout=**NN**, warmup=10, window=50, min_seconds=2, min_windows=3, max_seconds=6；10 个 shape（见 `config.constitution128.yaml` `bnmk_sweep.shapes`） |
| 样本字段 | `B,M,N,K,label,tflops,peak_tflops,windows,elapsed_s,dtype,layout,timing` |

| 文件 | 聚合规则 |
|------|----------|
| `bnmk_tflops_box_by_label.svg` | 按 `label` 分箱；x 轴按 **label 中位 TFLOPS 升序** |
| `bnmk_tflops_bar_median_by_label.svg` | 每 label **中位数**柱 + 柱顶标数；同序 |
| `bnmk_host_shape_heatmap.svg` | host × label；同 host+label 多卡取**均值** TFLOPS；色标 p5–p95；格内标整数 TFLOPS |

**已知坑（双计）**：`resolve_data_dir` 对 `results/**/*.jsonl` 递归加载。fillgap 目录同时有 `constitution128.merged.jsonl`（1280 行 bnmk）与 8 个 per-host jsonl（又 1280 行）→ 脚本报告 **2560 样本 / 每 label n=256**，实为 **1280 唯一 (host,device,label) 翻倍**。箱线/中位数值因重复拷贝不变，但 **n 与样本数虚高 2×**。应用 `--data-dir` 只指向 merged，或排除 `*/` 子目录。

---

## 4. `hccl_campaign_20260711_figs/`（23 SVG）

**出图**：`plot_hccl_campaign.py`（硬编码 `LOG_ROOT=logs/pipeline-comm-20260711_134811`）。  
**采集**：`scripts/cluster/launch_comm_kubectl.sh` → `hccl_torch_bench.py` / `hccl_p2p_bench.py` / `npu-smi info -t topo`。

### 4.1 Collective 公共条件

| 项 | 值 |
|----|-----|
| 启动 | `torchrun --nnodes={1,2,4,8} --nproc_per_node=16`（world=16/32/64/128） |
| 算子 | `all_reduce, all_gather, reduce_scatter, broadcast` |
| sizes | `1M,16M,64M,256M`（字节 1048576…268435456） |
| warmup / iters | **5 / 20**（bench 默认；launch 未改） |
| dtype | **fp32** |
| JSONL | `hccl-results/scale_{16,32,64,128}.jsonl`（由 `scale_*.rank*.jsonl` cat 合并） |
| record | `hccl_bench`：`op, world_size, rank, host, local_rank, nbytes, avg_s, alg_bw_GBps, bus_bw_GBps, dtype` |
| bus_bw 公式 | all_reduce: `alg × 2(n-1)/n`；all_gather/reduce_scatter: `alg × (n-1)/n`；broadcast: `alg`（`hccl_torch_bench.py`） |

### 4.2 Collective 图类

| 文件模式 | 函数 | 聚合 |
|----------|------|------|
| `hccl_bus_bw_vs_size_{op}.svg` ×4 | `plot_hccl_curves` | 每 (op, world, size) 对全体 rank 的 `bus_bw_GBps` 取 **中位数**，四 world 曲线 |
| `hccl_256mb_step_bus_bw.svg` | `plot_256mb_step_and_retention` | 固定 256MB；每 (op, world) **均值** bus_bw，四算子叠曲线 |
| `hccl_256mb_step_per_op.svg` | 同上 | 分算子阶梯 + 标数值 |
| `hccl_256mb_retention_bar.svg` | 同上 | 保持率 = `mean(world=W) / mean(world=16) × 100%`；分组柱 |
| `hccl_rank_violin_256mb_{op}.svg` ×4 | `plot_rank_distribution` | 256MB 下各 world 的 rank 级 bus_bw violin |
| `hccl_rank_box_256mb_all_ops.svg` | 同上 | 2×2 箱线 |
| `hccl_rank_hist_w{16,32,64,128}_256mb.svg` ×4 | 同上 | 每 world 四算子直方图 + 中位竖线 |

### 4.3 P2P 图类

| 项 | 值 |
|----|-----|
| 规模 | world=**16,128**（`P2P_SCALES`） |
| sizes | `64K,16M`；warmup=5, iters=20 |
| 策略 | ring + star；world≥64 时 bench 侧裁成 ring-only 防 rank0 堆积 |
| 边串行 | 全局一次一对；size 间 barrier |
| record | `hccl_p2p`：`src,dst,nbytes,bw_GBps,lat_us,role,world_size,…` |
| 去重 | `dedupe_p2p_edges` 按 `(world,src,dst,nbytes)` |

| 文件 | 函数 | 聚合 |
|------|------|------|
| `p2p_bw_violin_by_kind_size.svg` | `plot_p2p` | 边类型：星型(经 rank0)/环相邻/环闭合/跨节点块/其他；按 world 分面 violin |
| `p2p_box_compare_w16_w128_{65536,16777216}.svg` | 同上 | 同 size 下 16 vs 128 箱线 |
| `p2p_slow_edges_top15_16mb.svg` | 同上 | 16MB 边按 bw **升序** Top-15 |
| `p2p_fast_edges_top15_16mb.svg` | 同上 | 16MB 边按 bw **降序** Top-15（取排序末尾） |
| `p2p_kind_mean_compare_16mb.svg` | 同上 | 16MB、按边类型 **均值**，16 vs 128 并排柱 |

### 4.4 拓扑

| 项 | 内容 |
|----|------|
| 文件 | `topo_hccs_heatmap_master0.svg` |
| 函数 | `plot_topo_heatmap` |
| 原始 | `hccl-topo/raw/master-0.raw.txt`（`npu-smi info -t topo` 文本） |
| 聚合 | 解析 NPU×NPU 亲和字串 → 数值等级（SIO=5 … HCCS_SW=3.5 … X=0）；热力图；**Hs/HCCS_SW 不标字**，只标对角/`S` 等，减密网格噪声 |

---

## 5. `inter_bw_20260711_figs/`（1 SVG）

| 项 | 内容 |
|----|------|
| 文件 | `inter_vs_intra_bw.svg` |
| 出图 | `scripts/cluster/summarize_inter_bw.py`（`--plot`）；样式 `plot_style` |
| 原始 | `logs/inter-bw-20260711_141922/` → `all.jsonl` / `merged/probe.jsonl.rank*.jsonl` |
| record | `hccl_inter_bw`（`kind∈{intra,inter}`, `nbytes`, `bw_GBps`, `role`, `src/dst`, `hccl_buffsize`, `inflight`, …） |
| 采集 | `launch_inter_bw_kubectl.sh` + `hccl_inter_bw_probe.py`：`torchrun --nnodes=8 --nproc_per_node=16` |
| 条件 | sizes=`1M,16M,64M,256M`；warmup=**8**；iters=**30**；inflight=**4**；`HCCL_BUFFSIZE=**2048**`；严格串行（全员 barrier，每轮一对） |
| intra pairs | 同节点 `(0,1)(0,8)(7,8)(1,9)` 双向 |
| inter pairs | 相邻节点环 + 跨跳；`local_rank∈{0,5,10,15}` 对齐，双向 |
| 汇总规则 | 优先取 **recv** 侧（`role==recv`）；按 `(kind, nbytes)` 桶取 **median_GBps**；分组柱 intra vs inter |

交叉验证批次（不出本目录图）：`logs/inter-bw-20260711_142537/` ping-pong。

---

## 6. 指标字段 → 探针速查（体质系）

| 出图 metric key | JSONL record | 探针名 | 关键参数 |
|-----------------|--------------|--------|----------|
| `func_tflops` | card ← func_perf | GEMM | N=8192 bf16 w20/i50 |
| `hbm_gbps` | card ← hbm | HBM（High Bandwidth Memory，器件高带宽外存） copy | 1024MB w20/i50 |
| `sustained_tflops` + timeseries | card + gemm_sustained_sample | sustained | 30s window=50 |
| `vector_gflops` | card ← vector_fma | vector FMA | 64M fp32 |
| `scalar_elems_per_s` | card ← scalar_chain | scalar | 16M |
| `mte_gbps` | card ← mte_copy | MTE（Memory Transfer Engine，片上 Buffer 与 Global Memory 之间的数据搬运引擎；本字段多用 Tensor.copy_ 作纯搬运带宽代理，并非直接读该引擎指令计数器） | 512MB |
| `cube_vector_tflops` | card ← cube_vector_pipeline | cube+vector | n=4096 bf16 |
| `sfu_gflops` | card ← vector_sfu | SFU（特殊函数类吞吐代理；本探针默认 torch.exp，按 1 op/元素计，公开叙述常归在向量计算能力面） exp | 64M |
| `hbm_mode_*` | card ← hbm_modes | HBM（High Bandwidth Memory，器件高带宽外存） 四模式 | 512MB stride=16 |
| `launch_*` | card ← launch_latency | launch | 500 samples, burst=64, event |
| `health_*` / util / `power_w` | card ← health + round telemetry | npu-smi（昇腾 NPU 系统管理命令行，可查功耗/温度/usages 等） `-t` | |
| `shape_sweep_peak_tflops` | card（本批=BNMK max） | bnmk_sweep 回填 | 见 §1.9 |
| bnmk 图 | gemm_bnmk_sample | gemm_bnmk_sweep | 10 shapes bf16 NN |

---

## 7. 审计结论（可操作）

1. **体质 112+12 图**与 fillgap `constitution128.merged.jsonl` 一一对应；采集命令与 yaml 参数可复现。  
2. **timeseries 已纠正为跨卡按-iter 分位**；引用旧 md/旧 PNG 时勿沿用「两张代表卡」解读。  
3. **BNMK 三图存在 merged+per-host 双计**；数值中位无偏，样本数需 ÷2。  
4. **`shape_sweep_peak_*` 体质图实际是 BNMK peak**；方阵 sweep 曲线本批未出。  
5. **HCCL/P2P/inter_bw** 路径、warmup/iters、dtype、BUFFSIZE、串行策略均已钉死在 launch/bench；bus_bw 用标准 NCCL/HCCL 折算系数，出图侧对 collective 曲线用中位、256MB 阶梯/保持率用均值。

---

## 附录 · 图文件计数核对

```text
card_constitution_20260711_figs/          112 svg
constitution_extra_fillgap_20260711_figs/  12 svg
bnmk_shapes_20260711_figs/                  3 svg
hccl_campaign_20260711_figs/               23 svg
inter_bw_20260711_figs/                     1 svg
合计                                      151
```
