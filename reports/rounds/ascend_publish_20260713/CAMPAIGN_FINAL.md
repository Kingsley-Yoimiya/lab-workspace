# Ascend A3 96 卡战役终稿（2026-07-13）

> 对应计划：`ASCEND-8x16-CLUSTER-EXPERIMENT-PLAN.md`  
> 租约：`montyyin-moe96-r2`（6×16）  
> AFS 发布目录：`/afs-a3-241ceshi-shared/montyyin/results/reports/ascend_campaign_20260713/`

## 一句话结论

**Block A 主假设已基本印证**：`network_contrib = gap_real − gap_indep` 在节点边界 **N=16→32** 出现抬升；跨节点后网络侧贡献平台在 ~1.3–1.4 ms。  
**Block E 双信号 PASS**（抢占卡 14/15 同时被 step 排名与 AICore 命中）。  
**Block D**：在真实同步下「注入卡 vs 其他卡 median」会被集合通信拉齐（表观 WEAK）；按 **delayed iter 的全局 step 抬升** 重判为 **PASS_SYNC_EFFECT**（~+1974 ms ≈ DELAY_MS）。

---

## Block A：双轨分解（P0，主证据）

Stamp indep/real-16：`20260713_110548`；real 32/64/96 复用 `dense_failslow_gbsprop/20260713_071316`。

| N | gap_indep (ms) | gap_real (ms) | network_contrib (ms) | 跨节点 |
|---|---:|---:|---:|---|
| 8 | 0.29 | — | — | 半节点 |
| 16 | 0.53 | 1.56 | **1.02** | 否 |
| 32 | 0.58 | 1.97 | **1.40** | 是 |
| 64 | 1.15 | 2.43 | 1.28 | 是 |
| 96 | 1.26 | 2.67 | **1.40** | 是 |

**16→32**：indep +0.04，network_contrib **+0.37**，gap_real +0.42 → 抬升主因是网络侧，不是纯计算抖动。

图：`blockA_gap_decomp.svg`  
CSV：`network_contrib_final.csv` / `gap_indep.csv`

采集：indep=`virtual_sync_bench_npu.py`（无 HCCL）；real=Dense TP4PP2 GBS∝DP + `failslow_step_timer.py`。

局限：indep 步长 ~81 ms，real ~19–21 s，绝对毫秒相减是计划定义的一阶证据；外推到更大 N 看斜率而非相对百分比。

---

## Block E：多信号外部抢占（stamp `20260713_113423`）

- 注入：`npu_busy_preempt.py` 打卡 14/15，120s；并行 `npu-smi info` 采样。
- **dual_signal_verdict: PASS**
  - step 最慢四卡含 **14、15**
  - AICore 均值 Top2 = **15 (46.6%)、14 (46.0%)**，明显高于其余卡
- 产物：`dual_signal.json`

---

## Block D：延迟注入（修 timer 后）

Timer 修复：`delay` 移入 `perf_counter` 计时窗内（旧 bug 会导致注入卡 ms 反而偏小）。

### D1 rematch：`DELAY_RANKS=12-15`（与 Exp4 并行）

- `delayed_counts`：12–15 各 8 次，注入确实打上。
- 注入卡 vs 其他卡 median 差仅 ~2 ms → 表观 WEAK（**同步拉齐墙钟，属预期**）。
- **delayed vs normal iter**：
  - step 中位数：20129 → **22103**（**+1974 ms**）
  - gap 几乎不变（~1.7→1.8 ms）
  - 重判：**PASS_SYNC_EFFECT**（延迟被整步吸收，证明真实同步下局部慢会抬全局步长）

产物：`rank_contrast.json`、`exp3_delayed_iter_analysis.json`

### D2 AB：PP stage inject（worker-0，`20260713_113423_ab`）

| | baseline | inject | Δ |
|---|---:|---:|---:|
| global median (ms) | 18792 | 18837 | +45 |
| stage spread (ms) | 1.86 | 2.11 | +0.25 |

判读仍偏弱：短窗 + PP=2 下 stage 掩蔽信号不明显；全局几乎不动符合「切片掩蔽」叙事的一半，但 stage_spread 未显著拉开。建议更长窗或更大 `DELAY_MS` / 真实多 stage PP 再加做。

产物：`pp_inject_ab.json`

---

## Block C：功耗/利用率热力图

采样窗 `20260713_111100`（当时仅 real-16 占 master）→ master 忙 / 其余闲，采集链路验证通过。  
图：`blockC_power_aicore_heatmap.svg`（不宜单独当作「全负载均匀悖论」终局图）。

---

## 未做 / 后续

- Block F 排布缓解
- Block B Megatron collective 分段计时（若框架未暴露则跳过）
- 全负载 96 卡 Block C 重采
- Block D 更强 PP 配置加做

---

## 路径索引

| 内容 | 路径 |
|---|---|
| 本发布包（本地） | `reports/rounds/ascend_publish_20260713/` |
| AFS 发布包 | `/afs-a3-241ceshi-shared/montyyin/results/reports/ascend_campaign_20260713/` |
| Block A 详报 | `reports/rounds/blockA_20260713_110548/SUMMARY.md` |
| Exp45 | `.../results/exp45_parallel/20260713_113423/` |
| Exp3 AB | `.../results/dense_pp_inject/20260713_113423_ab/` |
