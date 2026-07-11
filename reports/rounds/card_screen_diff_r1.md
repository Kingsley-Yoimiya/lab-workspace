# Card Screen Diff R1（BNMK）— 128 卡全量

> 作者：Cursor Grok 4.5  
> 数据：`logs/card-diff-r1-20260710_234740/results`（AFS `card_screen-20260710_234742-perf128bnmk`）  
> 冒烟对照：`logs/card-diff-r1-smoke16-20260710_234407`  
> 图：`reports/rounds/card_screen_diff_r1_figs/`（每 shape 热力图 + sorted bars）  
> 调研：`reports/research/research_card_diff_r0.md`

## 1. 结论摘要

1. **128/128 卡 `verdict=good`**；`gemm_bnmk_sample` 共 **512** 条（128×4 shape）。
2. 不同 BNMK 的**绝对算力中位不同**（方阵/FFN ~277 TFLOPS，batched/tall-skinny ~309–310），说明单一 `func_tflops(N=8192)` 不能代表训练算子面。
3. 本轮 128 卡上四 shape 的 **CV 均很小（0.06%–0.15%）**，卡间相对偏差热力图接近「平」——与冒烟 16 卡上 FFN-like CV≈3% 不同，提示：**短跑/冷启动噪声会夸大差异**；全量稳态后差异面更平，但仍需按 shape 切片看绝对值分层。
4. Diff-first 交付齐全：每 shape 均有 host×device 相对中位数热力图 + sorted bars + `stats.json`。

## 2. 128 卡 BNMK 统计

| (B,M,N,K) | n | 中位 TFLOPS | CV% |
|-----------|--:|------------:|----:|
| (1,8192,8192,8192) | 128 | 277.6 | 0.15 |
| (1,4096,4096,11008) | 128 | 277.3 | 0.13 |
| (8,2048,2048,2048) | 128 | 308.8 | 0.13 |
| (1,16384,1024,1024) | 128 | 310.2 | 0.06 |

图例（相对中位数热力图）：

- `heatmap_host_device_relmed_B1_M8192_N8192_K8192_NN_bf16.png`
- `heatmap_host_device_relmed_B1_M4096_N4096_K11008_NN_bf16.png`
- `heatmap_host_device_relmed_B8_M2048_N2048_K2048_NN_bf16.png`
- `heatmap_host_device_relmed_B1_M16384_N1024_K1024_NN_bf16.png`

## 3. 冒烟 vs 全量

| 观察 | 16 卡冒烟 | 128 卡全量 |
|------|-----------|------------|
| FFN-like CV | ~3.07% | ~0.13% |
| tall-skinny 中位 | ~310 | ~310 |
| 解读 | 短 sustained/少 window 放大尾部 | 稳态后卡间更齐；差异主要体现在 **shape 间绝对层** |

## 4. 二次调研（R2 候选，未锁死）

1. **拉长 bnmk window / 多轮重复**：验证冒烟高 CV 是否可复现为真慢卡，还是噪声。
2. **Attention 代理 shape**（head_dim=128，S=4096）：是否出现新的相对偏差斑块。
3. **layout NT/TN** 与 **dtype fp16**：慢卡集合是否与 NN/bf16 一致。
4. 将 BNMK 中位写入 MFU peak 分母的「按层加权」方案（Track C 用）。

## 5. 代码锚点

- 探针：`CARD_SCREEN/card_screen/probes/stage_a.py` → `gemm_bnmk` / `gemm_bnmk_sweep`
- 扇出：`scripts/cluster/run_card_screen_128.sh`（`bnmk_sweep.enabled`）
- 出图：`reports/gen_card_screen_diff_r1.py`
