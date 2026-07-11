# Track B：Shape 多测矩阵 R0

**日期**: 2026-07-11  
**状态**: 调研 + 可落地配置草案  
**锚点数据**: `reports/rounds/card_screen_diff_r1.md`（现有 4 BNMK）  
**配置交付**: `projects/CARD_SCREEN/config.constitution128_shapes.yaml`

---

## 1. 结论摘要

1. 现有 4 个 BNMK 只覆盖「方阵锚点 / LLaMA 风 FFN / batched 方阵 / tall-skinny」四象限；**训练真实面**还缺 Qwen 系非 2 幂 FFN、Attention head_dim、microbatch/decode、短 K、宽短阵等。
2. **推荐分工**：`shape_sweep` 继续扫 **2 的幂方阵曲线**（吞吐 vs N）；`bnmk_sweep` 负责 **训练代理 (B,M,N,K)**。二者互补，勿用 bnmk 重复扫整条 2 幂方阵。
3. **P0 = 10 shape**（含原 4）：16 卡冒烟约 **+0.4–0.7 min/卡**；128 全量约 **+0.5–1.0 min/卡**（相对现有 4 shape；见 §5）。
4. constitution 默认仍可关 shape/bnmk；需要烤机多测时改用 `config.constitution128_shapes.yaml`（冒烟档 / 全量档）。

---

## 2. 现状与缺口

### 2.1 现有 4 shape（R1 已跑）

| (B,M,N,K) | 意图 | R1 中位 TFLOPS |
|-----------|------|---------------:|
| `(1,8192,8192,8192)` | 方阵锚点（对齐 `func_perf` N=8192） | 277.6 |
| `(1,4096,4096,11008)` | FFN-like（LLaMA-7B 风 intermediate=11008） | 277.3 |
| `(8,2048,2048,2048)` | batched 方阵 | 308.8 |
| `(1,16384,1024,1024)` | tall-skinny | 310.2 |

来源：`scripts/cluster/run_card_screen_128.sh` 内嵌 `bnmk_sweep.shapes`；报告见 `card_screen_diff_r1.md`。

### 2.2 代码侧分工

| 探针 | 实现 | 覆盖 |
|------|------|------|
| `shape_sweep` | `stage_a.sweep_shapes` → 方阵 `n×n`；2 的幂 + 可选端点 | 吞吐–边长曲线 |
| `bnmk_sweep` | 配置已在扇出脚本；`(B,M,N,K)` + `flops=2*B*M*N*K` | 训练代理非方阵 / batch |
| `func_perf` / `sustained` | 固定方阵 N=8192 | 正确性门控 + 稳态 |

R1 二次调研已点名：**Attention 代理（head_dim=128）**、拉长 window、layout/dtype 网格——本 R0 把 Attention / Qwen FFN / microbatch 收进 P0/P1 清单。

### 2.3 模型维度依据（推导 GEMM）

训练里线性层可写成 `C[B,M,N] = A[B,M,K] @ W[K,N]`（或等价转置），其中 `M ≈ tokens/GPU`（microbatch×seq），`K/N` 来自 hidden / intermediate / head_dim。

| 模型 | hidden | intermediate | 备注 |
|------|-------:|-------------:|------|
| Qwen2.5-72B | 8192 | 29568 | [HF config.json](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json) |
| Qwen2.5-7B | 3584 | 18944 | [Ollama qwen2.5:7b metadata](https://ollama.com/library/qwen2.5:7b-instruct/blobs/2bada8a74506)（`embedding_length` / `feed_forward_length`） |
| Qwen3-8B 对齐例 | 4096 | 12288 | [SWIFT VL 实践](https://swift.readthedocs.io/en/v3.8/BestPractices/Rapidly-Training-VL-model.html)（3584→4096，18944→12288） |
| 现有 FFN 代理 | 4096 | 11008 | R1 已测（LLaMA-7B 风） |

Attention：常见 `head_dim=128`；score/context 代理可用短 K 或 batched `(heads, S, D, D)` 风格 GEMM（见 R1 §4.2）。Transformer 层内 MLP/Attention 的列切分 GEMM 是 Megatron 类训练的主算子面（[Megatron-LM](https://github.com/nvidia/megatron-lm)）。

---

## 3. 推荐 Shape 清单

### 3.1 P0 必测（10 条，含原 4）

耗时按 `min_seconds≈2–3`、`max_seconds≈6–8`、`window=50` 估算；大 shape 多被 **时间下限** 卡住，小 shape 也因 FLOP 小而贴下限 → **单 shape 墙钟 ≈ 2–6 s**。

| # | (B,M,N,K) | 类别 | 为何测 | 预估耗时 |
|---|-----------|------|--------|----------|
| 1 | `(1,8192,8192,8192)` | 方阵锚点 | 与 `func_tflops` 对齐；跨轮可比 | ~2–4 s |
| 2 | `(1,4096,4096,11008)` | FFN（LLaMA 风） | R1 基线；非方阵 K | ~2–3 s |
| 3 | `(8,2048,2048,2048)` | batched | 批维调度 / 启动摊销 | ~2–3 s |
| 4 | `(1,16384,1024,1024)` | tall-skinny | 带宽敏感；R1 高位 TFLOPS 层 | ~2–3 s |
| 5 | `(1,4096,18944,3584)` | FFN up/gate（Qwen2.5-7B） | 非 2 幂 intermediate；tiling 余数 | ~2–4 s |
| 6 | `(1,4096,3584,18944)` | FFN down（Qwen2.5-7B） | 与 #5 对偶（N↔K）；epilogue 面不同 | ~2–4 s |
| 7 | `(1,4096,12288,4096)` | FFN（Qwen3-8B 风） | 较「整齐」的 3×hidden；对照 #2/#5 | ~2–3 s |
| 8 | `(1,4096,4096,128)` | Attention 代理 | head_dim=128 短 K；R1 候选 | ~2–3 s（贴下限） |
| 9 | `(1,8,8192,8192)` | microbatch / decode | 极小 M；launch/尾部开销主导 | ~2–3 s（贴下限） |
| 10 | `(1,1024,16384,1024)` | wide-short | 与 #4 对偶（M↔N）；宽输出 | ~2–3 s |

**P0 单卡合计**：约 **25–45 s（0.4–0.75 min）**；相对现有 4 shape 额外约 **+15–25 s（+0.25–0.4 min）**。

### 3.2 P1 扩展（按需加，默认全量档可选注释）

| # | (B,M,N,K) | 类别 | 为何测 | 预估耗时 |
|---|-----------|------|--------|----------|
| 11 | `(1,2048,29568,8192)` | FFN（Qwen2.5-72B） | 大 intermediate；显存 ~0.6 GB bf16 | ~3–5 s |
| 12 | `(32,4096,128,128)` | batched Attention | 多 head 批；与 #8 对照 | ~2–3 s |
| 13 | `(1,1,8192,8192)` | 极端 decode | M=1 病态；易暴露计时噪声 | ~2–3 s |
| 14 | `(1,8192,8192,256)` | short-K | 介于 Attention 与方阵之间 | ~2–3 s |
| 15 | `(1,4096,14336,4096)` | 非 2 幂「怪」FFN | 14336=7×2048；tiling 余数 | ~2–4 s |
| 16 | `(1,2048,32000,4096)` | vocab-lite | 词表投影缩水版（全 vocab 易 OOM） | ~2–4 s |

**P0+P1（16）单卡**：约 **0.7–1.3 min**。

### 3.3 明确不进默认矩阵

| Shape 想法 | 原因 |
|------------|------|
| 全 vocab `(…, 152064, …)` | 显存与时长爆炸 |
| layout×dtype 全网格 | 组合爆炸；单卡抽测即可（R1 §4） |
| 与 `shape_sweep` 重复的整条 2 幂方阵 | 已由方阵扫覆盖 |

---

## 4. 与 shape_sweep / bnmk_sweep 如何组合

```
┌─────────────────────┐     ┌──────────────────────────────┐
│ shape_sweep         │     │ bnmk_sweep                   │
│ 方阵 n∈{128…2^k…}   │     │ 训练代理 (B,M,N,K) 列表      │
│ 目的：曲线 / 峰值 N │     │ 目的：卡间 diff 按 shape 切片 │
│ 判定：中立          │     │ 判定：中立（先观测）         │
└─────────────────────┘     └──────────────────────────────┘
         │                              │
         └──────────┬───────────────────┘
                    ▼
         报告：方阵曲线 + 每 BNMK 热力图/CV
```

| 场景 | shape_sweep | bnmk_sweep |
|------|-------------|------------|
| 日常 constitution128 | **关**（控时） | **关** |
| 16 卡冒烟（多测） | 可选短扫 `stop:8192,max_seconds:4` | **P0 冒烟子集 6 条** |
| 128 全量多测 | 可选 `max_seconds:6–8` 或关 | **P0 全 10 条**；P1 注释备用 |
| 纯 diff 复现 R1 | 可关 | 至少原 4 条 |

**勿重复**：bnmk 已含 `(1,8192,8192,8192)` 时，不必为「再测一次 8192 方阵」单独加 bnmk；`shape_sweep` 的价值是 **多 N 曲线**，不是单点。

**时长封顶公式（单卡）**：

\[
T_{\text{bnmk}} \lesssim N_{\text{shapes}} \times \texttt{max\_seconds}
\]

\[
T_{\text{shape}} \lesssim N_{\text{pow2}} \times \texttt{max\_seconds}
\]

冒烟建议总 shape 相关预算 **≤ 2 min/卡**；全量 **≤ 3–4 min/卡**（含可选 shape_sweep）。

---

## 5. constitution128 建议

### 5.1 默认体质轮（现有 `config.constitution128.yaml`）

- 维持 `shape_sweep` / `bnmk_sweep` **enabled: false**（正交探针优先，控墙钟）。
- 在文件头注释指向 `config.constitution128_shapes.yaml`。

### 5.2 多测轮（新配置）

| 档 | 开哪些 | max_seconds | 单卡 shape 相关预估 |
|----|--------|-------------|---------------------|
| **smoke16** | bnmk P0 子集 6 条；shape_sweep **关** | bnmk 4 | **~0.4–0.5 min** |
| **full128** | bnmk P0 全 10 条；shape_sweep **关**（或短扫） | bnmk 6 | **~0.5–1.0 min** |
| **full128+curve** | 上 + shape_sweep `128→8192`（7 点） | shape 6 / bnmk 6 | **~1.5–2.5 min** |

smoke16 子集建议：`#1,#2,#4,#5,#8,#9`（锚点 + FFN 新旧 + tall + attn + micro）——用最少条数摸 CV/噪声。

### 5.3 额外分钟数（相对「仅原 4 BNMK」）

| 档 | shapes | 相对原 4 的额外 | 绝对墙钟（仅 bnmk） |
|----|-------:|----------------:|--------------------:|
| 原 R1 | 4 | 0 | ~0.2–0.5 min |
| smoke16 P0 子集 | 6 | **+0.1–0.2 min** | ~0.4–0.5 min |
| full128 P0 | 10 | **+0.25–0.5 min** | ~0.5–1.0 min |
| +P1 | 16 | **+0.5–0.9 min** | ~0.7–1.3 min |

**返回口径（给调度）**：P0 全量相对现网 4 shape，**单卡额外约 0.3–0.5 分钟**；冒烟子集额外约 **0.15 分钟**。

---

## 6. 落地文件

- `projects/CARD_SCREEN/config.constitution128_shapes.yaml`  
  - `tier: smoke16` / `full128` 用注释块切换；默认 **smoke 安全时长**。  
  - `probes.bnmk_sweep.enabled: true`；`shape_sweep` 默认 false。  
- `config.constitution128.yaml` 增加指向注释（不改默认开关）。

**实现缺口备注**：本树 `stage_a.py` 以方阵 `gemm_shape_sweep` 为主；R1 扇出已使用 `bnmk_sweep` schema。若当前 checkout 缺 `gemm_bnmk_sweep` 注册，合入探针后再挂本 YAML（schema 已与 `run_card_screen_128.sh` 对齐）。

---

## 7. 验收建议

1. 单卡：`python screen.py --device 0 --config config.constitution128_shapes.yaml`，确认 6/10 条均有 `gemm_bnmk_sample`。  
2. 16 卡冒烟：看各 shape CV；小 M（#8/#9）若 CV 高，先加长 `min_seconds` 再下结论（R1 教训）。  
3. 128 全量：复用 `gen_card_screen_diff_r1.py` 按 shape 出热力图。

---

## Sources

- [Qwen2.5-72B-Instruct config.json](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json) — hidden=8192, intermediate=29568  
- [qwen2.5:7b-instruct model blob](https://ollama.com/library/qwen2.5:7b-instruct/blobs/2bada8a74506) — embedding=3584, FFN=18944  
- [SWIFT Rapidly Training VL](https://swift.readthedocs.io/en/v3.8/BestPractices/Rapidly-Training-VL-model.html) — Qwen3-8B 维：4096 / 12288 / head_dim=128  
- [NVIDIA Megatron-LM](https://github.com/nvidia/megatron-lm) — 训练层内 GEMM / 并行切分语境  
- 内部：`reports/rounds/card_screen_diff_r1.md`，`reports/research/research_card_diff_r0.md`
