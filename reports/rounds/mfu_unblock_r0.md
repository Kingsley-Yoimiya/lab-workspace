# Track C C0 解阻报告：真 Qwen3-8B Dense 16 卡短跑

> 作者：Cursor Grok 4.5  
> 日志：`logs/mfu-unblock-r0-manual-20260711_002854`  
> AFS：`/afs-a3-241ceshi-shared/montyyin/logs/train-dense-manual/scale_16e/`  
> Peak：`peak_from_card_screen.py` → **292.79 TFLOPS/卡**（勿用 320）

## 1. 门禁清单

| 项 | 状态 |
|----|------|
| Wrapper 修 `/afs-grj`、`\| tee`、dense≠MoE | 通过（`wrappers/train_qwen3_8B_ascend.sh`） |
| 16 卡短跑可解析 throughput | **通过** |
| Peak 用 CARD_SCREEN 中位数 | **292.79** |
| Probing | rustc OK；`probing` 二进制仍缺（需 `make develop`） |

## 2. 排障路径（初步问题已排除）

1. `set_env.sh` 内含 `exit` → 不可在 `set -e` 下 source；改用 `bash -lc`  
2. 默认 `python3` 无 mindspeed → 强制 `llm_test` PATH + `PYTHONPATH=/MindSpeed-LLM/MindSpeed`  
3. TensorBoard 版本过旧 → `pip install tensorboard` + `SKIP_TB=1`  
4. `TB_ARG` 定义顺序错误 → 已修

## 3. 16 卡 Dense 结果（Qwen3-8B wrapper，TP=2 PP=2，GBS=128，seq=4096，5 iter）

| iter | ms/iter | TFLOP/s/GPU |
|-----:|--------:|------------:|
| 1 | 13080 | 72.4（冷启动） |
| 2 | 6829 | 138.7 |
| 3 | 6799 | 139.3 |
| 4 | 7302 | 129.7 |
| 5 | 6781 | **139.6** |

**稳态 MFU（相对 CARD_SCREEN peak）**  
`139.6 / 292.79 ≈ **47.7%**`

对比：上一轮微基准 ~10% MFU **不可外推**；真训练已进入可用量级。

## 4. 下一假设（进入 C1 优化环）

1. 冷启动 iter1 慢 2× → 忽略首 iter 报 MFU；或加 warmup  
2. 试 **TP=1 PP=1** 或 **TP=4 PP=1** 看通信/算力比  
3. 装 Probing 二进制后注入 step/comm 观测  
4. MoE 30B-A3B 16 卡短跑对照
