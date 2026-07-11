# Track C C0 解阻笔记（mfu_unblock_r0）

> 草稿骨架；最终叙事由父 agent 撰写。  
> 日期：2026-07-10  
> Job：`huawei-8node-copy`

## 1. 阻塞根因（已核实）

| # | 问题 | 证据 |
|---|------|------|
| 1 | `/afs-grj` 硬编码 | PT / muxi-30B 脚本；真实路径 `/afs-a3-241ceshi-shared/geruijun/` |
| 2 | `torchrun ... 2>&1` 后 `| tee` 缺行续接 | `PT_qwen3_32B.sh` / `PT_qwen3_30B_A3B.sh` 末尾 |
| 3 | **dense 误用 MoE 脚本** | `PT_qwen3_32B.sh` 与 `PT_qwen3_8B.sh` 均含 `NUM_EXPERTS=128` |
| 4 | muxi 原版偏 MACA | `MACA_PATH` / `MCCL_*`；不可直接在 Ascend 跑 |
| 5 | 旧 `run_train_mfu_scale.sh` | dense→`PT_qwen3_32B`（实为 MoE）；WORLD_SIZE 语义混乱 |

**健康基线（muxi 模型配置）**

- Dense：`muxi-Megatron/.../train_qwen3_8B.sh`（无 experts，layers=20, hidden=4096）
- MoE：`train_qwen3_30B_A3B.sh`（NUM_EXPERTS=128）

## 2. 本轮已修复 / 交付

- [x] `scripts/cluster/wrappers/train_qwen3_8B_ascend.sh` — Ascend dense wrapper（muxi 8B 配置）
- [x] `scripts/cluster/wrappers/train_qwen3_30B_A3B_ascend.sh` — Ascend MoE wrapper
- [x] `scripts/cluster/wrappers/patch_pt_train.sh` — PT 回退：修 `/afs-grj` + `| tee` + 分布式变量
- [x] `scripts/cluster/run_train_mfu_scale.sh` — 默认 `SOURCE=wrapper`；正确 `NNODES/RANK/MASTER_*/NPUS_PER_NODE=16`；上传 wrappers 到 AFS
- [x] `scripts/cluster/peak_from_card_screen.py` — 读 card-screen median `func_tflops` 作 MFU 分母
- [x] `scripts/cluster/check_probing_afs.sh` — vcctl 检查 rustc / Probing_plus 树，写 status
- [x] 本笔记骨架

### Peak 校准（本机已跑）

默认 `logs/card-screen-128-20260710_224218/results/perf128.cluster.json`：

- `medians.func_tflops` = **292.787744 TFLOPS/卡**
- 16 卡 denom = **4684.60 TFLOPS**（勿再盲用 320）

### Probing 就绪（已跑 `check_probing_afs.sh`）

- `STATUS=READY` / `PROBING_AFS_READY=1`
- `rustc 1.97.0`、`cargo` OK（`rust-env.sh`）
- Probing_plus / probing 源码树 OK
- `probing` 二进制仍缺（需 `MODE=develop ./scripts/cluster/run_probing_plus.sh`）
- status 日志：`logs/probing-check-*/probing_afs_status.txt`

### AFS 上传

wrappers 已写入 `/afs-a3-241ceshi-shared/montyyin/lab-workspace/scripts/cluster/wrappers/`；`patch_pt_train.sh` 对 PT_30B 验证：`NO_AFS_GRJ` + `PATCH_BASH_OK`。

## 3. 仍需 live 验证（C0 门禁）

- [ ] **16 卡短跑 smoke**（必做）  
  ```bash
  MODE=dense SCALES=16 TRAIN_ITERS=5 \
    ./scripts/cluster/run_train_mfu_scale.sh
  ```
  成功判据：rank0 日志出现 iteration / throughput；无 `| tee` 语法错误；无 `/afs-grj` FileNotFound。
- [ ] 从 smoke 日志抽取 tokens/s 或 TFLOP/s，用 `peak_from_card_screen.py --world-size 16` 算 MFU
- [ ] （可选）`MODE=moe SCALES=16 TRAIN_ITERS=5 MASTER_PORT=24700`
- [ ] （可选）`SOURCE=pt` 仅验证 patch 路径，**不可**当作 dense 结论
- [ ] 通过后再开 32/64/128 与更长 `TRAIN_ITERS`

## 4. 推荐命令速查

```bash
# Peak
python3 scripts/cluster/peak_from_card_screen.py --world-size 16

# Probing
./scripts/cluster/check_probing_afs.sh

# 首轮 smoke（dense 16 卡 × 5 iter）
MODE=dense SCALES=16 TRAIN_ITERS=5 \
  ./scripts/cluster/run_train_mfu_scale.sh
```

## 5. 风险 / 备注

- Wrapper 默认 `SKIP_SAVE=1` `SKIP_PROFILE=1`，缩短 smoke。
- Dense TP=2 PP=2；MoE TP=1 EP=4 PP=4（适配 16 NPU/节点）。
- `distributed-backend nccl` 保留（MindSpeed adaptor → HCCL）。
- 上一轮 `train-dense-20260710_225402` 全档 FAIL，与错误 BASE + tee/路径一致；本轮应用 wrapper 后重测。
