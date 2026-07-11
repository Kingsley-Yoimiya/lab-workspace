# 体质 / 烤机 R4 合并摘要（多 subagent 后）

**日期**: 2026-07-11  
**任务表**: [`research_constitution_r4_tracks.md`](research_constitution_r4_tracks.md)

---

## 已拍板

| 项 | 决定 |
|----|------|
| Shape | **直接 10 个 BNMK**（full128），写入 `config.constitution128.yaml` |
| 通信 | **烤机跑完接着跑**，同一 list 串行 |
| 入口 | `scripts/cluster/run_constitution_then_comm.sh` |

流水线顺序：

```
1 constitution128（10 shape + HBM 多模式 + Stage C + SDC）
2 plot_card_constitution.py（有 results 时）
3 HCCL collective scale 16/32/64/128
4 HCCL P2P 16 → 128（默认 ring）
```

---

## Track A — 可视化（Grok）✅

| 产物 | 路径 |
|------|------|
| 方案 | `research_constitution_viz_r0.md` |
| 脚本 | `reports/plot_card_constitution.py` |

```bash
python3 reports/plot_card_constitution.py \
  --data-dir logs/card-screen-128-20260710_224218/results \
  --out-dir reports/rounds
```

---

## Track B — Shape 矩阵（Grok）✅

P0 **10 shape** 已并入主配置 `config.constitution128.yaml`（`bnmk_sweep.enabled: true`）。

---

## Track C — HBM 多模式（Grok）✅

`hbm_modes_perf`：seq_copy / strided / read_heavy / write_heavy（constitution 已开）。

---

## Track D — 功耗×性能（Grok + Sonnet）✅

合并采纳：Grok 图清单 + Sonnet ramp/steady 门禁；前置修遥测。

---

## Track E — 卡间通信（Grok + Sonnet）✅

烤机后同窗口接着跑；报告分轨产出，再索引合并。  
P0：保持率重算、环境卫生、正确性校验、P2P 边矩阵。

---

## 集群通了之后

```bash
# 整条 list
CLUSTER_JOB=<running-job> ./scripts/cluster/run_constitution_then_comm.sh

# 或分段
SKIP_COMM=1 ./scripts/cluster/run_constitution_then_comm.sh
SKIP_CONSTITUTION=1 ./scripts/cluster/run_constitution_then_comm.sh
```
