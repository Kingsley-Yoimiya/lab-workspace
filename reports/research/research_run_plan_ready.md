# 机器到位后 · 一次性跑什么（相对旧结果的进步）

**日期**: 2026-07-11  
**入口**: `scripts/cluster/run_constitution_then_comm.sh`  
**前置**: Job Running（如 `huawei-8node-copy2`）+ CARD_SCREEN 同步到 AFS + 遥测修复已在代码里

---

## 总览（一条 list）

| # | 阶段 | 脚本 / 配置 | 相对旧跑的新东西 |
|---|------|-------------|------------------|
| 0 | （可选）1 卡遥测验收 | 机上手动 / calibrate | 确认 temp≠2、power/freq 有数 |
| 1 | **128 卡烤机体质** | `run_card_constitution_128.sh` + `config.constitution128.yaml` | 见下表 A |
| 2 | **本机出图** | `plot_card_constitution.py` | 见下表 B |
| 3 | **HCCL 拓扑探测** | `probe_hccl_topology.sh`（烤机后、collective **前**） | 见下表 T；方案 `research_comm_topology_r0.md` |
| 4 | **HCCL collective** | `run_hccl_scale.sh`（16→128） | 见下表 C |
| 5 | **HCCL P2P** | `run_hccl_p2p_128.sh`（16→128 ring） | 见下表 D |

预计墙钟（粗估，视空闲/失败重试）：烤机约 **数十分钟～2h 量级/节点并行**；拓扑分钟级；通信 scale+P2P 再 **数十分钟～数小时**（P2P@128 若稳）。

---

## A. 烤机体质（相对 `card-screen-128` / BNMK R1）

| 维度 | 旧有 | 本轮新增 / 进步 |
|------|------|-----------------|
| Cube | func + sustained | 同测；遥测挂上后可看 power–tflops |
| HBM | 单一 `src*2` | **+4 模式**：seq_copy / strided / read_heavy / write_heavy |
| Shape | 4 个 BNMK + 可选 2 幂 sweep | **固定 10 个 BNMK**（Qwen FFN、Attn hd128、microbatch、wide-short…） |
| Vector / Scalar / SFU | 仅 SDC 正确性 | **+吞吐**：`vector_gflops` / `scalar_elems_per_s` / `sfu_gflops` |
| MTE / Cube↔Vector | 无 | **+** `mte_gbps`、`cube_vector_tflops` |
| Launch | 无 | **+** sync/tiny/host_overhead p50/p99 + **burst** |
| 遥测 | temp≡2 有毒 | **修命令+card/chip** → 真温/功耗/频率/四路 util |
| 健康 | 粗 ECC | **+** ecc / pcie-err 等快照 |
| 元数据 | host/device | **+** card_id/chip_id、driver/firmware（若写入成功） |
| 判定 | slow_frac 换卡叙事 | **分布优先**：多指标落库，不纠结坏卡主键 |

产物：`logs/card-constitution-128-<stamp>/` + AFS `results/card_screen-*`

---

## B. 出图（相对旧 `card_screen_128_figs`）

| 旧图 | 本轮 |
|------|------|
| 三指标 hist / heatmap / box / sorted bar | **保留**，并对 **所有新数值列** 自动出（有字段才画） |
| shape TFLOPS vs N、BNMK 热力 | 旧有；本轮 10 shape 可再出 sorted/heatmap |
| — | **新散点**：Cube×Vector、HBM×MTE、power×perf（遥测好才有） |
| — | **功耗×性能**：云图 / 分位带 / 每卡曲线（steady 门禁，避免假低功耗曲线） |

产物：`reports/rounds/card_constitution_<stamp>.md` + `_figs/`

---

## T. HCCL 拓扑探测（相对旧「只有 Health=OK」）

| 旧 | 本轮 |
|----|------|
| `run_link_health` 仅 npu-smi + 找 hccn（8/8 缺） | **`probe_hccl_topology`**：topo 矩阵 / hccn.conf / ranktable 搜索 / env 快照 + JSON 摘要 |
| 通信报告无拓扑档位 | 产出 `topo_tier_hint`（1n16…8n128）；后续 bench 打 `intra_or_cross` |
| 慢边无法解释 | 拓扑图挂到 collective/P2P 前，用于区分机内 HCCS vs 跨机 RoCE |

产物：`logs/pipeline-…/hccl-topo/topo_summary.json`；`SKIP_TOPO=1` 可跳过。

---

## C. HCCL Collective（相对 `hccl_128.md`）

| 旧 | 本轮进步（目标） |
|----|------------------|
| 四算子 × 四消息 × 16/32/64/128 | **同矩阵再跑一遍**（可复现 + 可靠性） |
| 弱扩展写成「11.5%」 | 报告侧改用 **bus_bw 保持率**（解读纠正） |
| 失败可能被吞 | 流水线记失败；后续加「同规模重跑 / 不吞错」 |
| 无正确性 | 后续补最小校验（本 list 先采带宽分布） |

产物：`logs/pipeline-…/hccl-scale/` + AFS `hccl-*`

---

## D. HCCL P2P（相对 `hccl_cluster_r0`）

| 旧 | 本轮进步（目标） |
|----|------------------|
| 16 卡成功有边热力 | **再跑 16 作基线** |
| 128 SIGSEGV / ring 未证实 | **默认 ring**，16→128 串进 list；成则出慢边 TopK |
| 无与烤机串联 | **烤机后自动接着跑** |

产物：`logs/pipeline-…/hccl-p2p/` + 边级热力（成功时）

---

## 跑完你能「一眼看到」什么

1. **128 卡多部件分布**：Cube / HBM多模式 / Vector / Scalar / SFU / MTE / pipeline / launch / 温功耗频  
2. **节点 vs 单卡**：host×device 热力 + 正交散点（谁只慢在 HBM、谁只慢在 Vector）  
3. **功耗–性能画像**（遥测有效时）：不是只有时间曲线  
4. **拓扑**：机内 HCCS 矩阵 / RoCE 地址表 / 工具可用性（有无 hccn）  
5. **通信**：collective 带宽阶梯 +（若 128 P2P 通）边级慢边；口径用保持率；慢边用拓扑解释  

---

## 一键命令（机器 Ready 后）

```bash
cd project/lab-workspace
# 先 sync CARD_SCREEN 到 AFS（勿用会冲掉本地改动的旧 sync_to_afs 全量 main）
CLUSTER_JOB=<running-8node-job> \
  ./scripts/cluster/run_constitution_then_comm.sh
```

分段：`SKIP_COMM=1` 只烤机；`SKIP_CONSTITUTION=1` 只通信（含拓扑）；`SKIP_TOPO=1` 跳过拓扑。

---

## 仍依赖平台 / 上机验收的

- Volcano `validatepod.volcano.sh` 修好，Job 真正 Running  
- 1 卡确认：`temp` 合理、`power_w` 非空（否则功耗图自动跳过）  
- 128 P2P 仍可能挂：list 会记失败，不影响前面烤机结果已落盘  
