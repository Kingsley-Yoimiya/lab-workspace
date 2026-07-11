# 沐曦采集全流程核查清单 · 2026-07-11（执行后）

> 四路审计后已改代码、试采、全量重跑并重出图。

---

## 总判（更新）

| 面 | 结论 |
|----|------|
| 单卡算力（func/hbm/DMA/power） | **可信**；重跑中位 func≈**279.9** TFLOPS（127/128） |
| board_temp / XCORE clk / GPU util | **已落盘**；中位 board≈**54°C**、clk≈**1500**、util≈**98%** |
| BNMK | **已有** sample（本批开启） |
| 跨节点通信 | 仍为 eth0 Socket 面（未本轮重跑） |
| 判定 | good=119 / contended=8（master 撞残留进程）/ bad=1（worker-12:0） |

数据：`logs/muxi-constitution-20260711_232400-muxi-constitution128/results/constitution128.merged.jsonl`  
图：`reports/rounds/card_constitution_muxi_20260711_figs/`（**108** SVG）

---

## 本轮已落地

1. **遥测**：`--show-usage` + `-j` XCORE clk + `--show-temperature` board（不覆盖 dmon 温功耗）
2. **jsonl**：card 汇总从 vf/shape/sustained/hbm/gemm 多源回填
3. **sync** → montyyin AFS；master 试采 8/8 三字段 OK
4. **全量体质** durable 点火（`fire_constitution_durable_muxi.sh`）；补跑 worker-5/7
5. **报告**：措辞与 108 图已按新数据重写

---

## 仍待（可选）

| ID | 动作 |
|----|------|
| P0-2 | IB/xscale + `NCCL_DEBUG=INFO` 后再跑跨节点 |
| P1 | master 清残留后单节点复测消 contended；worker-12:0 单卡复测 |
| P1 | BNMK 专图入口（sample 已有） |
| P1 | 统一冒烟/体质 AFS 树 |

---

*审计 + 修正执行：2026-07-11 夜。*
