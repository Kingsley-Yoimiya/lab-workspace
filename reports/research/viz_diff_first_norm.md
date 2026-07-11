# Diff-First 可视化规范（短版）

**适用范围**: 128 卡 Ascend 集群 CARD_SCREEN 及后续新增性能 metric  
**配套调研**: `reports/research/research_card_diff_r0.md`  
**参考实现**: `reports/gen_card_screen_128_report.py`（`plot_host_device_heatmap`、TopK、CV）

---

## 核心规则

**每一个新的卡间可比性能 metric，交付时必须同时具备以下三项**（缺一不可）：

| # | 交付物 | 定义 |
|---|--------|------|
| 1 | **host×device 相对中位数热力图** | `dev% = (x − cluster_median) / cluster_median × 100`；行=host，列=device；与 `card_screen_128_figs/heatmap_host_device_deviation.png` 同构 |
| 2 | **按卡排序柱状图** | 全卡按该 metric 升序或降序排列的条形图，标中位线；用于快速扫尾部慢卡 |
| 3 | **CV + TopK 表** | 全集群 CV%（及 mean/median/std/min/max）；最慢/最快 TopK（建议 K=10）含 host、device、绝对值、相对中位数偏差 |

---

## 补充约定

- **先 diff 后绝对值**：正文解读优先相对中位数偏差与热力图斑块，再引用 TFLOPS/GB/s 原值。
- **中位数分桶**：硬件异构时按 `aggregate.py` 的 comparison group 分桶取中位数；单机型 128 卡可用全局中位数。
- **BNMK / 多 shape**：每个 `(B,M,N,K[,layout,dtype])` 切片视为独立 metric（或强制 facet），各自满足上述三项。
- **不替代判定逻辑**：本规范约束报告与图；是否把新 metric 写入 `PERF_KEYS` / `slow` 判定见调研文档二次问题，不在此默认开启。

---

## 最小检查清单（PR / 报告门禁）

- [ ] 热力图文件已生成且坐标为 host×device  
- [ ] sorted bars 已生成  
- [ ] CV% 与 TopK 表写入 markdown 或 `stats.json`  
- [ ] 文中引用相对偏差，而非只贴均值
