# 华为 ↔ Muxi 迁移对照总表（G10）

时间: 2026-07-11（报告流水线定稿）  
集群: Muxi `vc-c550-mohe-241` / `yushan-muxi-card-screen-128-cp-copy`（16×8 C550）

> **总汇报**：[`CAMPAIGN_FINAL_MUXI_20260711.md`](CAMPAIGN_FINAL_MUXI_20260711.md)  
> **语义手册**：[`METRIC_SEMANTICS_MUXI_20260711.md`](METRIC_SEMANTICS_MUXI_20260711.md)  
> **图溯源**：[`FIGURE_PROVENANCE_MUXI_20260711.md`](FIGURE_PROVENANCE_MUXI_20260711.md)  
> 昇腾同构：[`CAMPAIGN_FINAL_20260711.md`](CAMPAIGN_FINAL_20260711.md) · [`METRIC_SEMANTICS_20260711.md`](METRIC_SEMANTICS_20260711.md)

## 能力对照

| ID | 能力 | Muxi 状态 | 关键产物 |
|----|------|-----------|----------|
| G0 | 双集群隔离 | **done** | `muxi.env` / `huawei.env` |
| G1 | 快慢卡冒烟 | **done** | good=106 / slow=19 / bad=1；`muxi_smoke_20260711.md` |
| G2 | 体质 | **done** | 中位 func **279.3**；**91 SVG** + 含义优先 md |
| G3 | 拓扑 | **done** | `muxi_topo_20260711.md` |
| G4–G5 | NCCL + P2P | **done** | **23 SVG**；AR@256M w8≈190.5 GB/s；跨节点保持率 ~0.13% |
| G6 | 链路健康 | **done** | 16/16 |
| G7 | 流水线 | **done** | `run_constitution_then_comm_muxi.sh` |
| G8–G9 | MFU / 真训练 | **done** | dense@8=26.7%；tiny GPT 通 |
| G10 | 报告对齐 | **done** | plot_style SVG → `rewrite_meaning_mds_muxi.py` → 溯源 |

## 报告流水线（对齐昇腾）

```text
原始 JSONL
  → plot_card_constitution / plot_constitution_extra / plot_nccl_campaign_muxi  （plot_style SVG）
  → rewrite_meaning_mds_muxi.py   （图注：含义 + 底层 API；禁画图空话）
  → METRIC_SEMANTICS_MUXI + FIGURE_PROVENANCE_MUXI + CAMPAIGN_FINAL_MUXI
```

复跑：

```bash
cd project/lab-workspace
python3 reports/plot_card_constitution.py \
  --jsonl logs/../../logs/muxi-constitution-20260711_140024-muxi-constitution128/results/constitution128.merged.jsonl \
  --out-dir reports/rounds --stamp muxi_20260711
python3 reports/plot_constitution_extra.py \
  --jsonl <同上> --out-dir reports/rounds --stamp constitution_extra_muxi_20260711
python3 reports/plot_nccl_campaign_muxi.py
python3 reports/rewrite_meaning_mds_muxi.py
```

## 一眼对照昇腾

| | 昇腾 | 沐曦 |
|--|------|------|
| 体质中位 func | ~292 TFLOPS | ~279 TFLOPS |
| 通信保持率基线 | vs w16 | vs **w8** |
| AR@大消息跨节点 | w128 仍 ~89% | w16 即跌到 ~0.13%（eth0） |
| 出图 | plot_* + rewrite_meaning | **同构** |
