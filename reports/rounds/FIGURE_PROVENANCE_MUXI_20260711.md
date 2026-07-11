# 图溯源 · Muxi · 20260711（精简）

> 对标昇腾 [`FIGURE_PROVENANCE_AUDIT_20260711.md`](FIGURE_PROVENANCE_AUDIT_20260711.md) 结构精简版。  
> Job：`yushan-muxi-card-screen-128-cp-copy`，**16×8=128** MetaX C550-PL。  
> 出图规范：默认 `reports/plot_style.py`（大字号 / 去顶右边框 / y 点线网格 / hatch / **SVG**）。

## 0. 图目录 ↔ 绘图入口 ↔ 原始数据

| 图目录 | SVG 数 | 绘图入口 | 原始数据 |
|--------|--------|----------|----------|
| `card_constitution_muxi_20260711_figs/` | 108 | `reports/plot_card_constitution.py` | `logs/muxi-constitution-20260711_232400-muxi-constitution128/results/constitution128.merged.jsonl` |
| `constitution_extra_muxi_20260711_figs/` | 12 | `reports/plot_constitution_extra.py` | 同上 merged JSONL |
| `nccl_campaign_muxi_20260711_figs/` | 23 | （若有）NCCL 同构 plot 入口 | `logs/muxi-nccl-campaign-20260711/nccl-results/scale_*.jsonl`；AFS `/afs-a3-weight-share/montyyin/results/nccl-20260711_142129` |

样式统一走 `reports/plot_style.py`。

### 0.1 体质采集公共条件

- **Launch**（`logs/muxi-constitution-20260711_232400-muxi-constitution128/launch_one.sh`）：
  ```text
  python screen.py --device all --config config.constitution128.yaml \
    --sdc-rounds 5 --gemm-n 8192 --sustained-s 30 \
    --out .../constitution128.jsonl --no-plot
  ```
- **配置**：`projects/CARD_SCREEN/config.constitution128.yaml`
- **落库**：`card_screen/io/jsonl.py` → `record=card` + 各 round/sample 行
- **遥测**：`MxSmiProvider` → **`mx-smi`**（温度/功耗/拓扑）；**禁止**套用昇腾 `npu-smi -t …`
- **计时**：CUDA/MACA Event（`torch.cuda`），不是 NPU Event
- **合流**：各 pod JSONL → `constitution128.merged.jsonl`

### 0.2 本批明确缺口

- **有 BNMK sample**（`gemm_bnmk_sample`；出图可另开 bnmk 入口）
- **board_temp / GPU util / XCORE clk 已落盘**（出图可见）
- NCCL 跨节点走 **`SOCKET_IFNAME=eth0`**；拓扑可见 mlx5/xscale，本批未切 IB 数据面
- master 8 卡 **contended**（preflight 撞到残留 compute 进程）；worker-12:0 **bad**

## 1. NCCL 数据路径

| 项 | 路径 |
|----|------|
| 本地 campaign | `logs/muxi-nccl-campaign-20260711/nccl-results/scale_{8,16,32,64,128}.jsonl` |
| AFS collective | `/afs-a3-weight-share/montyyin/results/nccl-20260711_142129` |
| AFS P2P | `/afs-a3-weight-share/montyyin/results/nccl-p2p-20260711_150700` |
| 本地镜像 | `logs/muxi-nccl-campaign-20260711/{nccl-20260711_142129,nccl-p2p-20260711_150700,p2p-results}/` |

保持率现算：256MB、`bus_bw_GBps` 中位、相对 **w8**。

*配套语义：[`METRIC_SEMANTICS_MUXI_20260711.md`](METRIC_SEMANTICS_MUXI_20260711.md)；总汇报：[`CAMPAIGN_FINAL_MUXI_20260711.md`](CAMPAIGN_FINAL_MUXI_20260711.md)*

