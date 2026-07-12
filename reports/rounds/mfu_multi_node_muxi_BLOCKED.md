# Muxi Phase2 多机弱扩展 · 阻塞记录 · 2026-07-12

## 状态：**BLOCKED**（RoCE 跨节点门禁未过）

按修正计划：保持率 &lt;50% **不得**灌 eth0 假多机 MFU 主表。

| 项 | 内容 |
|----|------|
| 门禁报告 | `reports/rounds/muxi_ib_gate_20260712_gid4.md` |
| 阻塞根因 | RoCE peer ARP→网关 MAC，Proxy Connect |
| 平台请求 | `reports/research/MUXI_ROCE_PLATFORM_REQUEST_20260712.md` |
| 解锁后动作 | 立即跑 Dense TP×PP 扩 DP（8→16→32→…）+ MoE EP/扩 DP；编排 `run_mfu_tp_pp_scale_campaign_muxi.sh` |

## 机内基线（旁路，非 Phase2 替代）

见 `reports/rounds/mfu_single_node_muxi_ledger.md`（Dense 冒烟 + MoE 冒烟）。
