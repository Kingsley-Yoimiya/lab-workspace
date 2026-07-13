# 无限租约连环队列（campaign `20260713_135142`）

远程日志：`/tmp/infinite_campaign_20260713_135142.log`

| 阶段 | 内容 | 规模/并行 | 预计 |
|---|---|---|---|
| 0 | 等 Block C full96（进行中） | 96 + npu-smi | ~15–25 min |
| 1 | Exp45（E 双信号 + D rematch）∥ Exp3 strong AB | 16+16+16 | ~40–50 min |
| 2 | MoE failslow 32+64 → MoE 96 | 96 | ~1–1.5 h |
| 3 | Dense GBS∝DP 长窗 16+32 → 64+96（80 iter） | 至 96 | ~2–3 h |
| 4 | Block C full96 再采一轮热力图 | 96 | ~25 min |
| 5 | MoE32 长窗 60 iter | 32 | ~40 min |

完成后会追加 `offline_20260713/SUMMARY.md`，并写 `network_contrib_long.csv`。
