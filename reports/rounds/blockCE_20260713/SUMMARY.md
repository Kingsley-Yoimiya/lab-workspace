# Block C / E 证据速记（2026-07-13）

## Block C：功耗/利用率热力图

- 采样：`blockC_npusmi/20260713_111100`（real-16 训练期间，~10s×48 次）
- 解析：`parse_npusmi_heatmap.py` → `reports/rounds/blockC_20260713_111100/`
- 图：`blockC_power_aicore_heatmap.svg`
- 观察：采样窗内 **仅 master 跑 Dense-16**，master AICore≈43%，其余 5 节点≈0%。说明 `npu-smi info` 链路可用；**全负载 96 卡热力图**需在 indep-96 / 下次 full-job 时重采（当前图主要验证采集与「忙/闲」对比，不宜直接当「均匀悖论」终局证据）。

## Block D（lite，已有）

- Exp3 PP inject stamp `20260713_101626`：stage spread ↑，但 global median 也被抬高 → WEAK；且 hook 曾把 delay 放在 timer 外，需修后再判「切片掩蔽」。

## Block E（lite，已有）

- Exp4 preempt stamp `20260713_104631`：devices 14/15 上 `npu_busy_preempt.py` + step timer。
- 已知坑：delay/preempt 与 step 计时边界未对齐时，单靠 step 排名会误判；应用 **step gap + npu-smi AICore/功耗** 双信号（计划原文）。后续重跑时：先修 timer 边界，再同时落盘两路信号。
