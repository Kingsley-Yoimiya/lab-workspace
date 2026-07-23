# Muxi constitution512 干净重跑 · 2026-07-17

- Job: `yinjinrun-cs512-20260716-221823`（空闲、无通信并行）
- Config: `config.constitution128.yaml`，N=8192，sustained=30s，sdc=5
- AFS: `/afs-a3-weight-share/yinjinrun.p/results/card_screen-20260717_090525-muxi-constitution512-clean`
- 本地: `myportal/results/muxi-h3c/20260717_090525-constitution512-clean/`

## 结果（官方 aggregate）

| 项 | 值 |
|---|---:|
| 卡数 | 512 |
| good / slow / contended | **508 / 4 / 0** |
| func 中位 | 276.3 TFLOPS |
| HBM 中位 | 1474.8 GB/s |
| sustained 中位 | 277.6 TFLOPS |

相对昨夜脏跑（good=319 / contended=192 / slow=1）：本轮 **contended=0**，慢卡判定干净。

## Slow 卡（4）

- `yinjinrun-cs512-20260716-221823-worker-22` dev0: sust=220.9  func=273.1  hbm=1461.7  cause=no_throttle
- `yinjinrun-cs512-20260716-221823-worker-23` dev5: sust=202.6  func=273.2  hbm=1471.2  cause=power_cap
- `yinjinrun-cs512-20260716-221823-worker-37` dev2: sust=198.2  func=270.0  hbm=1463.0  cause=no_throttle
- `yinjinrun-cs512-20260716-221823-worker-51` dev4: sust=166.9  func=270.9  hbm=1470.4  cause=no_throttle

## Outline V3：训练前烤机失败率

$$\hat p = k/N = 4/512 = 0.007812$$

Wilson 95% CI ≈ [0.003042, 0.019914]
阈值：sustained < ~0.8×cluster median（本轮约 222.1，med=277.6）。

### $P_N=1-(1-\hat p)^N$（独立近似示意）

| N | P_N |
|--:|---:|
| 8 | 0.0608 |
| 16 | 0.1179 |
| 32 | 0.2220 |
| 64 | 0.3947 |
| 128 | 0.6336 |
| 256 | 0.8657 |
| 512 | 0.9820 |
| 1024 | 0.9997 |
| 2048 | 1.0000 |
| 4096 | 1.0000 |

说明：这是**一轮预检劣化率**，不是训练中每设备小时事件率；写 Fig.1 必须标注口径。
详见 `myportal/plans/outline-v3-cardscreen-estimates.md`。
