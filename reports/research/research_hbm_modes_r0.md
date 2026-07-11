# HBM 多模式探针（Track C / R0）

**日期**: 2026-07-11  
**目标**: 在现有单一 `hbm`（`dst = src * 2.0`）之外，用少量访问模式拆开「介质弱」vs「特定访问模式弱」。  
**探针名**: `hbm_modes_perf`（`default_enabled=false`，`verdict_neutral`）

## 背景

128 卡体质数据里 HBM 已有慢尾与节点聚集，但 Stage A `hbm` 只测「计算发起的顺序读写」（mul 一读一写），无法区分：

- HBM 介质 / 通道本身弱
- 顺序 DMA copy 通路弱（与 `mte_copy_perf` 交叉）
- 仅对跨步 / 读多 / 写多敏感

本探针用 **2–4 个 torch 可实现模式** 做轻量交叉，不替代 `hbm` / `mte_copy_perf`。

## 模式清单

| 模式名 | 内核（torch） | 流量记账 | 烤机解读 |
|--------|---------------|----------|----------|
| `seq_copy` | `dst.copy_(src)` | \(2 \times N \times 4\) 字节（R+W） | 顺序 DMA 式拷贝；与 `hbm`（mul）正交，可与 `mte_copy_perf` 对照 |
| `strided` | `dst[::stride].copy_(src[::stride])` | \(2 \times (N/\mathrm{stride}) \times 4\) | 跨步访存；压 bank / 行缓冲效率，**绝对 GB/s 本就更低** |
| `read_heavy` | `src.sum()` | \(N \times 4\)（近似纯读） | 读主导；介质读侧 / 读带宽敏感 |
| `write_heavy` | `dst.fill_(1.0)` | \(N \times 4\)（近似纯写） | 写主导；写合并 / 写带宽敏感 |

默认 `stride=16`，`mb=512`，`iters=30`（比 Stage A `hbm` 轻）。

未选模式（刻意不做）：随机 gather（torch 实现重、假阳性多）、极小块 ping-pong（易被 launch 主导）。

## JSONL 指标名

card 汇总行字段（中位数 GB/s）：

- `hbm_mode_seq_copy_gbps`
- `hbm_mode_strided_gbps`
- `hbm_mode_read_heavy_gbps`
- `hbm_mode_write_heavy_gbps`

明细行：`record=hbm_mode_round`，带 `mode` 字段。

结果 dict 同时含 `modes.<name>.gbps` 与扁平 `hbm_mode_<name>_gbps`。

## 与现有探针关系

| 探针 | 测什么 | 与本探针 |
|------|--------|----------|
| `hbm` | `src*2` 顺序 R+W | 保留为基线；勿与 `seq_copy` 绝对数值强对齐（mul vs copy） |
| `mte_copy_perf` | 纯 `copy_` | 与 `seq_copy` 同族；若两者都慢而 read/write 不慢 → 更像 copy/MTE 通路 |
| `hbm_modes_perf` | 多模式 | 模式间相对形状 + within-host 残差才有换卡意义 |

粗判启发式（仅分布解读，不进 verdict）：

1. 四模式相对集群中位数都慢 → 更像介质 / 全局带宽弱  
2. 仅 `strided` 慢 → 更像跨步敏感，勿直接判「坏 HBM」  
3. 仅 `seq_copy`（及 `mte_copy`）慢、read/write 正常 → 更像 copy/MTE  
4. 仅 `read_heavy` 或仅 `write_heavy` 慢 → 读写不对称，优先看遥测/驱动再谈换卡

## 假阳性注意

1. **跨模式绝对 GB/s 不可比**：流量公式不同；`strided` 触达元素更少，GB/s 定义按触达字节，数值天然偏低。  
2. **缓冲过小**：`mb` 太小会被 launch / sync 开销淹没，四模式一起「假慢」。constitution 用 512MB；冒烟勿用过小值下结论。  
3. **编译器 / 运行时优化**：`sum` / `fill_` 在部分后端可能被弱化；若某模式 GB/s 离谱偏高且无热功耗，视为测量无效而非「超强卡」。  
4. **与 `hbm` 数值差**：`seq_copy` ≠ `src*2`；差值大不等于故障。  
5. **争用 / 降频**：多模式串行烤机会抬温；后跑模式可能系统性偏低 → 比 within-host 残差，或固定模式顺序解读。  
6. **勿单独用模式比做 bad 判据**：本探针 `verdict_neutral`；只服务体质分布与换卡前置分析。

## 配置开关

- 代码默认：`probes.hbm_modes_perf.enabled: false`  
- `config.constitution128.yaml`：轻量开启（`mb=512`, `iters=30`）  
- 关闭：将该项改为 `{enabled: false}`

## 实现位置

- 内核：`card_screen/probes/stage_c.py` → `hbm_modes_perf` / `HBM_MODE_NAMES`  
- 注册：`card_screen/probes/builtin.py` → `HbmModesPerf`  
- 汇总：`card_screen/io/jsonl.py`
