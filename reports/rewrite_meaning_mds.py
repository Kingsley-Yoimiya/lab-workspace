#!/usr/bin/env python3
"""重写报告图注：只讲「指标是什么 / 底层 API」，不讲怎么画图。"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
import sys

_REP = Path(__file__).resolve().parent
if str(_REP) not in sys.path:
    sys.path.insert(0, str(_REP))
from glossary_links import linkify_abbr

ROUNDS = Path(__file__).resolve().parent / "rounds"
FILLGAP = Path(
    "/Users/yinjinrun/random-thing/logs/card-fillgap-20260711_140301/results/"
    "constitution128.merged.jsonl"
)

# 字段 → 人话含义（短）+ 底层 API（短）。详细版见 METRIC_SEMANTICS_20260711.md
SEM: dict[str, tuple[str, str]] = {
    "func_tflops": (
        "单卡方阵矩阵乘吞吐（TFLOPS）。对应 Cube（矩阵计算单元：AI Core 内专做大规模矩阵乘加的主算力部件）路径上的瞬时能力；"
        "不是热稳态，也不等于整网训练 MFU。",
        "torch 算子 `a@b`（bf16），FLOPs=`2·N³`，NPU Event 计时取中位；N=8192，warmup=20，iters=50。",
    ),
    "sustained_tflops": (
        "连续烤机后的方阵矩阵乘吞吐（TFLOPS）。仍走 Cube（矩阵计算单元）路径；"
        "看一段时间后还能维持多少算力，用来对比短窗 func_tflops，而不是替代整卡健康评分。",
        "循环 `a@b` 跑满 ~30s，每窗 50 次 GEMM 用 NPU Event 计时；**卡级字段取最后一个时间窗**（非中位）。N=8192 bf16。",
    ),
    "hbm_gbps": (
        "HBM（High Bandwidth Memory，器件高带宽外存）路径上的有效带宽代理（GB/s）。"
        "探针是「读+写 + 一次逐元素乘」，所以是访存+轻算混合，不是纯 DMA，也不是 npu-smi 的带宽占用率。",
        "设备侧大缓冲 `dst = src * 2.0`（fp32）；流量按 R+W；Event 计时中位。默认 1024MB，w20/i50。",
    ),
    "vector_gflops": (
        "Vector（向量计算单元：逐元素/向量运算，灵活度高于 Cube、峰值通常低于 Cube）路径上的 FMA 吞吐代理（GFLOPS）。"
        "测的不是矩阵乘主路径。",
        "逐元素 `a*b+c`，按 2 flops/elem；64M 元素 fp32；NPU Event 中位。w20/i50。",
    ),
    "scalar_elems_per_s": (
        "长依赖串行链吞吐（元素/秒）。更贴近 Scalar/控制流+同步，不是 SIMD 峰值。",
        "`torch.cumsum`；elems_per_s = elems/dt；16M fp32。量纲不是 GFLOPS，勿与 vector 直接比倍速。",
    ),
    "mte_gbps": (
        "纯设备侧拷贝带宽（GB/s）。字段名借 MTE（Memory Transfer Engine：片上 Buffer 与 Global Memory 之间的搬运引擎）；"
        "实现是 `Tensor.copy_`，用来和 hbm_gbps（带乘）对照「纯搬运 vs 访存+轻算」——不是直接读 MTE1/2/3 计数器。",
        "`Tensor.copy_`；流量按 R+W；512MB；Event 中位。w20/i50。",
    ),
    "cube_vector_tflops": (
        "Cube（矩阵乘）之后接 Vector（向量）epilogue（scale+bias）的端到端吞吐（TFLOPS）。"
        "看矩阵结果离开 Cube 再进向量后处理这条衔接路径，不是单独的 Cube 峰值。",
        "`c=a@b; c*scale+bias`；FLOPs=`2N³+3N²`；N=4096 bf16。数值通常低于纯 `func_tflops`。",
    ),
    "sfu_gflops": (
        "特殊函数类吞吐代理。公开 AI Core 叙述里 exp/sqrt 等常归 Vector 能力面；"
        "本字段名沿用 SFU，实现是 `torch.exp`，按 1 op/元素计，量纲更接近 Gops/s，不是 FMA GFLOPS。",
        "默认 `torch.exp(x)`；`gflops≈elems/dt/1e9`；64M fp32。与 SDC 正确性探针不是一回事。",
    ),
    "hbm_mode_seq_copy_gbps": (
        "HBM 多模式之一：顺序 copy 带宽（GB/s）。",
        "`dst.copy_(src)`；512MB；w10/i30。",
    ),
    "hbm_mode_strided_gbps": (
        "跨步访问有效带宽（只计触碰元素）。对 stride 敏感通路的探针。",
        "`src[::stride]`→`dst[::stride]`，stride=16；勿与顺序 copy 比绝对值判好坏。",
    ),
    "hbm_mode_read_heavy_gbps": (
        "读密集路径带宽代理（GB/s）。",
        "`src.sum()`，流量按只读计。",
    ),
    "hbm_mode_write_heavy_gbps": (
        "写密集路径带宽代理（GB/s）。",
        "`dst.fill_(1.0)`，流量按只写计。",
    ),
    "launch_sync_p50_us": (
        "空设备 `synchronize()` 往返延迟的 p50（µs）。反映驱动/设备响应基线。",
        "CPU `perf_counter` 包一层 `adapter.sync`；samples=500，warmup=50。与 kernel 发射无关。",
    ),
    "launch_sync_p99_us": (
        "同上的 p99（µs）。看调度抖动尾延迟。",
        "同 launch_latency 探针。",
    ),
    "launch_host_overhead_p50_us": (
        "Host 侧发射开销 p50（µs）≈ wall − device event。",
        "极小核 add 的墙钟与 NPU Event 差分；需 timing_method=event 才有意义。",
    ),
    "launch_host_overhead_p99_us": (
        "Host 发射开销 p99（µs）。",
        "同上。",
    ),
    "launch_burst_p50_us": (
        "连续 enqueue 64 个极小核后一次 sync 的总时延 p50（µs）。",
        "CPU 计时 burst；看队列深度下的发射成本。",
    ),
    "launch_burst_p99_us": (
        "突发总时延 p99（µs）。",
        "同上。",
    ),
    "launch_burst_per_kernel_p50_us": (
        "突发总时延 / 64，每核摊销 p50（µs）。",
        "由 burst 派生。",
    ),
    "launch_burst_per_kernel_p99_us": (
        "突发摊销 p99（µs）。",
        "同上。",
    ),
    "health_temp_c": (
        "流程早期轻载/开测温度快照（°C）。health 同样只是采样阶段标签；"
        "与负载探针回填的 board_temp_c 不是同一时刻的热状态。",
        "`npu-smi info -t temp -i <card> -c <chip>` 解析。",
    ),
    "health_power_w": (
        "constitution 流程早期、轻载时刻的芯片实时功耗（W）。"
        "来自 npu-smi -t power 的 Real-time Power；名称里的 health 只表示采样阶段，不是健康评分。"
        "与负载末的 power_w 是同一工具字段、不同时刻，差值未在本报告定义为降频幅度。",
        "`npu-smi info -t power -i -c` → Real-time Power。",
    ),
    "board_temp_c": (
        "板/NPU 温度（°C），取自负载遥测缓存。",
        "`npu-smi -t temp/board`；卡级常取 **vector_fma 探针末轮** 回填，不是 sustained 烤机峰值时刻。",
    ),
    "aicore_util_pct": (
        "AICore（即 AI Core：昇腾主计算核）占用率（%），来自 npu-smi usages 的 Aicore Usage Rate。"
        "本批多为某次负载探针末轮瞬时值，不是长时间平均。",
        "`npu-smi info -t usages`；卡级多为 vector_fma 末轮瞬时率。",
    ),
    "aicpu_util_pct": (
        "AICPU（器件侧 AI CPU，与 Cube/Vector 不是同一执行体）占用率（%）。"
        "来自 npu-smi 的 Aicpu Usage Rate；本批常为 0，表示该次采样为 0，不单独证明硬件缺失。",
        "同上 `-t usages`。",
    ),
    "ctrlcpu_util_pct": (
        "CtrlCPU（器件侧控制 CPU）占用率（%），来自 npu-smi 的 Ctrlcpu Usage Rate；"
        "不是宿主机 top 的 CPU%。与 launch_*（host 墙钟）不在同一观测面。",
        "同上 `-t usages`。",
    ),
    "mem_bw_util_pct": (
        "HBM Bandwidth Usage Rate（%）。",
        "同上 `-t usages`；瞬时率。",
    ),
    "power_w": (
        "负载探针时段（多为 vector_fma 末轮）的芯片实时功耗（W），同样来自 npu-smi Real-time Power。"
        "与 health_power_w 字段同源、采样时刻不同。",
        "`npu-smi -t power`；卡级取 vector_fma **末轮**。",
    ),
    "power_limit_w": (
        "功耗上限（W）。",
        "`npu-smi -t power` 解析；本批常缺失。",
    ),
    "shape_sweep_peak_tflops": (
        "名义「shape sweep 峰值」，本批实际是 **BNMK 各形状中位吞吐的最大值**。",
        "constitution128 关闭方阵 shape_sweep、开启 bnmk_sweep；`jsonl.py` 用 max(BNMK tflops) 回填此键。",
    ),
    "aicore_freq_mhz": (
        "AICore 频率（MHz）。",
        "`npu-smi -t board` 文本解析；本批常空。",
    ),
    "hbm_temp_c": (
        "HBM 温度（°C）。",
        "遥测解析；本批常空。",
    ),
}


def load_cards() -> list[dict]:
    out = []
    with FILLGAP.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("record") == "card":
                out.append(r)
    return out


def med(cards: list[dict], key: str) -> float | None:
    vs = [float(c[key]) for c in cards if c.get(key) is not None]
    return statistics.median(vs) if vs else None


def list_svgs(d: Path) -> list[str]:
    return sorted(p.name for p in d.glob("*.svg"))


def meaning_block(key: str) -> str:
    if key in SEM:
        what, how = SEM[key]
        body = f"**含义**：{what}  **底层**：{how}"
        return linkify_abbr(body)
    return (
        f"**含义**：字段 `{key}` 的语义见 "
        f"[`METRIC_SEMANTICS_20260711.md`](METRIC_SEMANTICS_20260711.md)。"
    )


def caption_metric_fig(name: str, cards: list[dict]) -> str:
    m = re.match(
        r"^(hist|heatmap_relmed|box_by_host|sorted_bar|bar_host_mean_std)_(.+)\.svg$",
        name,
    )
    if not m:
        return ""
    key = m.group(2)
    mv = med(cards, key)
    mv_s = f"{mv:.4g}" if mv is not None else "—"
    # 只保留必要的「这张图显示什么」半句，不讲画图法
    show = {
        "hist": "全卡分布",
        "heatmap_relmed": "host×device 相对集群中位偏差%（|Δ|≥1% 才标数）",
        "box_by_host": "分 host 箱线",
        "sorted_bar": "单卡升序一览",
        "bar_host_mean_std": "分 host 均值±σ",
    }[m.group(1)]
    return f"**`{key}`**（{show}）。本批中位≈**{mv_s}**。{meaning_block(key)}"


def write_constitution(cards: list[dict]) -> None:
    fig = ROUNDS / "card_constitution_20260711_figs"
    svgs = list_svgs(fig)
    lines = [
        "# Card Constitution · 20260711",
        "",
        "**怎么读**：关键硬件缩写在**本行第一次出现**时，后面直接跟括号附注（无需跳转）。"
        "完整对照表：[`ASCEND_HARDWARE_GLOSSARY_20260711.md`](ASCEND_HARDWARE_GLOSSARY_20260711.md)；"
        "测法：[`METRIC_SEMANTICS_20260711.md`](METRIC_SEMANTICS_20260711.md)。",
        "数据：`logs/card-fillgap-20260711_140301/results/constitution128.merged.jsonl`；"
        "job `whj4stu-copy-copy-copy` 8×16；`screen.py` + `config.constitution128.yaml`。",
        "",
        "## 关键中位",
        "",
        "| 字段 | 人话 | 中位 |",
        "|---|---|---:|",
    ]
    for k, lab in [
        ("func_tflops", "方阵 GEMM 吞吐代理（Cube 主算力路径）"),
        ("sustained_tflops", "稳态方阵 GEMM（Cube）"),
        ("hbm_gbps", "HBM 访存+轻算带宽代理"),
        ("vector_gflops", "向量 FMA 代理（Vector）"),
        ("mte_gbps", "纯 copy_ 带宽代理（字段名借 MTE）"),
        ("health_power_w", "轻载时刻 Real-time Power（非健康分）"),
        ("power_w", "负载末轮 Real-time Power"),
    ]:
        v = med(cards, k)
        lab_l = linkify_abbr(lab)
        mid = f"{v:.4g}" if v is not None else "—"
        lines.append(f"| `{k}` | {lab_l} | {mid} |")

    lines += ["", "## 逐图（含义优先）", ""]
    for name in svgs:
        if name.startswith("scatter_"):
            body = name[len("scatter_") : -4]
            parts = body.split("_vs_")
            if len(parts) == 2:
                x, y = parts
                cap = (
                    f"横轴 `{x}`，纵轴 `{y}`（每卡一点）。"
                    f"{meaning_block(x)} {meaning_block(y)}"
                )
            else:
                cap = f"散点 `{body}`。"
        elif name == "box_overview.svg":
            cap = "多指标全集群箱线总览；各轴字段含义见上表与语义手册。"
        elif name == "timeseries_sustained_p05_p50.svg":
            cap = (
                "**稳态 GEMM 时间序列的跨卡分位**。"
                "原始明细 `record=gemm_sustained_sample`（iter / t_s / tflops）。"
                "每个 iter 上对全部卡的 tflops 取 **p05 与 p50**（覆盖不足 90% 卡的 iter 丢弃）；"
                "横轴用该 iter 上各卡 t_s 的中位。"
                "含义上：p50≈集群典型可持续算力轨迹，p05≈尾部偏慢卡轨迹——"
                "**不是**挑两张代表卡各自画一条线。"
                f" {meaning_block('sustained_tflops')}"
            )
        else:
            cap = caption_metric_fig(name, cards) or f"`{name}`"
        lines.append(cap)
        lines.append("")
        lines.append(f"![{name}](card_constitution_20260711_figs/{name})")
        lines.append("")

    path = ROUNDS / "card_constitution_20260711.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path.name, len(svgs))


def write_extra() -> None:
    fig = ROUNDS / "constitution_extra_fillgap_20260711_figs"
    svgs = list_svgs(fig)
    lines = [
        "# Constitution 增强图 · 20260711",
        "",
        "字段含义见 [`METRIC_SEMANTICS_20260711.md`](METRIC_SEMANTICS_20260711.md)。同一 fillgap merged JSONL。",
        "",
    ]
    caps = {
        "radar_host_median_norm.svg": (
            "各 host 在多指标上的**中位相对集群中位**（1.0=集群水平）。"
            "用来看机间体质是否齐，不是单卡绝对值。"
        ),
        "parallel_host_median_norm.svg": "与雷达同一套 host 中位归一化，平行坐标展示。",
        "hbm_modes_grouped_bar.svg": (
            "四种 HBM 访问模式带宽："
            "`seq_copy` / `strided` / `read_heavy` / `write_heavy`。"
            "底层是 `hbm_modes_perf`（copy / 跨步 / sum / fill），单位 GB/s；"
            "**跨模式绝对值不可直接比「谁更好」**。"
        ),
        "corr_cube_vector_sfu_mte.svg": (
            "Cube / Vector / SFU / MTE 四路吞吐的 Pearson 相关。"
            "看子系统是否同涨同跌；相关≈0 表示彼此相对独立。"
        ),
        "box_launch_by_host.svg": (
            "Launch 延迟分 host："
            "空 sync p99、host 发射开销 p99、突发总时延 p50。"
            f"{meaning_block('launch_sync_p99_us')}"
        ),
        "cdf_core_metrics.svg": "核心吞吐指标的经验分布函数（CDF）。",
        "extreme10_small_multiples.svg": (
            "按 `sustained_tflops` 最慢/最快各 10 卡，多指标相对集群中位偏差。"
            "用来对照「慢卡是否多项一起慢」。"
        ),
        "scatter_sustained_vs_func.svg": (
            f"横轴短测 Cube，纵轴稳态 Cube。{meaning_block('func_tflops')} "
            f"{meaning_block('sustained_tflops')}"
        ),
    }
    for name in svgs:
        if name.startswith("heatmap_host_device_"):
            key = name.replace("heatmap_host_device_", "").replace(".svg", "")
            cap = f"host×device 上的 **`{key}` 绝对值**。{meaning_block(key)}"
        else:
            cap = caps.get(name, name)
        lines.append(f"**{name}**：{cap}")
        lines.append("")
        lines.append(f"![{name}](constitution_extra_fillgap_20260711_figs/{name})")
        lines.append("")
    path = ROUNDS / "constitution_extra_fillgap_20260711.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path.name, len(svgs))


def write_bnmk() -> None:
    fig = ROUNDS / "bnmk_shapes_20260711_figs"
    svgs = list_svgs(fig)
    lines = [
        "# BNMK · 20260711",
        "",
        "**BNMK 是什么**：按显式 `(B,M,N,K)` 做 batched GEMM（`a[B,M,K]@b[B,K,N]`），"
        "FLOPs=`2·B·M·N·K`，得到训练层形状代理吞吐（TFLOPS，bf16）。"
        "明细 `record=gemm_bnmk_sample`；本批 10 个 shape × 128 卡 = 1280 样本。",
        "底层：`gemm_bnmk_sweep`，NPU Event，每 shape 多窗取中位 tflops。",
        "",
    ]
    caps = {
        "bnmk_tflops_box_by_label.svg": "每个 shape label 在 128 卡上的 TFLOPS 分布。",
        "bnmk_tflops_bar_median_by_label.svg": "每个 shape 的跨卡中位 TFLOPS。",
        "bnmk_host_shape_heatmap.svg": "host×shape 的平均 TFLOPS（看某机某形状是否掉队）。",
    }
    for name in svgs:
        lines.append(f"**{name}**：{caps.get(name, '')}")
        lines.append("")
        lines.append(f"![{name}](bnmk_shapes_20260711_figs/{name})")
        lines.append("")
    (ROUNDS / "bnmk_shapes_20260711.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote bnmk", len(svgs))


def write_hccl() -> None:
    fig = ROUNDS / "hccl_campaign_20260711_figs"
    svgs = list_svgs(fig)
    lines = [
        "# HCCL 通信 · 20260711",
        "",
        "语义详见 [`METRIC_SEMANTICS_20260711.md`](METRIC_SEMANTICS_20260711.md) 通信章。",
        "",
        "**alg_bw**：业务字节 / 平均时延 → GB/s（算法视角）。",
        "**bus_bw**：按 NCCL-tests 同构公式把多跳折成可与链路比的总线带宽——"
        "AllReduce `×2(n-1)/n`，AG/RS `×(n-1)/n`，Broadcast `=alg`。",
        "扩展叙事用 **bus_bw 保持率 = bus_N/bus_16**，不要用 (bus_N/bus_16)/(N/16)。",
        "",
        "底层：`torch.distributed` + **HCCL**（`hccl_torch_bench.py`）；"
        "CPU `perf_counter` + `torch.npu.synchronize`；sizes 1M–256M；fp32；world 16→128。",
        "",
        "## 256MB bus_bw 保持率",
        "",
        "| op | w32 | w64 | w128 |",
        "|---|---:|---:|---:|",
        "| All-Reduce | 96.8% | 94.9% | 89.4% |",
        "| Broadcast | 91.4% | 86.8% | 86.8% |",
        "| All-Gather | 88.0% | 64.2% | 54.0% |",
        "| Reduce-Scatter | 91.8% | 71.0% | 46.4% |",
        "",
    ]
    for name in svgs:
        if name.startswith("hccl_bus_bw_vs_size_"):
            op = name.replace("hccl_bus_bw_vs_size_", "").replace(".svg", "")
            cap = (
                f"**`{op}` 的 bus_bw 随消息大小**。"
                f"底层 `dist.{op}`（all_gather/reduce_scatter 按 world 切分缓冲）；"
                f"每点是该 (world,size) 下各 rank bus_bw 的中位。"
            )
        elif "retention" in name:
            cap = "256MB 上各 collective 的 bus_bw 相对 world=16 的保持率（扩展健康度）。"
        elif "step" in name:
            cap = "固定 256MB，world 从 16→128 时 bus_bw 中位的阶梯变化。"
        elif name.startswith("hccl_rank_"):
            cap = (
                "同一 (op, world=?, 256MB) 下**每个 rank 各自的 bus_bw** 分布。"
                "看是否个别 rank 拖总线折算带宽。"
            )
        elif name.startswith("p2p_"):
            cap = (
                "**点对点 isend/irecv 单向带宽**（GB/s），不是 bus_bw 公式。"
                "边类型含 ring / star；大 world 默认仅 ring。"
                "底层 `hccl_p2p_bench.py`，严格串行单对。"
            )
        elif "topo" in name:
            cap = (
                "机内物理拓扑亲和：来自 **`npu-smi info -t topo`** 解析。"
                "S=SIO（die 内），空白=HCCS_SW（经交换机的 HCCS），·=self。"
                "这是静态拓扑关系，不是测速。"
            )
        else:
            cap = name
        lines.append(f"**{name}**：{cap}")
        lines.append("")
        lines.append(f"![{name}](hccl_campaign_20260711_figs/{name})")
        lines.append("")
    (ROUNDS / "hccl_campaign_20260711.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote hccl", len(svgs))


def write_inter() -> None:
    lines = [
        "# 机内 / 机间 P2P 带宽 · 20260711",
        "",
        "**测的是什么**：HCCL 点对点单向（或 ping-pong）有效带宽（GB/s），"
        "用来在没有 hccn 可读信息时反推机间链路能力。",
        "",
        "**底层**：`torch.distributed.isend/irecv` + HCCL；"
        "`hccl_inter_bw_probe.py`；全员 barrier 下严格串行单对；"
        "默认流水线 uni（inflight=4）；`HCCL_BUFFSIZE=2048`。",
        "",
        "- **intra**：同节点固定 local_rank 对 → 走机内 HCCS/SIO 平面",
        "- **inter**：不同节点、同 local_rank 对齐 → 走机间平面（A3 上常为 UB/UBoE 域）",
        "",
        "本批大包饱和区 inter≈119 GB/s、intra≈122 GB/s（recv 中位）；"
        "与 AllReduce bus_bw 定义不同，只能量级交叉。",
        "",
        "## 图",
        "",
        "**inter_vs_intra_bw.svg**：各消息大小上 intra/inter 中位带宽对比。",
        "",
        "![inter_vs_intra_bw.svg](inter_bw_20260711_figs/inter_vs_intra_bw.svg)",
        "",
    ]
    (ROUNDS / "inter_bw_20260711.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote inter")


def patch_campaign() -> None:
    p = ROUNDS / "CAMPAIGN_FINAL_20260711.md"
    t = p.read_text(encoding="utf-8")
    if "METRIC_SEMANTICS_20260711.md" not in t:
        t = t.replace(
            "FIGURE_PROVENANCE_AUDIT_20260711.md",
            "METRIC_SEMANTICS_20260711.md`](METRIC_SEMANTICS_20260711.md)（字段含义/底层 API）；"
            "采集链路见 [`FIGURE_PROVENANCE_AUDIT_20260711.md",
        )
    p.write_text(t, encoding="utf-8")


def main() -> None:
    cards = load_cards()
    write_constitution(cards)
    write_extra()
    write_bnmk()
    write_hccl()
    write_inter()
    patch_campaign()
    for md in [
        "card_constitution_20260711.md",
        "constitution_extra_fillgap_20260711.md",
        "bnmk_shapes_20260711.md",
        "hccl_campaign_20260711.md",
        "inter_bw_20260711.md",
    ]:
        text = (ROUNDS / md).read_text(encoding="utf-8")
        refs = re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", text)
        miss = [r for r in refs if not (ROUNDS / r).exists()]
        # 画图空话检测
        bad = sum(1 for w in ["直方图", "怎么看", "红虚线", "橙虚线", "出图脚本"] if w in text)
        print(f"{md}: refs={len(refs)} miss={len(miss)} fluff_hits≈{bad}")


if __name__ == "__main__":
    main()
