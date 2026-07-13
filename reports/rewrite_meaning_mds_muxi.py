#!/usr/bin/env python3
"""重写沐曦报告图注：只讲「指标是什么 / 底层 API」，不讲怎么画图。对标 rewrite_meaning_mds.py。"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROUNDS = Path(__file__).resolve().parent / "rounds"
MERGED = Path(
    "/Users/yinjinrun/random-thing/logs/"
    "muxi-constitution-20260711_232400-muxi-constitution128/results/"
    "constitution128.merged.jsonl"
)
NCCL_DIR = Path(
    "/Users/yinjinrun/random-thing/logs/muxi-nccl-campaign-20260711/nccl-results"
)
SEM_LINK = "METRIC_SEMANTICS_MUXI_20260711.md"
SIZE_256M = 256 * 1024 * 1024
FLUFF = ("直方图", "怎么看", "红虚线", "橙虚线", "出图脚本")

# 字段 → 人话含义（短）+ 底层 API（短）。详细版见 METRIC_SEMANTICS_MUXI_20260711.md
# 相对昇腾 SEM：NPU Event→CUDA/MACA Event；npu-smi→mx-smi；Cube→方阵 GEMM / MetaX 主算力路径
SEM: dict[str, tuple[str, str]] = {
    "func_tflops": (
        "单卡方阵 GEMM 吞吐（TFLOPS）。`func_*` 是跨后端同构键名；沐曦实际测 MetaX MACA "
        "GEMM 主算力路径，不对应昇腾 Cube 硬件。",
        "torch 算子 `a@b`（bf16），FLOPs=`2·N³`，CUDA/MACA Event（`torch.cuda`）计时取中位；"
        "N=8192，warmup=20，iters=50。",
    ),
    "sustained_tflops": (
        "稳态方阵 GEMM 吞吐（TFLOPS）。连续烤机后的可持续算力，用来看降频/争用，不是瞬时峰值。",
        "循环 `a@b` 跑满 ~30s，每窗 50 次 GEMM 用 CUDA Event 计时；"
        "**卡级字段取最后一个时间窗**（非中位）。N=8192 bf16。",
    ),
    "hbm_gbps": (
        "HBM 有效带宽代理（GB/s）。看读+写通路是否异常掉速或成簇偏低。",
        "设备侧大缓冲 `dst = src * 2.0`（fp32，含一次乘法，非纯 DMA）；流量按 R+W；"
        "Event 计时中位。默认 1024MB，w20/i50。",
    ),
    "vector_gflops": (
        "宽向量 FMA 吞吐代理（GFLOPS）。`vector_*` 是同构键名；沐曦实际走 MetaX MACA "
        "逐元素算子路径，不对应昇腾 Vector Core。",
        "逐元素 `a*b+c`，按 2 flops/elem；64M 元素 fp32；CUDA Event 中位。w20/i50。",
    ),
    "scalar_elems_per_s": (
        "长依赖串行链吞吐（元素/秒）。更贴近 Scalar/控制流+同步，不是 SIMD 峰值。",
        "`torch.cumsum`；elems_per_s = elems/dt；16M fp32。量纲不是 GFLOPS，勿与 vector 直接比倍速。",
    ),
    "mte_gbps": (
        "纯 copy / DMA 带宽（GB/s）。`mte_*` 只是昇腾同构遗留键名；沐曦实际走 MetaX MACA "
        "DMA copy 路径，不表示存在 Ascend MTE 硬件。",
        "`Tensor.copy_`；流量按 R+W；512MB；CUDA/MACA Event 中位。新别名 `dma_copy_gbps`。",
    ),
    "cube_vector_tflops": (
        "方阵 GEMM + Vector epilogue（scale+bias）端到端吞吐（TFLOPS）。"
        "`cube_*` / `vector_*` 只是昇腾同构遗留键名；沐曦实际走 MetaX MACA 的 "
        "GEMM→逐元素 epilogue 路径，不对应 Ascend Cube/Vector 硬件。",
        "`c=a@b; c*scale+bias`；FLOPs=`2N³+3N²`；N=4096 bf16。新别名 `gemm_epilogue_tflops`。",
    ),
    "sfu_gflops": (
        "特殊函数单元吞吐。字段叫 gflops，实现按 1 op/元素计，实质是 Gops/s 量级。",
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
        "CPU `perf_counter` 包一层 `adapter.sync`（`torch.cuda.synchronize`）；"
        "samples=500，warmup=50。与 kernel 发射无关。",
    ),
    "launch_sync_p99_us": (
        "同上的 p99（µs）。看调度抖动尾延迟。",
        "同 launch_latency 探针。",
    ),
    "launch_host_overhead_p50_us": (
        "Host 侧发射开销 p50（µs）≈ wall − device event。",
        "极小核 add 的墙钟与 CUDA Event 差分；需 timing_method=event 才有意义。",
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
        "开测早期轻载温度快照（°C）；`health` 是采样阶段标签，不是健康分。"
        "沐曦侧默认是 hotspot/结温代理。",
        "由 `MxSmiProvider` 调用 `mx-smi` 采集，不使用 `npu-smi`。"
        "与负载 `board_temp_c` / hotspot **不同时刻**；"
        "本批 JSONL 的 board_temp **已采集**（`--show-temperature` TTL；与 dmon hotspot 分传感器）。",
    ),
    "health_power_w": (
        "开测早期轻载实时功耗（W），常近空闲；`health` 不是健康分。",
        "由 `MxSmiProvider` 调用 `mx-smi` 采集实时功耗，不使用 `npu-smi`。"
        "**不要**和 `power_w`（负载末）直接相减当降频证据。",
    ),
    "board_temp_c": (
        "板温（°C）。跨厂商通用键名；本批取 Board Temperature 传感器峰值。",
        "`mx-smi --show-temperature` TTL 合并（不覆盖 dmon hotspot/power）。本批已落盘。",
    ),
    "aicore_util_pct": (
        "GPU 利用率（%）。JSONL 键名 `aicore_*` 为昇腾同构兼容；沐曦语义是 GPU util。",
        "`mx-smi --show-usage`（TTL）。本批已落盘；别名 `gpu_util_pct`。",
    ),
    "aicpu_util_pct": (
        "昇腾 AICPU 利用率（%）。沐曦无对应单元；字段保留兼容，常空。",
        "勿与 `vpu_util_pct`（视频处理单元）混读。",
    ),
    "ctrlcpu_util_pct": (
        "昇腾 CtrlCPU 利用率（%）。沐曦路径通常空。",
        "本批 **未采集 / 全空**。",
    ),
    "mem_bw_util_pct": (
        "显存带宽利用率（%）。",
        "可用 `mx-smi --show-hbm-bandwidth` 对照；本批体质 JSONL **未接线**。",
    ),
    "power_w": (
        "负载探针时段实时功耗（W）。",
        "`mx-smi` 功耗；卡级常取 vector_fma **末轮**。与 `health_power_w`（轻载开测快照）工况不同。",
    ),
    "power_limit_w": (
        "功耗上限 / 功耗墙（W）。",
        "`mx-smi --show-board-power`；本批中位 550 W。",
    ),
    "shape_sweep_peak_tflops": (
        "字段名沿用「shape sweep 峰值」，本批实为各 BNMK shape 中位 TFLOPS 的 max（名不副实）。",
        "旧独立 shape_sweep 开关已关；以 `gemm_bnmk_sample` 为准。",
    ),
    "aicore_freq_mhz": (
        "XCORE 时钟（MHz）。键名 `aicore_*` 兼容昇腾；沐曦来自 `clocks.XCORE.XCORE_CLK`。",
        "负载路径 TTL 合并 `mx-smi -j`；别名 `xcore_clk_mhz` / `sm_clock_mhz`。本批已落盘。",
    ),
    "hbm_temp_c": (
        "HBM 温度（°C）。",
        "`mx-smi` 多传感器；本批历史数据常空。",
    ),
}


def load_cards() -> list[dict]:
    out = []
    with MERGED.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("record") == "card":
                out.append(r)
    return out


def med(cards: list[dict], key: str) -> float | None:
    vs = [float(c[key]) for c in cards if c.get(key) is not None]
    return statistics.median(vs) if vs else None


def n_nonnull(cards: list[dict], key: str) -> int:
    return sum(1 for c in cards if c.get(key) is not None)


def fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}g}"


def list_svgs(d: Path) -> list[str]:
    return sorted(p.name for p in d.glob("*.svg"))


def meaning_block(key: str) -> str:
    if key in SEM:
        what, how = SEM[key]
        # Muxi 文案已在 SEM 中给出同构键名与 MetaX 实际路径；不要调用
        # glossary_links.linkify_abbr，以免注入 Ascend Cube/MTE/Vector 硬件释义。
        return f"**含义**：{what}  **底层**：{how}"
    return (
        f"**含义**：字段 `{key}` 的语义见 "
        f"[`{SEM_LINK}`]({SEM_LINK})。"
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
    mv_s = fmt(mv)
    show = {
        "hist": "全卡分布",
        "heatmap_relmed": "host×device 相对集群中位偏差%（|Δ|≥1% 才标数）",
        "box_by_host": "分 host 箱线",
        "sorted_bar": "单卡升序一览",
        "bar_host_mean_std": "分 host 均值±σ",
    }[m.group(1)]
    return f"**`{key}`**（{show}）。本批中位≈**{mv_s}**。{meaning_block(key)}"


def compute_nccl_retention() -> dict[str, dict[int, tuple[float, float | None]]]:
    """op -> world -> (median bus_bw @256MB, retention vs w8)."""
    by: dict[tuple[str, int], list[float]] = defaultdict(list)
    if not NCCL_DIR.is_dir():
        return {}
    for f in sorted(NCCL_DIR.glob("scale_*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("record") != "nccl_bench":
                    continue
                if int(r.get("nbytes") or 0) != SIZE_256M:
                    continue
                by[(str(r["op"]), int(r["world_size"]))].append(float(r["bus_bw_GBps"]))
    out: dict[str, dict[int, tuple[float, float | None]]] = {}
    for (op, world), vs in by.items():
        m = statistics.median(vs)
        out.setdefault(op, {})[world] = (m, None)
    for op, worlds in out.items():
        base = worlds.get(8, (None, None))[0]
        for w, (bw, _) in list(worlds.items()):
            ret = (bw / base) if base and base > 0 else None
            worlds[w] = (bw, ret)
    return out


def write_constitution(cards: list[dict]) -> int:
    fig = ROUNDS / "card_constitution_muxi_20260711_figs"
    svgs = list_svgs(fig)
    n_func = n_nonnull(cards, "func_tflops")
    lines = [
        "# Card Constitution · Muxi · 20260711",
        "",
        "**怎么读**：关键缩写在正文括号附注；完整硬件词条附录："
        "[`METAX_HARDWARE_GLOSSARY_20260711.md`](METAX_HARDWARE_GLOSSARY_20260711.md)。"
        f"测法：[`{SEM_LINK}`]({SEM_LINK})。  ",
        "JSON 键里的 Cube / MTE / AICore 是昇腾同构壳；沐曦侧架构对齐见 "
        "[`../research/METAX_ARCH_ALIGNMENT_20260711.md`](../research/METAX_ARCH_ALIGNMENT_20260711.md)"
        "（正文附注 + glossary 附录可并存）。下文在遗留键名后用括号附注沐曦对应语义，"
        "避免把同构键名误读为沐曦硬件名称。  ",
        "数据：`logs/muxi-constitution-20260711_232400-muxi-constitution128/results/"
        "constitution128.merged.jsonl`；"
        "job `yushan-muxi-card-screen-128-cp-copy` **16×8=128**；"
        "`screen.py` + `config.constitution128.yaml`；"
        "`--sdc-rounds 5 --gemm-n 8192 --sustained-s 30`。"
        "计时：CUDA/MACA Event（`torch.cuda`）；遥测：`mx-smi`。"
        "本批 **有 BNMK sample**；board_temp / GPU util / XCORE clk **已落盘**。"
        "`health_*` 的 `health` 只是采样阶段标签，**不是健康分**。",
        "",
        "## 关键中位",
        "",
        "| 字段 | 人话 | 中位 | 覆盖 |",
        "|---|---|---:|---:|",
    ]
    for k, lab in [
        ("func_tflops", "方阵 GEMM / MetaX 主算力"),
        ("sustained_tflops", "稳态 GEMM"),
        ("hbm_gbps", "HBM 带宽代理"),
        ("vector_gflops", "向量 FMA 代理（MACA 路径）"),
        ("mte_gbps", "纯 copy/DMA（同构键 MTE）"),
        ("cube_vector_tflops", "GEMM+epilogue（同构键 cube_*）"),
        ("sfu_gflops", "SFU/特殊函数（Gops/s 量级）"),
        ("health_power_w", "轻载开测功耗（≠健康分）"),
        ("power_w", "负载末功耗"),
        ("power_limit_w", "功耗墙"),
        ("health_temp_c", "轻载开测温度（≠健康分）"),
        ("board_temp_c", "板温快照"),
        ("aicore_util_pct", "GPU util（同构键 aicore_*）"),
        ("aicore_freq_mhz", "XCORE clk（同构键 aicore_*）"),
    ]:
        v = med(cards, k)
        n = n_nonnull(cards, k)
        lines.append(
            f"| `{k}` | {lab} | {fmt(v)} | {n}/{len(cards)} |"
            if v is not None
            else f"| `{k}` | {lab} | — | {n}/{len(cards)} |"
        )

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
        lines.append(f"![{name}](card_constitution_muxi_20260711_figs/{name})")
        lines.append("")

    path = ROUNDS / "card_constitution_muxi_20260711.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path.name, len(svgs), f"(func覆盖 {n_func}/{len(cards)})")
    return len(svgs)


def write_extra(cards: list[dict]) -> int:
    fig = ROUNDS / "constitution_extra_muxi_20260711_figs"
    if not fig.is_dir():
        print("skip constitution_extra: figs dir missing")
        return 0
    svgs = list_svgs(fig)
    lines = [
        "# Constitution 增强图 · Muxi · 20260711",
        "",
        "**怎么读**：硬件词条 [`METAX_HARDWARE_GLOSSARY_20260711.md`](METAX_HARDWARE_GLOSSARY_20260711.md)；"
        f"字段测法 [`{SEM_LINK}`]({SEM_LINK})。"
        "同一 constitution merged JSONL（`muxi-constitution-20260711_232400-…`）。"
        "Cube/MTE 等为同构壳键名，实测为 MetaX/MACA 探针路径。",
        "",
    ]
    caps = {
        "radar_host_median_norm.svg": (
            "各 host 在多指标上的**中位相对集群中位**（1.0=集群水平）。"
            "用来看机间体质是否齐，不是单卡绝对值。"
        ),
        "hbm_modes_grouped_bar.svg": (
            "四种 HBM 访问模式带宽："
            "`seq_copy` / `strided` / `read_heavy` / `write_heavy`。"
            "底层是 `hbm_modes_perf`（copy / 跨步 / sum / fill），单位 GB/s；"
            "**跨模式绝对值不可直接比「谁更好」**。"
        ),
        "corr_cube_vector_sfu_mte.svg": (
            "方阵 GEMM / 宽向量算子 / SFU / 纯 copy 四路吞吐的 Pearson 相关。"
            "图名中的 `cube` / `mte` 是昇腾同构遗留键名；沐曦实际走 MetaX MACA 的 "
            "GEMM、逐元素算子与 `Tensor.copy_` DMA copy 路径，不表示存在 Ascend Cube/MTE 硬件。"
            "看子系统是否同涨同跌；相关≈0 表示彼此相对独立。"
        ),
        "box_launch_by_host.svg": (
            "Launch 延迟分 host：`launch_sync_p99_us` / "
            "`launch_host_overhead_p99_us` / `launch_burst_p50_us`。"
            "**含义**：空 sync / host 发射开销 / 突发总时延（µs），看调度抖动尾延迟。"
            "  **底层**：`launch_latency` 探针（CPU 计时 + CUDA sync/Event）。"
        ),
        "cdf_core_metrics.svg": (
            "核心吞吐指标的经验分布函数（CDF）。"
            "**含义**：对 `func_tflops` / `hbm_gbps` / `vector_gflops` 等卡级字段做 ECDF，"
            "看集群齐性与尾部。  **底层**：同 constitution merged JSONL 的 stage_a/c 探针中位。"
        ),
        "parallel_host_median_norm.svg": (
            "与雷达同一套 host 中位归一化；平行坐标展示 "
            "`func_tflops` / `hbm_gbps` / `vector_gflops` / `mte_gbps` 等相对中位。"
        ),
        "extreme10_small_multiples.svg": (
            "按 `sustained_tflops` 最慢/最快各 10 卡，多指标相对集群中位偏差。"
            "**含义**：极端卡剖面；对照「慢卡是否多项一起慢」。"
            "  **底层**：同 merged JSONL 卡级字段。"
        ),
        "scatter_sustained_vs_func.svg": (
            f"横轴短测方阵 GEMM，纵轴稳态 GEMM。{meaning_block('func_tflops')} "
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
        lines.append(f"![{name}](constitution_extra_muxi_20260711_figs/{name})")
        lines.append("")
    path = ROUNDS / "constitution_extra_muxi_20260711.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path.name, len(svgs))
    return len(svgs)


def write_nccl(ret: dict[str, dict[int, tuple[float, float | None]]]) -> int:
    fig = ROUNDS / "nccl_campaign_muxi_20260711_figs"
    if not fig.is_dir():
        print("skip nccl figs: directory nccl_campaign_muxi_20260711_figs missing")
        # 仍写无图版含义+保持率（覆盖旧「画图空话」报告）
        svgs: list[str] = []
    else:
        svgs = list_svgs(fig)

    worlds = [8, 16, 32, 64, 128]
    op_order = ["all_reduce", "broadcast", "all_gather", "reduce_scatter"]
    op_label = {
        "all_reduce": "All-Reduce",
        "broadcast": "Broadcast",
        "all_gather": "All-Gather",
        "reduce_scatter": "Reduce-Scatter",
    }

    lines = [
        "# NCCL/MCCL 通信 · Muxi · 20260711",
        "",
        f"语义详见 [`{SEM_LINK}`]({SEM_LINK}) 通信章。",
        "",
        "**alg_bw**：业务字节 / 平均时延 → GB/s（算法视角）。",
        "**bus_bw**：按 NCCL-tests 同构公式把多跳折成可与链路比的总线带宽——"
        "AllReduce `×2(n-1)/n`，AG/RS `×(n-1)/n`，Broadcast `=alg`。",
        "扩展叙事用 **bus_bw 保持率 = bus_N / bus_8**（沐曦单节点 **8** 卡；"
        "**不是**昇腾的 `/bus_16`）。",
        "",
        "底层：`torch.distributed` + **NCCL/MCCL**（`nccl_torch_bench.py`）；"
        "CPU `perf_counter` + `torch.cuda.synchronize`；sizes 1M–256M；fp32；"
        "world **8→16→32→64→128**。",
        "环境：`NCCL/MCCL/GLOO_SOCKET_IFNAME=eth0`（多机必设）。",
        "P2P：`nccl_p2p_bench.py`（isend/irecv，严格串行单对）。",
        "",
        "数据：本地 `logs/muxi-nccl-campaign-20260711/nccl-results/scale_*.jsonl`；"
        "AFS `/afs-a3-weight-share/montyyin/results/nccl-20260711_142129`；"
        "P2P AFS `…/nccl-p2p-20260711_150700`。",
        "",
        "## 256MB bus_bw 保持率（相对 **w8**，中位）",
        "",
        "| op | w8 bus | w16 | w32 | w64 | w128 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for op in op_order:
        cells = []
        w8 = ret.get(op, {}).get(8, (None, None))[0]
        cells.append(fmt(w8, 4) if w8 is not None else "—")
        for w in (16, 32, 64, 128):
            _, r = ret.get(op, {}).get(w, (None, None))
            cells.append(f"{r * 100:.2f}%" if r is not None else "—")
        lines.append(f"| {op_label[op]} | " + " | ".join(cells) + " |")

    ar8 = ret.get("all_reduce", {}).get(8, (None, None))[0]
    ar16_r = ret.get("all_reduce", {}).get(16, (None, None))[1]
    lines += [
        "",
        f"→ All-Reduce@256MB：w8 中位 bus≈**{fmt(ar8)} GB/s**；"
        f"w16 保持率≈**{(ar16_r * 100):.2f}%**（断崖在首次跨节点）。"
        if ar8 is not None and ar16_r is not None
        else "",
        "",
    ]

    if svgs:
        lines += ["## 逐图（含义优先）", ""]
        for name in svgs:
            if name.startswith("nccl_bus_bw_vs_size_") or name.startswith(
                "hccl_bus_bw_vs_size_"
            ):
                op = (
                    name.replace("nccl_bus_bw_vs_size_", "")
                    .replace("hccl_bus_bw_vs_size_", "")
                    .replace(".svg", "")
                )
                cap = (
                    f"**`{op}` 的 bus_bw 随消息大小**。"
                    f"底层 `dist.{op}`（all_gather/reduce_scatter 按 world 切分缓冲）；"
                    f"每点是该 (world,size) 下各 rank bus_bw 的中位。"
                )
            elif "retention" in name:
                cap = "256MB 上各 collective 的 bus_bw 相对 world=8 的保持率（扩展健康度）。"
            elif "step" in name:
                cap = "固定 256MB，world 从 8→128 时 bus_bw 中位的阶梯变化。"
            elif name.startswith("nccl_rank_") or name.startswith("hccl_rank_"):
                wm = re.search(r"_w(\d+)_", name)
                wlab = f"world={wm.group(1)}" if wm else "各 world"
                cap = (
                    f"同一 (op, {wlab}, 256MB) 下**每个 rank 各自的 bus_bw** 分布。"
                    "看是否个别 rank 拖总线折算带宽。"
                )
            elif name.startswith("p2p_"):
                if "slow" in name:
                    cap = (
                        "**点对点慢边 TopK**（单向 isend/irecv GB/s）。"
                        "底层 `nccl_p2p_bench.py`；看跨节点/异常边是否拖后腿。"
                    )
                elif "fast" in name:
                    cap = (
                        "**点对点快边 TopK**（单向 isend/irecv GB/s）。"
                        "底层 `nccl_p2p_bench.py`；对照机内 MetaXLink 饱和区。"
                    )
                elif "kind" in name or "compare" in name or "violin" in name or "box" in name:
                    cap = (
                        "**点对点 isend/irecv 单向带宽**（GB/s）按边类型/规模对照，不是 bus_bw 公式。"
                        "底层 `nccl_p2p_bench.py`，严格串行单对；`torch.cuda.synchronize`。"
                    )
                else:
                    cap = (
                        "**点对点 isend/irecv 单向带宽**（GB/s），不是 bus_bw 公式。"
                        "边类型含 ring；大 world 默认仅 ring。"
                        "底层 `nccl_p2p_bench.py`，严格串行单对；`torch.cuda.synchronize`。"
                    )
            elif "topo" in name:
                cap = (
                    "机内物理拓扑亲和：来自 **`mx-smi topo`** 解析（MetaXLink / SYS）。"
                    "这是静态拓扑关系，不是测速。"
                )
            else:
                cap = name
            lines.append(f"**{name}**：{cap}")
            lines.append("")
            lines.append(f"![{name}](nccl_campaign_muxi_20260711_figs/{name})")
            lines.append("")
    else:
        lines += [
            "## 图",
            "",
            "本批尚未落地 `nccl_campaign_muxi_20260711_figs/`（无 SVG）。"
            "保持率与结论以上表为准；短版见 "
            "[`muxi_nccl_scale_20260711.md`](muxi_nccl_scale_20260711.md)。",
            "",
        ]

    path = ROUNDS / "nccl_campaign_muxi_20260711.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path.name, len(svgs), "(figs)" if svgs else "(no figs dir)")
    return len(svgs)


def write_provenance(n_const: int, n_extra: int, n_nccl: int) -> None:
    lines = [
        "# 图溯源 · Muxi · 20260711（精简）",
        "",
        "> 对标昇腾 [`FIGURE_PROVENANCE_AUDIT_20260711.md`](FIGURE_PROVENANCE_AUDIT_20260711.md) 结构精简版。  ",
        "> Job：`yushan-muxi-card-screen-128-cp-copy`，**16×8=128** MetaX C550-PL。  ",
        "> 出图规范：默认 `reports/plot_style.py`（大字号 / 去顶右边框 / y 点线网格 / hatch / **SVG**）。",
        "",
        "## 0. 图目录 ↔ 绘图入口 ↔ 原始数据",
        "",
        "| 图目录 | SVG 数 | 绘图入口 | 原始数据 |",
        "|--------|--------|----------|----------|",
        f"| `card_constitution_muxi_20260711_figs/` | {n_const} | `reports/plot_card_constitution.py` | "
        "`logs/muxi-constitution-20260711_232400-muxi-constitution128/results/constitution128.merged.jsonl` |",
        f"| `constitution_extra_muxi_20260711_figs/` | {n_extra} | `reports/plot_constitution_extra.py` | 同上 merged JSONL |",
        f"| `nccl_campaign_muxi_20260711_figs/` | {n_nccl} | （若有）NCCL 同构 plot 入口 | "
        "`logs/muxi-nccl-campaign-20260711/nccl-results/scale_*.jsonl`；"
        "AFS `/afs-a3-weight-share/montyyin/results/nccl-20260711_142129` |",
        "",
        "样式统一走 `reports/plot_style.py`。",
        "",
        "### 0.1 体质采集公共条件",
        "",
        "- **Launch**（`logs/muxi-constitution-20260711_232400-muxi-constitution128/launch_one.sh`）：",
        "  ```text",
        "  python screen.py --device all --config config.constitution128.yaml \\",
        "    --sdc-rounds 5 --gemm-n 8192 --sustained-s 30 \\",
        "    --out .../constitution128.jsonl --no-plot",
        "  ```",
        "- **配置**：`projects/CARD_SCREEN/config.constitution128.yaml`",
        "- **落库**：`card_screen/io/jsonl.py` → `record=card` + 各 round/sample 行",
        "- **遥测**：`MxSmiProvider` → **`mx-smi`**（温度/功耗/拓扑）；**禁止**套用昇腾 `npu-smi -t …`",
        "- **计时**：CUDA/MACA Event（`torch.cuda`），不是 NPU Event",
        "- **合流**：各 pod JSONL → `constitution128.merged.jsonl`",
        "",
        "### 0.2 本批数据状态",
        "",
        "- **BNMK sample 已落盘**（`gemm_bnmk_sample`；出图可另开 bnmk 入口）",
        "- **board_temp / GPU util / XCORE clk 已落盘**（出图可见）",
        "- NCCL 跨节点走 **`SOCKET_IFNAME=eth0`**；拓扑可见 mlx5/xscale，本批未切 IB 数据面",
        "- master 8 卡 **contended**（preflight 撞到残留 compute 进程）；worker-12:0 **bad**",
        "",
        "## 1. NCCL 数据路径",
        "",
        "| 项 | 路径 |",
        "|----|------|",
        "| 本地 campaign | `logs/muxi-nccl-campaign-20260711/nccl-results/scale_{8,16,32,64,128}.jsonl` |",
        "| AFS collective | `/afs-a3-weight-share/montyyin/results/nccl-20260711_142129` |",
        "| AFS P2P | `/afs-a3-weight-share/montyyin/results/nccl-p2p-20260711_150700` |",
        "| 本地镜像 | `logs/muxi-nccl-campaign-20260711/{nccl-20260711_142129,nccl-p2p-20260711_150700,p2p-results}/` |",
        "",
        "保持率现算：256MB、`bus_bw_GBps` 中位、相对 **w8**。",
        "",
        f"*配套语义：[`{SEM_LINK}`]({SEM_LINK})；总汇报：[`CAMPAIGN_FINAL_MUXI_20260711.md`](CAMPAIGN_FINAL_MUXI_20260711.md)*",
        "",
    ]
    path = ROUNDS / "FIGURE_PROVENANCE_MUXI_20260711.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path.name)


def write_campaign(
    cards: list[dict],
    ret: dict[str, dict[int, tuple[float, float | None]]],
    n_const: int,
    n_extra: int,
    n_nccl: int,
) -> None:
    def m(k: str) -> float | None:
        return med(cards, k)

    def cov(k: str) -> str:
        return f"{n_nonnull(cards, k)}/{len(cards)}"

    ar = ret.get("all_reduce", {})
    rows_ar = []
    for w in (8, 16, 32, 64, 128):
        bw, r = ar.get(w, (None, None))
        if w == 8:
            rows_ar.append(f"| {w} | **{fmt(bw)}** | 100% |")
        else:
            pct = f"**{r * 100:.2f}%**" if r is not None else "—"
            rows_ar.append(f"| {w} | {fmt(bw)} | {pct} |")

    # 四算子保持率表（vs w8）
    op_order = ["all_reduce", "broadcast", "all_gather", "reduce_scatter"]
    op_label = {
        "all_reduce": "All-Reduce",
        "broadcast": "Broadcast",
        "all_gather": "All-Gather",
        "reduce_scatter": "Reduce-Scatter",
    }
    ret_table = [
        "| 算子 | w16 | w32 | w64 | w128 |",
        "|------|-----|-----|-----|------|",
    ]
    for op in op_order:
        cells = []
        for w in (16, 32, 64, 128):
            _, r = ret.get(op, {}).get(w, (None, None))
            cells.append(f"{r * 100:.2f}%" if r is not None else "—")
        ret_table.append(f"| {op_label[op]} | " + " | ".join(cells) + " |")

    ar8 = ar.get(8, (None, None))[0]
    ar16r = ar.get(16, (None, None))[1]
    ar128r = ar.get(128, (None, None))[1]

    lines = [
        "# 128 卡体质 + 通信采集战役 · Muxi 最终汇报",
        "",
        "**日期**: 2026-07-11  ",
        "**Job**: `yushan-muxi-card-screen-128-cp-copy`（16×8 MetaX C550-PL = 128 卡）  ",
        "**集群**: `vc-c550-mohe-241`（kube 隔离：`scripts/cluster/muxi.env` → `CLUSTER_KUBECONFIG`）  ",
        "**对照计划**: [`../research/GOAL_MUXI_MIGRATE_STATUS.md`](../research/GOAL_MUXI_MIGRATE_STATUS.md)  ",
        f"**对标昇腾总汇报**: [`CAMPAIGN_FINAL_20260711.md`](CAMPAIGN_FINAL_20260711.md)  ",
        "**怎么读 / 硬件对照**：正文附注 + "
        "[`METAX_HARDWARE_GLOSSARY_20260711.md`](METAX_HARDWARE_GLOSSARY_20260711.md)；"
        "架构对齐笔记 [`../research/METAX_ARCH_ALIGNMENT_20260711.md`]"
        "(../research/METAX_ARCH_ALIGNMENT_20260711.md)（可并存）。",
        "",
        "---",
        "",
        "## 1. 结论（相对 GOAL G0–G10）",
        "",
        "| 计划项 | 状态 | 说明 |",
        "|--------|------|------|",
        "| G0 双集群隔离 | **完成** | `huawei.env` / `muxi.env`；不覆盖默认 kubeconfig |",
        f"| G1 快慢卡冒烟 128 | **完成** | good=106 / slow=19 / bad=1 / contended=2 |",
        f"| G2 体质 constitution | **完成** | {cov('func_tflops')} 有效；func 中位 "
        f"**{fmt(m('func_tflops'))} TFLOPS**；含义优先图 {n_const} SVG |",
        "| G3 拓扑 | **完成** | 16/16 `mx-smi`；机内 MetaXLink；NIC mlx5+xscale |",
        f"| G4 NCCL collective | **完成** | 8→128；单机 AR@256M ≈**{fmt(ar8)} GB/s**；"
        f"跨节点保持率 ~**{(ar16r * 100):.2f}%** |"
        if ar8 is not None and ar16r is not None
        else "| G4 NCCL collective | **完成** | 8→128；见 NCCL 报告 |",
        "| G5 NCCL P2P | **完成** | ring 16/128；机内 16M ≈30–33 GB/s；跨节点 ≈0.35 |",
        "| G6 链路健康 | **完成** | 16/16 mx-smi + ibv 文本 |",
        "| G7 一键流水线 | **完成** | `run_constitution_then_comm_muxi.sh` |",
        "| G8 MFU 微基准 | **完成** | dense@8=**26.7%**；跨节点 ~0.2%；moe@8=15.0% |",
        "| G9 真训练 MFU | **完成** | tiny GPT 8 卡 5iter 通；稳态≈54ms；估算 MFU≈4.5% |",
        "| G10 报告对齐 | **完成** | 本文件 + 语义手册 + 溯源 + 含义优先分报告 |",
        "",
        "**计划达成度：GOAL 主路径 100%。**  ",
        "剩余为 **可选增强**：跨节点切 IB/`net*` 重测；TE fused attn 符号补齐后冲高真训练 MFU；"
        "master contended / worker-12:0 单卡复测。",
        "",
        "---",
        "",
        "## 2. 关键数字",
        "",
        "### 体质（constitution128 merged，现算中位）",
        "",
        "| 指标 | 中位 | 覆盖 |",
        "|------|------|------|",
        f"| 方阵 GEMM func TFLOPS | **{fmt(m('func_tflops'))}** | {cov('func_tflops')} |",
        f"| Sustained TFLOPS | **{fmt(m('sustained_tflops'))}** | {cov('sustained_tflops')} |",
        f"| HBM 带宽 GB/s | **{fmt(m('hbm_gbps'))}** | {cov('hbm_gbps')} |",
        f"| Vector GFLOPS（MACA 路径） | **{fmt(m('vector_gflops'))}** | {cov('vector_gflops')} |",
        f"| SFU（Gops/s 量级） | **{fmt(m('sfu_gflops'))}** | {cov('sfu_gflops')} |",
        f"| 纯 copy / DMA GB/s | **{fmt(m('mte_gbps'))}** | {cov('mte_gbps')} |",
        f"| GEMM+epilogue TFLOPS | **{fmt(m('cube_vector_tflops'))}** | {cov('cube_vector_tflops')} |",
        f"| 早期轻载功耗 health_power_w | **{fmt(m('health_power_w'))} W** | {cov('health_power_w')} |",
        f"| 满载功耗 power_w | **{fmt(m('power_w'))} W** | {cov('power_w')} |",
        f"| 功耗墙 power_limit_w | **{fmt(m('power_limit_w'))} W** | {cov('power_limit_w')} |",
        f"| 早期轻载温度 health_temp_c | **{fmt(m('health_temp_c'))} °C** | {cov('health_temp_c')} |",
        f"| GPU util（aicore_util_pct） | **{fmt(m('aicore_util_pct'))}%** | {cov('aicore_util_pct')} |",
        f"| XCORE clk（aicore_freq_mhz） | **{fmt(m('aicore_freq_mhz'))}** | {cov('aicore_freq_mhz')} |",
        f"| board_temp_c | **{fmt(m('board_temp_c'))} °C** | {cov('board_temp_c')} |",
        "| 判定（本批体质） | good **119** / contended **8** / bad **1** | 128 |",
        "| BNMK | **有 sample** | 本批已开 |",
        "",
        "`health_power_w` / `health_temp_c` 是流程早期轻载阶段通过 `mx-smi` 取得的快照；"
        "`health` 只是采样阶段标签，不表示“健康分”，也不等同于负载阶段遥测。",
        "",
        "**判定口径不要混用**：冒烟判定为 good=106 / slow=19 / bad=1（另有 contended=2），"
        "体质判定为 good=119 / contended=8 / bad=1；两者采样阶段与规则不同。",
        "",
        "### 通信（All-Reduce @ 256MB bus_bw 保持率 vs **w8**，现算）",
        "",
        "| world | bus_bw (GB/s) | 保持率 vs w8 |",
        "|------:|--------------:|-------------:|",
        *rows_ar,
        "",
        "四算子保持率（256MB，相对 w8）：",
        "",
        *ret_table,
        "",
        "→ **断崖在 8→16（首次跨节点）**；其后几乎持平。机内健康，跨节点 eth0 打穿。",
        "",
        "### MFU / 训练",
        "",
        "| 项 | 值 |",
        "|----|-----|",
        "| dense MFU @8 | **26.7%**（peak=279×8） |",
        "| dense MFU @16–128 | **0.22–0.32%** |",
        "| moe MFU @8 | **15.0%** |",
        "| 真训练 tiny GPT @8 | 稳态 **53.9 ms/iter**；估算 MFU **~4.5%**（local/unfused） |",
        "",
        "---",
        "",
        "## 3. 产物索引（看图从这里进）",
        "",
        "> 出图默认 `reports/plot_style.py`（大字号 / 去顶右边框 / y 点线网格 / hatch 柱 / **SVG**）。  ",
        "> **图注优先讲清：字段人话含义 + 底层 API/命令/算子**（禁止画图空话）。  ",
        f"> 语义手册：[`{SEM_LINK}`]({SEM_LINK})；"
        "采集链路溯源：[`FIGURE_PROVENANCE_MUXI_20260711.md`](FIGURE_PROVENANCE_MUXI_20260711.md)。",
        "",
        "### 体质主报告",
        f"- [`card_constitution_muxi_20260711.md`](card_constitution_muxi_20260711.md)",
        f"- [`card_constitution_muxi_20260711_figs/`](card_constitution_muxi_20260711_figs/)（**{n_const}** svg）",
        "  - 注意：`timeseries_sustained_p05_p50.svg` = **跨卡** p05/p50（按 iter 对齐），不是代表卡时序",
        "",
        "### 体质增强",
        f"- [`constitution_extra_muxi_20260711.md`](constitution_extra_muxi_20260711.md)",
        f"- [`constitution_extra_muxi_20260711_figs/`](constitution_extra_muxi_20260711_figs/)（{n_extra}）",
        "",
        "### NCCL/MCCL + P2P + 拓扑",
        f"- [`nccl_campaign_muxi_20260711.md`](nccl_campaign_muxi_20260711.md)"
        + (f" / [`nccl_campaign_muxi_20260711_figs/`](nccl_campaign_muxi_20260711_figs/)（{n_nccl}）" if n_nccl else "（本批暂无 `_figs/`）"),
        "- 短版：[`muxi_nccl_scale_20260711.md`](muxi_nccl_scale_20260711.md) · [`muxi_nccl_p2p_20260711.md`](muxi_nccl_p2p_20260711.md)",
        "- 拓扑：[`muxi_topo_20260711.md`](muxi_topo_20260711.md) · 链路：[`muxi_link_health_20260711.md`](muxi_link_health_20260711.md)",
        "",
        "### MFU / 真训练",
        "- [`muxi_mfu_bench_20260711.md`](muxi_mfu_bench_20260711.md)",
        "- [`muxi_train_mfu_20260711.md`](muxi_train_mfu_20260711.md)",
        "",
        "### 对照与总览",
        "- [`muxi_vs_huawei_align_20260711.md`](muxi_vs_huawei_align_20260711.md)",
        "- 昇腾总汇报：[`CAMPAIGN_FINAL_20260711.md`](CAMPAIGN_FINAL_20260711.md)",
        "",
        "### 原始数据",
        "- 体质: `logs/muxi-constitution-20260711_232400-muxi-constitution128/results/constitution128.merged.jsonl`",
        "- NCCL 本地: `logs/muxi-nccl-campaign-20260711/nccl-results/`",
        "- NCCL AFS: `/afs-a3-weight-share/montyyin/results/nccl-20260711_142129`",
        "- 冒烟: `logs/muxi-card-screen-20260711_133828-muxi-smoke/`",
        "",
        "---",
        "",
        "## 4. 本轮修过的坑（便于复现）",
        "",
        "1. **双集群 kubeconfig**：只用 `CLUSTER_KUBECONFIG`，永不覆盖 weibozhen 默认 config。",
        "2. **长任务**：本机 `nohup`/长 SSH 易被 IDE 杀掉 → **pod 内 `setsid nohup` + 短连 fire/poll**。",
        "3. **`pod exec -i` 上传与启动分离**：合并会导致空文件。",
        "4. **多机 NCCL**：必须 `*_SOCKET_IFNAME=eth0`，否则 Proxy Connect 失败。",
        "5. **扇出并发**：16 路过猛会 SSH 踢人 → `CLUSTER_FANOUT_PARALLEL≈4–6`。",
        "6. **AFS 写结果**：经 pod 写；登录机假挂载不可用。",
        "7. **G9 nvcc**：`CUDA_HOME/bin/nvcc` ← symlink `cucc`；TE fused attn 缺符号 → `local/unfused`。",
        "8. **假成功**：`torchrun | tee` 必须用 `PIPESTATUS[0]`。",
        "9. **遥测完整性**：本批 merged JSONL（`20260711_232400`）已包含 board_temp、"
        "GPU util、XCORE clk 与 BNMK sample，与第 2 节统计口径一致。",
        "",
        "---",
        "",
        "## 5. 一眼叙事",
        "",
        f"这批 128 张 C550 **机内算力很齐**（func 中位 **{fmt(m('func_tflops'))} TFLOPS**，"
        f"覆盖 {cov('func_tflops')}），HBM 中位高但有整节点掉速簇；冒烟有 1 张正确性坏卡。",
        "",
        f"通信上 **单机 All-Reduce@256MB 中位 ≈{fmt(ar8)} GB/s**，但 "
        f"**一跨节点保持率就掉到 ~{(ar16r * 100):.2f}%**（w128 ≈{(ar128r * 100):.2f}%），"
        if ar8 is not None and ar16r is not None and ar128r is not None
        else "通信上单机健康、跨节点断崖，",
        "P2P/MFU 同步复现同一断崖——根因是当前走 **eth0 socket**，不是 MetaXLink 坏了。"
        "拓扑已看到 mlx5/xscale，下一步应切 IB 重测。",
        "",
        "训练侧：微基准单机 MFU 26.7% 可用；Megatron tiny 冒烟已打通（需 nvcc shim + 避开 TE fused）。",
        "",
        "与昇腾同日战役对照：昇腾跨节点 All-Reduce 保持率仍约 89%，沐曦跨节点是 **路径问题**；"
        "机内两边都健康，可并行作为双集群基线。",
        "",
        "> 索引更新时间：2026-07-11（`rewrite_meaning_mds_muxi.py` 现算关键数字）",
        "",
    ]
    # clean accidental blank from conditional join
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    path = ROUNDS / "CAMPAIGN_FINAL_MUXI_20260711.md"
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    print("wrote", path.name)


def write_semantics(cards: list[dict], ret: dict[str, dict[int, tuple[float, float | None]]]) -> None:
    ar8 = ret.get("all_reduce", {}).get(8, (None, None))[0]
    ar16 = ret.get("all_reduce", {}).get(16, (None, None))[0]
    ar16r = ret.get("all_reduce", {}).get(16, (None, None))[1]

    def mv(k: str) -> str:
        return fmt(med(cards, k))

    def cov(k: str) -> str:
        return f"{n_nonnull(cards, k)}/{len(cards)}"

    lines = [
        "# 指标语义审计 · Muxi / MetaX C550 · 20260711",
        "",
        "> **用途**：为沐曦侧体质筛查（`record=card`，`backend=metax`）与通信战役"
        "（`record=nccl_bench` / `nccl_p2p`）提供字段级语义字典。  ",
        "> **对标**：昇腾 [`METRIC_SEMANTICS_20260711.md`](METRIC_SEMANTICS_20260711.md)"
        "（同构字段名；后端/遥测命令不同）。  ",
        "> **代码锚点**：`projects/CARD_SCREEN/.../stage_a.py`、`stage_c.py`、`io/jsonl.py`、"
        "`MetaxAdapter` / `MxSmiProvider`；`scripts/cluster/nccl_torch_bench.py`、"
        "`nccl_p2p_bench.py`、`mfu_train_bench_nccl.py`。  ",
        "> **本批数据路径**：  ",
        "> - 体质 merged：`logs/muxi-constitution-20260711_232400-muxi-constitution128/results/"
        "constitution128.merged.jsonl`  ",
        "> - NCCL 本地：`logs/muxi-nccl-campaign-20260711/nccl-results/scale_*.jsonl`  ",
        "> - NCCL AFS：`/afs-a3-weight-share/montyyin/results/nccl-20260711_142129`  ",
        "> - P2P AFS：`…/results/nccl-p2p-20260711_150700`  ",
        "> **遥测统一命令**：`mx-smi`（温度/功耗/拓扑/MetaXLink）；"
        "**禁止**把昇腾 `npu-smi -t …` 套到沐曦。  ",
        "> **计时**：CUDA/MACA Event（`torch.cuda`），不是 NPU Event。  ",
        "> **架构对齐**：[`../research/METAX_ARCH_ALIGNMENT_20260711.md`]"
        "(../research/METAX_ARCH_ALIGNMENT_20260711.md)（XCORE / MACA / MCCL / mx-smi）；"
        "报告附录 [`METAX_HARDWARE_GLOSSARY_20260711.md`](METAX_HARDWARE_GLOSSARY_20260711.md)。  ",
        "> **本批状态**：有 BNMK；board_temp / GPU util / XCORE clk 已落盘。",
        "",
        "---",
        "",
        "## 一、体质字段（`record=card`，backend=metax）",
        "",
        "卡级字段由 `jsonl.py` 从各探针 `perf` 子树扁平化写入。"
        "探针逻辑与昇腾同构（torch 算子 + Event 计时），设备路径走 **CUDA/MACA**"
        "（`torch.cuda` Event），不是 NPU Event。",
        "",
        "---",
        "",
        "### `func_tflops`（方阵 GEMM / MetaX 主算力 TFLOPS）",
        "",
        "- **是什么**：单卡方阵 GEMM 峰值吞吐代理（TFLOPS）。沐曦上反映 MetaX 主算力路径在方阵乘下的瞬时能力。",
        "- **怎么得到**：探针 `func_perf`。`c = a @ b`（bf16）；理论 FLOPs = `2·N³`；"
        "CUDA/MACA Event 计时；卡级取各轮 tflops **中位数**。",
        "- **关键参数**：`N=8192`，warmup=20，iters=50（`config.constitution128.yaml` +"
        " `launch_one.sh`：`--gemm-n 8192 --sdc-rounds 5 --sustained-s 30`）。",
        f"- **本批中位**：**{mv('func_tflops')} TFLOPS**（覆盖 {cov('func_tflops')}）。",
        "- **注意**：短窗峰值；与 `sustained_tflops` 对比看热稳态。"
        "勿与昇腾 Cube/矩阵绝对值直接当「谁更快」——硬件代差与探针参数需对齐后再比。",
        "",
        "---",
        "",
        "### `sustained_tflops`（稳态 GEMM TFLOPS）",
        "",
        "- **是什么**：连续烤机后的稳态方阵 GEMM 吞吐（TFLOPS）。",
        "- **怎么得到**：循环 `a @ b`，按时间窗聚合；卡级字段取 **最后一个时间窗**（非中位）。"
        "CUDA Event；~30s。",
        f"- **关键参数**：`--sustained-s 30`，每窗 50 次 GEMM，N=8192 bf16。",
        f"- **本批中位**：**{mv('sustained_tflops')} TFLOPS**（{cov('sustained_tflops')}）。",
        "- **注意**：末窗可能略高于/低于 func；含自洽检查。",
        "",
        "---",
        "",
        "### `hbm_gbps`（HBM GB/s）",
        "",
        "- **是什么**：HBM 有效带宽代理（GB/s）。`dst = src * 2.0`（fp32，含一次乘法，非纯 DMA）；流量按读+写计。",
        "- **怎么得到**：探针 `hbm`；Event 中位；默认 1024MB，w20/i50。",
        f"- **关键参数**：1024 MB fp32。",
        f"- **本批中位**：**{mv('hbm_gbps')} GB/s**（{cov('hbm_gbps')}）；分布有双峰（部分节点掉到 ~1000–1050）。",
        "- **注意**：慢卡簇多与整节点 HBM 掉速相关（冒烟已见 worker-7/14）。",
        "",
        "---",
        "",
        "### `vector_gflops` / `scalar_elems_per_s` / `mte_gbps` / `sfu_gflops` / `cube_vector_tflops`",
        "",
        "- **是什么**：Vector FMA、cumsum 串行链、纯 `copy_`、`exp` SFU、GEMM+epilogue 流水"
        "（与昇腾同构字段名）。",
        "- **怎么得到**：对应 stage_c 探针；CUDA Event 中位。",
        "- **关键参数**：Vector/SFU 64M fp32；Scalar 16M；DMA copy 512MB；pipeline N=4096 bf16。",
        f"- **本批中位**：Vector **{mv('vector_gflops')}** GFLOPS；DMA copy **{mv('mte_gbps')}** GB/s；"
        f"SFU **{mv('sfu_gflops')}**；GEMM+epilogue **{mv('cube_vector_tflops')}** TFLOPS；"
        f"Scalar **{mv('scalar_elems_per_s')}** elems/s。",
        "- **注意**：`sfu_gflops` 仍按 1 op/元素计，实质偏 Gops/s；勿与 Vector FMA 按 2× 换算。",
        "",
        "---",
        "",
        "### Launch 延迟族（`launch_sync_*` / `launch_host_overhead_*` / `launch_burst_*`）",
        "",
        "- **是什么**：空 sync、host−device 差分、burst 发射成本（µs）。",
        "- **怎么得到**：`launch_latency`；CPU `perf_counter` + `torch.cuda.synchronize` / Event 差分。",
        "- **关键参数**：samples=500，warmup=50，burst_count=64，timing_method=event。",
        f"- **本批中位**：sync p50 **{mv('launch_sync_p50_us')} µs**；"
        f"host overhead p50 **{mv('launch_host_overhead_p50_us')} µs**；"
        f"burst p50 **{mv('launch_burst_p50_us')} µs**。",
        "- **注意**：CV 明显高于算力字段——更适合看尾延迟/驱动抖动，不宜作为主判定指标。",
        "",
        "---",
        "",
        "### 遥测：`health_*` / `power_*` / `board_temp_c` / GPU util / XCORE clk",
        "",
        "- **是什么**：轻载与负载阶段的功耗/温度快照，以及负载阶段的板温、GPU 利用率和 XCORE 时钟；"
        "兼容键分别为 `board_temp_c`、`aicore_util_pct`、`aicore_freq_mhz`。"
        "`health` 只是采样阶段标签，**不是健康分**。",
        "- **怎么得到（沐曦）**：`mx-smi` 解析（`MxSmiProvider`），不是 `npu-smi`。",
        f"- **关键参数**：开测早期轻载快照；负载末常取 vector_fma 末轮回填；"
        "board_temp / util / XCORE 为负载路径 TTL 合并。",
        f"- **本批中位**：轻载 temp **{mv('health_temp_c')} °C**；"
        f"轻载 power **{mv('health_power_w')} W**；"
        f"负载 power **{mv('power_w')} W**；power limit **{mv('power_limit_w')} W**；"
        f"board_temp **{mv('board_temp_c')} °C**；"
        f"GPU util **{mv('aicore_util_pct')}%**；"
        f"XCORE clk **{mv('aicore_freq_mhz')} MHz**。",
        "- **注意**：",
        "  - **不要**把 health 与 load power 相减当降频证据。",
        "  - 本批 JSONL（`20260711_232400`）中 `board_temp_c` / `aicore_util_pct` / "
        "`aicore_freq_mhz` **已采集并落盘**；是采样时刻快照，不等同于完整热稳态。",
        "  - 相对昇腾：空闲/满载功耗量级不同（昇腾满载中位 ~872 W；沐曦 ~467 W / 550 W 墙）。",
        "",
        "---",
        "",
        "### `shape_sweep_peak_tflops`（名不副实：BNMK max）",
        "",
        "- **是什么**：字段名沿用「shape sweep 峰值」，本批实为各 BNMK shape 中位 TFLOPS 的 **max**。",
        "- **怎么得到**：`gemm_bnmk_sample`；旧独立 shape_sweep 开关已关。",
        f"- **本批中位**：字段中位 **{mv('shape_sweep_peak_tflops')}**（覆盖 {cov('shape_sweep_peak_tflops')}）。",
        "- **注意**：勿按字面理解成独立 shape sweep 扫参结果。",
        "",
        "---",
        "",
        "### 判定字段（冒烟 vs 体质，勿混用）",
        "",
        "- **是什么**：相对集群中位的偏离 + 正确性/计时质量门控。",
        "- **怎么得到**：冒烟 job `logs/muxi-card-screen-20260711_133828-muxi-smoke/`；"
        "体质判定来自 constitution merged。",
        "- **关键参数**：相对中位阈值 + `max_rel_err`（冒烟与体质规则不完全相同）。",
        "- **注意**：本批冒烟 good=106 / slow=19 / bad=1 / contended=2；"
        "体质 good=119 / contended=8 / bad=1。两者采样阶段与规则不同，勿混读。",
        "",
        "---",
        "",
        "## 二、通信字段（NCCL / MCCL）",
        "",
        "### `alg_bw_GBps`（算法带宽）",
        "",
        "- **是什么**：业务字节 / 平均 collective 耗时 → GB/s。",
        "- **怎么得到**：`nccl_torch_bench.py`；`torch.distributed` + **NCCL**（MetaX 栈常叠 MCCL）；"
        "CPU `perf_counter` + `torch.cuda.synchronize`。",
        "- **关键参数**：sizes 1M–256M；fp32；`SOCKET_IFNAME=eth0`。",
        "- **注意**：与 `bus_bw` 差一个 NCCL-tests 折算因子。",
        "",
        "---",
        "",
        "### `bus_bw_GBps`（总线带宽）",
        "",
        "- **是什么**：NCCL-tests 同构折算后的总线带宽（GB/s）——扩展叙事的核心指标。",
        "- **怎么得到**：由 alg_bw 按公式折算（见下）；本批保持率用各 rank 中位。",
        "- **关键参数 / 公式**：",
        "  - All-Reduce：`alg × 2(n−1)/n`",
        "  - All-Gather / Reduce-Scatter：`alg × (n−1)/n`",
        "  - Broadcast：`= alg`",
        "  - 沐曦基线世界大小：单节点 **8 卡** → 保持率 **`bus_N / bus_8`**"
        "（昇腾用 `/bus_16`）。",
        f"- **本批关键事实**：w8@256MB All-Reduce bus 中位 ≈ **{fmt(ar8)} GB/s**；"
        f"w16 ≈ **{fmt(ar16)} GB/s**（保持率 ≈ **{(ar16r * 100):.2f}%**）。"
        if ar8 is not None and ar16 is not None and ar16r is not None
        else "- **本批关键事实**：见 `nccl_campaign_muxi_20260711.md`。",
        "- **注意**：跨节点数字是「功能通、链路错」的证据，不是 MetaXLink 机内上限。",
        "",
        "---",
        "",
        "### Collective Ops / P2P",
        "",
        "- **是什么**：`all_reduce` / `all_gather` / `reduce_scatter` / `broadcast`；"
        "P2P 为 `isend/irecv` 单向有效带宽（**不用** bus_bw 公式）。",
        "- **怎么得到**：`nccl_torch_bench.py` / `nccl_p2p_bench.py`。",
        "- **关键参数**：多机必须 `*_SOCKET_IFNAME=eth0`；大 world P2P 默认 ring。",
        "- **注意**：P2P 机内 16M ≈30–33 GB/s、跨节点 ≈0.35 GB/s，可与 collective 交叉验证断崖，"
        "但定义不同勿 1:1 对齐。",
        "",
        "---",
        "",
        "## 三、MFU 字段",
        "",
        "### 微基准 MFU（G8，`mfu_train_bench_nccl.py`）",
        "",
        "- **是什么**：合成 dense/moe 训练步的 Model FLOPs Utilization ="
        " `achieved_tflops / (peak_per_gpu × world)`。",
        f"- **怎么得到**：峰值分母取体质 `func_tflops` 中位 **{mv('func_tflops')} TFLOPS/卡**。",
        "- **关键参数**：dense/moe 合成步；world 8→128。",
        "- **注意**：dense@8=**26.7%**；dense@16+≈**0.2–0.3%**（通信打穿）；moe@8=**15.0%**。",
        "",
        "### 真训练 MFU（G9，tiny GPT）",
        "",
        "- **是什么**：Megatron `pretrain_gpt` 冒烟上的估算吞吐 / 同峰值分母。",
        "- **怎么得到**：稳态 ms/iter → 聚合 TFLOPS / (peak×world)。",
        "- **关键参数**：4L/H1024 + `local/unfused`；需 `nvcc`←`cucc` shim。",
        "- **注意**：验收是「链路跑通」，不是冲高 MFU；勿与 G8 GEMM 微基准直接比高低。"
        "本批稳态 ~54 ms/iter → MFU ≈ **4.5%**。",
        "",
        "---",
        "",
        "## 四、拓扑与链路健康",
        "",
        "| 数据 | 含义 | 底层 |",
        "|------|------|------|",
        "| `mx-smi topo` / MetaXLink | 机内互联类型（MX / SYS） | `probe_muxi_topology.sh` |",
        "| NIC `mlx5_*` + `xscale_*` | IB/加速网卡可见 | 同 topo + `ibv_devinfo` |",
        "| link-health 文本 | 每节点温度/ECC/PCIe/链路摘要 | `run_link_health_muxi.sh` |",
        "",
        "**解读**：设备健康 + IB 设备可见 ≠ 当前 NCCL 已走 IB。本批集体通信实测仍走 eth0 socket。",
        "",
        "---",
        "",
        "## 五、易混对照速查（沐曦特化）",
        "",
        "| 对比 | 正确读法 |",
        "|------|----------|",
        "| 沐曦保持率基线 vs 昇腾 | 沐曦用 **w8**；昇腾用 **w16** |",
        f"| `func_tflops` 沐曦 vs 昇腾 | 本批 ~{mv('func_tflops')} vs 昇腾 ~292；需同 N/dtype/计时再比 |",
        f"| `hbm_gbps` 沐曦 vs 昇腾 | 沐曦中位更高（~{mv('hbm_gbps')} vs ~1240），但有整节点掉速簇 |",
        f"| `health_power` vs `power_w` | ~{mv('health_power_w')} W vs ~{mv('power_w')} W；墙 {mv('power_limit_w')} W |",
        "| G8 MFU vs G9 MFU | G8=合成 GEMM 步；G9=真 Megatron tiny；目的不同 |",
        f"| P2P 机内 30 GB/s vs AR {fmt(ar8)} GB/s | 协议不同；AR 是 collective bus 折算饱和区 |"
        if ar8 is not None
        else "| P2P vs AR | 协议不同，勿 1:1 |",
        "| eth0 0.2 GB/s vs MetaXLink | 跨节点当前路径 vs 机内路径；不是「卡不行」 |",
        "| `nvcc` shim | cu-bridge 用 `cucc`；Megatron fused_kernels 硬查 `nvcc` |",
        "| BNMK / board_temp / GPU util / XCORE clk | **本批（232400）有 sample / 已落盘** |",
        "",
        "---",
        "",
        "## 六、与昇腾语义手册的关系",
        "",
        "- **字段名、公式、图注话术**尽量同构，便于对照阅读。",
        "- **差异只写清楚**：后端（NCCL/MCCL vs HCCL）、遥测（`mx-smi` vs `npu-smi`）、"
        "世界大小基线（8 vs 16）、计时设备（CUDA Event vs NPU Event）；"
        "同构键名（Cube/MTE/AICore）在沐曦侧是兼容壳，不表示昇腾硅上同名部件。",
        "",
        "*文档版本：2026-07-11 · 由 `rewrite_meaning_mds_muxi.py` 现算中位 · "
        "配套 [`CAMPAIGN_FINAL_MUXI_20260711.md`](CAMPAIGN_FINAL_MUXI_20260711.md) · "
        "[`FIGURE_PROVENANCE_MUXI_20260711.md`](FIGURE_PROVENANCE_MUXI_20260711.md)*",
        "",
    ]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    path = ROUNDS / SEM_LINK
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    print("wrote", path.name)


def audit_md(name: str) -> None:
    text = (ROUNDS / name).read_text(encoding="utf-8")
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", text)
    miss = [r for r in refs if not (ROUNDS / r).exists()]
    bad = sum(1 for w in FLUFF if w in text)
    print(f"{name}: refs={len(refs)} miss={len(miss)} fluff_hits={bad}")
    if miss:
        print("  miss:", miss[:5], ("..." if len(miss) > 5 else ""))


def main() -> None:
    cards = load_cards()
    print(f"loaded cards={len(cards)} from {MERGED}")
    ret = compute_nccl_retention()
    if ret:
        ar8 = ret.get("all_reduce", {}).get(8, (None, None))[0]
        ar16r = ret.get("all_reduce", {}).get(16, (None, None))[1]
        print(
            f"nccl AR@256M w8_med={fmt(ar8)} "
            f"w16_ret={(ar16r * 100):.2f}%" if ar8 and ar16r else "nccl ret partial"
        )
    n_const = write_constitution(cards)
    n_extra = write_extra(cards)
    n_nccl = write_nccl(ret)
    write_provenance(n_const, n_extra, n_nccl)
    write_campaign(cards, ret, n_const, n_extra, n_nccl)
    write_semantics(cards, ret)

    for md in [
        "card_constitution_muxi_20260711.md",
        "constitution_extra_muxi_20260711.md",
        "nccl_campaign_muxi_20260711.md",
        "CAMPAIGN_FINAL_MUXI_20260711.md",
        "METRIC_SEMANTICS_MUXI_20260711.md",
        "FIGURE_PROVENANCE_MUXI_20260711.md",
    ]:
        audit_md(md)


if __name__ == "__main__":
    main()
