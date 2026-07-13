#!/usr/bin/env python3
"""重写 jiuding 风格 SVG 报告的中文解读（保留路径/文件名，补全逐图图注）。"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROUNDS = Path(__file__).resolve().parent / "rounds"
CONSTITUTION_JSONL = Path(
    "/Users/yinjinrun/random-thing/logs/card-fillgap-20260711_140301/results/constitution128.merged.jsonl"
)
HCCL_LOG = Path("/Users/yinjinrun/random-thing/logs/pipeline-comm-20260711_134811")

METRIC_CN: dict[str, str] = {
    "func_tflops": "Cube func TFLOPS",
    "hbm_gbps": "HBM GB/s",
    "sustained_tflops": "Sustained TFLOPS",
    "vector_gflops": "Vector GFLOPS",
    "scalar_elems_per_s": "Scalar elems/s",
    "mte_gbps": "MTE copy GB/s",
    "cube_vector_tflops": "Cube+Vector TFLOPS",
    "sfu_gflops": "SFU GFLOPS",
    "launch_sync_p50_us": "Launch sync p50 (μs)",
    "launch_sync_p99_us": "Launch sync p99 (μs)",
    "launch_host_overhead_p50_us": "Host overhead p50 (μs)",
    "launch_host_overhead_p99_us": "Host overhead p99 (μs)",
    "launch_burst_p50_us": "Burst total p50 (μs)",
    "launch_burst_per_kernel_p50_us": "Burst/kernel p50 (μs)",
    "health_temp_c": "Health temp (°C)",
    "health_power_w": "Health power (W)",
    "board_temp_c": "Board temp (°C)",
    "aicore_util_pct": "AICore util %",
    "aicpu_util_pct": "AICPU util %",
    "ctrlcpu_util_pct": "CtrlCPU util %",
    "mem_bw_util_pct": "MemBW util %",
    "power_w": "Power (W)",
    "shape_sweep_peak_tflops": "Shape sweep peak TFLOPS",
}

METRIC_HINT: dict[str, str] = {
    "func_tflops": "单卡 Cube GEMM 峰值算力，反映 AICore 算力体质",
    "hbm_gbps": "HBM 顺序带宽，决定大模型权重/激活搬运上限",
    "sustained_tflops": "持续 GEMM 采样中位，反映长时间跑分稳定性",
    "vector_gflops": "Vector 单元吞吐，与 Cube 正交",
    "scalar_elems_per_s": "Scalar 控制流吞吐，通常极窄",
    "mte_gbps": "MTE 拷贝引擎带宽，与 HBM 强相关",
    "cube_vector_tflops": "Cube+Vector 混合算子峰值",
    "sfu_gflops": "SFU 特殊函数单元吞吐",
    "launch_sync_p50_us": "kernel launch→sync 中位延迟（μs）",
    "launch_sync_p99_us": "launch→sync 尾延迟，对抖动敏感",
    "launch_host_overhead_p50_us": "Host 侧排队/调度中位开销",
    "launch_host_overhead_p99_us": "Host 侧尾延迟，易受 CPU 争用影响",
    "launch_burst_p50_us": "burst 发射窗口总时长中位",
    "launch_burst_per_kernel_p50_us": "burst 内单 kernel 均摊时长",
    "health_temp_c": "空闲/health 采样温度",
    "health_power_w": "空闲/health 采样功耗",
    "board_temp_c": "满载烤机板温",
    "aicore_util_pct": "AICore 利用率（0=未跑满，~92=满载）",
    "aicpu_util_pct": "AICPU 利用率",
    "ctrlcpu_util_pct": "CtrlCPU 利用率",
    "mem_bw_util_pct": "MemBW 利用率",
    "power_w": "满载采样功耗",
    "shape_sweep_peak_tflops": "10-shape sweep 中的峰值 TFLOPS",
}

HIGHER_BETTER = {
    "func_tflops", "hbm_gbps", "sustained_tflops", "vector_gflops", "scalar_elems_per_s",
    "mte_gbps", "cube_vector_tflops", "sfu_gflops", "shape_sweep_peak_tflops",
    "aicore_util_pct", "mem_bw_util_pct",
}
LOWER_BETTER = {
    "launch_sync_p50_us", "launch_sync_p99_us", "launch_host_overhead_p50_us",
    "launch_host_overhead_p99_us", "launch_burst_p50_us", "launch_burst_per_kernel_p50_us",
    "health_temp_c", "board_temp_c", "ctrlcpu_util_pct", "health_power_w", "power_w",
}


def short_host(host: str) -> str:
    if not host:
        return "?"
    if "master" in host:
        return "master-0"
    m = re.search(r"worker-(\d+)", host)
    if m:
        return f"worker-{m.group(1)}"
    return host


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if abs(v) >= 1e8:
            return f"{v:.2e}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.4g}"
    return str(v)


def relmed(value: float, median: float) -> float:
    if not median:
        return 0.0
    return (value - median) / median * 100.0


def metric_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    ordered = sorted(values)
    mean = statistics.mean(ordered)
    std = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    cv = (std / mean * 100.0) if mean else 0.0
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "mean": mean,
        "std": std,
        "cv_pct": cv,
        "min": ordered[0],
        "max": ordered[-1],
        "p5": ordered[max(0, int(len(ordered) * 0.05))],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def load_cards(path: Path) -> list[dict]:
    cards = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("record") == "card":
                cards.append(o)
    return cards


def values_for(cards: list[dict], key: str) -> list[float]:
    out: list[float] = []
    for c in cards:
        v = c.get(key)
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    return out


def host_medians(cards: list[dict], key: str) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for c in cards:
        v = c.get(key)
        if v is None:
            continue
        by[short_host(c.get("host", ""))].append(float(v))
    return {h: statistics.median(vs) for h, vs in by.items()}


def outlier_cards(cards: list[dict], key: str, med: float, threshold_pct: float = 1.0) -> list[tuple[str, float]]:
    rows = []
    for c in cards:
        v = c.get(key)
        if v is None:
            continue
        d = relmed(float(v), med)
        if abs(d) >= threshold_pct:
            rows.append((f"{short_host(c.get('host',''))}/dev{c.get('device_id','?')}", d))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    return rows


def list_svgs(fig_dir: Path) -> list[str]:
    return sorted(p.name for p in fig_dir.glob("*.svg"))


def verify_refs(md_text: str, fig_dir: Path) -> tuple[int, int]:
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", md_text)
    missing = [r for r in refs if not (fig_dir.parent / r).is_file() and not (fig_dir / Path(r).name).is_file()]
    # refs are relative like figdir/file.svg
    really_missing = []
    for r in refs:
        if not (ROUNDS / r).is_file():
            really_missing.append(r)
    return len(refs), len(really_missing)


# ── Constitution captions ────────────────────────────────────────────────

def constitution_caption(fn: str, cards: list[dict], stats: dict[str, dict]) -> str:
    base = fn.replace(".svg", "")

    if base == "box_overview":
        return (
            "128 卡核心指标箱线总览（jiuding 大字号）。Cube 中位 **292.4 TFLOPS**（CV 1.9%）、"
            "HBM **1240.7 GB/s**（CV 4.3%，右尾有低带宽离群）、Sustained **306.9**、"
            "Vector **98.8 GFLOPS**、Launch p99 尾延迟箱体最宽（CV 36.9%）。"
            "一眼可见：算力/HBM 主体紧凑，launch 与功耗类指标离散更大。"
        )

    if base.startswith("hist_"):
        key = base.replace("hist_", "")
        key = key.replace("_c", "_c")  # health_temp_c etc kept
        # fix keys with trailing _c vs _pct
        for k in METRIC_CN:
            if base == f"hist_{k}":
                key = k
                break
        st = stats.get(key, {})
        if not st:
            return f"128 卡 `{key}` 直方图：展示全集群频数分布与 jiuding 风格 bin 划分。"
        tail = ""
        if key in HIGHER_BETTER:
            tail = f"区间 [{fmt(st['min'])}, {fmt(st['max'])}]，p5–p95 为 {fmt(st['p5'])}–{fmt(st['p95'])}。"
        elif key in LOWER_BETTER:
            tail = f"区间 [{fmt(st['min'])}, {fmt(st['max'])}]；越低通常越好。"
        if key == "power_w":
            tail += " 可见 **双峰**：~27 卡读数 <300 W（采样窗口未满载），~91 卡 >800 W（满载中位 **871.5 W**）。"
        if key == "aicore_util_pct":
            tail += " **27 卡为 0%**（health 窗口未跑 kernel），**67 卡 ~92%** 满载。"
        if key == "hbm_gbps":
            tail += f" 左尾最低 **1012.4 GB/s**（worker-0），比中位低约 18.4%。"
        return (
            f"128 卡 **{METRIC_CN.get(key, key)}** 直方图：中位 **{fmt(st['median'])}**，"
            f"CV **{fmt(st['cv_pct'])}%**。{METRIC_HINT.get(key, '')} {tail}"
        )

    if base.startswith("heatmap_relmed_"):
        key = base.replace("heatmap_relmed_", "")
        st = stats.get(key, {})
        med = st.get("median", 0)
        outs = outlier_cards(cards, key, med, 1.0)
        n_out = len(outs)
        top = outs[:3]
        top_txt = "；".join(f"{n} {d:+.1f}%" for n, d in top) if top else "无 |Δ|≥1% 离群"
        return (
            f"**{METRIC_CN.get(key, key)}** 的 host×device 相对中位偏差热力图（行 **master-0 / worker-0…6**，列 device 0–15）。"
            f"**读图规则（新版）**：色阶看全局偏差，格内数字**仅标注 |Δ|≥1%** 的离群格（本指标约 **{n_out}** 格），"
            f"避免满屏数字干扰色阶。集群中位 **{fmt(med)}**；最大离群：{top_txt}。"
            f"{METRIC_HINT.get(key, '')}"
        )

    if base.startswith("box_by_host_"):
        key = base.replace("box_by_host_", "")
        hm = host_medians(cards, key)
        if not hm:
            return f"按 host 分组的 {METRIC_CN.get(key, key)} 箱线图。"
        best_h = max(hm, key=hm.get) if key in HIGHER_BETTER else min(hm, key=hm.get)
        worst_h = min(hm, key=hm.get) if key in HIGHER_BETTER else max(hm, key=hm.get)
        spread = max(hm.values()) - min(hm.values())
        return (
            f"8 节点（**master-0 + worker-0…6**）**{METRIC_CN.get(key, key)}** 箱线对比。"
            f"节点中位最高 **{best_h}={fmt(hm[best_h])}**，最低 **{worst_h}={fmt(hm[worst_h])}**（跨节点极差 **{fmt(spread)}**）。"
            f"每节点 16 卡，箱体宽度反映 node 内离散；{METRIC_HINT.get(key, '')}"
        )

    if base.startswith("sorted_bar_"):
        key = base.replace("sorted_bar_", "")
        st = stats.get(key, {})
        vals = values_for(cards, key)
        if not vals:
            return f"128 卡 {METRIC_CN.get(key, key)} 排序柱状图。"
        med = st.get("median", statistics.median(vals))
        if key in HIGHER_BETTER:
            worst_v, best_v = min(vals), max(vals)
        else:
            worst_v, best_v = max(vals), min(vals)
        return (
            f"128 卡 **{METRIC_CN.get(key, key)}** 从低到高排序柱（jiuding hatch 描边）。"
            f"全集群中位 **{fmt(med)}**；排序两端 **{fmt(worst_v)}**（{'低' if key in HIGHER_BETTER else '高'}端）"
            f" ↔ **{fmt(best_v)}**（{'高' if key in HIGHER_BETTER else '低'}端），"
            f"极差 **{fmt(abs(best_v - worst_v))}**。用于定位单卡离群，而非直接判 BAD。"
        )

    if base.startswith("bar_host_mean_std_"):
        key = base.replace("bar_host_mean_std_", "")
        hm = host_medians(cards, key)
        if not hm:
            return f"各 host 均值 ± 标准差柱：{METRIC_CN.get(key, key)}。"
        means = hm
        stds = {}
        by: dict[str, list[float]] = defaultdict(list)
        for c in cards:
            v = c.get(key)
            if v is not None:
                by[short_host(c.get("host", ""))].append(float(v))
        for h, vs in by.items():
            stds[h] = statistics.pstdev(vs) if len(vs) > 1 else 0
        max_std_h = max(stds, key=stds.get)
        return (
            f"8 节点 **{METRIC_CN.get(key, key)}** 均值柱 + 误差线（±1σ）。"
            f"节点间均值极差 **{fmt(max(means.values()) - min(means.values()))}**；"
            f"node 内离散最大为 **{max_std_h}**（σ={fmt(stds[max_std_h])}）。"
            f"host 轴已缩短为 **master-0 / worker-N**。"
        )

    scatter_map = {
        "scatter_func_tflops_vs_vector_gflops": (
            "Cube func TFLOPS × Vector GFLOPS 正交散点。Pearson 相关约 **-0.04**，"
            "两轴几乎独立——Cube 快不代表 Vector 快。128 点按 host 着色，无单一坏簇。"
        ),
        "scatter_hbm_gbps_vs_mte_gbps": (
            "HBM × MTE 散点。中位 **1240.7 / 1267.9 GB/s**，点云沿对角线聚集（MTE 拷贝吃 HBM 带宽），"
            "低 HBM 卡（如 1012 GB/s）MTE 同步偏低，符合内存瓶颈传导。"
        ),
        "scatter_power_w_vs_func_tflops": (
            "满载功耗 × Cube 算力。功耗中位 **871.5 W**（health 空闲 **167.9 W**），"
            "算力 273–303 TFLOPS 与 180–959 W 交叉——低功耗点来自采样窗口未跑满（aicore util=0），"
            "不是算力差。"
        ),
        "scatter_health_power_w_vs_func_tflops": (
            "空闲 health 功耗 × Cube。health 中位 **167.9 W**（p5–p95: 159–253 W），"
            "与 func 几乎无关；少数 health_power 偏高（+182% 离群）是节点电源管理差异，不影响算力排序。"
        ),
        "scatter_power_w_vs_hbm_gbps": (
            "满载功耗 × HBM。高功耗（>850 W）卡 HBM 集中在 1200–1270 GB/s；"
            "低功耗点（<300 W）HBM 分布仍正常——再次说明功耗采样窗口混杂 idle/满载。"
        ),
        "scatter_health_power_w_vs_hbm_gbps": (
            "空闲功耗 × HBM。health **~168 W** 与 HBM **~1240 GB/s** 无相关；"
            "HBM 左尾低带宽卡（1012 GB/s）health 功耗仍正常，指向 HBM 本体而非平台供电。"
        ),
        "scatter_launch_host_overhead_p50_us_vs_ctrlcpu_util_pct": (
            "Host overhead p50（中位 **240.1 μs**）× CtrlCPU util（中位 **7%**）。"
            "高 CtrlCPU（>14%）时 overhead 略抬升，但相关系数弱——launch 抖动更多来自 host 调度而非 CtrlCPU 单因子。"
        ),
    }
    if base in scatter_map:
        return scatter_map[base]

    if base == "timeseries_sustained_p05_p50":
        return (
            "Sustained GEMM 采样时序带：p50 中位轨迹与 p05 下界。"
            "128 卡 sustained 中位 **306.9 TFLOPS**（func **292.4**），"
            "时序 p05 不低于 **294.8 TFLOPS**，说明烤机过程无系统性热降频断崖。"
        )

    return f"**{base}**：128 卡 constitution 指标可视化（jiuding SVG）。"


def build_constitution_md(cards: list[dict], fig_dir: Path, md_path: Path) -> str:
    figs = list_svgs(fig_dir)
    stats = {k: metric_stats(values_for(cards, k)) for k in METRIC_CN}
    stats = {k: v for k, v in stats.items() if v}

    skipped = [
        "`aicore_freq_mhz`（AICore freq (MHz)）：字段缺失或全空，跳过",
        "`hbm_temp_c`（HBM temp (C)）：字段缺失或全空，跳过",
        "`power_limit_w`（Power limit (W)）：字段缺失或全空，跳过",
        "`gemm_shape_sample`：无可用曲线，跳过 shape",
    ]

    lines = [
        "# Card Constitution 分布报告（jiuding · 中文解读）",
        "",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 卡数: {len(cards)}",
        f"- 数据源: `{CONSTITUTION_JSONL}`",
        f"- 图目录: `{fig_dir.name}/`（**{len(figs)}** 张 SVG）",
        "",
        "> 本报告只做分布统计与可视化，不强调 slow / 坏卡判定。",
        "",
        "## 读图说明（本轮 jiuding 重绘变更）",
        "",
        "1. **热力图 `heatmap_relmed_*`**：颜色表示相对集群中位的偏差百分比；格内数字**仅标注 |Δ|≥1%** 的离群格，其余留白，用色阶读全局。",
        "2. **Host 轴标签**已统一缩短为 **master-0 / worker-0 … worker-6**（每节点 16 卡）。",
        "3. 柱图/曲线为大字号 jiuding 风格；热力图格内注释为紧凑字号。",
        "4. 全部交付 **SVG**，便于缩放审阅。",
        "",
        "## 跳过说明",
        "",
    ]
    for s in skipped:
        lines.append(f"- {s}")

    lines += [
        "",
        "## 指标分布",
        "",
        "| 指标 | n | median | mean | std | CV% | min | max | p5 | p50 | p95 |",
        "|------|---|--------|------|-----|-----|-----|-----|----|----|-----|",
    ]
    key_order = [
        "func_tflops", "hbm_gbps", "sustained_tflops", "vector_gflops", "scalar_elems_per_s",
        "mte_gbps", "cube_vector_tflops", "sfu_gflops", "launch_sync_p50_us", "launch_sync_p99_us",
        "launch_host_overhead_p50_us", "launch_host_overhead_p99_us", "launch_burst_p50_us",
        "launch_burst_per_kernel_p50_us", "health_temp_c", "health_power_w", "board_temp_c",
        "aicore_util_pct", "aicpu_util_pct", "ctrlcpu_util_pct", "mem_bw_util_pct", "power_w",
        "shape_sweep_peak_tflops",
    ]
    for key in key_order:
        st = stats.get(key)
        if not st:
            continue
        lines.append(
            f"| {METRIC_CN[key]} | {st['n']} | {fmt(st['median'])} | {fmt(st['mean'])} | "
            f"{fmt(st['std'])} | {fmt(st['cv_pct'])} | {fmt(st['min'])} | "
            f"{fmt(st['max'])} | {fmt(st['p5'])} | {fmt(st['median'])} | {fmt(st['p95'])} |"
        )

    lines += ["", "## 相对中位数偏差", "", "偏差 = `(值 - 集群中位数) / 集群中位数 × 100%`。", ""]
    for key in key_order:
        st = stats.get(key)
        if not st:
            continue
        med = st["median"]
        devs = [relmed(float(c[key]), med) for c in cards if c.get(key) is not None]
        if not devs:
            continue
        abs_mean = sum(abs(d) for d in devs) / len(devs)
        lines.append(
            f"- **{METRIC_CN[key]}** (`{key}`): [{min(devs):+.2f}%, {max(devs):+.2f}%]，"
            f"|偏差|均值 {abs_mean:.2f}%"
        )

    hosts = sorted({short_host(c.get("host", "")) for c in cards})
    lines += [
        "",
        "## 元数据",
        "",
        f"- hosts ({len(hosts)}): {', '.join(hosts)}",
        "- backends: npu",
        "- launch_timing_method: event",
        "",
        "## 图表解读",
        "",
    ]

    for fn in figs:
        title = fn.replace(".svg", "").replace("_", " ")
        cap = constitution_caption(fn, cards, stats)
        lines += [f"### {title}", "", cap, "", f"![{title}]({fig_dir.name}/{fn})", ""]

    return "\n".join(lines) + "\n"


# ── Fillgap captions ───────────────────────────────────────────────────────

def fillgap_caption(fn: str, cards: list[dict], stats: dict[str, dict]) -> str:
    base = fn.replace(".svg", "")

    if base == "radar_host_median_norm":
        return (
            "8 节点 **雷达图**（各轴为 Cube/HBM/Sustained/Vector/MTE/SFU 等中位归一化）。"
            "worker-4 Cube 中位最高（**295.7 TFLOPS**），master-0 最低（**289.6**）；"
            "各节点轮廓接近，无单一「全面弱 node」。"
        )
    if base == "parallel_host_median_norm":
        return (
            "**平行坐标**：每折线代表一节点，纵轴 0–1 归一化。"
            "Launch 类指标（取倒数归一化）在 master-0 / worker-6 上略长，"
            "与 launch_sync_p99 尾延迟离群（master-0 最高 **26.3 μs**）一致。"
        )
    if base == "hbm_modes_grouped_bar":
        return (
            "HBM **四模式**分组柱：顺序拷贝中位 **1268 GB/s**、读密集 **1454.5**、写密集 **1468.1**、"
            "跨步仅 **20.04 GB/s**（跨步访存天然受限）。"
            "四模式 CV 均 <1%，说明 HBM 控制器行为一致。"
        )
    if base == "corr_cube_vector_sfu_mte":
        return (
            "Cube / Vector / SFU / MTE **Pearson 相关矩阵**。"
            "Cube–MTE 正相关最强（共享内存带宽），Cube–Vector 近零（**≈-0.04**），"
            "SFU 与其他轴弱相关——调优算力不应假设 Vector 会随 Cube 同涨。"
        )
    if base == "box_launch_by_host":
        return (
            "三指标 launch 箱线（sync p99 / host overhead p99 / burst p50）按 host 并列。"
            "sync p99 中位 **6.78 μs** 但 master-0 尾部长（**26.3 μs**）；"
            "overhead p99 中位 **628.7 μs**；burst p50 中位 **472.5 μs**。"
        )
    if base == "cdf_core_metrics":
        return (
            "六项核心指标 **CDF 叠图**（Cube/HBM/Vector/MTE/SFU/Sustained）。"
            "Cube/HBM/Vector 曲线陡峭（CV 1–4%）；launch 类 CDF 拖尾长，"
            "p95 处 Cube 仍贴近 **299 TFLOPS**，而 launch_sync_p99 可达 **11.4 μs**（p95）。"
        )
    if base == "extreme10_small_multiples":
        return (
            "Sustained **最快/最慢各 10 卡**小 multiples（6 项指标）。"
            "最快 sustained **313.7 TFLOPS** vs 最慢 **294.8 TFLOPS**（差 **4.9%**）；"
            "快卡 HBM/MTE 同步偏高，慢卡多在 worker-0/master-0，但 HBM 左尾才是更大缺口。"
        )
    if base.startswith("heatmap_host_device_"):
        key = base.replace("heatmap_host_device_", "")
        st = stats.get(key, {})
        med = st.get("median", 0)
        return (
            f"**{METRIC_CN.get(key, key)}** host×device **绝对值**热力图。"
            f"**读图规则（新版）**：颜色看全局，格内数字**仅标注偏离该指标集群中位 ≥0.5%** 的格子。"
            f"集群中位 **{fmt(med)}**；用于看 node 内 device 0–15 是否有系统性弱槽位。"
        )
    if base == "scatter_sustained_vs_func":
        ratios = [
            float(c["sustained_tflops"]) / float(c["func_tflops"])
            for c in cards if c.get("sustained_tflops") and c.get("func_tflops")
        ]
        mr = statistics.median(ratios)
        return (
            f"Sustained vs Cube func 散点（y=x 虚线 + 中位比 **{mr:.3f}** 点线）。"
            f"128 点整体在 y=x **上方**，sustained 中位 **306.9** 高于 func **292.4**，"
            f"说明持续 GEMM 采样窗口内未出现大幅降频；离群点不超过 ±4%。"
        )
    return f"**{base}** 增强可视化。"


def build_fillgap_md(cards: list[dict], fig_dir: Path) -> str:
    figs = list_svgs(fig_dir)
    stats = {k: metric_stats(values_for(cards, k)) for k in METRIC_CN}
    stats = {k: v for k, v in stats.items() if v}

    extra_keys = [
        "hbm_mode_seq_copy_gbps", "hbm_mode_strided_gbps",
        "hbm_mode_read_heavy_gbps", "hbm_mode_write_heavy_gbps",
    ]
    extra_cn = {
        "hbm_mode_seq_copy_gbps": "HBM 顺序拷贝 GB/s",
        "hbm_mode_strided_gbps": "HBM 跨步 GB/s",
        "hbm_mode_read_heavy_gbps": "HBM 读密集 GB/s",
        "hbm_mode_write_heavy_gbps": "HBM 写密集 GB/s",
    }
    for k in extra_keys:
        st = metric_stats(values_for(cards, k))
        if st:
            stats[k] = st
            METRIC_CN[k] = extra_cn[k]

    lines = [
        "# Constitution 增强可视化报告（jiuding · 中文解读）",
        "",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 卡数: {len(cards)}",
        f"- 数据源: `{CONSTITUTION_JSONL}`",
        f"- 图目录: `{fig_dir.name}/`（**{len(figs)}** 张 SVG）",
        "",
        "> 补充 plot_card_constitution 未覆盖的多维对比、CDF、相关矩阵、快慢卡分析。",
        "",
        "## 读图说明",
        "",
        "- **`heatmap_host_device_*`**：颜色看全局绝对值；数字仅标偏离集群中位 **≥0.5%** 的格。",
        "- Host 轴：**master-0 / worker-N**；device 列 0–15。",
        "",
        "## 核心指标 median / CV 摘要",
        "",
        "| 指标 | n | median | CV% | min | max |",
        "|------|---|--------|-----|-----|-----|",
    ]
    summary_keys = [
        "func_tflops", "hbm_gbps", "sustained_tflops", "vector_gflops", "scalar_elems_per_s",
        "mte_gbps", "cube_vector_tflops", "sfu_gflops",
        "hbm_mode_seq_copy_gbps", "hbm_mode_strided_gbps",
        "hbm_mode_read_heavy_gbps", "hbm_mode_write_heavy_gbps",
        "launch_sync_p99_us", "launch_host_overhead_p99_us", "launch_burst_p50_us",
    ]
    for key in summary_keys:
        st = stats.get(key)
        if not st:
            continue
        lines.append(
            f"| {METRIC_CN.get(key, key)} | {st['n']} | {fmt(st['median'])} | "
            f"{fmt(st['cv_pct'])} | {fmt(st['min'])} | {fmt(st['max'])} |"
        )

    hosts = sorted({short_host(c.get("host", "")) for c in cards})
    lines += [
        "",
        "## 元数据",
        "",
        f"- hosts ({len(hosts)}): {', '.join(hosts)}",
        "",
        "## 图表解读",
        "",
    ]
    for fn in figs:
        title = fn.replace(".svg", "").replace("_", " ")
        lines += [
            f"### {title}",
            "",
            fillgap_caption(fn, cards, stats),
            "",
            f"![{title}]({fig_dir.name}/{fn})",
            "",
        ]
    return "\n".join(lines) + "\n"


# ── BNMK ───────────────────────────────────────────────────────────────────

def load_bnmk(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("record") == "gemm_bnmk_sample":
                out.append(o)
    return out


def build_bnmk_md(samples: list[dict], fig_dir: Path) -> str:
    figs = list_svgs(fig_dir)
    by_label: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        by_label[s["label"]].append(float(s["tflops"]))
    stats_rows = []
    for label in sorted(by_label):
        vs = by_label[label]
        stats_rows.append({
            "label": label,
            "n": len(vs),
            "median": statistics.median(vs),
            "mean": statistics.mean(vs),
            "min": min(vs),
            "max": max(vs),
        })
    hosts = sorted({short_host(s.get("host", "")) for s in samples})

    captions = {
        "bnmk_tflops_box_by_label.svg": (
            f"10 种 BNMK shape 的 TFLOPS **箱线图**（每 shape **{len(samples)//len(by_label)}** 卡）。"
            f"最高中位 **B1_M16384_N1024_K1024 = 310.53 TFLOPS**；"
            f"最低 **B1_M8_N8192_K8192 = 15.27 TFLOPS**（K 维极小导致利用率塌陷，属 shape 特性而非坏卡）。"
            f"**B1_M4096_N4096_K11008** 箱体最宽（min **259.9**），对 tile 配置敏感。"
        ),
        "bnmk_tflops_bar_median_by_label.svg": (
            "各 shape **中位 TFLOPS 柱**（jiuding 大字号）。"
            "第一梯队 **310+ TFLOPS**（M16384×N1024、B8 batch）；"
            "第二梯队 **275–307**；孤立短柱 **15.27** 为超宽 K 的 B1_M8 shape。"
            "与 Cube 单测中位 **292.4** 对比，多数 shape 可达或超过单测水平。"
        ),
        "bnmk_host_shape_heatmap.svg": (
            "Host × Shape **TFLOPS 热力图**（8×10）。"
            "颜色看全局吞吐；数字仅标偏离该 shape 集群中位较大的格子。"
            "最弱格 **worker-2 × B1_M8_N8192_K8192**（中位 **15.23 TFLOPS**）；"
            "最强 **worker-1 × B1_M16384_N1024_K1024**（**310.61**）。"
            "除极小-K shape 外，节点间色阶均匀，无 host 级系统性弱行。"
        ),
    }

    lines = [
        f"# BNMK Shape 报告 · 20260711（jiuding · 中文解读）",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> 数据：`{CONSTITUTION_JSONL}`",
        f"> 图目录：`{fig_dir.name}/`（**{len(figs)}** 张 SVG）",
        "",
        "## 摘要",
        "",
        f"- 样本数：**{len(samples)}**（10 shape × 128 卡）",
        f"- Shape 种类：**{len(by_label)}**",
        f"- 节点数：**{len(hosts)}**（{', '.join(hosts)}）",
        "",
        "## 各 Shape TFLOPS 统计",
        "",
        "| Label | N | 中位数 | 均值 | 最小 | 最大 |",
        "|-------|---|--------|------|------|------|",
    ]
    for st in stats_rows:
        lines.append(
            f"| {st['label']} | {st['n']} | {st['median']:.2f} | {st['mean']:.2f} | "
            f"{st['min']:.2f} | {st['max']:.2f} |"
        )

    lines += ["", "## 图表解读", ""]
    for fn in figs:
        title = {
            "bnmk_tflops_box_by_label.svg": "TFLOPS 箱线图（按 label）",
            "bnmk_tflops_bar_median_by_label.svg": "各 label 中位 TFLOPS 柱状图",
            "bnmk_host_shape_heatmap.svg": "Host × Shape 热力图",
        }.get(fn, fn)
        lines += [
            f"### {title}",
            "",
            captions.get(fn, f"BNMK 图 {fn}"),
            "",
            f"![{title}]({fig_dir.name}/{fn})",
            "",
        ]
    return "\n".join(lines) + "\n"


# ── HCCL ───────────────────────────────────────────────────────────────────

def load_hccl_rows() -> list[dict]:
    rows = []
    for ws in (16, 32, 64, 128):
        p = HCCL_LOG / "hccl-results" / f"scale_{ws}.jsonl"
        if p.is_file():
            rows.extend(json.loads(l) for l in p.read_text().splitlines() if l.strip())
    return rows


def hccl_medians(rows: list[dict]) -> tuple[dict, dict]:
    retention: dict[str, dict[int, float]] = {}
    means_256: dict[tuple[str, int], float] = {}
    base: dict[str, float] = {}
    for op in ("all_reduce", "all_gather", "reduce_scatter", "broadcast"):
        for ws in (16, 32, 64, 128):
            vals = [
                float(r["bus_bw_GBps"]) for r in rows
                if r.get("record") == "hccl_bench" and r["op"] == op
                and int(r["world_size"]) == ws and int(r["nbytes"]) == 268435456
            ]
            if vals:
                means_256[(op, ws)] = statistics.mean(vals)
        if (op, 16) in means_256 and means_256[(op, 16)]:
            base[op] = means_256[(op, 16)]
            retention[op] = {
                ws: means_256[(op, ws)] / means_256[(op, 16)] * 100
                for ws in (16, 32, 64, 128) if (op, ws) in means_256
            }
    return retention, means_256


def load_p2p_edges() -> list[tuple[tuple[int, int], float]]:
    p = HCCL_LOG / "p2p-results" / "scale_128.jsonl"
    by: dict[tuple[int, int], list[float]] = defaultdict(list)
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("record") != "hccl_p2p" or int(r["nbytes"]) != 16777216:
            continue
        by[(int(r["src"]), int(r["dst"]))].append(float(r["bw_GBps"]))
    return [(k, statistics.median(v)) for k, v in by.items()]


def hccl_caption(fn: str, retention: dict, means_256: dict, hccl_rows: list[dict], p2p_edges: list) -> str:
    op_cn = {
        "all_reduce": "All-Reduce", "all_gather": "All-Gather",
        "reduce_scatter": "Reduce-Scatter", "broadcast": "Broadcast",
    }

    m = re.match(r"hccl_bus_bw_vs_size_(.+)\.svg", fn)
    if m:
        op = m.group(1)
        cn = op_cn.get(op, op)
        w16 = means_256.get((op, 16), 0)
        w128 = means_256.get((op, 128), 0)
        ret = retention.get(op, {}).get(128, 0)
        return (
            f"**{cn}** Bus 带宽 vs 消息大小（1/16/64/256 MB），四 world 色线。"
            f"256 MB 时 w16 均值 **{w16:.2f} GB/s** → w128 **{w128:.2f} GB/s**，"
            f"保持率 **{ret:.1f}%**。小消息区受延迟主导，256 MB 才进入带宽饱和区。"
        )

    if fn == "hccl_256mb_step_bus_bw.svg":
        return (
            "四算子 256 MB **Bus 带宽阶梯**（world 16→32→64→128）。"
            "All-Reduce 从 **154.8** 降至 **138.4 GB/s**；"
            "All-Gather **119.3→64.4**；Reduce-Scatter **110.5→51.3**——"
            "后两者在 64 卡后断崖，是 128 卡通信瓶颈主因。"
        )
    if fn == "hccl_256mb_step_per_op.svg":
        return (
            "分算子 **256 MB 扩展曲线**（四线并列）。"
            "All-Reduce / Broadcast 曲线平缓；All-Gather / Reduce-Scatter 在 w64–w128 段陡降，"
            "与 ring 算法跨节点步数增加一致。"
        )
    if fn == "hccl_256mb_retention_bar.svg":
        return (
            "256 MB **保持率柱**（相对 w16=100%）。"
            "All-Reduce w128 **89.4%**；Broadcast **86.8%**；"
            "All-Gather **54.0%**；Reduce-Scatter **46.4%**。"
            "AR 仍健康；AG/RS 需作为后续 NCCL/HCCL 调优重点。"
        )

    m = re.match(r"hccl_rank_violin_256mb_(.+)\.svg", fn)
    if m:
        op = m.group(1)
        vals = [
            float(r["bus_bw_GBps"]) for r in hccl_rows
            if r.get("record") == "hccl_bench" and r["op"] == op
            and int(r["world_size"]) == 128 and int(r["nbytes"]) == 268435456
        ]
        spread = (max(vals) - min(vals)) if vals else 0
        return (
            f"**{op_cn.get(op, op)}** @ w128 / 256 MB **Rank 小提琴图**。"
            f"中位 **{statistics.median(vals):.2f} GB/s**，rank 极差 **{spread:.2f} GB/s**。"
            f"{'AG/RS 小提琴呈双峰/长尾，反映跨 node 步长差异。' if op in ('all_gather','reduce_scatter') else 'AR/BC 分布紧凑，rank 间公平性较好。'}"
        )

    if fn == "hccl_rank_box_256mb_all_ops.svg":
        return (
            "四算子 256 MB @ w128 **Rank 箱线汇总**。"
            "All-Reduce 箱体最窄（中位 **138.4 GB/s**）；"
            "Reduce-Scatter 中位仅 **51.3 GB/s** 且下须长——"
            "慢 rank 拖累全局 RS 扩展。"
        )

    m = re.match(r"hccl_rank_hist_w(\d+)_256mb\.svg", fn)
    if m:
        ws = int(m.group(1))
        vals = [
            float(r["bus_bw_GBps"]) for r in hccl_rows
            if r.get("record") == "hccl_bench"
            and int(r["world_size"]) == ws and int(r["nbytes"]) == 268435456
        ]
        return (
            f"world=**{ws}**、256 MB、四算子合并 **Rank 直方图**（{len(vals)} 条记录）。"
            f"随 world 增大，直方图左移且拖尾变长；w128 时 AG/RS 低带宽 rank 占比显著上升。"
        )

    if fn == "p2p_bw_violin_by_kind_size.svg":
        return (
            "P2P **带宽小提琴**（按边类型 × 消息大小）。"
            "w128 ring 上 **环相邻** 边 16 MB 中位 **~100 GB/s**；"
            "**星型(经 rank0)** 中位 **~107 GB/s**。"
            "64 KB 仍在延迟区（<1 GB/s），16 MB 进入带宽区。"
        )

    if fn == "p2p_box_compare_w16_w128_65536.svg":
        return (
            "P2P **64 KB** w16 vs w128 箱线对比。"
            "小消息带宽极低（延迟主导），两 world 差异主要来自 rank 数与环长度，"
            "不代表链路饱和能力。"
        )

    if fn == "p2p_box_compare_w16_w128_16777216.svg":
        return (
            "P2P **16 MB** w16 vs w128 箱线。"
            "w128 环相邻边带宽中位 **~100 GB/s**，与机间探针 **~119 GB/s** 同量级；"
            "w16 机内边更高（~114 GB/s 起），符合 node 内 SIO/HCCS 混合拓扑。"
        )

    if fn == "p2p_slow_edges_top15_16mb.svg":
        slow = sorted(p2p_edges, key=lambda x: x[1])[:5]
        txt = "，".join(f"({s},{d})={v:.1f}" for (s, d), v in slow)
        return (
            f"w128 / 16 MB **最慢 15 条 P2P 边**柱图。最慢五边：{txt} GB/s。"
            f"慢边多在 **环相邻跨 node 边界**（如 15→16、127→0），"
            f"对应机间链路而非单卡故障。"
        )

    if fn == "p2p_fast_edges_top15_16mb.svg":
        fast = sorted(p2p_edges, key=lambda x: -x[1])[:5]
        txt = "，".join(f"({s},{d})={v:.1f}" for (s, d), v in fast)
        return (
            f"w128 / 16 MB **最快 15 条边**：{txt} GB/s。"
            f"最快 **(10,11)=120.8 GB/s**，多为 **同 node 内 SIO/HCCS 短跳**，"
            f"高于慢边 ~20%，与 topo 中 SIO 直连对一致。"
        )

    if fn == "p2p_kind_mean_compare_16mb.svg":
        return (
            "P2P 16 MB **边类型均值柱**：环相邻 **~100 GB/s** vs 星型经 rank0 **~107 GB/s**。"
            "星型略高因路径经过 rank0 聚合，但二者同量级，无数量级差异。"
        )

    if fn == "topo_hccs_heatmap_master0.svg":
        return (
            "**master-0 机内 HCCS 拓扑热力图**（16×16 Phy-ID）。"
            "**空白格 = HCCS_SW**（经交换机，约 112 条边）；**S = SIO** 片内直连（8 对）；"
            "**· = self** 对角。不再用满屏 `H` 标注——读图时先看 SIO 邻接带，再看 SW 全互联背景。"
            "8 节点矩阵同构（各 8 SIO + 112 HCCS_SW）。"
        )

    return f"HCCL 战役图 **{fn}**。"


def build_hccl_md(fig_dir: Path) -> str:
    figs = list_svgs(fig_dir)
    hccl_rows = load_hccl_rows()
    retention, means_256 = hccl_medians(hccl_rows)
    p2p_edges = load_p2p_edges()

    op_cn = {
        "all_reduce": "All-Reduce", "all_gather": "All-Gather",
        "reduce_scatter": "Reduce-Scatter", "broadcast": "Broadcast",
    }

    lines = [
        "# HCCL 通信战役报告 · 20260711（jiuding · 中文解读）",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> 数据源：`{HCCL_LOG}`",
        f"> 图目录：`{fig_dir.name}/`（**{len(figs)}** 张 SVG）",
        "",
        "## 摘要",
        "",
        "本报告汇总 All-Reduce / All-Gather / Reduce-Scatter / Broadcast 在 world=16/32/64/128 下的 Bus 带宽，"
        "256 MB 扩展性、Rank 分布、P2P 边级对比与机内拓扑。",
        "",
        "### 256 MB 保持率（相对 world=16）",
        "",
        "| 算子 | w=16 | w=32 | w=64 | w=128 |",
        "|------|------|------|------|-------|",
    ]
    for op in ("all_reduce", "all_gather", "reduce_scatter", "broadcast"):
        cells = [op_cn[op]]
        for ws in (16, 32, 64, 128):
            cells.append(f"{retention[op][ws]:.1f}%")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "### 256 MB 平均 Bus 带宽 (GB/s)",
        "",
        "| 算子 | w=16 | w=32 | w=64 | w=128 |",
        "|------|------|------|------|-------|",
    ]
    for op in ("all_reduce", "all_gather", "reduce_scatter", "broadcast"):
        cells = [op_cn[op]]
        for ws in (16, 32, 64, 128):
            cells.append(f"{means_256[(op, ws)]:.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        f"- HCCL 记录数：{len(hccl_rows)}",
        f"- P2P 去重边数（w128/16MB）：{len(p2p_edges)}",
        "- 拓扑：已解析 `master-0.raw.txt`",
        "",
        "## 读图说明",
        "",
        "- 拓扑图：**空白 = HCCS_SW**，**S = SIO**，**· = self**（新版符号，替代满屏 H）。",
        "- Host 名：**master-0 / worker-N**。",
        "- 柱/曲线：jiuding 大字号；热力图紧凑字号。",
        "",
        "## 图表解读",
        "",
    ]

    sections = [
        ("1. Collective · Bus 带宽 vs 消息大小", [f for f in figs if f.startswith("hccl_bus_bw_vs_size_")]),
        ("2. 256 MB 大消息扩展性", [f for f in figs if f.startswith("hccl_256mb_")]),
        ("3. Rank 分布（256 MB）", [f for f in figs if f.startswith("hccl_rank_")]),
        ("4. P2P", [f for f in figs if f.startswith("p2p_")]),
        ("5. 机内拓扑", [f for f in figs if f.startswith("topo_")]),
    ]
    for sec_title, sec_figs in sections:
        if not sec_figs:
            continue
        lines += [f"## {sec_title}", ""]
        for fn in sec_figs:
            title = fn.replace(".svg", "").replace("_", " ")
            lines += [
                f"### {title}",
                "",
                hccl_caption(fn, retention, means_256, hccl_rows, p2p_edges),
                "",
                f"![{title}]({fig_dir.name}/{fn})",
                "",
            ]

    return "\n".join(lines) + "\n"


def main() -> None:
    cards = load_cards(CONSTITUTION_JSONL)
    reports = []

    specs = [
        ("card_constitution_20260711_jiuding.md", "card_constitution_20260711_jiuding_figs",
         lambda fd, mp: build_constitution_md(cards, fd, mp)),
        ("fillgap_jiuding_20260711.md", "fillgap_jiuding_20260711_figs",
         lambda fd, mp: build_fillgap_md(cards, fd)),
        ("bnmk_shapes_20260711.md", "bnmk_shapes_20260711_figs",
         lambda fd, mp: build_bnmk_md(load_bnmk(CONSTITUTION_JSONL), fd)),
        ("hccl_campaign_20260711.md", "hccl_campaign_20260711_figs",
         lambda fd, mp: build_hccl_md(fd)),
    ]

    for md_name, fig_name, builder in specs:
        md_path = ROUNDS / md_name
        fig_dir = ROUNDS / fig_name
        text = builder(fig_dir, md_path)
        md_path.write_text(text, encoding="utf-8")
        refs, missing = verify_refs(text, fig_dir)
        svgs = len(list_svgs(fig_dir))
        reports.append((md_name, refs, missing, svgs))
        print(f"Wrote {md_path}  refs={refs} missing={missing} svgs_on_disk={svgs}")

    print("\n=== Summary ===")
    for name, refs, missing, svgs in reports:
        print(f"{name}: refs={refs}, missing={missing}, fig_dir={svgs}")


if __name__ == "__main__":
    main()
