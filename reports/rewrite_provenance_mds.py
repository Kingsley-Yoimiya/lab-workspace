#!/usr/bin/env python3
"""按 FIGURE_PROVENANCE_AUDIT 与出图目录，重写带出处的报告 MD（少读图空话）。"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROUNDS = Path(__file__).resolve().parent / "rounds"
FILLGAP = Path(
    "/Users/yinjinrun/random-thing/logs/card-fillgap-20260711_140301/results/"
    "constitution128.merged.jsonl"
)


def load_cards() -> list[dict]:
    cards = []
    with FILLGAP.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("record") == "card":
                cards.append(r)
    return cards


def med(cards: list[dict], key: str) -> float | None:
    vs = [float(c[key]) for c in cards if c.get(key) is not None]
    return statistics.median(vs) if vs else None


def list_svgs(fig_dir: Path) -> list[str]:
    return sorted(p.name for p in fig_dir.glob("*.svg"))


def caption_constitution(name: str, cards: list[dict]) -> str:
    """为 constitution 单图生成出处型图注。"""
    common = (
        f"数据：`logs/card-fillgap-20260711_140301/results/constitution128.merged.jsonl`，"
        f"`record=card`。采集：`screen.py --config config.constitution128.yaml "
        f"--device all --gemm-n 8192 --sustained-s 30 --sdc-rounds 5` "
        f"（job `whj4stu-copy-copy-copy`，8×16=128）。出图：`plot_card_constitution.py`。"
    )
    m = re.match(
        r"^(hist|heatmap_relmed|box_by_host|sorted_bar|bar_host_mean_std)_(.+)\.svg$",
        name,
    )
    if m:
        kind, key = m.group(1), m.group(2)
        mv = med(cards, key)
        mv_s = f"{mv:.4g}" if mv is not None else "—"
        rules = {
            "hist": f"聚合：128 卡直方图；红虚线=集群中位数（{mv_s}）。",
            "heatmap_relmed": (
                f"聚合：host×device 格 = (v−median)/median×100%；"
                f"中位={mv_s}；**仅 |Δ|≥1% 标数**；host 短名 master-0/worker-N。"
            ),
            "box_by_host": "聚合：按 host 分组箱线 + 卡级散点。",
            "sorted_bar": "聚合：按指标值**升序**；橙虚线=集群中位数；hatch 区分 ≥/< 中位。",
            "bar_host_mean_std": "聚合：每 host 均值±σ（总体标准差）。",
        }
        probe = {
            "func_tflops": "探针 `func_perf` GEMM N=8192 bf16 warmup=20 iters=50",
            "hbm_gbps": "探针 `hbm` 1024MB w20/i50",
            "sustained_tflops": "探针 `sustained` 30s window=50",
            "vector_gflops": "探针 `vector_fma_perf` 64M elems fp32",
            "scalar_elems_per_s": "探针 `scalar_chain_perf` 16M elems",
            "cube_vector_tflops": "探针 `cube_vector_pipeline` n=4096 bf16",
            "mte_gbps": "探针 `mte_copy_perf` 512MB",
            "sfu_gflops": "探针 `vector_sfu_perf` op=exp",
            "power_w": "负载探针末次 `npu-smi -t power` 遥测回填",
            "health_power_w": "health 快照 `npu-smi -t power`",
            "health_temp_c": "health 快照 `npu-smi -t temp`",
            "shape_sweep_peak_tflops": "本批无 shape_sweep；字段实为 max(BNMK tflops) 回填",
        }
        hint = "见 constitution128 探针表 / FIGURE_PROVENANCE_AUDIT"
        for k, v in probe.items():
            if key == k:
                hint = v
                break
        else:
            if "launch" in key:
                hint = "探针 `launch_latency` samples=500 warmup=50 burst=64 event timing"
            elif "util" in key or "board_temp" in key:
                hint = "探针 round 末次 `npu-smi -t usages/temp` 回填"
        return (
            f"**{name}**：字段 `{key}`。{hint}。{rules[kind]} {common} "
            f"集群中位≈**{mv_s}**。"
        )

    if name.startswith("scatter_"):
        body = name.replace("scatter_", "").replace(".svg", "")
        return (
            f"**{name}**：卡级散点，轴字段见文件名 `{body}`；按 host 着色。"
            f"缺任一轴则跳过。{common}"
        )
    if name == "box_overview.svg":
        return f"**{name}**：CORE 指标全集群箱线总览（≤6 个有数据轴）。{common}"
    if name == "timeseries_sustained_p05_p50.svg":
        return (
            "**timeseries_sustained_p05_p50.svg**：`record=gemm_sustained_sample` "
            "（iter/t_s/tflops；~2e4 行）。**现行语义**：按 iter 对齐全部卡，"
            "仅保留覆盖≥90%卡的 iter，对该步跨卡序列取 **p05 / p50**；"
            "横轴为同 iter 上 t_s 的中位。**不是**「两张分位代表卡」的单卡曲线。"
            f"条件：sustained 30s。{common}"
        )
    return f"**{name}**。{common}"


def write_constitution(cards: list[dict]) -> None:
    fig_dir = ROUNDS / "card_constitution_20260711_figs"
    svgs = list_svgs(fig_dir)
    lines = [
        "# Card Constitution 分布报告 · 20260711",
        "",
        "- Job：`whj4stu-copy-copy-copy`（8×16 Ascend A3 = 128 卡）",
        f"- 原始数据：`{FILLGAP}`（`record=card` 等）",
        "- 采集：`CARD_SCREEN/screen.py` + `config.constitution128.yaml`；"
        "遥测 `npu-smi info -t … -i card -c chip`",
        "- 出图：`reports/plot_card_constitution.py` → SVG（默认 `plot_style`）",
        f"- 图数：**{len(svgs)}**；溯源总表见 [`FIGURE_PROVENANCE_AUDIT_20260711.md`](FIGURE_PROVENANCE_AUDIT_20260711.md)",
        "",
        "## 关键数字（card 中位）",
        "",
        f"| 指标 | 中位 |",
        f"|---|---:|",
    ]
    for k, lab in [
        ("func_tflops", "Cube func TFLOPS"),
        ("sustained_tflops", "Sustained TFLOPS"),
        ("hbm_gbps", "HBM GB/s"),
        ("vector_gflops", "Vector GFLOPS"),
        ("health_power_w", "Health power W"),
        ("power_w", "Power W（负载末）"),
        ("health_temp_c", "Health temp C"),
    ]:
        v = med(cards, k)
        lines.append(f"| {lab} (`{k}`) | {v:.4g} |" if v is not None else f"| {lab} | — |")
    lines += ["", "## 逐图出处", ""]
    for name in svgs:
        lines.append(caption_constitution(name, cards))
        lines.append("")
        lines.append(f"![{name}](card_constitution_20260711_figs/{name})")
        lines.append("")
    path = ROUNDS / "card_constitution_20260711.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path, "refs", len(svgs))


def write_extra() -> None:
    fig_dir = ROUNDS / "constitution_extra_fillgap_20260711_figs"
    svgs = list_svgs(fig_dir)
    caps = {
        "radar_host_median_norm.svg": (
            "每 host 对多指标取中位再 ÷ 集群中位；极坐标；1.0=集群中位。"
            "同一 fillgap merged JSONL；`plot_constitution_extra.plot_radar_and_parallel`。"
        ),
        "parallel_host_median_norm.svg": "与雷达同一归一化矩阵的平行坐标。",
        "hbm_modes_grouped_bar.svg": (
            "字段 `hbm_mode_{seq_copy,strided,read_heavy,write_heavy}_gbps`；"
            "探针 hbm_modes_perf 512MB；全集群中位 + 各 host 中位分组柱。"
        ),
        "corr_cube_vector_sfu_mte.svg": (
            "Cube/Vector/SFU/MTE 四字段齐全的卡对齐后 Pearson corrcoef。"
        ),
        "box_launch_by_host.svg": (
            "launch_sync_p99 / host_overhead_p99 / burst_p50 按 host 箱线；"
            "探针 launch_latency。"
        ),
        "cdf_core_metrics.svg": "func/hbm/vector/mte/sfu/sustained 经验 CDF + 中位竖线。",
        "extreme10_small_multiples.svg": (
            "按 sustained_tflops 升序取最慢/最快各 10 卡，多指标相对集群中位偏差 %。"
        ),
        "scatter_sustained_vs_func.svg": "x=func_tflops y=sustained_tflops；y=x 与中位比参考线。",
    }
    lines = [
        "# Constitution 增强可视化 · 20260711",
        "",
        "- 数据：同 fillgap `constitution128.merged.jsonl`",
        "- 出图：`plot_constitution_extra.py`，stamp=`constitution_extra_fillgap_20260711`",
        f"- 图数：**{len(svgs)}**；总溯源见 [`FIGURE_PROVENANCE_AUDIT_20260711.md`](FIGURE_PROVENANCE_AUDIT_20260711.md)",
        "",
        "## 逐图出处",
        "",
    ]
    for name in svgs:
        if name.startswith("heatmap_host_device_"):
            key = name.replace("heatmap_host_device_", "").replace(".svg", "")
            cap = (
                f"host×device **绝对值**热力图，字段 `{key}`；色标 p5–p95；"
                f"**偏离集群中位 ≥0.5% 才标数**（与 constitution 的 relmed 热力不同）。"
            )
        else:
            cap = caps.get(name, "见溯源审计。")
        lines.append(f"**{name}**：{cap}")
        lines.append("")
        lines.append(f"![{name}](constitution_extra_fillgap_20260711_figs/{name})")
        lines.append("")
    path = ROUNDS / "constitution_extra_fillgap_20260711.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path, "refs", len(svgs))


def write_bnmk() -> None:
    fig_dir = ROUNDS / "bnmk_shapes_20260711_figs"
    svgs = list_svgs(fig_dir)
    lines = [
        "# BNMK Shape 报告 · 20260711",
        "",
        "- 数据：fillgap merged 中 `record=gemm_bnmk_sample`（现优先只读 merged，**1280** 行 = 10 shape × 128 卡）",
        "- 采集：`BnmkSweep` / `gemm_bnmk_sweep`（constitution128 配置）",
        "- 出图：`plot_bnmk_shapes.py`",
        "",
        "## 逐图出处",
        "",
    ]
    caps = {
        "bnmk_tflops_box_by_label.svg": "按 shape label 分组的 tflops 箱线（每 label 128 卡）。",
        "bnmk_tflops_bar_median_by_label.svg": "每 label 取跨卡中位 TFLOPS 柱图（升序）。",
        "bnmk_host_shape_heatmap.svg": "host×label 格 = 该 host 上该 label 的 tflops 均值；色标 YlOrRd。",
    }
    for name in svgs:
        lines.append(f"**{name}**：{caps.get(name, '')}")
        lines.append("")
        lines.append(f"![{name}](bnmk_shapes_20260711_figs/{name})")
        lines.append("")
    path = ROUNDS / "bnmk_shapes_20260711.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path, "refs", len(svgs))


def write_hccl() -> None:
    fig_dir = ROUNDS / "hccl_campaign_20260711_figs"
    svgs = list_svgs(fig_dir)
    lines = [
        "# HCCL 通信战役报告 · 20260711",
        "",
        "- 数据根：`logs/pipeline-comm-20260711_134811/`",
        "- Collective：`hccl-results/scale_{16,32,64,128}.jsonl`，`record=hccl_bench`",
        "- P2P：`p2p-results/scale_{16,128}.jsonl`，`record=hccl_p2p`",
        "- 拓扑：`hccl-topo/raw/master-0.raw.txt`（`npu-smi info -t topo`）",
        "- 发射：`launch_comm_kubectl.sh` → torchrun + `hccl_torch_bench.py` / `hccl_p2p_bench.py`",
        "- 条件摘要：ops=AR/AG/RS/BC；sizes=1M/16M/64M/256M；backend=hccl；nproc=16",
        "- 出图：`plot_hccl_campaign.py`",
        "",
        "## 关键保持率（256MB bus_bw 相对 world=16）",
        "",
        "| 算子 | w32 | w64 | w128 |",
        "|---|---:|---:|---:|",
        "| All-Reduce | 96.8% | 94.9% | 89.4% |",
        "| Broadcast | 91.4% | 86.8% | 86.8% |",
        "| All-Gather | 88.0% | 64.2% | 54.0% |",
        "| Reduce-Scatter | 91.8% | 71.0% | 46.4% |",
        "",
        "## 逐图出处",
        "",
    ]
    for name in svgs:
        if name.startswith("hccl_bus_bw_vs_size_"):
            op = name.replace("hccl_bus_bw_vs_size_", "").replace(".svg", "")
            cap = (
                f"op=`{op}`；x=消息大小，y=各 rank `bus_bw_GBps` 中位；"
                f"曲线按 world∈{{16,32,64,128}}。聚合：同 (op,world,nbytes) 跨 rank 中位。"
            )
        elif "retention" in name:
            cap = (
                "256MB 各算子 bus_bw 相对 world=16 的保持率柱；"
                "retention[op][w]=med(w)/med(16)×100%。"
            )
        elif "step" in name:
            cap = "256MB 阶梯：world 增大时各算子 bus_bw 中位变化。"
        elif name.startswith("hccl_rank_"):
            cap = "256MB 下各 rank 的 bus_bw 分布（箱线/直方图/小提琴）；数据为每 rank 一行。"
        elif name.startswith("p2p_"):
            cap = (
                "P2P JSONL：isend/irecv 测边带宽；含 w16/w128、64KB/16MB；"
                "慢/快边 TopK 按 bw 排序。"
            )
        elif "topo" in name:
            cap = (
                "解析 `npu-smi -t topo` 矩阵；色=亲和等级（SIO/HCCS/HCCS_SW…）；"
                "空白格=HCCS_SW，S=SIO，·=self；**不再满屏写 HCCS_SW**。"
            )
        else:
            cap = "见 FIGURE_PROVENANCE_AUDIT。"
        lines.append(f"**{name}**：{cap}")
        lines.append("")
        lines.append(f"![{name}](hccl_campaign_20260711_figs/{name})")
        lines.append("")
    path = ROUNDS / "hccl_campaign_20260711.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path, "refs", len(svgs))


def write_inter() -> None:
    fig_dir = ROUNDS / "inter_bw_20260711_figs"
    svgs = list_svgs(fig_dir)
    lines = [
        "# 机内/机间 P2P 带宽探针 · 20260711",
        "",
        "- 数据：`logs/inter-bw-20260711_141922/`（`record=hccl_inter_bw`）",
        "- 采集：`hccl_inter_bw_probe.py`，torchrun 8×16；严格串行单对 + 全员 barrier；"
        "流水线 uni `inflight=4`；sizes=1M/16M/64M/256M；`HCCL_BUFFSIZE=2048`",
        "- 边集：intra=同节点固定 local pairs；inter=跨节点同 local_rank 对齐",
        "- 汇总：`summarize_inter_bw.py`（recv 侧）；本图为各 size 的 kind 中位柱",
        "- 交叉：ping-pong RTT/2 见 `logs/inter-bw-20260711_142537/`（inter@256M≈117 GB/s）",
        "",
        "## 关键数字（recv 中位 GB/s）",
        "",
        "| kind | 1M | 16M | 64M | 256M |",
        "|---|---:|---:|---:|---:|",
        "| intra | 7.70 | 91.27 | 114.04 | 122.35 |",
        "| inter | 8.64 | 89.39 | 110.52 | 119.32 |",
        "",
        "## 图",
        "",
    ]
    for name in svgs:
        lines.append(
            f"**{name}**：intra vs inter 在各消息大小上的中位带宽（GB/s）；"
            f"来自上述 JSONL 聚合，非 hccn 静态读数。"
        )
        lines.append("")
        lines.append(f"![{name}](inter_bw_20260711_figs/{name})")
        lines.append("")
    path = ROUNDS / "inter_bw_20260711.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path, "refs", len(svgs))


def patch_campaign() -> None:
    path = ROUNDS / "CAMPAIGN_FINAL_20260711.md"
    text = path.read_text(encoding="utf-8")
    # rewrite section 3 index cleanly
    block = """## 3. 产物索引（看图从这里进）

> 出图默认 `reports/plot_style.py`（大字号 / 去顶右边框 / y 点线网格 / hatch 柱 / **SVG**）。  
> **图注以数据出处与测量条件为主**；逐类溯源见 [`FIGURE_PROVENANCE_AUDIT_20260711.md`](FIGURE_PROVENANCE_AUDIT_20260711.md)。

### 体质主报告（含功耗/温度）
- [`card_constitution_20260711.md`](card_constitution_20260711.md)
- [`card_constitution_20260711_figs/`](card_constitution_20260711_figs/)（**112** svg）
  - 注意：`timeseries_sustained_p05_p50.svg` = **跨卡** p05/p50（按 iter 对齐），不是代表卡时序

### 体质增强
- [`constitution_extra_fillgap_20260711.md`](constitution_extra_fillgap_20260711.md)
- [`constitution_extra_fillgap_20260711_figs/`](constitution_extra_fillgap_20260711_figs/)（12）

### BNMK 10 shape
- [`bnmk_shapes_20260711.md`](bnmk_shapes_20260711.md) / [`bnmk_shapes_20260711_figs/`](bnmk_shapes_20260711_figs/)（3；样本 1280）

### HCCL + P2P + 拓扑
- [`hccl_campaign_20260711.md`](hccl_campaign_20260711.md) / [`hccl_campaign_20260711_figs/`](hccl_campaign_20260711_figs/)（23）
- [`topo_summary_20260711.md`](topo_summary_20260711.md)

### 机间带宽
- [`inter_bw_20260711.md`](inter_bw_20260711.md) / [`INTER_BW_PROBE_20260711.md`](INTER_BW_PROBE_20260711.md)

### 原始数据
- 体质 fillgap: `logs/card-fillgap-20260711_140301/results/constitution128.merged.jsonl`
- 通信: `logs/pipeline-comm-20260711_134811/`
- 机间: `logs/inter-bw-20260711_141922/`
"""
    if "## 3. 产物索引" in text:
        pre = text.split("## 3. 产物索引")[0]
        # keep section 4+ if present
        rest = text.split("## 4.", 1)
        post = ("## 4." + rest[1]) if len(rest) > 1 else ""
        text = pre + block + "\n" + post
    path.write_text(text, encoding="utf-8")
    print("patched", path)


def main() -> None:
    cards = load_cards()
    write_constitution(cards)
    write_extra()
    write_bnmk()
    write_hccl()
    write_inter()
    patch_campaign()
    # validate
    for md_name, fig in [
        ("card_constitution_20260711.md", "card_constitution_20260711_figs"),
        ("constitution_extra_fillgap_20260711.md", "constitution_extra_fillgap_20260711_figs"),
        ("bnmk_shapes_20260711.md", "bnmk_shapes_20260711_figs"),
        ("hccl_campaign_20260711.md", "hccl_campaign_20260711_figs"),
        ("inter_bw_20260711.md", "inter_bw_20260711_figs"),
    ]:
        md = (ROUNDS / md_name).read_text(encoding="utf-8")
        refs = re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", md)
        miss = [r for r in refs if not (ROUNDS / r).exists()]
        print(f"check {md_name}: refs={len(refs)} miss={len(miss)}")


if __name__ == "__main__":
    main()
