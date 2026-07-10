#!/usr/bin/env python3
"""根据 dense/moe MFU JSONL 生成 reports/train_mfu_128.md 与图。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[3]  # random-thing
REPORTS = Path(__file__).resolve().parent
FIGS = REPORTS / "train_mfu_128_figs"
DENSE = ROOT / "logs/mfu-dense-20260710_225626/results"
MOE = ROOT / "logs/mfu-moe-20260710_225844/results"


def load(d: Path) -> list[dict]:
    rows: list[dict] = []
    if not d.exists():
        return rows
    for f in sorted(d.glob("scale_*.jsonl")):
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("record") == "train_mfu" and o.get("rank", 0) == 0:
                rows.append(o)
    rows.sort(key=lambda x: x["world_size"])
    return rows


def weak_eff(rows: list[dict]) -> dict[int, float]:
    if not rows:
        return {}
    base = next((r for r in rows if r["world_size"] == 16), rows[0])
    base_per = base["tokens_per_sec"] / base["world_size"]
    return {r["world_size"]: 100.0 * (r["tokens_per_sec"] / r["world_size"]) / base_per for r in rows}


def plot_all(dense: list[dict], moe: list[dict]) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if dense:
        ax.plot([r["world_size"] for r in dense], [r["mfu"] * 100 for r in dense], "o-", label="Dense", linewidth=2)
    if moe:
        ax.plot([r["world_size"] for r in moe], [r["mfu"] * 100 for r in moe], "s-", label="MoE", linewidth=2)
    ax.set_xlabel("World size (NPUs)")
    ax.set_ylabel("MFU (%)")
    ax.set_title("训练 MFU 随规模变化（微基准）")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks([16, 32, 64, 128])
    fig.tight_layout()
    fig.savefig(FIGS / "mfu_vs_scale.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if dense:
        ax.plot([r["world_size"] for r in dense], [r["tokens_per_sec"] / 1e6 for r in dense], "o-", label="Dense", linewidth=2)
    if moe:
        ax.plot([r["world_size"] for r in moe], [r["tokens_per_sec"] / 1e6 for r in moe], "s-", label="MoE", linewidth=2)
    ax.set_xlabel("World size (NPUs)")
    ax.set_ylabel("Tokens/s（百万）")
    ax.set_title("吞吐随规模变化")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks([16, 32, 64, 128])
    fig.tight_layout()
    fig.savefig(FIGS / "toks_vs_scale.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, rows, marker in [("Dense", dense, "o"), ("MoE", moe, "s")]:
        eff = weak_eff(rows)
        if not eff:
            continue
        xs = sorted(eff)
        ax.plot(xs, [eff[x] for x in xs], f"{marker}-", label=name, linewidth=2)
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5, label="理想=100%")
    ax.set_xlabel("World size (NPUs)")
    ax.set_ylabel("弱扩展效率（%，相对 16 卡单卡吞吐）")
    ax.set_title("弱扩展效率")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks([16, 32, 64, 128])
    fig.tight_layout()
    fig.savefig(FIGS / "weak_scaling.png", dpi=140)
    plt.close()


def fmt_table(rows: list[dict], eff: dict[int, float]) -> str:
    lines = [
        "| world | MFU | tokens/s | step_ms | achieved TFLOPS | peak TFLOPS | 弱扩展效率 |",
        "|------:|----:|---------:|--------:|----------------:|------------:|----------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['world_size']} | {r['mfu']*100:.2f}% | {r['tokens_per_sec']:.0f} | "
            f"{r['step_ms']:.1f} | {r['achieved_tflops']:.1f} | {r['peak_tflops']:.0f} | "
            f"{eff.get(r['world_size'], float('nan')):.1f}% |"
        )
    return "\n".join(lines)


def main() -> None:
    dense = load(DENSE)
    moe = load(MOE)
    plot_all(dense, moe)
    de = weak_eff(dense)
    me = weak_eff(moe)

    dense_cfg = dense[0] if dense else {}
    moe_cfg = moe[0] if moe else {}
    dmap = {r["world_size"]: r for r in dense}
    mmap = {r["world_size"]: r for r in moe}

    cmp_lines = [
        "| world | Dense MFU | MoE MFU | MoE/Dense |",
        "|------:|----------:|--------:|----------:|",
    ]
    for w in [16, 32, 64, 128]:
        dm, mm = dmap.get(w), mmap.get(w)
        d_s = f"{dm['mfu']*100:.2f}%" if dm else "—"
        m_s = f"{mm['mfu']*100:.2f}%" if mm else "—"
        r_s = f"{(mm['mfu']/dm['mfu'])*100:.1f}%" if dm and mm else "—"
        cmp_lines.append(f"| {w} | {d_s} | {m_s} | {r_s} |")

    dense_mfu16 = f"{dense[0]['mfu']*100:.2f}%" if dense else "—"
    dense_mfu128 = f"{dense[-1]['mfu']*100:.2f}%" if dense else "—"
    dense_tok16 = f"{dense[0]['tokens_per_sec']/1e6:.2f}" if dense else "?"
    dense_tok128 = f"{dense[-1]['tokens_per_sec']/1e6:.2f}" if dense else "?"
    weak128 = f"{de.get(128, float('nan')):.1f}" if dense else "?"
    moe_mfu16 = f"{moe[0]['mfu']*100:.2f}%" if moe else "—"
    moe_worlds = ", ".join(str(r["world_size"]) for r in moe) or "无"

    md = f"""# 训练 MFU 128 卡扩展报告

> 数据：Dense `{DENSE}`；MoE `{MOE}`  
> 生成时间：2026-07-10

## 1. 说明（重要）

本报告使用自研微基准 [`mfu_train_bench.py`](../scripts/cluster/mfu_train_bench.py)（torchrun + HCCL），**不是**完整 MindSpeed/Megatron Qwen3-8B / Qwen3-30B-A3B 训练。

原版 `geruijun` 脚本阻塞原因：

1. 路径写死 `/afs-grj`（实际为 `/afs-a3-241ceshi-shared/geruijun/`）
2. 脚本末尾 `| tee` 语法损坏
3. `PT_qwen3_8B.sh` 实际带 `NUM_EXPERTS`（MoE 形态）

因此按计划交付 **scale 阶梯 MFU/吞吐**，用可控微基准覆盖 16→128；完整 Qwen 需另修 wrapper。

峰值算力按 **320 TFLOPS/卡（bf16 估计）**；MFU = achieved / (peak × world_size)。

## 2. 关键结论

1. **Dense** 全档成功：16 卡 MFU **{dense_mfu16}** → 128 卡 **{dense_mfu128}**；吞吐从 {dense_tok16}M 升至 {dense_tok128}M tokens/s。
2. Dense 弱扩展效率 128 相对 16：**{weak128}%**（单卡吞吐略降，与 HCCL 跨节点开销一致）。
3. **MoE**（experts=8, topk=2）MFU 显著低于 Dense（约 **{moe_mfu16}** @16），符合专家路由 + 额外通信开销预期。
4. MoE 已测档位：{moe_worlds}。

## 3. Dense 结果

配置：seq={dense_cfg.get("seq","?")}, hidden={dense_cfg.get("hidden","?")}, layers={dense_cfg.get("layers","?")}（约 {dense_cfg.get("n_params",0)/1e6:.0f}M 参数）

{fmt_table(dense, de) if dense else "_无数据_"}

## 4. MoE 结果

配置：seq={moe_cfg.get("seq","?")}, hidden={moe_cfg.get("hidden","?")}, layers={moe_cfg.get("layers","?")}, experts=8, topk=2

{fmt_table(moe, me) if moe else "_无数据_"}

## 5. Dense vs MoE

{chr(10).join(cmp_lines)}

## 6. 图

![MFU vs scale](train_mfu_128_figs/mfu_vs_scale.png)

![tokens/s vs scale](train_mfu_128_figs/toks_vs_scale.png)

![弱扩展效率](train_mfu_128_figs/weak_scaling.png)

## 7. 复现

```bash
# Dense
MODE=dense SCALES=16,32,64,128 ./scripts/cluster/run_mfu_bench_scale.sh

# MoE（注意递增 MASTER_PORT，避免 Bind_IP_Port）
MODE=moe SCALES=16,32,64,128 MASTER_PORT=31001 ./scripts/cluster/run_mfu_bench_scale.sh

# 出报告
python3 reports/gen_train_mfu_128_report.py
```

本地日志：`logs/mfu-dense-*`、`logs/mfu-moe-*`；AFS：`/afs-a3-241ceshi-shared/montyyin/results/mfu-*`。
"""
    out = REPORTS / "train_mfu_128.md"
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}")
    print("dense worlds", [r["world_size"] for r in dense])
    print("moe worlds", [r["world_size"] for r in moe])


if __name__ == "__main__":
    main()
