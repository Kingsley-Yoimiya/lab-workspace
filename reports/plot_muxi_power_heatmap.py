#!/usr/bin/env python3
"""从 telemetry JSONL 画 128 卡功耗/频率时间平均热力图。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPORTS = Path(__file__).resolve().parents[2] / "reports"
sys.path.insert(0, str(_REPORTS))
from plot_style import apply_plot_style, save_fig, style_axes  # noqa: E402


def _extract_gpus(smi: dict | list) -> list[dict]:
    """mx-smi -j：顶层 bus-id 键 → 每卡 dict；或 gpus 列表。"""
    if isinstance(smi, list):
        return [x for x in smi if isinstance(x, dict)]
    if not isinstance(smi, dict):
        return []
    for key in ("gpus", "GPU", "gpu", "Devices", "devices"):
        if key in smi and isinstance(smi[key], list):
            return [x for x in smi[key] if isinstance(x, dict)]
    bus = []
    for k, v in smi.items():
        if isinstance(v, dict) and ("device_id" in v or "board_power" in v or "clocks" in v):
            v = dict(v)
            v["_bus"] = k
            bus.append(v)
    if bus:
        bus.sort(key=lambda x: int(x.get("device_id", 0)))
        return bus
    return []


def _num(d: dict, keys: list[str]) -> float | None:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(str(d[k]).replace("W", "").replace("MHz", "").strip())
            except ValueError:
                continue
        for v in d.values():
            if isinstance(v, dict):
                n = _num(v, keys)
                if n is not None:
                    return n
    return None


def _power_w(g: dict) -> float | None:
    bp = g.get("board_power")
    if isinstance(bp, dict):
        pw = bp.get("power")
        if isinstance(pw, dict):
            try:
                return float(pw.get("Way0") or pw.get("way0") or 0)
            except (TypeError, ValueError):
                pass
        if "power" in bp and not isinstance(bp["power"], dict):
            try:
                return float(bp["power"])
            except (TypeError, ValueError):
                pass
    return _num(g, ["power_w", "power", "Power", "Pwr:Usage", "usage"])


def _clock_mhz(g: dict) -> float | None:
    clocks = g.get("clocks")
    if isinstance(clocks, dict):
        xcore = clocks.get("XCORE") or clocks.get("xcore")
        if isinstance(xcore, dict) and "XCORE_CLK" in xcore:
            try:
                return float(xcore["XCORE_CLK"])
            except (TypeError, ValueError):
                pass
        csc = clocks.get("CSC") or {}
        if isinstance(csc, dict) and "CSC_SMPCLK" in csc:
            try:
                return float(csc["CSC_SMPCLK"])
            except (TypeError, ValueError):
                pass
    return _num(g, ["sm_clock_mhz", "clock", "Clock", "sm_clock", "graphics_clock"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--telem-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metric", choices=["power", "clock"], default="power")
    args = ap.parse_args()

    # node -> local_gpu -> list values
    acc: dict[tuple[int, int], list[float]] = {}
    for path in sorted(args.telem_dir.glob("node*.jsonl")):
        for line in path.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            node = int(rec.get("node", -1))
            smi = rec.get("smi") or {}
            if isinstance(smi, dict) and "raw" in smi and isinstance(smi["raw"], str):
                raw = smi["raw"]
                i = raw.find("{")
                if i >= 0:
                    try:
                        smi = json.loads(raw[i:])
                    except json.JSONDecodeError:
                        continue
            gpus = _extract_gpus(smi)
            for li, g in enumerate(gpus[:8]):
                val = _power_w(g) if args.metric == "power" else _clock_mhz(g)
                if val is None:
                    continue
                acc.setdefault((node, li), []).append(val)

    grid = np.full((16, 8), np.nan)
    for (n, li), vals in acc.items():
        if 0 <= n < 16 and 0 <= li < 8 and vals:
            grid[n, li] = float(np.mean(vals))

    apply_plot_style((10, 6))
    fig, ax = plt.subplots()
    label = "板级功耗 Way0 (W)" if args.metric == "power" else "XCORE 频率 (MHz)"
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("节点内 GPU 编号")
    ax.set_ylabel("节点编号")
    ax.set_title(f"MUXI 128 卡{label}时间平均热力图")
    style_axes(ax)
    out = save_fig(fig, args.out, also_png=True)
    cap = (
        f"{label}：对各卡在采集窗口内对 mx-smi -j 采样取时间平均，"
        "排布为 16 节点 × 8 卡。"
        "与虚拟同步差距指标对照：热力图可看起来均匀，但 gap 仍随规模上升。"
    )
    args.out.with_suffix(".caption.md").write_text(cap + "\n", encoding="utf-8")
    meta = {
        "n_cells_filled": int(np.sum(~np.isnan(grid))),
        "metric": args.metric,
        "mean": float(np.nanmean(grid)) if np.any(~np.isnan(grid)) else None,
        "std": float(np.nanstd(grid)) if np.any(~np.isnan(grid)) else None,
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"WROTE {out} meta={meta}")


if __name__ == "__main__":
    main()
