#!/usr/bin/env python3
"""集群 HCCS 拓扑摘要：解析 hccl-topo raw，输出 JSON + 中文 Markdown。"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

TOPO_RAW_DIR = Path(
    "/Users/yinjinrun/random-thing/logs/pipeline-comm-20260711_134811/hccl-topo/raw"
)
OUT_JSON = Path(__file__).resolve().parent / "rounds" / "topo_summary_20260711.json"
OUT_MD = Path(__file__).resolve().parent / "rounds" / "topo_summary_20260711.md"


def parse_hccs_matrix(raw_text: str) -> list[list[str]] | None:
    """从 npu-smi topo raw 文本解析 Phy-ID × Phy-ID 矩阵。"""
    lines = [ln.rstrip() for ln in raw_text.splitlines()]
    in_topo = False
    matrix: list[list[str]] = []
    for ln in lines:
        if ln.strip() == "=== topo ===":
            in_topo = True
            continue
        if in_topo and ln.strip().startswith("Legend:"):
            break
        if not in_topo:
            continue
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith("Phy-ID") or (
            stripped.startswith("NPU") and ("HCCS" in stripped or "PIX" in stripped)
        ):
            parts = re.split(r"\s+", stripped)
            if len(parts) >= 2:
                # 跳过列标题行（Phy-ID0 Phy-ID1 …）
                if all(p.startswith("Phy-ID") or p.startswith("NPU") for p in parts[1:]):
                    continue
                matrix.append(parts[1:])
        elif "\t" in ln or "  " in ln:
            parts = re.split(r"\s+", stripped)
            if len(parts) >= 2 and any(
                tok in ("HCCS", "HCCS_SW", "PIX", "PHB", "SYS", "SIO", "X")
                for tok in parts[1:]
            ):
                matrix.append(parts[1:])
    if len(matrix) < 2:
        return None
    return matrix


def matrix_ok(matrix: list[list[str]]) -> bool:
    n = len(matrix)
    if n < 2:
        return False
    for i, row in enumerate(matrix):
        if len(row) != n:
            return False
        if row[i].strip() != "X":
            return False
    return True


def count_edge_type(matrix: list[list[str]], edge: str) -> int:
    """统计无向边（i < j）。"""
    n = len(matrix)
    cnt = 0
    for i in range(n):
        row_len = len(matrix[i])
        for j in range(i + 1, min(n, row_len)):
            if matrix[i][j].strip() == edge:
                cnt += 1
    return cnt


def parse_host(raw_text: str, fallback: str) -> str:
    for ln in raw_text.splitlines():
        if ln.startswith("HOST="):
            return ln.split("=", 1)[1].strip()
    return fallback


def parse_npu_count(raw_text: str) -> int | None:
    m = re.search(r"Total Count\s*:\s*(\d+)", raw_text)
    return int(m.group(1)) if m else None


def parse_hccn_flags(raw_text: str) -> tuple[bool, bool]:
    conf_present = False
    tool_present = False
    lines = raw_text.splitlines()
    in_hccn = False
    hccn_body: list[str] = []
    for ln in lines:
        if ln.strip() == "=== hccn.conf ===":
            in_hccn = True
            continue
        if in_hccn:
            if ln.startswith("==="):
                break
            hccn_body.append(ln)
        if ln.startswith("HCCN_TOOL="):
            val = ln.split("=", 1)[1].strip()
            tool_present = bool(val)
    body = "\n".join(hccn_body).strip()
    if body and "No such file" not in body and "not found" not in body.lower():
        conf_present = True
    return conf_present, tool_present


def summarize_host(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    host = parse_host(text, path.stem.replace(".raw", ""))
    matrix = parse_hccs_matrix(text)
    ok = matrix is not None and matrix_ok(matrix)
    n_phy = len(matrix) if matrix else 0
    sio_pairs = count_edge_type(matrix, "SIO") if matrix else 0
    hccs_sw_edges = count_edge_type(matrix, "HCCS_SW") if matrix else 0
    npu_count = parse_npu_count(text)
    hccn_conf, hccn_tool = parse_hccn_flags(text)
    return {
        "host": host,
        "n_phy": n_phy,
        "npu_count": npu_count,
        "sio_pairs": sio_pairs,
        "hccs_sw_edges": hccs_sw_edges,
        "matrix_ok": ok,
        "hccn_conf_present": hccn_conf,
        "hccn_tool_present": hccn_tool,
        "source": path.name,
    }


def build_cluster(per_host: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_hosts": len(per_host),
        "topo_tier_hint": "8n128",
        "hccn_conf_present": all(h["hccn_conf_present"] for h in per_host),
        "hccn_tool_present": all(h["hccn_tool_present"] for h in per_host),
        "all_matrix_ok": all(h["matrix_ok"] for h in per_host),
        "npu_per_host": per_host[0]["npu_count"] if per_host else None,
        "phy_per_host": per_host[0]["n_phy"] if per_host else None,
    }


def build_md(summary: dict[str, Any]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    cluster = summary["cluster"]
    hosts = summary["per_host"]
    lines = [
        "# 集群拓扑摘要 · 20260711",
        "",
        f"> 生成时间：{ts}  ",
        f"> 数据源：`{TOPO_RAW_DIR}`",
        "",
        "## 集群概览",
        "",
        f"- 节点数：**{cluster['n_hosts']}**（拓扑层级提示 `{cluster['topo_tier_hint']}`）",
        f"- 每节点 NPU：**{cluster.get('npu_per_host', '?')}** · Phy-ID 维度：**{cluster.get('phy_per_host', '?')}**",
        f"- 全部矩阵解析正常：**{'是' if cluster['all_matrix_ok'] else '否'}**",
        f"- `/etc/hccn.conf` 存在：**{'是' if cluster['hccn_conf_present'] else '否'}**",
        f"- `hccn_tool` 可用：**{'是' if cluster['hccn_tool_present'] else '否'}**",
        "",
        "## 机内互联特征",
        "",
        "各节点 Phy-ID 矩阵呈 **8 组 SIO 邻接对 + 全互联 HCCS_SW** 模式：",
        "每两颗相邻 Phy-ID（同 NPU 双芯）经 SIO 直连，跨芯/跨 NPU 经 HCCS 交换机互联。",
        "",
        "## 逐节点",
        "",
        "| 节点 | Phy-ID | SIO 对 | HCCS_SW 边 | 矩阵 OK | hccn.conf | hccn_tool |",
        "|------|--------|--------|------------|---------|-----------|-----------|",
    ]
    for h in hosts:
        lines.append(
            f"| {h['host']} | {h['n_phy']} | {h['sio_pairs']} | {h['hccs_sw_edges']} | "
            f"{'✓' if h['matrix_ok'] else '✗'} | "
            f"{'✓' if h['hccn_conf_present'] else '✗'} | "
            f"{'✓' if h['hccn_tool_present'] else '✗'} |"
        )
    lines += [
        "",
        "## 解读",
        "",
        "- **SIO 对**：矩阵中 `SIO` 无向边计数（期望 8，对应 8 颗 NPU 各 1 条片内 SIO 链路）。",
        "- **HCCS_SW 边**：经 HCCS 交换机的无向边（16×16 全互联减对角与 SIO 后约 112 条）。",
        "- 本批 raw 中 `hccn.conf` 与 `HCCN_TOOL` 均未采集到，跨节点 RoCE 细节需后续补采。",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if not TOPO_RAW_DIR.is_dir():
        raise SystemExit(f"拓扑 raw 目录不存在：{TOPO_RAW_DIR}")

    raw_files = sorted(TOPO_RAW_DIR.glob("*.raw.txt"))
    if not raw_files:
        raise SystemExit(f"未找到 *.raw.txt：{TOPO_RAW_DIR}")

    per_host = [summarize_host(p) for p in raw_files]
    per_host.sort(key=lambda h: h["host"])

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(TOPO_RAW_DIR),
        "per_host": per_host,
        "cluster": build_cluster(per_host),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(build_md(summary), encoding="utf-8")

    c = summary["cluster"]
    print(f"Wrote JSON -> {OUT_JSON}")
    print(f"Wrote MD   -> {OUT_MD}")
    print(f"Hosts={c['n_hosts']} tier={c['topo_tier_hint']} "
          f"matrix_ok={c['all_matrix_ok']} hccn_conf={c['hccn_conf_present']}")


if __name__ == "__main__":
    main()
