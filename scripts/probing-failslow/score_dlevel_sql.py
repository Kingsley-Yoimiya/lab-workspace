#!/usr/bin/env python3
"""D0–D5 判分：离线训练埋点 + C2 Probing SQL dump。

D4 规则（decisions A5）：
  - 必须先离线到 D3
  - 读 probing/query_manifest.json
  - 缺 EXT 所需表 → 停 D3，tool_probing_sql=TABLE_MISSING
  - 表在但无外部争用信号 → 停 D3，SQL_NO_EXT_EVIDENCE
  - 表在且信号命中且 grid 对 → D4
绝不把 injection.log 升为 D4。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score_dlevel_offline import GT, score_case  # noqa: E402


def load_manifest(root: Path, case: str) -> dict | None:
    paths = list(root.glob(f"{case}/by_pod/*/round_*/C2_probing/probing/query_manifest.json"))
    if not paths:
        paths = list(root.glob(f"{case}/**/C2_probing/probing/query_manifest.json"))
    if not paths:
        return None
    paths.sort(key=lambda p: (0 if "h14410" in str(p) else 1, str(p)))
    return json.loads(paths[0].read_text())


def read_query_ok(root: Path, case: str, name: str) -> tuple[bool, str]:
    paths = list(root.glob(f"{case}/**/C2_probing/probing/query_{name}.txt"))
    if not paths:
        return False, "missing_file"
    text = paths[0].read_text(errors="ignore")
    if "error=" in text or "not found" in text.lower() or "QueryError" in text:
        return False, text[-400:]
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("SQL:") and l != "----"]
    return (len(lines) > 2), text[:800]


def ext_evidence(case: str, manifest: dict, root: Path) -> tuple[bool, str]:
    present = manifest.get("tables_present") or {}
    missing = manifest.get("tables_missing") or []

    if case.startswith("P1-EXT"):
        if not present.get("process.gpu_users"):
            if not present.get("gpu.utilization"):
                miss = [t for t in ("gpu.utilization", "process.gpu_users") if not present.get(t)]
                return False, "TABLE_MISSING:" + ",".join(miss)
            return False, "SQL_NO_EXT_EVIDENCE:gpu.utilization_present_but_no_process.gpu_users"
        ok, _ = read_query_ok(root, case, "process_gpu_users")
        return (ok, "process.gpu_users_rows" if ok else "process.gpu_users_empty")

    if case.startswith("P3-EXT"):
        if not present.get("process.cpu_stats"):
            if not present.get("cpu.utilization"):
                return False, "TABLE_MISSING:process.cpu_stats,cpu.utilization"
            return False, "SQL_NO_EXT_EVIDENCE:cpu.utilization_is_self_process_only"
        ok, snippet = read_query_ok(root, case, "process_cpu_stats")
        if ok and re.search(r"stress", snippet, re.I):
            return True, "process.cpu_stats_stress"
        return False, "SQL_NO_EXT_EVIDENCE:no_external_pid_in_process.cpu_stats"

    _ = missing
    return False, "unsupported_case"


def score_with_sql(root: Path, case: str, dose: str) -> dict:
    base = score_case(root, case, dose)
    # normalize d_level to int then string label
    d_num = int(base.get("d_level") or 0)
    manifest = load_manifest(root, case)
    notes = [base.get("notes") or ""]

    if manifest is None:
        base["tool_probing_sql"] = "SQL_PENDING"
        notes.append("D4_pending: no probing/query_manifest.json")
        base["notes"] = "; ".join(x for x in notes if x)
        base["d_level"] = f"D{d_num}"
        return base

    missing = manifest.get("tables_missing") or []
    base["tool_probing_sql"] = "DUMP_OK"
    notes.append("sql_dump=ok")
    if missing:
        notes.append("tables_missing=" + ",".join(missing))

    if d_num < 3:
        notes.append(f"D4_skipped: offline_d_level=D{d_num}")
        base["notes"] = "; ".join(x for x in notes if x)
        base["d_level"] = f"D{d_num}"
        return base

    hit, evid = ext_evidence(case, manifest, root)
    notes.append(evid)
    if evid.startswith("TABLE_MISSING"):
        base["tool_probing_sql"] = "TABLE_MISSING"
        base["notes"] = "; ".join(x for x in notes if x)
        base["d_level"] = "D3"
        return base
    if not hit:
        base["tool_probing_sql"] = "SQL_NO_EXT_EVIDENCE"
        base["notes"] = "; ".join(x for x in notes if x)
        base["d_level"] = "D3"
        return base

    grid = GT.get(case, {}).get("grid", "")
    base["d_level"] = "D4"
    base["grid_reported"] = grid
    base["tool_probing_sql"] = "PASS_D4"
    notes.append(f"D4 grid={grid}")
    base["notes"] = "; ".join(x for x in notes if x)
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--cases", default="P1-EXT-A,P1-EXT-B,P3-EXT-A")
    ap.add_argument("--dose", default="Loud")
    args = ap.parse_args()
    root = Path(args.result_root)
    run_id = root.name
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]

    rows = []
    for case in cases:
        r = score_with_sql(root, case, args.dose)
        r["run_id"] = run_id
        r["case"] = r.get("case_id", case)
        rows.append(r)

    csv_path = root / f"scoring_table_SQL_{args.dose}.csv"
    fields = [
        "run_id", "dose", "case", "d_level", "c1_c0", "target_reported", "target_truth",
        "grid_reported", "grid_truth", "tool_probing_sql", "tool_greyhound", "tool_xputimer", "notes",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md = [f"# Verdict SQL — {run_id} ({args.dose})", ""]
    md.append("| case | C1/C0 | d_level | SQL | notes |")
    md.append("|---|---:|---|---|---|")
    for r in rows:
        note = (r.get("notes") or "")[:140].replace("|", "/")
        md.append(
            f"| {r['case']} | {r.get('c1_c0') or '—'} | **{r['d_level']}** | "
            f"{r['tool_probing_sql']} | {note} |"
        )
    md.append("")
    md.append("- 主证据：C2 `probing/query_manifest.json`；训练 jsonl 仅离线验证到 D3。")
    md.append("- Greyhound / XPUTimer = ENV-BLOCKED（不记 D0）。")
    md.append(f"- CSV: `{csv_path}`")
    (root / f"VERDICT_SQL_{args.dose}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
