#!/usr/bin/env python3
"""汇总敏感度图谱矩阵的剂量—响应统计量。

Example:
  python3 reports/analyze_sensitivity_atlas.py \
    --result-dir /path/to/sensitivity-atlas-run
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


MANIFEST_NAME = "matrix_manifest.jsonl"
JSON_OUTPUT_NAME = "atlas_analysis.json"
CSV_OUTPUT_NAME = "atlas_rows.csv"
MANIFEST_FIELDS = ("summary_path", "inject_kind", "workload", "profile", "pattern")
CSV_FIELDS = [
    "summary_path",
    "inject_kind",
    "workload",
    "profile",
    "pattern",
    "selected_target_duty",
    "throughput_drop_pct_at_target_0_5_or_max",
    "linear_slope",
    "linear_intercept",
    "linear_r2",
    "quadratic_a",
    "quadratic_b",
    "quadratic_c",
    "linear_max_residual_relative_to_peak",
    "auc_0_to_max_dose",
    "baseline_throughput",
    "max_dose_p90_amplification",
    "max_dose_p99_amplification",
    "sidecar_actual_busy_at_selected_target",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help=(
            "结果目录；必须包含 matrix_manifest.jsonl，且 manifest 引用的 "
            "*.summary.json 必须存在"
        ),
    )
    return parser.parse_args()


def _fail(message: str) -> "NoReturn":
    raise ValueError(message)


def _number(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{where} 必须是数字，实际为 {value!r}")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{where} 必须是有限数字，实际为 {value!r}")
    return result


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"缺少{label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"{label}不是合法 JSON: {path}:{exc.lineno}:{exc.colno}: {exc.msg}")
    if not isinstance(value, dict):
        _fail(f"{label}顶层必须是 JSON object: {path}")
    return value


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        _fail(f"缺少 manifest: {path}")
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            _fail(
                f"manifest 第 {line_number} 行不是合法 JSON: "
                f"{exc.msg}（列 {exc.colno}）"
            )
        if not isinstance(record, dict):
            _fail(f"manifest 第 {line_number} 行必须是 JSON object")
        missing = [field for field in MANIFEST_FIELDS if field not in record]
        if missing:
            _fail(
                f"manifest 第 {line_number} 行缺少字段: {', '.join(missing)}"
            )
        for field in MANIFEST_FIELDS:
            if not isinstance(record[field], str) or not record[field].strip():
                _fail(f"manifest 第 {line_number} 行字段 {field} 必须是非空字符串")
        records.append(record)
    if not records:
        _fail(f"manifest 没有有效记录: {path}")
    return records


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if math.isclose(sxx, 0.0, abs_tol=1e-15):
        _fail("线性拟合至少需要两个不同的 target_duty")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    sse = sum(residual * residual for residual in residuals)
    sst = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 if math.isclose(sst, 0.0, abs_tol=1e-15) else 1.0 - sse / sst
    return slope, intercept, r2


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            _fail("二次拟合至少需要三个不同的 target_duty")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][3] for row in range(3)]


def _quadratic_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    sums = [sum(x**power for x in xs) for power in range(5)]
    matrix = [
        [sums[4], sums[3], sums[2]],
        [sums[3], sums[2], sums[1]],
        [sums[2], sums[1], sums[0]],
    ]
    vector = [
        sum(y * x * x for x, y in zip(xs, ys)),
        sum(y * x for x, y in zip(xs, ys)),
        sum(ys),
    ]
    a, b, c = _solve_3x3(matrix, vector)
    return a, b, c


def _analyze_summary(
    manifest_record: dict[str, Any],
    summary: dict[str, Any],
    *,
    summary_path: Path,
    result_dir: Path,
) -> dict[str, Any]:
    raw_points = summary.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        _fail(f"summary 的 points 必须是非空数组: {summary_path}")

    points: list[dict[str, float]] = []
    for index, raw_point in enumerate(raw_points):
        where = f"{summary_path} points[{index}]"
        if not isinstance(raw_point, dict):
            _fail(f"{where} 必须是 JSON object")
        point = {
            "target": _number(raw_point.get("target_duty"), where=f"{where}.target_duty"),
            "drop": _number(
                raw_point.get("victim_throughput_drop_pct"),
                where=f"{where}.victim_throughput_drop_pct",
            ),
            "throughput": _number(
                raw_point.get("victim_iters_per_s_median"),
                where=f"{where}.victim_iters_per_s_median",
            ),
            "p90": _number(
                raw_point.get("victim_iter_ms_p90_median"),
                where=f"{where}.victim_iter_ms_p90_median",
            ),
            "p99": _number(
                raw_point.get("victim_iter_ms_p99_median"),
                where=f"{where}.victim_iter_ms_p99_median",
            ),
        }
        if math.isclose(point["target"], 0.0, abs_tol=1e-12):
            point["actual_busy"] = 0.0
        else:
            point["actual_busy"] = _number(
                raw_point.get("sidecar_busy_wall_ratio_median"),
                where=f"{where}.sidecar_busy_wall_ratio_median",
            )
        points.append(point)

    points.sort(key=lambda point: point["target"])
    targets = [point["target"] for point in points]
    if len(set(targets)) != len(targets):
        _fail(f"summary 含重复 target_duty: {summary_path}")
    if len(points) < 3:
        _fail(f"summary 至少需要 3 个剂量点以完成二次拟合: {summary_path}")
    if not math.isclose(targets[0], 0.0, abs_tol=1e-12):
        _fail(f"summary 缺少 target_duty=0 的 baseline 点: {summary_path}")

    selected = next(
        (
            point
            for point in points
            if math.isclose(point["target"], 0.5, rel_tol=0.0, abs_tol=1e-9)
        ),
        points[-1],
    )
    baseline = points[0]
    highest = points[-1]
    if baseline["p90"] <= 0:
        _fail(f"baseline p90 必须大于 0: {summary_path}")
    if baseline["p99"] <= 0:
        _fail(f"baseline p99 必须大于 0: {summary_path}")

    drops = [point["drop"] for point in points]
    actual_busy = [point["actual_busy"] for point in points]
    slope, intercept, r2 = _linear_fit(actual_busy, drops)
    quadratic_a, quadratic_b, quadratic_c = _quadratic_fit(actual_busy, drops)
    linear_residuals = [
        abs(y - (slope * x + intercept)) for x, y in zip(actual_busy, drops)
    ]
    peak = max(abs(value) for value in drops)
    relative_residual = 0.0 if math.isclose(peak, 0.0) else max(linear_residuals) / peak
    auc = sum(
        (right["actual_busy"] - left["actual_busy"])
        * (left["drop"] + right["drop"])
        / 2.0
        for left, right in zip(points, points[1:])
    )

    summary_path_text: str
    try:
        summary_path_text = str(summary_path.relative_to(result_dir))
    except ValueError:
        summary_path_text = str(summary_path)
    return {
        "summary_path": summary_path_text,
        "inject_kind": manifest_record["inject_kind"],
        "workload": manifest_record["workload"],
        "profile": manifest_record["profile"],
        "pattern": manifest_record["pattern"],
        "selected_target_duty": selected["target"],
        "throughput_drop_pct_at_target_0_5_or_max": selected["drop"],
        "linear_slope": slope,
        "linear_intercept": intercept,
        "linear_r2": r2,
        "quadratic_a": quadratic_a,
        "quadratic_b": quadratic_b,
        "quadratic_c": quadratic_c,
        "linear_max_residual_relative_to_peak": relative_residual,
        "auc_0_to_max_dose": auc,
        "baseline_throughput": _number(
            summary.get("baseline_iters_per_s", baseline["throughput"]),
            where=f"{summary_path}.baseline_iters_per_s",
        ),
        "max_dose_p90_amplification": highest["p90"] / baseline["p90"],
        "max_dose_p99_amplification": highest["p99"] / baseline["p99"],
        "sidecar_actual_busy_at_selected_target": selected["actual_busy"],
    }


def _write_outputs(result_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    json_path = result_dir / JSON_OUTPUT_NAME
    csv_path = result_dir / CSV_OUTPUT_NAME
    analysis = {
        "manifest": MANIFEST_NAME,
        "combination_count": len(rows),
        "fit_x": "sidecar_actual_busy_wall_ratio",
        "fit_y": "victim_throughput_drop_pct",
        "linear_max_residual_relative_to_peak_formula": (
            "max(abs(observed - linear_prediction)) / "
            "max(abs(victim_throughput_drop_pct))"
        ),
        "rows": rows,
    }

    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    try:
        json_tmp.write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        json_tmp.replace(json_path)
        csv_tmp.replace(csv_path)
    finally:
        if json_tmp.exists():
            json_tmp.unlink()
        if csv_tmp.exists():
            csv_tmp.unlink()
    return json_path, csv_path


def main() -> int:
    args = _parse_args()
    result_dir = args.result_dir.expanduser().resolve()
    if not result_dir.is_dir():
        _fail(f"--result-dir 不存在或不是目录: {result_dir}")

    records = _read_manifest(result_dir / MANIFEST_NAME)
    rows = []
    for line_number, record in enumerate(records, start=1):
        path = Path(record["summary_path"]).expanduser()
        summary_path = path if path.is_absolute() else result_dir / path
        if not summary_path.is_file():
            _fail(
                f"manifest 第 {line_number} 行引用的 summary 不存在: {summary_path}"
            )
        summary = _read_json(summary_path, label="summary")
        rows.append(
            _analyze_summary(
                record,
                summary,
                summary_path=summary_path.resolve(),
                result_dir=result_dir,
            )
        )

    json_path, csv_path = _write_outputs(result_dir, rows)
    print(f"atlas_analysis: {json_path}")
    print(f"atlas_rows: {csv_path}")
    print(f"combinations: {len(rows)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
