#!/usr/bin/env python3
"""NCCL/MCCL torch bench 纯计算：带宽公式与 timing/JSONL 字段（无 GPU 依赖）。

供 nccl_torch_bench.py 与本地单测共用，避免把汇总 collective 算进目标计时。
"""
from __future__ import annotations

from statistics import mean
from typing import Any, Iterable, Sequence


def parse_bytes_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip().upper()
        if not part:
            continue
        if part.endswith("K"):
            out.append(int(float(part[:-1]) * 1024))
        elif part.endswith("M"):
            out.append(int(float(part[:-1]) * 1024**2))
        elif part.endswith("G"):
            out.append(int(float(part[:-1]) * 1024**3))
        else:
            out.append(int(part))
    return out


def bus_bw_factor(op: str, world_size: int) -> float:
    """NCCL-tests 同构 bus_bw 因子：bus_bw = alg_bw * factor。"""
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if op == "all_reduce":
        return 2.0 * (world_size - 1) / world_size
    if op in ("all_gather", "reduce_scatter"):
        return (world_size - 1) / world_size
    # broadcast / 其它：按算法字节口径，不做多跳折算
    return 1.0


def alg_bw_gbps(data_bytes: int, time_s: float) -> float:
    if time_s <= 0:
        return float("inf") if data_bytes > 0 else 0.0
    return data_bytes / time_s / 1e9


def bus_bw_gbps(op: str, world_size: int, data_bytes: int, time_s: float) -> float:
    return alg_bw_gbps(data_bytes, time_s) * bus_bw_factor(op, world_size)


def mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def elementwise_max(rows: Sequence[Sequence[float]]) -> list[float]:
    """对多 rank 的同长度 per-iter 时间取逐轮 max。"""
    if not rows:
        return []
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise ValueError("all rank timing rows must have the same length")
    return [max(col) for col in zip(*rows)]


def build_bench_record(
    *,
    op: str,
    world_size: int,
    rank: int,
    host: str,
    local_rank: int,
    nbytes: int,
    dtype: str,
    iters_s_local: Sequence[float],
    iters_s_global_max: Sequence[float],
    backend: str = "nccl",
    record: str = "nccl_bench",
    timing_version: str = "w0.1",
) -> dict[str, Any]:
    """构造 JSONL 记录：旧字段兼容 + 明确 local / global_max 口径。

    兼容约定：
    - avg_s / alg_bw_GBps / bus_bw_GBps 继续存在；
    - 自 W0.1 起，旧三字段对齐 **global_max**（集体完成由最慢 rank 决定）；
    - 同时写出 *_local / *_global_max 与每轮原始延迟，避免歧义。
    """
    avg_local = mean_or_zero(iters_s_local)
    avg_global = mean_or_zero(iters_s_global_max)
    alg_local = alg_bw_gbps(nbytes, avg_local)
    bus_local = bus_bw_gbps(op, world_size, nbytes, avg_local)
    alg_global = alg_bw_gbps(nbytes, avg_global)
    bus_global = bus_bw_gbps(op, world_size, nbytes, avg_global)
    return {
        "record": record,
        "backend": backend,
        "op": op,
        "world_size": world_size,
        "rank": rank,
        "host": host,
        "local_rank": local_rank,
        "nbytes": nbytes,
        "dtype": dtype,
        "timing_version": timing_version,
        "bw_basis": "global_max",
        # 兼容旧字段：吞吐按全局最慢完成时间
        "avg_s": avg_global,
        "alg_bw_GBps": alg_global,
        "bus_bw_GBps": bus_global,
        # 明确口径
        "avg_s_local": avg_local,
        "alg_bw_GBps_local": alg_local,
        "bus_bw_GBps_local": bus_local,
        "avg_s_global_max": avg_global,
        "alg_bw_GBps_global_max": alg_global,
        "bus_bw_GBps_global_max": bus_global,
        "iters_s_local": [float(x) for x in iters_s_local],
        "iters_s_global_max": [float(x) for x in iters_s_global_max],
        "n_iters": len(iters_s_local),
    }


def validate_bench_record(rec: dict[str, Any]) -> list[str]:
    """静态校验 JSONL schema；返回错误列表（空=通过）。"""
    errors: list[str] = []
    required = [
        "record",
        "op",
        "world_size",
        "rank",
        "nbytes",
        "avg_s",
        "alg_bw_GBps",
        "bus_bw_GBps",
        "avg_s_local",
        "avg_s_global_max",
        "iters_s_local",
        "iters_s_global_max",
        "bw_basis",
        "timing_version",
    ]
    for k in required:
        if k not in rec:
            errors.append(f"missing field: {k}")
    if "iters_s_local" in rec and "iters_s_global_max" in rec:
        a = rec["iters_s_local"]
        b = rec["iters_s_global_max"]
        if not isinstance(a, list) or not isinstance(b, list):
            errors.append("iters_s_* must be lists")
        elif len(a) != len(b):
            errors.append("iters_s_local and iters_s_global_max length mismatch")
        elif a and b:
            for i, (la, ga) in enumerate(zip(a, b)):
                if ga + 1e-15 < la:
                    errors.append(f"iters_s_global_max[{i}] < local")
    if rec.get("bw_basis") == "global_max":
        for old, new in (
            ("avg_s", "avg_s_global_max"),
            ("alg_bw_GBps", "alg_bw_GBps_global_max"),
            ("bus_bw_GBps", "bus_bw_GBps_global_max"),
        ):
            if old in rec and new in rec and abs(float(rec[old]) - float(rec[new])) > 1e-12:
                errors.append(f"{old} must equal {new} when bw_basis=global_max")
    return errors


def summarize_case_print(
    op: str,
    world_size: int,
    nbytes: int,
    avg_s_local: float,
    avg_s_global_max: float,
    alg_local: float,
    bus_local: float,
    alg_global: float,
    bus_global: float,
) -> str:
    return (
        f"op={op} world={world_size} size={nbytes} "
        f"avg_ms_local={avg_s_local * 1e3:.3f} avg_ms_global_max={avg_s_global_max * 1e3:.3f} "
        f"alg_local={alg_local:.2f} bus_local={bus_local:.2f} "
        f"alg_global_max={alg_global:.2f} bus_global_max={bus_global:.2f} GB/s"
    )


def assert_timing_contract_notes() -> Iterable[str]:
    """文档化 W0.1 计时契约（供测试断言字符串稳定）。"""
    return (
        "barrier_before_timed_region",
        "warmup_then_barrier",
        "timed_region_collective_plus_device_sync_only",
        "stop_clock_before_result_reduction",
        "global_bw_from_max_rank_time",
        "keep_per_iter_raw_times",
    )
