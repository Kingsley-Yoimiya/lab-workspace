"""每卡 step time 采集钩子（Dense FailSlow 实验一主指标）。

通过 site-packages 的 .pth 仅 import 本模块；本模块不在 import 时拉 torch/megatron，
而是轮询等待 megatron.training.training 进入 sys.modules 后再 monkeypatch train_step。

落盘：RUN_DIR/step_times_rank{global_rank}.jsonl
  {"iter": N, "ms": wall_ms, "rank": R, "ts": unix}
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

_INSTALLED = False
_LOCK = threading.Lock()


def _enabled() -> bool:
    if os.environ.get("FAILSLOW_STEP_LOG", "").strip().lower() in ("1", "true", "yes"):
        return True
    p = os.environ.get("PROBING", "0").strip().lower()
    return p not in ("", "0", "false", "no")


def _out_path(rank: int) -> Path:
    run_dir = os.environ.get("RUN_DIR") or os.environ.get("LOG_DIR") or "."
    return Path(run_dir) / f"step_times_rank{rank}.jsonl"


def _global_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    try:
        node = int(os.environ.get("NODE_RANK") or os.environ.get("RANK") or 0)
        local = int(os.environ.get("LOCAL_RANK") or 0)
        if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
            npu = max(
                1,
                len([x for x in os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",") if x != ""]),
            )
        else:
            npu = int(os.environ.get("NPUS_PER_NODE") or 16)
        return node * npu + local
    except Exception:
        return int(os.environ.get("LOCAL_RANK") or 0)


def install() -> bool:
    """若 megatron.training.training 已在 sys.modules，则 patch train_step。"""
    global _INSTALLED
    with _LOCK:
        if _INSTALLED or not _enabled():
            return _INSTALLED
        mod = sys.modules.get("megatron.training.training")
        if mod is None:
            return False
        if getattr(mod, "_failslow_step_timer_wrapped", False):
            _INSTALLED = True
            return True
        orig = getattr(mod, "train_step", None)
        if orig is None:
            return False

        state = {"iter": 0}

        def _maybe_delay(iter_idx: int, rank: int) -> bool:
            """实验三：按 PP stage 或 DELAY_RANKS 间歇 sleep。DELAY_INJECT=1 开启。"""
            if os.environ.get("DELAY_INJECT", "").strip().lower() not in ("1", "true", "yes"):
                return False
            every = max(1, int(os.environ.get("DELAY_EVERY", "5")))
            burst = max(1, int(os.environ.get("DELAY_BURST", "2")))
            delay_ms = float(os.environ.get("DELAY_MS", "500"))
            phase = (iter_idx - 1) % every
            if phase >= burst:
                return False
            ranks_env = os.environ.get("DELAY_RANKS", "").strip()
            if ranks_env:
                want = {int(x) for x in ranks_env.split(",") if x.strip().isdigit()}
                if rank not in want:
                    return False
            else:
                target = int(os.environ.get("DELAY_STAGE", "1"))
                pp = max(1, int(os.environ.get("PP_SIZE") or os.environ.get("PP") or 2))
                world = int(os.environ.get("WORLD_SIZE_NPUS") or os.environ.get("WORLD_NPU") or 0)
                if world <= 0:
                    nnodes = int(os.environ.get("NNODES") or os.environ.get("WORLD_SIZE") or 1)
                    npu = int(os.environ.get("NPUS_PER_NODE") or 16)
                    world = nnodes * npu
                ranks_per_stage = max(1, world // pp)
                if rank // ranks_per_stage != target:
                    return False
            time.sleep(delay_ms / 1000.0)
            return True

        def wrapped(*args, **kwargs):
            state["iter"] += 1
            rank = _global_rank()
            # delay 必须落在计时窗内，否则 injected ranks 的 ms 反而偏小（旧 bug）
            t0 = time.perf_counter()
            delayed = _maybe_delay(state["iter"], rank)
            out = orig(*args, **kwargs)
            ms = (time.perf_counter() - t0) * 1000.0
            pp = max(1, int(os.environ.get("PP_SIZE") or os.environ.get("PP") or 2))
            nnodes = int(os.environ.get("NNODES") or os.environ.get("WORLD_SIZE") or 1)
            npu = int(os.environ.get("NPUS_PER_NODE") or 16)
            world = int(os.environ.get("WORLD_SIZE_NPUS") or (nnodes * npu))
            ranks_per_stage = max(1, world // pp)
            rec = {
                "iter": state["iter"],
                "ms": round(ms, 3),
                "rank": rank,
                "ts": time.time(),
                "delayed": delayed,
                "pp_stage": rank // ranks_per_stage,
            }
            try:
                path = _out_path(rank)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as exc:
                if state["iter"] <= 2:
                    print(f"[failslow_step_timer] write fail: {exc}", flush=True)
            return out

        mod.train_step = wrapped  # type: ignore[assignment]
        mod._failslow_step_timer_wrapped = True
        _INSTALLED = True
        print(
            f"[failslow_step_timer] installed train_step hook → {_out_path(_global_rank())}",
            flush=True,
        )
        return True


def _watch() -> None:
    if not _enabled():
        return
    # 最多等 30 分钟（覆盖慢启动）
    for _ in range(3600):
        if install():
            return
        time.sleep(0.5)


def start_watcher() -> None:
    if not _enabled():
        return
    t = threading.Thread(target=_watch, name="failslow-step-timer", daemon=True)
    t.start()


# 仅启动 watcher，绝不在 import 时 import torch/megatron
start_watcher()
