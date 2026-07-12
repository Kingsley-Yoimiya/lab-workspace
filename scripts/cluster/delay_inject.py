#!/usr/bin/env python3
"""轻量 PP-stage 延迟注入：按 GLOBAL_RANK // ranks_per_stage 映射 stage，间歇 sleep。

环境变量:
  DELAY_INJECT=1
  DELAY_STAGE=1          # 被注入的 PP stage（0-based）
  DELAY_MS=50            # 每次注入额外延迟
  DELAY_EVERY=20         # 每 N step 注入一次
  DELAY_BURST=3          # 连续注入步数
  PP_SIZE=4
  WORLD_SIZE / 或 GPUS
"""
from __future__ import annotations

import os
import threading
import time

_INSTALLED = False


def _enabled() -> bool:
    return os.environ.get("DELAY_INJECT", "").strip().lower() in ("1", "true", "yes")


def _stage_of(rank: int) -> int:
    pp = max(1, int(os.environ.get("PP_SIZE", "4")))
    world = int(os.environ.get("WORLD_SIZE") or os.environ.get("GPUS_PER_NODE") or 8)
    # Megatron: rank 按 TP 再 PP；简化：假定 TP=world/pp，同一 stage 连续 rank 块
    ranks_per_stage = max(1, world // pp)
    return rank // ranks_per_stage


def install_on_step(step_fn):
    """包装一个 step 可调用对象。"""
    every = max(1, int(os.environ.get("DELAY_EVERY", "20")))
    burst = max(1, int(os.environ.get("DELAY_BURST", "3")))
    delay_ms = float(os.environ.get("DELAY_MS", "50"))
    target = int(os.environ.get("DELAY_STAGE", "1"))
    rank = int(os.environ.get("GLOBAL_RANK") or os.environ.get("RANK") or 0)
    state = {"i": 0}

    def wrapped(*a, **k):
        state["i"] += 1
        i = state["i"]
        # 注入窗口：every 周期内前 burst 步
        in_burst = (i % every) < burst and (i % every) > 0 or (i % every) == 0 and burst > 0
        # 更清晰：周期起点后的 burst 步
        phase = ((i - 1) % every)
        do_delay = _enabled() and _stage_of(rank) == target and phase < burst
        if do_delay:
            time.sleep(delay_ms / 1000.0)
        return step_fn(*a, **k)

    return wrapped


def maybe_sleep_for_iter(iter_idx: int) -> bool:
    """供 virtual_sync_bench / 独立循环直接调用。返回是否注入。"""
    if not _enabled():
        return False
    every = max(1, int(os.environ.get("DELAY_EVERY", "20")))
    burst = max(1, int(os.environ.get("DELAY_BURST", "3")))
    delay_ms = float(os.environ.get("DELAY_MS", "50"))
    target = int(os.environ.get("DELAY_STAGE", "1"))
    rank = int(os.environ.get("GLOBAL_RANK") or os.environ.get("RANK") or 0)
    phase = (iter_idx - 1) % every
    if _stage_of(rank) == target and phase < burst:
        time.sleep(delay_ms / 1000.0)
        return True
    return False
