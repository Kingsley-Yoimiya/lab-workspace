#!/usr/bin/env python3
"""最小真 Tensor Parallel Block 微基准（Ascend + HCCL）。

与 mfu_train_bench / virtual_sync real_sync 的区别：
  - 那些是 DP + grad AllReduce，不是 TP。
  - 本脚本：ColumnParallel Linear（local matmul + all_gather）
            + RowParallel Linear（local matmul + reduce_scatter 或 all_reduce）。

主指标说明：
  - 本文件每 rank 记录 local_step_ms（及可选 compute/collective 分段）。
  - global_step_ms[i] = max over ranks of local_step_ms[r,i]，由下游汇总；
    禁止只用被注入 rank 的 local ms 当作训练损失。

用法:
  torchrun --nproc_per_node=4 tp_block_bench_npu.py --tp 4 --iters 40 --out-dir /tmp/tp4
  # --tp 应等于 world size（由 torchrun 隐含）
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path


def _write(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tp", type=int, required=True, help="TP 度；应等于 WORLD_SIZE")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--ffn", type=int, default=0, help="0 → 4*hidden；须被 tp 整除")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument(
        "--row-mode",
        choices=["reduce_scatter", "all_reduce"],
        default="reduce_scatter",
        help="RowParallel 聚合方式",
    )
    ap.add_argument("--profile-segments", action="store_true", help="分段计时 compute/collective")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_npu  # noqa: F401

    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local)
    device = torch.device(f"npu:{local}")
    hostname = socket.gethostname()

    if world != args.tp:
        raise SystemExit(f"--tp={args.tp} must equal world_size={world}")

    H = args.hidden
    FFN = args.ffn if args.ffn > 0 else 4 * H
    if FFN % world != 0 or H % world != 0:
        raise SystemExit(f"hidden={H} and ffn={FFN} must be divisible by tp={world}")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    B, S = args.batch, args.seq
    class ColumnParallelLinear(nn.Module):
        """输出按列切分：local Y = X @ W_local；再 all_gather 拼满。"""

        def __init__(self, in_f: int, out_f: int, gather_output: bool = True):
            super().__init__()
            assert out_f % world == 0
            self.out_local = out_f // world
            self.gather_output = gather_output
            self.weight = nn.Parameter(
                torch.empty(in_f, self.out_local, device=device, dtype=dtype)
            )
            nn.init.normal_(self.weight, std=0.02)

        def forward(self, x: torch.Tensor, coll_ms: list | None = None):
            # x: [*, in_f]
            y_local = x @ self.weight  # [*, out_local]
            if not self.gather_output:
                return y_local
            t0 = time.time()
            parts = [torch.empty_like(y_local) for _ in range(world)]
            dist.all_gather(parts, y_local.contiguous())
            if coll_ms is not None:
                torch.npu.synchronize()
                coll_ms.append((time.time() - t0) * 1000.0)
            return torch.cat(parts, dim=-1)

    class RowParallelLinear(nn.Module):
        """输入按列切分：local partial = X_local @ W_local；再 reduce_scatter / all_reduce。"""

        def __init__(self, in_f: int, out_f: int, mode: str = "reduce_scatter"):
            super().__init__()
            assert in_f % world == 0
            self.in_local = in_f // world
            self.mode = mode
            self.weight = nn.Parameter(
                torch.empty(self.in_local, out_f, device=device, dtype=dtype)
            )
            nn.init.normal_(self.weight, std=0.02)

        def forward(self, x: torch.Tensor, coll_ms: list | None = None):
            # x expected full [*, in_f] → take local shard；或已是 local
            if x.shape[-1] == self.in_local * world:
                x_local = x.narrow(-1, rank * self.in_local, self.in_local)
            else:
                x_local = x
            partial = x_local @ self.weight  # [*, out_f]
            t0 = time.time()
            if self.mode == "all_reduce":
                dist.all_reduce(partial)
                out = partial
            else:
                # reduce_scatter：沿 batch*seq 维切分输出；为保持形状，先 flatten tokens
                flat = partial.reshape(-1, partial.shape[-1]).contiguous()
                # 若 token 数不能被 world 整除，pad
                n = flat.shape[0]
                pad = (world - n % world) % world
                if pad:
                    flat = torch.cat([flat, flat.new_zeros(pad, flat.shape[-1])], dim=0)
                chunk = flat.shape[0] // world
                out_flat = torch.empty(chunk, flat.shape[-1], device=device, dtype=dtype)
                dist.reduce_scatter_tensor(out_flat, flat)
                # 为简化下游，再 all_gather 回全量（仍算 TP 路径上的 RS+AG；或仅测 RS）
                # 这里保持与全量 token 对齐：all_gather 拼回
                parts = [torch.empty_like(out_flat) for _ in range(world)]
                dist.all_gather(parts, out_flat)
                gathered = torch.cat(parts, dim=0)[:n]
                out = gathered.view_as(partial)
            if coll_ms is not None:
                torch.npu.synchronize()
                coll_ms.append((time.time() - t0) * 1000.0)
            return out

    class TPBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(H).to(device=device, dtype=dtype)
            self.fc1 = ColumnParallelLinear(H, FFN, gather_output=True)
            self.fc2 = RowParallelLinear(FFN, H, mode=args.row_mode)
            self.ln2 = nn.LayerNorm(H).to(device=device, dtype=dtype)

        def forward(self, x, coll_ms=None):
            h = self.ln1(x)
            h = F.gelu(self.fc1(h, coll_ms=coll_ms))
            h = self.fc2(h, coll_ms=coll_ms)
            x = x + h
            return self.ln2(x)

    model = TPBlock()
    # 仅优化本地参数
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)

    def one_step(profile: bool = False) -> dict:
        coll_ms: list[float] = [] if profile else None  # type: ignore
        x = torch.randn(B, S, H, device=device, dtype=dtype)
        opt.zero_grad(set_to_none=True)
        t_compute0 = time.time()
        y = model(x, coll_ms=coll_ms)
        loss = y.float().pow(2).mean()
        loss.backward()
        # grad all-reduce on local params is NOT DP here — grads already local to shards
        opt.step()
        torch.npu.synchronize()
        t_compute1 = time.time()
        out = {"compute_wall_ms": (t_compute1 - t_compute0) * 1000.0}
        if profile and coll_ms is not None:
            out["collective_ms"] = sum(coll_ms)
            out["collective_ms_parts"] = [round(v, 3) for v in coll_ms]
        return out

    out_path = Path(args.out_dir) / f"step_times_rank{rank:03d}.jsonl"
    for _ in range(args.warmup):
        one_step(profile=False)
    dist.barrier()

    t_start = time.time()
    for i in range(1, args.iters + 1):
        t0 = time.time()
        seg = one_step(profile=args.profile_segments)
        t1 = time.time()
        local_ms = (t1 - t0) * 1000.0
        rec = {
            "iter": i,
            "t0": t0,
            "t1": t1,
            "ms": round(local_ms, 3),
            "local_step_ms": round(local_ms, 3),
            "rank": rank,
            "global_rank": rank,
            "local": local,
            "tp": world,
            "world_size": world,
            "row_mode": args.row_mode,
            "arch": "tp_block",
            "hostname": hostname,
            "hidden": H,
            "ffn": FFN,
            "seq": S,
            "batch": B,
            "tag": args.tag,
            "t_start": t_start,
            "note_global_step": "global_step_ms = max_r local_step_ms[r]; aggregate downstream",
        }
        if args.profile_segments:
            rec["compute_wall_ms"] = round(seg.get("compute_wall_ms", 0.0), 3)
            rec["collective_ms"] = round(seg.get("collective_ms", 0.0), 3)
            rec["collective_ms_parts"] = seg.get("collective_ms_parts")
        _write(out_path, rec)

    Path(args.out_dir).joinpath(f"done_rank{rank:03d}.txt").write_text(
        f"OK tp={world} rank={rank} row_mode={args.row_mode}\n", encoding="utf-8"
    )
    print(
        f"TP_DONE tp={world} rank={rank} iters={args.iters} row_mode={args.row_mode} → {out_path}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
