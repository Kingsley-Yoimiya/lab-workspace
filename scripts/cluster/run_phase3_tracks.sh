#!/usr/bin/env bash
# Phase3：indep / real_sync(dp_allreduce) / 真 TP2+TP4 + HCCL 基线
# 在 npu-dev-1 上执行；镜像内需有 torchrun。
set -uo pipefail
HOST_CS="${HOST_CS:-$HOME/CARD_SCREEN}"
HOST_SCRIPTS="${HOST_SCRIPTS:-$HOME/lab-workspace/scripts/cluster}"
IMG="${IMG:-quay.io/ascend/vllm-ascend:v0.19.1rc1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)-phase3-tracks}"
OUT_HOST="$HOST_CS/results/$RUN_ID"
mkdir -p "$OUT_HOST"

DEVICES=()
for i in $(seq 0 15); do DEVICES+=(--device="/dev/davinci${i}"); done
DEVICES+=(--device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc)

sudo -n docker run --rm --name "phase3-${RUN_ID}" --network=host --ipc=host \
  "${DEVICES[@]}" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$HOST_CS":/workspace/CARD_SCREEN \
  -v "$HOST_SCRIPTS":/workspace/inject \
  -w /workspace/inject \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  -e PYTHONUNBUFFERED=1 \
  "$IMG" bash -lc "
set -uo pipefail
OUT=/workspace/CARD_SCREEN/results/${RUN_ID}
mkdir -p \"\$OUT/hccl\" \"\$OUT/indep\" \"\$OUT/real_sync\" \"\$OUT/tp2\" \"\$OUT/tp4\"

echo '=== HCCL baseline (AG/RS) ==='
torchrun --nproc_per_node=16 hccl_torch_bench.py \
  --ops all_gather,reduce_scatter,all_reduce \
  --sizes 1M,16M,64M \
  --warmup 3 --iters 10 \
  --out \"\$OUT/hccl/hccl.jsonl\" || echo HCCL_FAIL

echo '=== independent 16 (arch=block, NO HCCL) ==='
python3 - <<'PY' || echo INDEP_FAIL
import os, subprocess, sys
out = '/workspace/CARD_SCREEN/results/${RUN_ID}/indep'
procs = []
base_vis = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15')
for local in range(16):
    env = os.environ.copy()
    env['LOCAL_RANK'] = str(local)
    env['RANK'] = str(local)
    env['LOCAL_WORLD_SIZE'] = '16'
    env['ASCEND_RT_VISIBLE_DEVICES'] = base_vis
    procs.append(subprocess.Popen(
        [sys.executable, 'virtual_sync_bench_npu.py',
         '--mode', 'independent', '--arch', 'block',
         '--iters', '40', '--warmup', '5',
         '--hidden', '4096', '--seq', '1024', '--batch', '1', '--layers', '2',
         '--out-dir', out, '--tag', 'indep16'],
        env=env))
rc = 0
for p in procs:
    rc = max(rc, p.wait())
print('indep_rc', rc)
raise SystemExit(rc)
PY

echo '=== real_sync 16 = dp_allreduce (NOT TP) ==='
torchrun --nproc_per_node=16 virtual_sync_bench_npu.py \
  --mode real_sync --arch block \
  --iters 40 --warmup 5 --hidden 4096 --seq 1024 --batch 1 --layers 2 \
  --out-dir \"\$OUT/real_sync\" --tag 'dp_allreduce_16' || echo REAL_SYNC_FAIL

echo '=== true TP2 ==='
torchrun --nproc_per_node=2 tp_block_bench_npu.py \
  --tp 2 --iters 40 --warmup 5 --out-dir \"\$OUT/tp2\" || echo TP2_FAIL

echo '=== true TP4 ==='
torchrun --nproc_per_node=4 tp_block_bench_npu.py \
  --tp 4 --iters 40 --warmup 5 --out-dir \"\$OUT/tp4\" || echo TP4_FAIL

echo PHASE3_DONE
"
echo "$OUT_HOST"
