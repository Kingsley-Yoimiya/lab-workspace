#!/usr/bin/env bash
# 将 Megatron-LM-0.12.3 examples/qwen3/PT_*.sh 就地修补到 OUT_SH：
# 1) /afs-grj → 真实 DATA_ROOT
# 2) 修复 torchrun 后「缺行续接的 | tee」
# 3) 覆盖 TRAIN_ITERS / NPUS_PER_NODE / NNODES / RANK / MASTER_*
#
# 用法:
#   patch_pt_train.sh <src_pt.sh> <out.sh> \
#     DATA_ROOT ITERS NPUS NNODES RANK MASTER_ADDR MASTER_PORT RUN_DIR
set -euo pipefail

SRC="${1:?src}"
OUT="${2:?out}"
DATA_ROOT="${3:?data_root}"
ITERS="${4:?iters}"
NPUS="${5:?npus}"
NNODES="${6:?nnodes}"
RANK="${7:?rank}"
MASTER_ADDR="${8:?master_addr}"
MASTER_PORT="${9:?master_port}"
RUN_DIR="${10:?run_dir}"

cp "$SRC" "$OUT"

# 路径
sed -i "s|/afs-grj/|${DATA_ROOT}/|g" "$OUT"
sed -i "s|TOKENIZER_MODEL=/afs-grj/Qwen3-32B|TOKENIZER_MODEL=${DATA_ROOT}/Qwen3-32B|g" "$OUT"
sed -i "s|VOCAB_FILE=/afs-grj/Qwen3-32B/vocab.json|VOCAB_FILE=${DATA_ROOT}/Qwen3-32B/vocab.json|g" "$OUT"
sed -i "s|DATA_PATH=\"/afs-grj/dataset/data_text_document\"|DATA_PATH=\"${DATA_ROOT}/dataset/data_text_document\"|g" "$OUT"

# 迭代与卡数
sed -i "s/^TRAIN_ITERS=.*/TRAIN_ITERS=${ITERS}/" "$OUT"
sed -i "s/^NPUS_PER_NODE=.*/NPUS_PER_NODE=${NPUS}/" "$OUT"
sed -i "s/^PROC_PER_NODE=.*/PROC_PER_NODE=${NPUS}/" "$OUT"
sed -i "s/^GPUS_PER_NODE=.*/GPUS_PER_NODE=${NPUS}/" "$OUT"

# 分布式（PT 用 NNODES/RANK；部分旧脚本用 WORLD_SIZE 当 nnodes）
if grep -qE '^NNODES=' "$OUT"; then
  sed -i "s/^NNODES=.*/NNODES=${NNODES}/" "$OUT"
else
  # 在文件前部注入，避免被后面的探测逻辑覆盖：改用环境变量优先
  sed -i "1a export NNODES=${NNODES}" "$OUT"
fi
sed -i "s/^WORLD_SIZE=.*/WORLD_SIZE=${NNODES}/" "$OUT" || true
# 强制 RANK / MASTER（覆盖硬编码 127.0.0.1）
if grep -qE '^RANK=' "$OUT"; then
  sed -i "s/^RANK=.*/RANK=${RANK}/" "$OUT"
else
  sed -i "1a export RANK=${RANK}" "$OUT"
fi
# 把 MASTER_ADDR=... 默认赋值改成我们的值（保留 ${MASTER_ADDR:-} 形式时在顶部 export）
sed -i "1a export MASTER_ADDR=${MASTER_ADDR}" "$OUT"
sed -i "1a export MASTER_PORT=${MASTER_PORT}" "$OUT"
sed -i "1a export WORLD_SIZE=${NNODES}" "$OUT"
sed -i "1a export NNODES=${NNODES}" "$OUT"
sed -i "1a export RANK=${RANK}" "$OUT"
sed -i "1a export NPUS_PER_NODE=${NPUS}" "$OUT"

# 日志 / ckpt 目录
sed -i "s|^LOG_DIR=.*|LOG_DIR=\"${RUN_DIR}/\"|" "$OUT" || true
sed -i "s|^tensorboard_dir=.*|tensorboard_dir=\"${RUN_DIR}/tb\"|" "$OUT" || true
sed -i "s|^TENSORBOARD_DIR=.*|TENSORBOARD_DIR=\"${RUN_DIR}/tb\"|" "$OUT" || true
sed -i "s|^CKPT_SAVE_DIR=.*|CKPT_SAVE_DIR=\"${RUN_DIR}/ckpt\"|" "$OUT" || true

# 修复损坏的 | tee：去掉 torchrun 参数列表中间的「裸 | tee」行，
# 并在 --distributed-backend 行后保证管道正确。
python3 - "$OUT" <<'PY'
import re, sys
path = sys.argv[1]
text = open(path, encoding="utf-8", errors="replace").read()
# 常见坏法：某行以 --xxx 结尾无反斜杠，下一行是 "    | tee ..."
# 或 "--distributed-backend nccl 2>&1 \" 后跟若干参数，再裸 "| tee"
# 策略：把独立的 "| tee ..." 行并入前一个非空命令行，并去掉错误的行续接。
lines = text.splitlines(True)
out = []
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.lstrip()
    if stripped.startswith("| tee"):
        # 回溯找上一条非空、非纯注释行
        j = len(out) - 1
        while j >= 0 and out[j].strip() == "":
            j -= 1
        if j >= 0:
            prev = out[j].rstrip("\n")
            # 去掉末尾孤立反斜杠（若有），改为管道
            if prev.rstrip().endswith("\\"):
                prev = prev.rstrip()[:-1].rstrip()
            # 若上一行已是 2>&1，直接接 | tee；否则补 2>&1 |
            if prev.rstrip().endswith("2>&1"):
                out[j] = prev + " " + stripped
            else:
                out[j] = prev + " 2>&1 " + stripped
            i += 1
            continue
    out.append(line)
    i += 1
open(path, "w", encoding="utf-8").writelines(out)
print(f"patched tee → {path}")
PY

chmod +x "$OUT"
echo "PATCH_PT_OK → $OUT"
