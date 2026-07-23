#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FIRE="$ROOT/scripts/cluster/fire_nccl_scale_muxi_jump.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

bash -n "$FIRE"

# 多 case 参数、进度信号和严格 case 数校验必须存在。
grep -Fq 'OPS="${OPS:-all_reduce}"' "$FIRE"
grep -Fq 'SIZES="${SIZES:-256M}"' "$FIRE"
grep -Fq 'CASES=\$c LOG_BYTES=\$b LOG_MTIME=\$m ERRORS=\$e' "$FIRE"
grep -Fq 'EXPECTED_OPS=' "$FIRE"
grep -Fq 'EXPECTED_SIZES=' "$FIRE"
grep -Fq 'expected_case_count' "$FIRE"
grep -Fq 'NCCL_IB_HCA_CFG="${NCCL_IB_HCA:-xscale_0,xscale_1,xscale_2,xscale_3}"' "$FIRE"
grep -Fq "export NCCL_IB_HCA='\${NCCL_IB_HCA_CFG}'" "$FIRE"
grep -Fq "export MCCL_IB_HCA='\${MCCL_IB_HCA_CFG}'" "$FIRE"
grep -Fq 'HCA_EFFECTIVE=$MCCL_IB_HCA_CFG' "$FIRE"
grep -Fq 'UNEXPECTED_HCA=\$h' "$FIRE"
grep -Fq 'nccl_ib_hca: $NCCL_IB_HCA_CFG' "$FIRE"
grep -Fq 'rank_mapping_file: rank_mapping.csv' "$FIRE"
grep -Fq 'EXPECTED_HOST_ORDER=' "$FIRE"
grep -Fq -- '--master_addr=${RENDEZVOUS_POD}.${JOB}' "$FIRE"
grep -Fq 'kill -TERM -- -\$pid' "$FIRE"
grep -Fq 'kill -KILL -- -\$pid' "$FIRE"
grep -Fq 'MCCL_ALGO_CFG="${MCCL_ALGO:-}"' "$FIRE"
grep -Fq "export MCCL_ALGO='\${MCCL_ALGO_CFG}'" "$FIRE"
grep -Fq "export MCCL_PROTO='\${MCCL_PROTO_CFG}'" "$FIRE"
grep -Fq "export MCCL_MIN_NCHANNELS='\${MCCL_MIN_NCHANNELS_CFG}'" "$FIRE"
grep -Fq 'software_control_evidence.log' "$FIRE"
grep -Fq 'CUDA_VISIBLE_DEVICES_CFG="${CUDA_VISIBLE_DEVICES:-}"' "$FIRE"
grep -Fq "export CUDA_VISIBLE_DEVICES='\${CUDA_VISIBLE_DEVICES_CFG}'" "$FIRE"
grep -Fq 'gpu_mapping_evidence.log' "$FIRE"
if grep -Eq 'local (rank|idx)="\\$1"[^[:cntrl:]]*pod="\\$\\{(pods|round_pods)\\[\\$(rank|idx)\\]\\}"' "$FIRE"; then
  echo "dependent local assignment is forbidden" >&2
  exit 1
fi

# dry-run：node_rank block permutation 必须覆盖同一16 pod集合，local_rank不变。
map_out="$(
  RUN_ID=mapping-dry-run MASTER_PORT=29999 BUNDLE_DIR=/tmp/unused JOB=test-job \
  WORLD=64 NNODES=16 NPROC_PER_NODE=4 POD_ORDER_DRY_RUN=1 \
  POD_ORDER=15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
  bash "$FIRE"
)"
grep -Fq 'POD_ORDER_DRY_RUN nnodes=16 nproc=4 order=15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14' <<<"$map_out"
grep -Fq 'rendezvous=test-job-worker-14' <<<"$map_out"
printf '%s\n' "$map_out" >"$tmp/mapping.out"
python3 - "$tmp/mapping.out" <<'PY'
import csv, io, sys
lines=[x.removeprefix("MAPPING ") for x in open(sys.argv[1]) if x.startswith("MAPPING ")]
rows=list(csv.DictReader(io.StringIO("".join(lines))))
assert len(rows)==64
assert sorted(int(x["global_rank"]) for x in rows)==list(range(64))
assert sorted(set(x["pod"] for x in rows))==sorted(
    ["test-job-master-0"]+[f"test-job-worker-{i}" for i in range(15)]
)
assert rows[0]["node_rank"]=="0" and rows[0]["pod"]=="test-job-worker-14"
for node_rank in range(16):
    local=[int(x["local_rank"]) for x in rows if int(x["node_rank"])==node_rank]
    assert local==[0,1,2,3]
PY

if RUN_ID=mapping-invalid MASTER_PORT=29998 BUNDLE_DIR=/tmp/unused JOB=test-job \
  WORLD=64 NNODES=16 NPROC_PER_NODE=4 POD_ORDER_DRY_RUN=1 \
  POD_ORDER=0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
  bash "$FIRE" >/dev/null 2>&1; then
  echo "duplicate POD_ORDER should fail" >&2
  exit 1
fi

# 抽出内嵌纯 Python validator，用 2 rank × 4 case 合成数据验证。
awk '
  /cat >"\$WORK\/validate.py" <<'\''PY'\''/ {copy=1; next}
  copy && /^PY$/ {exit}
  copy {print}
' "$FIRE" >"$tmp/validate.py"

python3 - "$tmp" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
(root / "raw").mkdir()
ops = ["all_reduce", "broadcast"]
sizes = [4096, 1048576]
for rank in range(2):
    rows = []
    for size in sizes:
        for op in ops:
            vals = [0.001, 0.002, 0.003]
            rows.append({
                "op": op, "world_size": 2, "rank": rank,
                "host": f"host{rank}", "local_rank": 0,
                "physical_gpu": 0,
                "nbytes": size, "timing_version": "w0.1",
                "bw_basis": "global_max", "n_iters": 3,
                "iters_s_local": vals, "iters_s_global_max": vals,
                "avg_s_global_max": sum(vals) / len(vals),
                "alg_bw_GBps_global_max": 1.0,
                "bus_bw_GBps_global_max": 2.0,
            })
    with open(root / f"scale_2.rank{rank}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
PY

AFS_OUT="$tmp" EXPECTED_WORLD=2 EXPECTED_NNODES=2 EXPECTED_NPROC=1 \
  EXPECTED_HOST_ORDER=host0,host1 EXPECTED_VISIBLE_DEVICES=0 \
  EXPECTED_OPS=all_reduce,broadcast EXPECTED_SIZES=4K,1M EXPECTED_ITERS=3 \
  python3 "$tmp/validate.py"

python3 - "$tmp/validation_summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
assert s["valid"]
assert s["rank_files"] == 2
assert s["records"] == 8
assert s["expected_case_count"] == 4
assert len(s["cases"]) == 4
PY

[[ "$(wc -l <"$tmp/raw/scale_2.jsonl")" -eq 8 ]]

# 单 case 继续保留旧版 top-level 汇总字段，避免破坏既有调用方。
mkdir -p "$tmp/single/raw"
python3 - "$tmp" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for rank in range(2):
    rows = [json.loads(x) for x in open(root / f"scale_2.rank{rank}.jsonl")]
    row = next(x for x in rows if x["op"] == "all_reduce" and x["nbytes"] == 4096)
    (root / "single" / f"scale_2.rank{rank}.jsonl").write_text(json.dumps(row) + "\n")
PY
AFS_OUT="$tmp/single" EXPECTED_WORLD=2 EXPECTED_NNODES=2 EXPECTED_NPROC=1 \
  EXPECTED_HOST_ORDER=host0,host1 EXPECTED_VISIBLE_DEVICES=0 \
  EXPECTED_OPS=all_reduce EXPECTED_SIZES=4K EXPECTED_ITERS=3 \
  python3 "$tmp/validate.py"
python3 - "$tmp/single/validation_summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
assert s["valid"] and s["expected_case_count"] == 1
assert {"avg_ms", "alg_bw_GBps", "bus_bw_GBps", "iters_s_global_max"} <= s.keys()
PY

echo "test_fire_nccl_scale_muxi_jump: OK"
