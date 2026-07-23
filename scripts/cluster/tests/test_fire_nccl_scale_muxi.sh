#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FIRE="$ROOT/scripts/cluster/fire_nccl_scale_muxi.sh"
RETRY_LIB="$ROOT/scripts/cluster/parallel_retry.sh"

out="$(
  CLUSTER_JOB_OVERRIDE=test-world64 \
  CLUSTER_POD_OVERRIDE=test-world64-master-0 \
  CLUSTER_N_WORKERS=63 \
  AFS_OUT=/afs-a3-weight-share/yinjinrun.p/results/dry-run \
  NPROC_OVERRIDE=1 \
  FIRE_PARALLELISM=64 \
  FIRE_REQUIRE_FULL_PARALLEL=1 \
  FIRE_DRY_RUN=1 \
  bash "$FIRE" 64 29991
)"

grep -q "DRY_RUN stages=write,start nnodes=64 parallel=64" <<<"$out"
grep -q "PLAN node_rank=0 pod=test-world64-master-0" <<<"$out"
grep -q "PLAN node_rank=63 pod=test-world64-worker-62" <<<"$out"
[[ "$(grep -c '^PLAN node_rank=' <<<"$out")" -eq 64 ]]

# 环境 override 必须在 source muxi.env 后仍生效，且 64n1r 映射完整。
grep -q "FIRE scale=64 nnodes=64 nproc=1" <<<"$out"

# prepare 必须先于并行 write/start；共享上传串行、带 hash 门禁。
prepare_line="$(grep -n '^prepare_exec mkdir ' "$FIRE" | cut -d: -f1)"
write_line="$(grep -n '^if ! run_parallel_stage write ' "$FIRE" | cut -d: -f1)"
start_line="$(grep -n '^if ! run_parallel_stage start ' "$FIRE" | cut -d: -f1)"
[[ "$prepare_line" -lt "$write_line" && "$write_line" -lt "$start_line" ]]
grep -q 'HASH_VERIFIED=1' "$FIRE"
grep -q 'UPLOAD_GATE_OK bench_and_metrics_hash_verified=1' "$FIRE"

# 每个后台 write 使用独立 launcher 文件重定向，不共享父 stdin。
grep -q 'render_run_body "$r" >"$run_src"' "$FIRE"
grep -q '<"$run_src"' "$FIRE"
if grep -Eq 'render_run_body.*\|.*cluster_pod_exec_i' "$FIRE"; then
  echo "parallel writes must not share a render pipeline stdin" >&2
  exit 1
fi

# 写入失败必须阻止 start；启动失败必须触发并行清理。
grep -q 'FIRE_FAIL stage=write' "$FIRE"
grep -q 'FIRE_FAIL stage=start; cleaning all target nodes' "$FIRE"
grep -q 'ALREADY_LAUNCHER_RUNNING' "$FIRE"
grep -q 'LAUNCHER_SUBMITTED_AT' "$FIRE"
grep -q 'atomic_marker.*startedf' "$FIRE"
grep -q 'atomic_marker.*donef' "$FIRE"
grep -q 'atomic_marker.*failf' "$FIRE"
grep -q 'run_parallel_stage cleanup cleanup_one_rank' "$FIRE"

# macOS 系统 Bash 3.2 不支持 ${var^^}；并行 stage 记时必须保持可移植。
if grep -Eq '\$\{[^}]*\^\^[^}]*\}' "$FIRE"; then
  echo "bash-4-only uppercase expansion is forbidden" >&2
  exit 1
fi
[[ "$(printf '%s' write | tr '[:lower:]' '[:upper:]')" == "WRITE" ]]

# 纯逻辑：首次必须64并行；仅瞬时失败节点重试，成功节点不重跑。
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
source "$RETRY_LIB"

mock_transient_once() {
  local rank="$1" attempt="$2" log="$3"
  local count_file="$tmp/count.$rank"
  local count=0
  [[ -f "$count_file" ]] && count="$(cat "$count_file")"
  count=$((count + 1))
  echo "$count" >"$count_file"
  if [[ "$rank" -eq 17 && "$attempt" -eq 1 ]]; then
    echo "error: EOF" >"$log"
    return 1
  fi
  echo "OK rank=$rank attempt=$attempt" >"$log"
}

parallel_retry_run write 64 64 3 "$tmp/transient" mock_transient_once
grep -q 'ATTEMPT=1 PENDING=64' "$tmp/transient/write_retry_summary.log"
grep -q 'ATTEMPT=2 PENDING=1 RANKS=17' "$tmp/transient/write_retry_summary.log"
[[ "$(cat "$tmp/count.17")" -eq 2 ]]
[[ "$(cat "$tmp/count.0")" -eq 1 ]]
[[ "$(cat "$tmp/count.63")" -eq 1 ]]

# 永久语义/权限失败不得重试，且必须向调用方传播失败。
mock_permanent_failure() {
  local rank="$1" attempt="$2" log="$3"
  local count_file="$tmp/permanent.count.$rank"
  local count=0
  [[ -f "$count_file" ]] && count="$(cat "$count_file")"
  count=$((count + 1))
  echo "$count" >"$count_file"
  if [[ "$rank" -eq 23 ]]; then
    echo "Error from server (Forbidden): permission denied" >"$log"
    return 2
  fi
  echo "OK" >"$log"
}
if parallel_retry_run start 64 64 3 "$tmp/permanent" mock_permanent_failure; then
  echo "expected permanent failure propagation" >&2
  exit 1
fi
[[ "$(cat "$tmp/permanent.count.23")" -eq 1 ]]
[[ "$(cat "$tmp/permanent.count.0")" -eq 1 ]]
grep -q 'PERMANENT_FAILURES=23:2' "$tmp/permanent/start_retry_summary.log"

if CLUSTER_JOB_OVERRIDE=test-world64 \
  CLUSTER_POD_OVERRIDE=test-world64-master-0 \
  CLUSTER_N_WORKERS=63 \
  AFS_OUT=/afs-a3-weight-share/yinjinrun.p/results/dry-run \
  NPROC_OVERRIDE=1 \
  FIRE_PARALLELISM=16 \
  FIRE_REQUIRE_FULL_PARALLEL=1 \
  FIRE_DRY_RUN=1 \
  bash "$FIRE" 64 29992 >/dev/null 2>&1; then
  echo "expected full-parallel validation failure" >&2
  exit 1
fi

if CLUSTER_JOB_OVERRIDE=test-world64 \
  CLUSTER_POD_OVERRIDE=test-world64-master-0 \
  CLUSTER_N_WORKERS=63 \
  AFS_OUT=/afs-a3-weight-share/yinjinrun.p/results/dry-run \
  NPROC_OVERRIDE=1 \
  FIRE_PREPARE_RETRIES=0 \
  FIRE_DRY_RUN=1 \
  bash "$FIRE" 64 29993 >/dev/null 2>&1; then
  echo "expected prepare retry validation failure" >&2
  exit 1
fi

echo "test_fire_nccl_scale_muxi: OK"
