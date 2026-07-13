#!/usr/bin/env bash
# 本机入口：上传 Dense FailSlow 编排到跳板并 nohup 启动
set +eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
JOB="${JOB:-montyyin-moe96-r2}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export STAMP
SCALES="${SCALES:-32+64,96,16}"
TRAIN_ITERS="${TRAIN_ITERS:-220}"
GBS="${GBS:-1920}"
PROBING="${PROBING:-1}"
FAILSLOW_STEP_LOG="${FAILSLOW_STEP_LOG:-1}"
MASTER_PORT="${MASTER_PORT:-25600}"
SCALE_TIMEOUT_SEC="${SCALE_TIMEOUT_SEC:-10800}"
SSH_HOST="${SSH_HOST:-ais-jump}"
LOG_DIR="${LOG_DIR:-/Users/yinjinrun/random-thing/logs/dense-failslow-${STAMP}}"
mkdir -p "$LOG_DIR"
echo "$LOG_DIR" > /tmp/dense_failslow_logdir.txt
echo "$STAMP" > /tmp/dense_failslow_stamp.txt

SCRIPT_LOCAL="$ROOT/scripts/cluster/jumphost_dense_failslow.sh"
HOOK_DIR="$ROOT/scripts/cluster/hooks"
WRAP_LOCAL="$ROOT/scripts/cluster/wrappers/train_qwen3_8B_ascend.sh"

echo "START $(date -Iseconds) STAMP=$STAMP SCALES=$SCALES ITERS=$TRAIN_ITERS JOB=$JOB"

ssh -o BatchMode=yes -o ConnectTimeout=30 "$SSH_HOST" \
  'cat > /tmp/jumphost_dense_failslow.sh && chmod +x /tmp/jumphost_dense_failslow.sh' \
  < "$SCRIPT_LOCAL"

python3 - <<PY
import base64, pathlib, subprocess

host = "${SSH_HOST}"
job = "${JOB}"
hook = pathlib.Path("${HOOK_DIR}")
wrap = pathlib.Path("${WRAP_LOCAL}")
pairs = [
    (hook / "failslow_step_timer.py",
     "/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/hooks/failslow_step_timer.py", "644"),
    (hook / "sitecustomize.py",
     "/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/hooks/sitecustomize.py", "644"),
    (wrap,
     "/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/wrappers/train_qwen3_8B_ascend.sh", "755"),
]
# ensure dirs
subprocess.check_call([
    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host,
    "export KUBECONFIG=/root/.kube/config.huawei-a3-241ceshi; "
    f"vcctl pod exec {job}-master-0 -- bash -lc "
    "'mkdir -p /afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/hooks "
    "/afs-a3-weight-share/yinjinrun.p-huawei/lab-workspace/scripts/cluster/wrappers'",
])
for src, dst, mode in pairs:
    b64 = base64.b64encode(src.read_bytes()).decode()
    tmp = f"/tmp/_dense_up_{src.name}.b64"
    pathlib.Path(tmp).write_text(b64)
    subprocess.check_call([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host,
        f"cat > {tmp}",
    ], stdin=open(tmp, "rb"))
    remote = (
        "export KUBECONFIG=/root/.kube/config.huawei-a3-241ceshi; "
        f"base64 -d {tmp} | vcctl pod exec -i {job}-master-0 -- bash -lc "
        f"'cat > {dst} && chmod {mode} {dst} && wc -c {dst}'"
    )
    out = subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=40", host, remote], text=True)
    print(src.name, out.strip())
    n = int(out.strip().split()[0])
    if n < 100:
        raise SystemExit(f"upload too small: {src} -> {n}")
print("SYNC_OK")
PY

ssh -o BatchMode=yes -o ConnectTimeout=30 "$SSH_HOST" bash -s <<REMOTE | tee "$LOG_DIR/start.log"
export KUBECONFIG=/root/.kube/config.huawei-a3-241ceshi
export JOB=$JOB STAMP=$STAMP SCALES='$SCALES' GBS=$GBS TRAIN_ITERS=$TRAIN_ITERS
export PROBING=$PROBING FAILSLOW_STEP_LOG=$FAILSLOW_STEP_LOG MASTER_PORT=$MASTER_PORT
export SCALE_TIMEOUT_SEC=$SCALE_TIMEOUT_SEC TP=4 PP=2 EP=1
export RUN_ROOT=/afs-a3-weight-share/yinjinrun.p-huawei/results/dense_failslow/${STAMP}
export LOCAL_LOG=/tmp/dense-failslow-${STAMP}.log
pkill -f /tmp/jumphost_dense_failslow.sh || true
sleep 1
nohup bash -lc 'export KUBECONFIG=/root/.kube/config.huawei-a3-241ceshi
export JOB='"$JOB"' STAMP='"$STAMP"' SCALES='"'$SCALES'"' GBS='"$GBS"' TRAIN_ITERS='"$TRAIN_ITERS"'
export PROBING='"$PROBING"' FAILSLOW_STEP_LOG='"$FAILSLOW_STEP_LOG"' MASTER_PORT='"$MASTER_PORT"'
export SCALE_TIMEOUT_SEC='"$SCALE_TIMEOUT_SEC"' TP=4 PP=2 EP=1
export RUN_ROOT=/afs-a3-weight-share/yinjinrun.p-huawei/results/dense_failslow/'"$STAMP"'
export LOCAL_LOG=/tmp/dense-failslow-'"$STAMP"'.log
exec bash /tmp/jumphost_dense_failslow.sh' > /tmp/dense-failslow-${STAMP}_nohup.out 2>&1 &
echo STARTED_\$! STAMP=$STAMP
sleep 10
tail -40 /tmp/dense-failslow-${STAMP}.log
REMOTE

echo "DONE_LAUNCH $(date -Iseconds) log=$LOG_DIR"
