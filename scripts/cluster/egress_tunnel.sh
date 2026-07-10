#!/usr/bin/env bash
# 本机 Clash HTTP → SSH RemoteForward → weibozhen:18080
# 用法:
#   ./scripts/cluster/egress_tunnel.sh start
#   ./scripts/cluster/egress_tunnel.sh status
#   ./scripts/cluster/egress_tunnel.sh stop
#   ./scripts/cluster/egress_tunnel.sh test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=job_helpers.sh
source "$SCRIPT_DIR/job_helpers.sh"

LOCAL_PROXY="${LOCAL_PROXY:-http://127.0.0.1:7897}"
REMOTE_PORT="${REMOTE_PORT:-18080}"
# 解析 host:port
LOCAL_PROXY_HOSTPORT="${LOCAL_PROXY#*://}"
LOCAL_PROXY_HOSTPORT="${LOCAL_PROXY_HOSTPORT%%/*}"
LOCAL_HOST="${LOCAL_PROXY_HOSTPORT%%:*}"
LOCAL_PORT="${LOCAL_PROXY_HOSTPORT##*:}"
PID_FILE="${PID_FILE:-/tmp/weibozhen-egress-tunnel.pid}"
LOG_FILE="${LOG_FILE:-/tmp/weibozhen-egress-tunnel.log}"

start_tunnel() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already running pid=$(cat "$PID_FILE")"
    return 0
  fi
  # 先测本机代理
  curl -fsSI --connect-timeout 5 -x "$LOCAL_PROXY" https://static.rust-lang.org >/dev/null \
    || { echo "本机代理不可用: $LOCAL_PROXY"; exit 1; }

  echo "==> ssh -R ${REMOTE_PORT}:${LOCAL_HOST}:${LOCAL_PORT} $CLUSTER_SSH_HOST"
  # -f 后台；-N 不执行远程命令；ExitOnForwardFailure 保证转发成功
  ssh -f -N \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -R "127.0.0.1:${REMOTE_PORT}:${LOCAL_HOST}:${LOCAL_PORT}" \
    "$CLUSTER_SSH_HOST"

  # 找 ssh 进程 pid（按 RemoteForward 特征）
  sleep 1
  PID="$(pgrep -f "ssh.*-R.*${REMOTE_PORT}:${LOCAL_HOST}:${LOCAL_PORT}.*${CLUSTER_SSH_HOST}" | head -1 || true)"
  if [[ -z "${PID}" ]]; then
    # fallback: 最近启动的 ssh 到该 host
    PID="$(pgrep -n -f "ssh.*${CLUSTER_SSH_HOST}" || true)"
  fi
  echo "${PID:-}" > "$PID_FILE"
  echo "started pid=${PID:-unknown} remote=127.0.0.1:${REMOTE_PORT} -> ${LOCAL_HOST}:${LOCAL_PORT}"
}

stop_tunnel() {
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
      kill "$PID" || true
      echo "stopped pid=$PID"
    fi
    rm -f "$PID_FILE"
  fi
  # 清掉残留
  pkill -f "ssh.*-R.*${REMOTE_PORT}:.*${CLUSTER_SSH_HOST}" 2>/dev/null || true
}

status_tunnel() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "running pid=$(cat "$PID_FILE")"
  else
    echo "stopped"
  fi
  cluster_ssh "ss -lntp 2>/dev/null | grep ${REMOTE_PORT} || netstat -lntp 2>/dev/null | grep ${REMOTE_PORT} || echo 'remote port not listening'"
}

test_tunnel() {
  echo "==> 经反代拉 rust CDN HEAD"
  cluster_ssh "curl -fsSI --connect-timeout 10 --max-time 30 -x http://127.0.0.1:${REMOTE_PORT} https://static.rust-lang.org | head -8"
  echo "==> 经反代拉一小段 rustup-init"
  cluster_ssh "curl -fsSL --connect-timeout 10 --max-time 60 -x http://127.0.0.1:${REMOTE_PORT} -r 0-1023 -o /tmp/rustup-init.head https://static.rust-lang.org/rustup/dist/aarch64-unknown-linux-gnu/rustup-init && ls -la /tmp/rustup-init.head && echo PROXY_OK"
}

case "${1:-}" in
  start) start_tunnel ;;
  stop) stop_tunnel ;;
  status) status_tunnel ;;
  test) test_tunnel ;;
  restart) stop_tunnel; start_tunnel ;;
  *)
    echo "用法: $0 {start|stop|status|test|restart}"
    echo "环境变量: LOCAL_PROXY (默认 http://127.0.0.1:7897) REMOTE_PORT (默认 18080)"
    exit 1
    ;;
esac
