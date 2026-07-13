#!/usr/bin/env bash
# AFS 写盘守卫：破坏性操作目标必须落在 $AFS_HOME 下。
# 用法:
#   source scripts/cluster/muxi.env   # 或 huawei.env（需已设 AFS_HOME）
#   source scripts/cluster/afs_guard.sh
#   afs_assert_under_home /path/to/target
#
# 约定见 AFS_LAYOUT.md

afs_normalize_path() {
  local p="$1"
  # 去掉末尾多余 /
  while [[ "$p" == */ && "$p" != / ]]; do
    p="${p%/}"
  done
  printf '%s' "$p"
}

# 若 path 不在 AFS_HOME 下则返回非 0 并打印错误
afs_assert_under_home() {
  local path="${1:?usage: afs_assert_under_home <path>}"
  local home="${AFS_HOME:?AFS_HOME 未设置；请先 source huawei.env 或 muxi.env}"

  path="$(afs_normalize_path "$path")"
  home="$(afs_normalize_path "$home")"

  if [[ "$path" != "$home" && "$path" != "$home"/* ]]; then
    echo "afs_guard: 拒绝 — 路径不在 AFS_HOME 下" >&2
    echo "  path: $path" >&2
    echo "  home: $home" >&2
    return 1
  fi
  return 0
}

# 便捷：断言多个路径
afs_assert_under_home_all() {
  local p
  for p in "$@"; do
    afs_assert_under_home "$p" || return 1
  done
  return 0
}
