#!/usr/bin/env bash
# 新机器恢复 montyyin 工作区
set -euo pipefail

REPO_SSH="${1:-}"
if [[ -z "$REPO_SSH" ]]; then
  echo "用法: $0 git@github.com:<user>/<repo>.git"
  exit 1
fi

WORKSPACE="${HOME:-/montyyin}"
echo "==> 工作区: $WORKSPACE"

# 1. SSH 密钥需事先放到 ~/.ssh/（私钥不能从 git 恢复）
if [[ ! -f "$HOME/.ssh/id_ed25519_github_montyyin" ]]; then
  echo "错误: 找不到专用密钥 ~/.ssh/id_ed25519_github_montyyin"
  echo "请先从安全位置拷贝私钥，并配置 ~/.ssh/config（见 README）"
  exit 1
fi
chmod 700 "$HOME/.ssh"
chmod 600 "$HOME/.ssh/id_ed25519_github_montyyin"
chmod 644 "$HOME/.ssh/id_ed25519_github_montyyin.pub" 2>/dev/null || true

# 2. 若当前目录还不是该仓库，则 clone
if [[ ! -d "$WORKSPACE/.git" ]]; then
  echo "==> clone 元仓库（含 submodule）"
  # HOME 已存在时：先 clone 到临时目录再 rsync 内容，避免覆盖 HOME
  TMP="$(mktemp -d)"
  git clone --recurse-submodules "$REPO_SSH" "$TMP/ws"
  rsync -a --exclude '.ssh' "$TMP/ws"/ "$WORKSPACE"/
  rm -rf "$TMP"
else
  echo "==> 已有 .git，拉取并更新 submodule"
  git -C "$WORKSPACE" pull --ff-only
  git -C "$WORKSPACE" submodule update --init --recursive
fi

echo "==> 完成。进入 projects/ 开始工作。"
