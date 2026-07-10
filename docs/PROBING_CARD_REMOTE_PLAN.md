# Probing + CARD_SCREEN 远程与分支计划

## 目标

1. Ascend 开发推送到 `Kingsley-Yoimiya/Probing`（新仓），不直接改 `banzhijiangshan/Probing_plus`
2. 接入 `ShilohYu/CARD_SCREEN`，在分支 `montyyin_develop` 上开发，与上游分离

## 远程布局

### projects/Probing_plus

| remote | URL | 用途 |
|--------|-----|------|
| `upstream` | `banzhijiangshan/Probing_plus` | 只读拉基线 |
| `origin` | `Kingsley-Yoimiya/Probing` | 推送 `kingsley/ascend-lab` |

本地继续在 `kingsley/ascend-lab` 开发；代码质量问题后续在该分支上改，不混进上游 master。

### projects/CARD_SCREEN

| remote | URL | 用途 |
|--------|-----|------|
| `upstream`（或暂 `origin`） | `ShilohYu/CARD_SCREEN` | 上游 |
| 若无写权限 | 需用户 fork 后再设 `origin` | 推 `montyyin_develop` |

分支：`montyyin_develop`（从默认分支切出）。

## 步骤

1. 探测 `Kingsley-Yoimiya/Probing` 与 `CARD_SCREEN` 可达性/写权限
2. 改 Probing_plus remotes → push 分支
3. submodule add CARD_SCREEN → 开 `montyyin_develop`
4. 元仓库 commit + push

## 执行结果（2026-07-10）

- `Probing_plus` → origin=`Kingsley-Yoimiya/Probing`，upstream=`banzhijiangshan/Probing_plus`，分支 `kingsley/ascend-lab` 已推送
- `CARD_SCREEN` submodule 已加；分支 `montyyin_develop` 已推到 `ShilohYu/CARD_SCREEN`（有写权限）
