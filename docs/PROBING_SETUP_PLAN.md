# Probing 三仓 Setup 计划

## 目标

在元仓库 `lab-workspace` 的 `projects/` 下，用 **SSH + submodule** 接入三个相关仓库，并保证后续在 Ascend 适配仓上的修改有独立分支，不与上游/旧测试混写。

## 仓库角色（第一性原理）

| 仓库 | 角色 | 路径 |
|------|------|------|
| `DeepLink-org/probing` | 上游原版（只读参考） | `projects/probing` |
| `banzhijiangshan/Probing_plus` | Ascend 适配基线（开发主战场） | `projects/Probing_plus` |
| `Kingsley-Yoimiya/probing-test` | 历史 Nvidia 测试（只读参考） | `projects/probing-test` |

三者 **各自独立 submodule**，不互相嵌套、不把代码拷进元仓库。

## 分支策略（Probing_plus）

问题：直接在 `main`/`master` 上改会和上游混在一起；且对 `banzhijiangshan/Probing_plus` 未必有写权限。

方案：

1. submodule 跟踪上游 `Probing_plus`
2. 本地开工作分支：`kingsley/ascend-lab`（或类似）
3. 若无上游 push 权限 → fork 到 `Kingsley-Yoimiya/Probing_plus`，加 `origin`/`upstream`：
   - `upstream` = `banzhijiangshan/Probing_plus`（拉更新）
   - `origin` = 自己的 fork（推分支）
4. 元仓库只记录 submodule **commit 指针**；日常开发在 submodule 内 commit/push 到自己的 fork 分支

## 执行步骤

1. `git submodule add` 三个 SSH 地址到 `projects/`
2. 检查 `Probing_plus` 默认分支；创建 `kingsley/ascend-lab`
3. 探测对上游的 push 权限；无权限则 fork 并改 remote
4. 元仓库 commit + push（更新 `.gitmodules` 与指针）
5. 简要验证：`git submodule status`

## 不做的事

- 不把三个仓的代码合并成一个目录
- 不在 `probing` / `probing-test` 上直接开开发分支（除非后续明确要求）
- 不提交密钥、大数据、构建产物
