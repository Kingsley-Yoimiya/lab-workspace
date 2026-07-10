# montyyin 工作区（元仓库）

公共机上的个人工作区备份与恢复入口。实验项目以 **git submodule** 形式放在 `projects/`。

远程：`git@github.com:Kingsley-Yoimiya/lab-workspace.git`

## 结构

```
.
├── projects/
│   ├── probing/         # DeepLink-org/probing（上游参考）
│   ├── Probing_plus/    # → Kingsley-Yoimiya/Probing（分支 kingsley/ascend-lab）
│   ├── probing-test/    # 历史 Nvidia 测试参考
│   └── CARD_SCREEN/     # ShilohYu/CARD_SCREEN（分支 montyyin_develop）
├── scripts/             # 恢复与 SSH 代理脚本
├── configs/
├── docs/
└── README.md
```

## SSH（含本机 32262 代理）

本机 egress 是 **HTTP CONNECT** 代理 `127.0.0.1:32262`（`PASEO_EGRESS_PROXY`）。  
`http_proxy` 只影响 HTTPS；**git SSH 必须**在 `~/.ssh/config` 里配 `ProxyCommand`：

```
Host github.com
  IdentityFile /montyyin/.ssh/id_ed25519_github_montyyin
  IdentitiesOnly yes
  ProxyCommand /montyyin/scripts/github-proxy-command.py %h %p
```

验证：`ssh -F ~/.ssh/config -T git@github.com`

## Probing（原 Probing_plus）开发分支

路径仍为 `projects/Probing_plus`，远程：

- `origin` → `Kingsley-Yoimiya/Probing`（推送）
- `upstream` → `banzhijiangshan/Probing_plus`（拉基线）

工作分支：`kingsley/ascend-lab`（已与 origin 跟踪）。

## CARD_SCREEN

- 路径：`projects/CARD_SCREEN`
- 工作分支：`montyyin_develop`（已推到 `ShilohYu/CARD_SCREEN`）

## 新机器恢复

1. 恢复私钥到 `~/.ssh/id_ed25519_github_montyyin`
2. 恢复 `~/.ssh/config`（含 ProxyCommand，若仍有 32262 代理）
3. `git clone --recurse-submodules git@github.com:Kingsley-Yoimiya/lab-workspace.git`

## 安全提醒

- **私钥永不入库**
- 元仓库建议 **private**
- 大数据 / 模型 / 日志默认在 `.gitignore`
