# montyyin 工作区（元仓库）

公共机上的个人工作区备份与恢复入口。实验项目以 **git submodule** 形式放在 `projects/`。

远程：`git@github.com:Kingsley-Yoimiya/lab-workspace.git`

## 结构

```
.
├── projects/
│   ├── probing/         # DeepLink-org/probing（上游参考）
│   ├── Probing_plus/    # Ascend 适配（开发主战场，分支 kingsley/ascend-lab）
│   └── probing-test/    # 历史 Nvidia 测试参考
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

## Probing_plus 开发分支（勿与上游混写）

本地已切到 `kingsley/ascend-lab`。对 `banzhijiangshan/Probing_plus` **无 push 权限**，需要先 fork 到自己账号，再：

```bash
cd projects/Probing_plus
git remote rename origin upstream
git remote add origin git@github.com:Kingsley-Yoimiya/Probing_plus.git
git push -u origin kingsley/ascend-lab
```

之后改代码只在该分支 commit/push 到自己的 fork。

## 新机器恢复

1. 恢复私钥到 `~/.ssh/id_ed25519_github_montyyin`
2. 恢复 `~/.ssh/config`（含 ProxyCommand，若仍有 32262 代理）
3. `git clone --recurse-submodules git@github.com:Kingsley-Yoimiya/lab-workspace.git`

## 安全提醒

- **私钥永不入库**
- 元仓库建议 **private**
- 大数据 / 模型 / 日志默认在 `.gitignore`
