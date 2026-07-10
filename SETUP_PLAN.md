# 工作区 GitHub 同步 Setup 计划

## 目标

在公共机器上的个人工作区 `/montyyin` 实现：

1. **独立 SSH 密钥** → 只用于本工作区与 GitHub 通信
2. **工作区元仓库（meta-repo）** → 服务器崩了可一键从 GitHub 恢复
3. **实验项目以 submodule 接入** → 各自独立历史，元仓库只记录指针

## 第一性原理

| 问题 | 结论 |
|------|------|
| 公共机上的密钥不能和本机混用 | 生成专用 ed25519 密钥，SSH config 限定 `github.com` |
| 整个 HOME 不能直接 git 化 | `.cursor` / `.npm` / `.local` / `.paseo` 等是机器态，必须 ignore |
| 实验仓库有自己的上游 | 用 submodule，不把别人的完整历史塞进元仓库 |
| 恢复要可重复 | 元仓库只存：配置、脚本、文档、submodule 指针 |

## 目录约定

```
/montyyin/                          # 元仓库根（HOME）
├── .gitignore                      # 严格忽略机器态与密钥
├── .gitmodules                     # submodule 注册表
├── README.md                       # 恢复说明
├── SETUP_PLAN.md                   # 本计划
├── scripts/
│   └── bootstrap.sh                # 新机器恢复脚本
├── configs/                        # 可同步的点配置（按需）
└── projects/                       # 所有 clone 的实验项目（submodule）
    └── <repo-name>/
```

## 执行步骤

### Phase 1 — SSH（本机可完成大半）

1. `mkdir -p ~/.ssh && chmod 700 ~/.ssh`
2. 生成密钥：`~/.ssh/id_ed25519_github_montyyin`
3. 写 `~/.ssh/config`：`Host github.com` → 使用该 IdentityFile
4. **你**把公钥加到 GitHub → Settings → SSH keys
5. `ssh -T git@github.com` 验证

### Phase 2 — 元仓库结构

1. 写严格 `.gitignore`（忽略 `.ssh`、缓存、IDE 态、密钥）
2. 建 `projects/`、`scripts/`、`README.md`
3. `git init`，本地 git user 仅对本仓库生效（不改全局，避免污染公共机）
4. 首次 commit

### Phase 3 — 推到 GitHub

1. 你提供：GitHub 用户名 + 期望的私有仓库名（建议 `montyyin-workspace`）
2. 创建空私有仓库（`gh` 或网页）
3. `git remote add origin git@github.com:<user>/<repo>.git`
4. `git push -u origin main`

### Phase 4 — Submodule 工作流（之后每次）

```bash
# 加入实验项目
git submodule add git@github.com:<org>/<repo>.git projects/<repo>
git commit -m "add submodule: <repo>"

# 新机器恢复
git clone --recurse-submodules git@github.com:<user>/<repo>.git
# 或已 clone 后：
git submodule update --init --recursive
```

## 明确不入库的内容

- `~/.ssh/`（私钥绝不能进 git）
- `.cursor/`、`.cursor-server/`、`.paseo/`、`.npm/`、`.cache/`、`.local/`
- `node_modules/`、大型模型权重、数据集、日志
- `.env`、credentials、token

## 需要你确认的信息

1. GitHub 用户名
2. 元仓库名称（默认 `montyyin-workspace`）
3. git commit 用的 name / email
4. 仓库可见性：建议 **private**
