# montyyin 工作区（元仓库）

公共机上的个人工作区备份与恢复入口。实验项目以 **git submodule** 形式放在 `projects/`。

## 结构

```
.
├── projects/          # 实验仓库（submodule）
├── scripts/           # 恢复与辅助脚本
├── configs/           # 可同步的配置片段
├── SETUP_PLAN.md      # setup 计划
└── README.md
```

## 首次 Setup（本机）

1. 生成专用 SSH 密钥（已完成则跳过）
2. 将公钥添加到 GitHub → Settings → SSH and GPG keys
3. 验证：`ssh -T git@github.com`
4. 创建私有空仓库后：

```bash
cd ~
git remote add origin git@github.com:<USER>/<REPO>.git
git push -u origin main
```

## 加入实验项目（submodule）

```bash
git submodule add git@github.com:<org>/<repo>.git projects/<repo>
git commit -m "add submodule: <repo>"
git push
```

## 新机器恢复

1. 从安全位置恢复私钥到 `~/.ssh/id_ed25519_github_montyyin`
2. 写入 `~/.ssh/config`（Host github.com → 该 IdentityFile）
3. 执行：

```bash
bash scripts/bootstrap.sh git@github.com:<USER>/<REPO>.git
# 或全新目录：
git clone --recurse-submodules git@github.com:<USER>/<REPO>.git
```

## 安全提醒

- **私钥永不入库**；元仓库只同步公钥指纹说明与流程
- 建议元仓库设为 **private**
- 大数据、模型权重、日志默认在 `.gitignore` 中
