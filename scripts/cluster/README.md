# 集群运维脚本（分支 `ops/cluster`）

本机 pull → 经 `ssh weibozhen` + `vcctl pod exec` 写入真 AFS；**不要**在登录机/worker 上直连 GitHub。

## 路径约定

| 位置 | 路径 |
|------|------|
| 本机 ops 仓库 | `project/lab-workspace`（本分支） |
| 本机 main worktree | `project/lab-workspace-main`（`sync_to_afs.sh` 自动创建） |
| AFS 工作区 | `/afs-a3-241ceshi-shared/montyyin/lab-workspace` |
| 默认 job/pod | `huawei-8node-copy` / `…-master-0` |

## 脚本

```bash
# 带宽/连通性探测
./scripts/cluster/probe_net.sh

# 把 main（含 submodule）同步到 AFS
./scripts/cluster/sync_to_afs.sh

# CARD_SCREEN / Probing_plus 冒烟（在 master pod）
./scripts/cluster/run_card_screen.sh
./scripts/cluster/run_probing_plus.sh

# 镜像骨架（默认不 push）
./scripts/cluster/build_image.sh
PUSH=1 ./scripts/cluster/build_image.sh

# 辅助
./scripts/cluster/job_helpers.sh pods
```

## 环境变量

- `CLUSTER_SSH_HOST`（默认 `weibozhen`）
- `CLUSTER_JOB` / `CLUSTER_POD`
- `AFS_WORKSPACE`
- `LOG_DIR`（默认写到仓库上两级的 `logs/cluster-*`，即 `random-thing/logs/`）

## 注意

- `ssh weibozhen` 登录容器上的 `/afs-a3-weight-share` 是假挂载，写不进 worker 真盘。
- AFS 多节点共享，同步一次即可。
