# AFS 布局与写盘规则

任务侧共享盘约定的**唯一真相源**。网络 / Clash / 跳板见其它文档。

## 布局

| 角色 | 华为 (`241ceshi-shared`) | 沐曦 (`weight-share`，共享盘) |
|------|--------------------------|-------------------------------|
| 根 `AFS_ROOT` | `/afs-a3-241ceshi-shared` | `/afs-a3-weight-share` |
| 默认 `AFS_USER` | `montyyin` | **`yinjinrun.p`** |
| 前缀 `AFS_HOME` | `$AFS_ROOT/$AFS_USER` | 同左 |
| 代码 | `$AFS_HOME/lab-workspace` | 同左 |
| 结果 | `$AFS_HOME/results/<name>-<YYYYMMDD_HHMMSS>/` | 同左 |
| 只读依赖 | 显式变量（如 `DATA_ROOT`、`SHARED_CS_READONLY`） | 同左（如 `$AFS_ROOT/yushan/CARD_SCREEN`） |

weight-share 上先建**自己姓名目录**再开发；本组默认 `/afs-a3-weight-share/yinjinrun.p/{lab-workspace,results}`。

Profile：`source scripts/cluster/huawei.env` / `muxi.env`（可用 `AFS_USER_OVERRIDE` 等覆盖）。

## 硬规则

1. **只写 `$AFS_HOME`**：同步、日志、ckpt、报告、临时文件只落在自己的 `lab-workspace` 或 `results`。
2. **他人目录只读**：`yushan`、weight-share 上的 `montyyin`、`geruijun` 等只能读；要跑别人的 CARD_SCREEN 用 `SHARED_CS_READONLY`，或拷到自己的 `lab-workspace`，**禁止原地改**。
3. **误写在别人底下的自有内容要迁走**：迁到 `$AFS_HOME`，不留可写尾巴。迁移脚本：`migrate_weight_share_home.sh`。
4. **真盘只在 pod**：跳板登录壳里 `/afs-a3-*` 常是假挂载；写 / 测 / 迁移一律 `cluster_pod_exec`。
5. **破坏性操作必须过守卫**：`rm -rf`、全量覆盖 sync、清空远端目录的目标须在 `$AFS_HOME` 下（`afs_assert_under_home`），越界失败。
6. **Job / kubeconfig 隔离**：不覆盖跳板默认 `~/.kube/config`；不把别人 Running job 当自己的工作区写配置。
7. **结果带时间戳子目录**，避免互相覆盖。

## 守卫

```bash
source scripts/cluster/huawei.env   # 或 muxi.env
source scripts/cluster/afs_guard.sh
afs_assert_under_home "$AFS_RESULTS/foo"   # 通过
afs_assert_under_home /afs-a3-weight-share/yushan/x   # 失败
```
