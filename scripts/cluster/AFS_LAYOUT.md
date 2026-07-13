# AFS 布局与写盘规则

任务侧共享盘约定的**唯一真相源**。网络 / Clash / 跳板见其它文档。

## 布局

两端个人可写目录都放在 **`/afs-a3-weight-share`**，用姓名后缀区分集群：

| 角色 | 华为 | 沐曦 |
|------|------|------|
| 根 `AFS_ROOT` | `/afs-a3-weight-share` | `/afs-a3-weight-share` |
| 默认 `AFS_USER` | **`yinjinrun.p-huawei`** | **`yinjinrun.p`** |
| 前缀 `AFS_HOME` | `$AFS_ROOT/$AFS_USER` | 同左 |
| 代码 | `$AFS_HOME/lab-workspace` | 同左 |
| 结果 | `$AFS_HOME/results/<name>-<YYYYMMDD_HHMMSS>/` | 同左 |
| 只读依赖 | `/afs-a3-241ceshi-shared/...`（`AFS_SHARED_READONLY` / `DATA_ROOT`） | `$AFS_ROOT/yushan/CARD_SCREEN`（`SHARED_CS_READONLY`） |

- 华为：`/afs-a3-weight-share/yinjinrun.p-huawei/{lab-workspace,results,...}`
- 沐曦：`/afs-a3-weight-share/yinjinrun.p/{lab-workspace,results}`
- **不要**再往 `/afs-a3-241ceshi-shared`（测试 Shared）写自有内容；该盘仅作只读依赖。

Profile：`source scripts/cluster/huawei.env` / `muxi.env`（可用 `AFS_USER_OVERRIDE` 等覆盖）。

## 硬规则

1. **只写 `$AFS_HOME`**：同步、日志、ckpt、报告、临时文件只落在自己的 `lab-workspace` 或 `results`。
2. **他人目录 / 测试 Shared 只读**：`yushan`、`geruijun`、`/afs-a3-241ceshi-shared` 下他人前缀等只能读；要跑别人的 CARD_SCREEN 用只读变量，或拷到自己的 `lab-workspace`，**禁止原地改**。
3. **误写在别人底下的自有内容要迁走**：迁到 `$AFS_HOME`，不留可写尾巴。
4. **真盘只在 pod**：跳板登录壳里 `/afs-a3-*` 常是假挂载；写 / 测 / 迁移一律 `cluster_pod_exec`。
5. **破坏性操作必须过守卫**：目标须在 `$AFS_HOME` 下（`afs_assert_under_home`），越界失败。
6. **Job / kubeconfig 隔离**：不覆盖跳板默认 `~/.kube/config`。
7. **结果带时间戳子目录**，避免互相覆盖。

## 守卫

```bash
source scripts/cluster/huawei.env   # 或 muxi.env
source scripts/cluster/afs_guard.sh
afs_assert_under_home "$AFS_RESULTS/foo"   # 通过
afs_assert_under_home /afs-a3-241ceshi-shared/montyyin/x   # 失败（已迁走/禁写）
afs_assert_under_home /afs-a3-weight-share/yushan/x        # 失败
```

## 迁移（历史）

- 沐曦旧 `weight-share/montyyin` → `yinjinrun.p`：`migrate_weight_share_home.sh`
- 华为旧 `241ceshi-shared/montyyin` → `weight-share/yinjinrun.p-huawei`（已迁，源目录已删）
