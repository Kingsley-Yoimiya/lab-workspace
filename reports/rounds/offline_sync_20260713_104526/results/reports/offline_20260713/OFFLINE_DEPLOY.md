# 远程 2h 收口部署说明（本机可离线）

> 部署时间：2026-07-13  
> 跳板：`ais-jump`  
> Job：`montyyin-moe96-r2`

## 已在跳板启动的进程

| 组件 | 脚本 | 日志 |
|------|------|------|
| MoE FailSlow 编排 | `/root/montyyin-lab-remote/jumphost_moe_failslow.sh` | `/tmp/moe-failslow-20260713_003312.log` |
| 远程监控+收口 | `/root/montyyin-lab-remote/remote_monitor_finalize.sh` | `/tmp/remote_monitor_20260713_003312.log` |
| 总启动日志 | `launch_2h_remote.sh` | `/tmp/launch_2h_remote_20260713_003312.log` |

## MoE 短跑配置

- Stamp：`20260713_003312`
- 波次：`32+64` 并行，ITERS=40
- TP1/PP4/EP4，GBS=1920，`failslow_step_timer` 落盘
- AFS：`/afs-a3-241ceshi-shared/montyyin/results/moe_failslow/20260713_003312/`

## 归档（已打包）

`/afs-a3-241ceshi-shared/montyyin/archive/20260713_offline/`

- `dense_failslow_20260713_001230.tgz`
- `dense_failslow_gbsprop_20260713_071316.tgz`
- `mfu_moe_scale_181247.tgz`（含 32/64 MFU）
- `mfu_moe_scale_221912.tgz`（含 96 MFU）

## 收口产物（监控完成后）

`/afs-a3-241ceshi-shared/montyyin/results/reports/offline_20260713/SUMMARY.md`

各实验目录下 `gap_vs_n.csv` 由 pod 内 `parse_failslow_gap.py` 生成。

## 手动检查（ssh ais-jump）

```bash
export KUBECONFIG=/root/.kube/config.huawei-a3-241ceshi
tail -f /tmp/moe-failslow-20260713_003312.log
tail -f /tmp/remote_monitor_20260713_003312.log
grep DONE /tmp/moe-failslow-20260713_003312.log
```
