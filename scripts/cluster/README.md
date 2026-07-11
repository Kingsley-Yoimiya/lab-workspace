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
# 本机 Clash → weibozhen:18080 反代（装 rustup / docker build 用）
./scripts/cluster/egress_tunnel.sh start
./scripts/cluster/egress_tunnel.sh test

# 把 rust 装到 AFS（当前 job 立刻可用，不换镜像）
./scripts/cluster/install_rust_afs.sh

# 打带 rustup 的镜像（默认基于本地 a3-cann）
./scripts/cluster/build_image.sh
PUSH=1 ./scripts/cluster/build_image.sh

# 带宽/连通性探测
./scripts/cluster/probe_net.sh

# 把 main（含 submodule）同步到 AFS
./scripts/cluster/sync_to_afs.sh

# CARD_SCREEN / Probing_plus 冒烟（在 master pod）
./scripts/cluster/run_card_screen.sh
./scripts/cluster/run_probing_plus.sh

# 128 卡筛卡扇出（perf + 轻量 SDC）
./scripts/cluster/run_card_screen_128.sh

# HCCL scale（16/32/64/128）+ 链路健康
./scripts/cluster/run_hccl_scale.sh
./scripts/cluster/run_link_health.sh

# 训练 MFU 微基准（dense / moe）— 非完整 Qwen
MODE=dense ./scripts/cluster/run_mfu_bench_scale.sh
MODE=moe MASTER_PORT=31001 ./scripts/cluster/run_mfu_bench_scale.sh

# 真 MindSpeed Qwen 训练 MFU（C0：默认 Ascend wrapper，dense=8B / moe=30B-A3B）
MODE=dense SCALES=16 TRAIN_ITERS=5 ./scripts/cluster/run_train_mfu_scale.sh
# Peak 分母（card-screen median func_tflops）
python3 scripts/cluster/peak_from_card_screen.py --world-size 16
# Probing AFS 就绪检查
./scripts/cluster/check_probing_afs.sh

# 报告（本机）
python3 reports/gen_card_screen_128_report.py
python3 reports/gen_hccl_128_report.py
python3 reports/gen_train_mfu_128_report.py

# 辅助
./scripts/cluster/job_helpers.sh pods
```

## 环境变量

- `CLUSTER_SSH_HOST`（默认 `weibozhen`）
- `CLUSTER_JOB` / `CLUSTER_POD`
- `AFS_WORKSPACE`
- `LOCAL_PROXY`（默认 `http://127.0.0.1:7897` Clash）
- `REMOTE_PORT` / `REMOTE_PROXY_PORT`（默认 `18080`）
- `LOG_DIR`（默认写到仓库上两级的 `logs/cluster-*`，即 `random-thing/logs/`）

## 注意

- `ssh weibozhen` 登录容器上的 `/afs-a3-weight-share` 是假挂载，写不进 worker 真盘。
- AFS 多节点共享，同步一次即可。
- 登录机直连 GitHub/大文件 CDN 常超时；用 `egress_tunnel.sh` 走本机 Clash。
- `scripts/cluster/docker/rustup-init` 为 aarch64 二进制，不入库，构建时自动下载。
