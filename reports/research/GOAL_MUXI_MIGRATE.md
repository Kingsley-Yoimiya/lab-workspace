# GOAL: 华为脚本全量迁移到 Muxi 并跑出结果

**状态**: **已完成**（2026-07-11）  
**设立**: 2026-07-11  
**Owner**: montyyin / Cursor agent  
**成功标准**: 华为侧每条主能力在沐曦集群都有**可复跑脚本 + 落盘结果 + 对照报告**；双集群可并存互不踩 kubeconfig。  
**进度板**: `GOAL_MUXI_MIGRATE_STATUS.md`  
**正式总汇报**: [`../rounds/CAMPAIGN_FINAL_MUXI_20260711.md`](../rounds/CAMPAIGN_FINAL_MUXI_20260711.md)（体例对齐昇腾 `CAMPAIGN_FINAL`；语义见 `METRIC_SEMANTICS_MUXI`）

## 范围（必须全部完成）

对标华为 `scripts/cluster/` 主能力，在 muxi（`vc-c550-mohe-241` / MetaX C550 / 16×8）落地：

| # | 能力 | 华为入口 | Muxi 目标 | 验收 |
|---|------|----------|-----------|------|
| G0 | 双集群运维隔离 | `huawei.env` | `muxi.env` + KUBECONFIG | 两边可并行 vcctl |
| G1 | 快慢卡冒烟 128 | `run_card_screen*.sh` | `run_card_screen_muxi.sh` | 128 卡 JSONL + cluster.json |
| G2 | 体质 constitution | `run_card_constitution_128.sh` | `run_card_constitution_muxi.sh` | 128 卡多探针 JSONL + 出图报告 |
| G3 | 拓扑探测 | `probe_hccl_topology.sh` | `probe_muxi_topology.sh` | topo_summary.json |
| G4 | Collective scale | `run_hccl_scale.sh` | `run_nccl_scale.sh` + NCCL bench | scale_8/16/32/64/128 JSONL |
| G5 | P2P 慢边 | `run_hccl_p2p_128.sh` | `run_nccl_p2p.sh` | 边级结果 + TopK |
| G6 | 链路健康 | `run_link_health.sh` | `run_link_health_muxi.sh` | 每节点健康文本/JSON |
| G7 | 一键流水线 | `run_constitution_then_comm.sh` | `run_constitution_then_comm_muxi.sh` | pipeline 日志齐套 |
| G8 | MFU 微基准 | `run_mfu_bench_scale.sh` | NCCL 版 | dense/moe scale 结果 |
| G9 | 真训练 MFU | `run_train_mfu_scale.sh` | 沐曦训练 wrapper（若镜像具备） | 至少 1 档 scale 出数 |
| G10 | 报告对齐 | `reports/gen_*` / `plot_*` | 同格式 rounds 报告 | 华为↔沐曦可对照 |

**明确不做 / 延后**: 覆盖默认 `~/.kube/config`；在登录机假 AFS 上写结果。

## 约束

1. **永不覆盖** weibozhen 默认 kubeconfig；只用 `CLUSTER_KUBECONFIG`。
2. 扇出并发有界（默认 ≤6），避免 SSH 踢连接。
3. 结果写 `/afs-a3-weight-share/montyyin/results/` + 本机 `logs/`；**不覆盖**历史目录。
4. 华为与沐曦可同时测；改 muxi 脚本不得破坏华为默认路径。
5. 每完成一项：更新 `GOAL_MUXI_MIGRATE_STATUS.md`，并在 `reports/rounds/` 留报告。

## 执行节奏

- **本会话 / loop**: 按 P0→P1→P2 推进；长任务后台跑，tick 时检查进度并续跑。
- **P0**: G2 体质 + G4 NCCL scale（冒烟 G0/G1 已完成）
- **P1**: G3 拓扑 + G5 P2P + G7 流水线 + G8 MFU 微基准
- **P2**: G6 链路 + G9 真训练 + G10 报告打磨

## 进度指针

见同目录 [`GOAL_MUXI_MIGRATE_STATUS.md`](./GOAL_MUXI_MIGRATE_STATUS.md)。
