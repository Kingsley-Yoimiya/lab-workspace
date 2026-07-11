# Muxi（沐曦）128 卡冒烟适配计划

## 目标

华为 Ascend 上已跑通的 CARD_SCREEN 冒烟，转写到 muxi，并与华为集群**长期并存**（不互相覆盖 kubeconfig）。

## 第一性原理

1. 筛卡逻辑与厂商无关；差异在 backend 探测与 `*-smi`。
2. 沐曦走 CUDA 兼容层 + `mx-smi`（`MetaxAdapter`）。
3. 阻塞项在运维面：kube 隔离、job/AFS、扇出并发。
4. 跳板共用 weibozhen，**两套独立 KUBECONFIG**。

## 双集群协调（已落地）

| 规则 | 说明 |
|------|------|
| 禁止覆盖 | 脚本不得 `cp` 改写 `~/.kube/config` |
| 独立文件 | `config.huawei-a3-241ceshi` / `config.muxi-mohe` |
| 切换方式 | `source huawei.env` 或 `muxi.env`（强制赋值 `CLUSTER_KUBECONFIG`） |
| 默认 config | 保持华为，供其他会话 |
| 扇出并发 | muxi 默认 6，避免 SSH 被踢 |

## 状态（2026-07-11）

- [x] muxi job 连通，16×8 MetaX C550
- [x] master 冒烟 + 128 卡扇出齐套
- [x] 结果：good=106 / slow=19 / contended=2 / bad=1；中位 ~240.5 TFLOPS / ~1487 GB/s
- [x] 运维脚本双 profile + 本地 CARD_SCREEN Metax 接线
- [ ] NCCL/MACA 通信线（非冒烟阻塞）

详见 `reports/rounds/muxi_smoke_20260711.md`。
