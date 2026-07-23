# probing-failslow — 沐曦 Fail-Slow 实验编排

从 `myportal/tmp/baseline-compare-bundle/` 回收的可跑脚本（2026-07-23）。

## 入口

| 脚本 | 作用 |
|---|---|
| `provision_priv_pods.sh` | 起/删 privileged 整机 pod；`deploy_local` 铺代码到 pod 本地 |
| `run_case_pipeline_v4.sh` | 单 case：C0/C1/C2…，支持多节点参数 |
| `run_case_abc.sh` | 论文最小 A/B/C：基线、注入、注入+Probing，并立即回拉 |
| `run_campaign.sh` | 多 case 战役 |
| `pull_campaign_results.sh` | 从 pod 回拉结果 |
| `train_bench_probe.py` | GPT-2 124M（默认；tiny 仅回退）+ within-step 计时 |
| `sidecar_inject_v2.py` | 外部注入（cube/hbm/stress/…） |
| `collect.py` | 离线定位汇总 |

## 落盘约定

见 myportal：`plans/probing-experiment-layout.md`。

- **不依赖 AFS**（raw pod 挂 weight-share 实测不可靠）；默认 pod 本地 + 回拉本机 `results/muxi-mohe|muxi-h3c/<run_id>/`。
- 顺序：先 mohe-241，再 h3c-test。
- 调频：先 `mount -o remount,rw /sys`；用完恢复 `xcore,9` / `mc,3`；删 pod 前 `provision_priv_pods.sh delete` 会尝试复位。

## mohe 单 case（16 卡）

先以 `NODES=... RUN_ID=... KUBECONFIG=... bash provision_priv_pods.sh apply` 起 pod；
随后不传子命令运行 `provision_priv_pods.sh` 即默认 `deploy_local`。论文的 A/B/C 入口：

```bash
CASE_ID=P1-EXT-A RUN_ID=20260723_180000-p1exta \
PODS=yjr-case-h...,yjr-case-h... \
KUBECONFIG="$HOME/.kube/config-vc-c550-mohe-241.yaml" \
bash scripts/probing-failslow/run_case_abc.sh
```

默认 `NNODES=2 NPROC=8 ITERS=500 WARMUP=50 MODEL=gpt2 SEQ=1024 BATCH=8`。
P1-EXT-A/`3a` 映射为 `cube,duty=0.8`，P1-EXT-B/`3b` 映射为 `hbm,duty=0.8`。
训练 warmup 完成后在第 100 个测量步起 sidecar（总 step 150），第 300 个测量步停（总 step 350）；sidecar 自身先连续预热 5 秒。目标卡由 `SIDECAR_LOCAL_RANK`（默认 7）覆盖。结果从 pod 本地
`/workspace/probe-bundle/out/<case>/` 回拉至 `results/muxi-mohe/<RUN_ID>/<CASE>/by_pod/`。
Greyhound/XPUTimer 在此环境标为 `ENV-BLOCKED`，不纳入运行集合。

## 相关文档

- SOP：`plans/probing-experiment-sop.md`
- 待补清单：`plans/probing-experiment-open-questions.md`
- Privilege：`docs/muxi/privileged-freq-nic-guide.md`
