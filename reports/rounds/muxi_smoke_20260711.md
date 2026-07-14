# Muxi 128 卡冒烟结果（2026-07-11）

## 运维面（双集群协调）

| 项 | 做法 |
|----|------|
| kube 隔离 | `CLUSTER_KUBECONFIG` 指向独立文件；**禁止**覆盖 `~/.kube/config` |
| 华为默认 | 跳板默认 config 保持 `vc-a3-241ceshi` |
| 沐曦 | `/root/.kube/config.muxi-mohe` → `vc-c550-mohe-241` |
| 切换 | `source scripts/cluster/{huawei,muxi}.env`（强制赋值，可来回切） |
| 扇出并发 | `CLUSTER_FANOUT_PARALLEL=6`（16 路曾导致 SSH `Connection closed`） |

## 集群与代码

- Job: `yushan-muxi-card-screen-128-cp-copy`（16 pods Running）
- 卡: MetaX C550-PL，每节点 8 卡，目标 128
- 代码: AFS `/afs-a3-weight-share/yushan/CARD_SCREEN`（`backend=metax` + `mx-smi`）
- 本地已移植: `MetaxAdapter` 接线、`MxSmiProvider`、`launch_vcctl.py`

## 冒烟参数

```text
--device all --sdc-rounds 3 --gemm-n 4096 --sustained-s 10
```

## 结果摘要（128 卡完整）

本地日志: `logs/muxi-card-screen-20260711_133828-muxi-smoke/`  
AFS: `/afs-a3-weight-share/montyyin/results/card_screen-20260711_133828-muxi-smoke/`

| 指标 | 值 |
|------|-----|
| 节点 / 卡 | **16 / 128** |
| 汇总判定 | **good=106, slow=19, contended=2, bad=1** |
| 中位 func_tflops | **240.5** |
| 中位 hbm_gbps | **1487.6** |
| 中位 sustained_tflops | **260.7** |

首轮 16 路并行时 worker-3/9/13 SSH 被踢；限流补跑后齐套。

### 明确问题卡

| Host | Dev | Verdict | 原因 |
|------|-----|---------|------|
| worker-12 | 0 | **bad** | GEMM correctness `max_rel_err=0.0762` |
| worker-0 | 5 | **contended** | `perf_quality:timing_flatline` |
| worker-9 | 3 | **contended** | `perf_quality:timing_flatline` |

### 慢卡模式（19 张，多为 HBM ~1040–1050 GB/s vs 中位 1487）

- **整机偏慢**: worker-14（8/8）、worker-7（8/8）
- **部分慢**: worker-0 的 dev0/3/6
- slow-cause 多为 `intrinsic` 或 `power_cap`（满载贴近 550W 墙）

相对华为 Ascend 冒烟：沐曦中位算力略低（~240 vs ~260 func TFLOPS），HBM 中位更高（~1487 vs ~1250 GB/s），但存在整节点 HBM 掉速簇。

## 代码适配清单（已完成）

- [x] `job_helpers.sh`：`CLUSTER_KUBECONFIG` + 有界扇出辅助
- [x] `huawei.env` / `muxi.env`：强制 profile 赋值
- [x] `switch_kube_context.sh`：只 status/ensure，不再覆盖
- [x] `run_card_screen_muxi.sh`：动态 pods + 限并发
- [x] 本地 CARD_SCREEN：接入 Metax 检测与 `mx-smi`
- [x] 128 卡冒烟齐套并落盘
- [ ] 通信线（NCCL/MACA）另开，冒烟不阻塞

## 复跑命令

```bash
source scripts/cluster/muxi.env
./scripts/cluster/switch_kube_context.sh ensure
CLUSTER_FANOUT_PARALLEL=6 ./scripts/cluster/run_card_screen_muxi.sh all

# 补跑失败节点
PREV_RUN=<run_id> FAIL_FILE=logs/muxi-card-screen-<run_id>/fail_pods.txt \
  ./scripts/cluster/run_card_screen_muxi.sh retry
```
