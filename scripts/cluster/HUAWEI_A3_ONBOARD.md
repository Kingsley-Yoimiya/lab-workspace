# 华为 A3（一卡两芯 / 8 卡节点）上机教程

面向：**网络、kube、job、镜像名、AFS 路径已经齐了**，只缺「怎么传代码 / 换镜像 / 跑通第一把」的人。

- **组员外发包（可 zip 转发）**：仓库 `docs/huawei-a3-onboard/`（`bash docs/huawei-a3-onboard/pack.sh`）
- **不算本教程**：Clash / lab-gw / Token（见 `docs/PJLAB_TEAM_ONBOARDING.md`）

---

## 0. 你在跑什么拓扑

| 项 | 值 |
|----|-----|
| 机器形态 | Atlas **A3**：每节点 **8 物理卡 × 2 芯 = 16 逻辑 NPU** |
| 典型 job | `huawei-8node-copy`（8 节点 × 16 = **128 NPU**） |
| 跳板 | `ssh weibozhen` |
| kubeconfig（跳板上，独立文件） | `/root/.kube/config.huawei-a3-241ceshi` |
| 真 AFS（华为个人） | `/afs-a3-weight-share/yinjinrun.p-huawei/{lab-workspace,results}` |
| 写盘规则 | 见 [`AFS_LAYOUT.md`](./AFS_LAYOUT.md)（与沐曦同盘 weight-share，华为用 `-huawei` 后缀） |
| 默认训练/筛卡镜像 | `registry2.d.pjlab.org.cn/ccr-yangxiaolei/mindspeed-llm:openeuler22.03-mindspeed-llm-2.3.0-a3-arm` |

本机进入仓库后固定：

```bash
cd /path/to/lab-workspace   # 或 random-thing/project/lab-workspace
source scripts/cluster/huawei.env
./scripts/cluster/switch_kube_context.sh status
./scripts/cluster/job_helpers.sh pods
```

**禁止**：把沐曦的 kubeconfig `cp` 覆盖到 `~/.kube/config`（会踢掉别人的华为会话）。脚本一律走 `CLUSTER_KUBECONFIG`。

---

## 1. 前置检查（5 分钟）

按顺序确认，任一失败先别 sync / 别跑卡。

```bash
# 1) 跳板可达
ssh -o BatchMode=yes -o ConnectTimeout=10 weibozhen 'hostname; which vcctl'

# 2) 华为 kube + job pods
source scripts/cluster/huawei.env
./scripts/cluster/job_helpers.sh pods
# 期望：master-0 + worker-* 均为 Running

# 3) 真 AFS 可写（必须在 pod 里测，登录机上的 /afs-a3-* 是假挂载）
source scripts/cluster/job_helpers.sh
cluster_pod_exec 'ls -la /afs-a3-241ceshi-shared/montyyin; touch /afs-a3-241ceshi-shared/montyyin/.write_test && rm -f /afs-a3-241ceshi-shared/montyyin/.write_test && echo AFS_OK'

# 4) NPU 可见
cluster_pod_exec 'npu-smi info | head -40'
```

假挂载陷阱：在 `weibozhen` 登录容器里直接 `ls /afs-a3-241ceshi-shared` 可能「看得到」，但**写不进 worker 真盘**。传文件、跑实验一律经 `vcctl pod exec`（或本仓库的 `sync_to_afs.sh` / `cluster_pod_exec`）。

---

## 2. 传文件（日常最高频）

### 2.1 全量同步（推荐）

本机 `main` 工作区（含 submodule）→ `ssh weibozhen` → `vcctl pod exec` → 真 AFS：

```bash
source scripts/cluster/huawei.env
./scripts/cluster/sync_to_afs.sh
```

脚本会：

1. 准备/更新 `../lab-workspace-main` worktree（与 ops 分支分离）
2. 拉 submodule（CARD_SCREEN → `montyyin_develop`，Probing_plus → `kingsley/ascend-lab`）
3. `tar` 管道上传到 `$AFS_WORKSPACE`（默认 `/afs-a3-241ceshi-shared/montyyin/lab-workspace`）
4. 远端校验 `README.md` / `projects/CARD_SCREEN` 等

日志默认落在仓库上两级的 `logs/cluster-sync-<时间戳>/`。

### 2.2 只传单个目录 / 单文件（增量）

全量太慢时，用同一管道自己打 tar（仍必须进 pod）：

```bash
source scripts/cluster/huawei.env
source scripts/cluster/job_helpers.sh

# 例：只更新 CARD_SCREEN
export COPYFILE_DISABLE=1
LOCAL_CS=../lab-workspace-main/projects/CARD_SCREEN   # 按你本机路径改
REMOTE_CS="${AFS_CS}"

tar -C "$(dirname "$LOCAL_CS")" -cf - "$(basename "$LOCAL_CS")" \
| cluster_pod_exec_i "$CLUSTER_POD" "mkdir -p '$(dirname "$REMOTE_CS")' && rm -rf '$REMOTE_CS' && tar -xpf - -C '$(dirname "$REMOTE_CS")' && echo OK"
```

单文件：

```bash
cat ./my_script.py | cluster_pod_exec_i "$CLUSTER_POD" "cat > '${AFS_CS}/my_script.py'"
```

### 2.3 结果拉回本机

在 pod 里结果一般在 `$AFS_RESULTS` 或 CARD_SCREEN 下的 `results/`。回拉示例：

```bash
source scripts/cluster/huawei.env
source scripts/cluster/job_helpers.sh
OUT_LOCAL="./logs/pull-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_LOCAL"

cluster_pod_exec_i "$CLUSTER_POD" "tar -C '${AFS_RESULTS}' -cf - ." \
  > /tmp/afs-results.tar
tar -xpf /tmp/afs-results.tar -C "$OUT_LOCAL"
```

AFS 多节点共享，**sync 一次即可**，不必对每个 worker 各传一遍。

---

## 3. 镜像：默认用现成的，还是自己打？

### 3.1 默认：直接用 MindSpeed A3 镜像（多数场景）

`huawei.env` 里的 `CLUSTER_IMAGE` 已是：

```text
registry2.d.pjlab.org.cn/ccr-yangxiaolei/mindspeed-llm:openeuler22.03-mindspeed-llm-2.3.0-a3-arm
```

筛卡 / HCCL / MindSpeed 训练 MFU **通常不需要换镜像**。确认 job 已挂该镜像、pod Running 即可。

进 master 看一眼：

```bash
cluster_pod_exec 'python -c "import torch; import torch_npu; print(torch.__version__)"; npu-smi info | head -20'
```

### 3.2 需要 rust / cargo / Probing 时：两条路

| 方式 | 何时用 | 命令 |
|------|--------|------|
| **A. 装到 AFS**（不换镜像） | 当前 job 立刻要 rust，不想改镜像 | `./scripts/cluster/install_rust_afs.sh` |
| **B. 打带 rust 的镜像** | 长期、多 job 复用 | 见下 |

打镜像（在 weibozhen 上 `docker build`，经本机 Clash 反代拉 rustup）：

```bash
# 本机 Clash 默认 7897；先打通反代
./scripts/cluster/egress_tunnel.sh start
./scripts/cluster/egress_tunnel.sh test

# 只本地 build（不 push）
./scripts/cluster/build_image.sh

# 有 registry 写权限再 push
PUSH=1 ./scripts/cluster/build_image.sh
```

默认产物：

```text
registry2.d.pjlab.org.cn/ccr-yangxiaolei/lab-workspace-env:v0.1.0-rust
基座: registry2.d.pjlab.org.cn/lepton-trainingjob/a3-cann:8.3.rc2-a3-openeuler24.03-py3.11
```

换基座（例如基于 mindspeed）：

```bash
BASE_IMAGE=registry2.d.pjlab.org.cn/ccr-yangxiaolei/mindspeed-llm:openeuler22.03-mindspeed-llm-2.3.0-a3-arm \
  PUSH=1 ./scripts/cluster/build_image.sh
```

### 3.3 把新镜像挂到 job

```bash
source scripts/cluster/huawei.env
source scripts/cluster/job_helpers.sh

# 克隆现有 job 并换镜像（新名字按平台规范改）
cluster_job_clone my-128-rust registry2.d.pjlab.org.cn/ccr-yangxiaolei/lab-workspace-env:v0.1.0-rust

# push 后可选预热（需 pull secret 权限，默认 huawei-dev2）
# 在 weibozhen 上：
#   KUBECONFIG=/root/.kube/config.huawei-a3-241ceshi \
#     vcctl image load -i <FULL_IMAGE> --imagepullsecret huawei-dev2
```

之后用新 job：

```bash
export CLUSTER_JOB_OVERRIDE=my-128-rust
source scripts/cluster/huawei.env   # 重新 source 使 OVERRIDE 生效
./scripts/cluster/job_helpers.sh pods
```

---

## 4. 最小跑通（建议顺序）

### 4.1 Master 单机冒烟（几分钟）

```bash
source scripts/cluster/huawei.env
./scripts/cluster/sync_to_afs.sh          # 若 AFS 上还没有代码
./scripts/cluster/run_card_screen.sh
```

期望日志里出现 `CARD_SCREEN_SMOKE_OK`，远端 `results/<时间戳>-ops-smoke/smoke.jsonl`。

### 4.2 128 卡扇出筛卡

```bash
source scripts/cluster/huawei.env
# 跳板 SSH 易被打爆时降并发（默认华为 profile 已是 4）
CLUSTER_FANOUT_PARALLEL=4 ./scripts/cluster/run_card_screen_128.sh
```

### 4.3 常用战役入口（config 已齐时）

```bash
# HCCL 规模 + 链路
./scripts/cluster/run_hccl_scale.sh
./scripts/cluster/run_link_health.sh

# 体质 128
./scripts/cluster/run_card_constitution_128.sh

# MindSpeed 真训练 MFU 微基准（dense / moe）
MODE=dense SCALES=16 TRAIN_ITERS=5 ./scripts/cluster/run_train_mfu_scale.sh
```

更完整的命令表见同目录 [`README.md`](./README.md)。

---

## 5. 手动进 pod（调试用）

```bash
ssh weibozhen
export KUBECONFIG=/root/.kube/config.huawei-a3-241ceshi
vcctl pod get --job huawei-8node-copy
vcctl pod exec -it huawei-8node-copy-master-0 -- bash

cd /afs-a3-241ceshi-shared/montyyin/lab-workspace/projects/CARD_SCREEN
npu-smi info
python screen.py --device all --sdc-rounds 3 --gemm-n 4096 --sustained-s 10 --out /tmp/smoke.jsonl
```

跳板上若还没有 `vcctl`（arm64）：

```bash
wget http://sdoc.pjlab.org.cn/vcctl-tools/vcctl_linux_arm64
chmod +x vcctl_linux_arm64 && sudo mv vcctl_linux_arm64 /usr/local/bin/vcctl
```

---

## 6. 排障速查

| 现象 | 怎么处理 |
|------|----------|
| `Connection closed by UNKNOWN port` | 降 `CLUSTER_FANOUT_PARALLEL`（试 2～4）；少开并发 SSH |
| 登录机写 AFS「成功」但 worker 看不到 | 假挂载；改走 `sync_to_afs.sh` / `cluster_pod_exec` |
| `ssh weibozhen` / HTTPS 407 | 回 lab-gw / Clash 文档，本教程不管网络 |
| kube Unauthorized / token 过期 | 平台重新下发 kubeconfig，放到跳板上 `config.huawei-a3-241ceshi`，**不要**覆盖默认 `~/.kube/config` |
| 镜像 pull 失败 | 确认 pull secret（如 `huawei-dev2`）、registry 权限；`vcctl image load` |
| 登录机拉 GitHub / rustup 超时 | `./scripts/cluster/egress_tunnel.sh start` 走本机 Clash |
| 误 source 了 `muxi.env` | 再 `source scripts/cluster/huawei.env` 强制切回 |

---

## 7. 你「该有」的东西 checklist

拿到本教程的人通常已经具备：

- [ ] 能 `ssh weibozhen`
- [ ] 跳板上有华为独立 kubeconfig
- [ ] 平台上有 Running 的 `huawei-8node-*` job（或可 clone）
- [ ] AFS 上有自己的目录前缀（如 `montyyin/`）
- [ ] 本机有 `lab-workspace` 仓库 + `scripts/cluster/huawei.env`

你还需要做的通常只有：

1. `source huawei.env` → 检查 pods / 真 AFS  
2. `sync_to_afs.sh`（或增量 tar）  
3. 默认镜像直接跑；要 rust 再 `install_rust_afs` 或 `build_image`  
4. `run_card_screen.sh` → 再上 128 / HCCL / MFU  

脚本索引与双集群约定：[`README.md`](./README.md)。
