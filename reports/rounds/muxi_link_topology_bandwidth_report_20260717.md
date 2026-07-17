# 沐曦 64 节点集群链路拓扑与并发带宽复测报告（2026-07-17）

本文面向网络、交换机和集群平台同事。测试对象为 Volcano 作业 `yinjinrun-cs512-20260716-221823`，共 64 节点、每节点 8 卡。2026-07-17 15:23 再次确认作业状态为 `Running`，`MINAVAILABLE=64`、`RUNNINGS=64`；本报告没有删除或重启该作业。

口径说明：

- `bus_bw_GBps` 是 NCCL-tests 同构的总线带宽：AllReduce 的算法带宽乘 `2(n-1)/n`；表中 scale 数值取各 rank 的中位数。
- P2P `bw_GBps` 是 `torch.distributed.isend/irecv` 传输 16 MiB 或 256 MiB 数据后，以字节数除以平均耗时得到的单向有效带宽。
- incast 的 `per_sender_bw_GBps` 是多个发送者到同一接收者时，每个发送者的有效带宽；并发汇总带宽另列，不能把二者混读。
- `GB/s` 按十进制 `1e9 B/s` 计算；消息大小中的 `M`/`MiB` 按脚本实际的二进制字节数计算。

## 1. 机器与链路排布（topo 原文/表 + 文字）

### 1.1 作业与主节点

```text
NAME                              STATUS    MINAVAILABLE   RUNNINGS   AGE   QUEUE
yinjinrun-cs512-20260716-221823   Running   64             64         17h   default

master pod: yinjinrun-cs512-20260716-221823-master-0
pod IP:     10.120.19.4
node:       host-10-12-144-137
```

### 1.2 主节点 `mx-smi topo -m` 原文

抓取时间：`2026-07-17T15:23:05+08:00`，`mx-smi 2.2.9`。

```text
Attached GPUs : 8
Device link type matrix
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    Node Affinity  CPU Affinity
GPU0    X       MX      MX      MX      MX      MX      MX      MX      0              0-15,128-143
GPU1    MX      X       MX      MX      MX      MX      MX      MX      0              0-15,128-143
GPU2    MX      MX      X       MX      MX      MX      MX      MX      2              32-47,160-175
GPU3    MX      MX      MX      X       MX      MX      MX      MX      2              32-47,160-175
GPU4    MX      MX      MX      MX      X       MX      MX      MX      4              64-79,192-207
GPU5    MX      MX      MX      MX      MX      X       MX      MX      4              64-79,192-207
GPU6    MX      MX      MX      MX      MX      MX      X       MX      6              96-111,224-239
GPU7    MX      MX      MX      MX      MX      MX      MX      X       6              96-111,224-239
```

`MX` 表示卡间经过 MetaXLink。8 卡两两均显示 `MX`，不是经 CPU/PCIe Host Bridge 绕行。CPU/NUMA 亲和分为 4 对：GPU0–1 对应 NUMA 0，GPU2–3 对应 NUMA 2，GPU4–5 对应 NUMA 4，GPU6–7 对应 NUMA 6。

### 1.3 主节点 4 个 RoCE 口

同一时刻从 `/sys/class/infiniband/xscale_*/ports/1/` 读取：

| HCA | state | phys_state | rate | link_layer |
|---|---|---|---:|---|
| xscale_0 | ACTIVE | LinkUp | 200 Gb/s | Ethernet |
| xscale_1 | ACTIVE | LinkUp | 200 Gb/s | Ethernet |
| xscale_2 | ACTIVE | LinkUp | 200 Gb/s | Ethernet |
| xscale_3 | ACTIVE | LinkUp | 200 Gb/s | Ethernet |

此前 16 节点抽样覆盖 `16 × 4 = 64` 个 xscale 口，结果也是 64/64 `ACTIVE`、64/64 `LinkUp`、64/64 `200 Gb/s`。因此静态口状态不支持“端口掉线或协商降速”解释。

AR64 的 MCCL INFO 原文进一步确认数据面：

```text
MCCL INFO Using network IB
MCCL INFO NET/IB : Using [0]xscale_0:1/RoCE [1]xscale_1:1/RoCE
                    [2]xscale_2:1/RoCE [3]xscale_3:1/RoCE [RO];
                    OOB eth0:10.120.19.4<0>
MCCL INFO Channel ... [send] via NET/IB/1/GDRDMA
MCCL_SOCKET_IFNAME set by environment to eth0
MCCL_IB_HCA set to xscale_0,xscale_1,xscale_2,xscale_3
```

这里 `eth0` 是 OOB/控制面 socket；大数据传输明确使用 `xscale_0..3/RoCE` 和 `NET/IB/*/GDRDMA`，不是误走 eth0 数据慢路径。

## 2. 复测/复现脚本与命令（完整可复制，含环境变量）

### 2.1 公共环境和拓扑抓取

所有命令从本机执行，工作目录固定为 `~/random-thing/project/lab-workspace`。不要使用 `ais vcctl upload`。

```bash
cd ~/random-thing/project/lab-workspace

export HTTPS_PROXY=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export https_proxy="$HTTPS_PROXY"
export http_proxy="$HTTP_PROXY"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export KUBECONFIG="$HOME/.kube/config-vc-c550-h3c-test.yaml"
export CLUSTER_JOB_OVERRIDE=yinjinrun-cs512-20260716-221823
source scripts/cluster/muxi.env

JOB="$CLUSTER_JOB_OVERRIDE"
POD="${JOB}-master-0"
LOCAL_OUT="$HOME/Codespace/myportal/results/muxi-h3c/20260717_link-topo-bw-report"
mkdir -p "$LOCAL_OUT"

kubectl get vcjob "$JOB" -o wide | tee "$LOCAL_OUT/job_status.txt"
kubectl exec "$POD" -- mx-smi topo -m | tee "$LOCAL_OUT/master_topo.txt"
kubectl exec "$POD" -- bash -lc '
for d in xscale_0 xscale_1 xscale_2 xscale_3; do
  p=/sys/class/infiniband/$d/ports/1
  echo "=== $d ==="
  printf "state=";      sed "s/[[:space:]]*$//" "$p/state"
  printf "phys_state="; sed "s/[[:space:]]*$//" "$p/phys_state"
  printf "rate=";       sed "s/[[:space:]]*$//" "$p/rate"
  printf "link_layer="; sed "s/[[:space:]]*$//" "$p/link_layer"
done
' | tee "$LOCAL_OUT/master_xscale_rate.txt"
```

### 2.2 AllReduce 规模曲线复现

完整 w8→w512 已于 `20260717_103913` 复现，不应在仍占用 512 卡时无目的重复。以下是原始点火入口；每个 world 必须用不同端口。脚本会在目标 pod 的 `/tmp` 写 launcher，并设置生产 RoCE 环境。

```bash
cd ~/random-thing/project/lab-workspace
export HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897
export https_proxy="$HTTPS_PROXY" http_proxy="$HTTP_PROXY"
export NO_PROXY=127.0.0.1,localhost no_proxy="$NO_PROXY"
export KUBECONFIG="$HOME/.kube/config-vc-c550-h3c-test.yaml"
export CLUSTER_JOB_OVERRIDE=yinjinrun-cs512-20260716-221823
export CLUSTER_N_WORKERS=63
source scripts/cluster/muxi.env

STAMP=$(date +%Y%m%d_%H%M%S)
export AFS_OUT="/afs-a3-weight-share/yinjinrun.p/results/muxi-ib-rerun-${STAMP}"
export LOG_DIR="$HOME/random-thing/logs/muxi-ib-rerun-${STAMP}/comm"
export OPS=all_reduce
export SIZES=256M
mkdir -p "$LOG_DIR"

# 生产 RoCE 参数由 fire_nccl_scale_muxi.sh 固化到远端 launcher：
# xscale_0..3，GID index 5，MCCL_IB_TC=128，MCCL_ENABLE_VSWITCH=1，
# NCCL/MCCL/GLOO_SOCKET_IFNAME=eth0。
# 轻量复现可只跑 w8、w16；不要与其他 torchrun 共用端口。
bash scripts/cluster/fire_nccl_scale_muxi.sh 8  29781
# 等 scale_8.node_0.done 出现后再跑下一规模
bash scripts/cluster/fire_nccl_scale_muxi.sh 16 29782
```

w512 必须用本机 kubectl 并行点火，已用脚本：

```bash
PARALLEL=16 \
bash ~/random-thing/logs/muxi-ib-rerun-20260717_103913/fire512_parallel.sh 29774
```

原始结果在 `$AFS_OUT/scale_<world>.jsonl`，rank 完成标记为 `scale_<world>.node_<node_rank>.done`。本次完整复现的固定 AFS 路径为：

```text
/afs-a3-weight-share/yinjinrun.p/results/muxi-ib-rerun-20260717_103913/
```

### 2.3 Incast 8 节点 @16 MiB 轻量复测脚本

测试程序是 `~/random-thing/logs/muxi-congest-ab-20260717/incast_ab.py`。它用 `torch.distributed` 的 `isend/irecv`：serial 阶段让 7 个发送者依次向 rank0 发，concurrent 阶段让 7 个发送者同时向 rank0 发；默认 warmup=3、iters=20。以下脚本可整体保存为 `/tmp/fire_incast8.sh` 后执行，预计 2～5 分钟。

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/random-thing/project/lab-workspace"

export HTTPS_PROXY=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export https_proxy="$HTTPS_PROXY" http_proxy="$HTTP_PROXY"
export NO_PROXY=127.0.0.1,localhost no_proxy="$NO_PROXY"
export KUBECONFIG="$HOME/.kube/config-vc-c550-h3c-test.yaml"
export CLUSTER_JOB_OVERRIDE=yinjinrun-cs512-20260716-221823
source scripts/cluster/muxi.env

JOB="$CLUSTER_JOB_OVERRIDE"
WORLD=8
PORT="${PORT:-29818}"
STAMP=$(date +%Y%m%d_%H%M%S)
AFS_OUT="/afs-a3-weight-share/yinjinrun.p/results/muxi-incast8-${STAMP}"
AFS_SCRIPT="/afs-a3-weight-share/yinjinrun.p/lab-workspace/scripts/cluster/incast_ab.py"
LOCAL_OUT="$HOME/Codespace/myportal/results/muxi-h3c/${STAMP}-incast8-retest"
mkdir -p "$LOCAL_OUT"

MASTER_POD="${JOB}-master-0"
MASTER_ADDR=$(kubectl get pod "$MASTER_POD" -o jsonpath='{.status.podIP}')
kubectl exec "$MASTER_POD" -- mkdir -p "$AFS_OUT"
kubectl exec -i "$MASTER_POD" -- bash -c "cat > '$AFS_SCRIPT'" \
  < "$HOME/random-thing/logs/muxi-congest-ab-20260717/incast_ab.py"

fire_one() {
  r="$1"
  if [ "$r" -eq 0 ]; then pod="${JOB}-master-0"; else pod="${JOB}-worker-$((r-1))"; fi
  kubectl exec "$pod" -- bash -lc "
    cat > /tmp/run_incast8_${PORT}_${r}.sh <<'REMOTE'
#!/usr/bin/env bash
set -e
export PATH=/opt/conda/bin:\$PATH
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0 MCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=xscale_0,xscale_1,xscale_2,xscale_3
export MCCL_IB_HCA=xscale_0,xscale_1,xscale_2,xscale_3
export NCCL_IB_GID_INDEX=5 MCCL_IB_GID_INDEX=5
export MCCL_IB_TC=128 MCCL_ENABLE_VSWITCH=1 MCCL_PCIE_BUFFER_MODE=0
export FORCE_ACTIVE_WAIT=2
cp '$AFS_SCRIPT' /tmp/incast_ab.py
/opt/conda/bin/torchrun --nnodes=$WORLD --node_rank=$r --nproc_per_node=1 \
  --master_addr=$MASTER_ADDR --master_port=$PORT \
  /tmp/incast_ab.py --nbytes 16777216 --warmup 3 --iters 20 \
  --out '$AFS_OUT/inc' > '$AFS_OUT/node_${r}.log' 2>&1
echo OK > '$AFS_OUT/node_${r}.done'
REMOTE
    chmod +x /tmp/run_incast8_${PORT}_${r}.sh
    setsid nohup /tmp/run_incast8_${PORT}_${r}.sh </dev/null >/dev/null 2>&1 &
  "
}
export -f fire_one
export JOB WORLD PORT MASTER_ADDR AFS_OUT AFS_SCRIPT KUBECONFIG
export HTTPS_PROXY HTTP_PROXY https_proxy http_proxy NO_PROXY no_proxy
seq 0 7 | xargs -P 8 -I{} bash -c 'fire_one "$@"' _ {}

# 最多等 5 分钟；若 done 数不增长，立即检查 node_*.log，不要盲等。
for _ in $(seq 1 60); do
  n=$(kubectl exec "$MASTER_POD" -- bash -lc \
    "ls '$AFS_OUT'/node_*.done 2>/dev/null | wc -l")
  echo "done=$n/8"
  [ "$n" -eq 8 ] && break
  sleep 5
done

# rank0 JSONL 含 serial/concurrent 两条原始记录；同时备份所有节点日志。
kubectl exec "$MASTER_POD" -- bash -lc "cat '$AFS_OUT/inc.rank0.jsonl'" \
  > "$LOCAL_OUT/inc.rank0.jsonl"
kubectl exec "$MASTER_POD" -- tar -C "$AFS_OUT" -cf - . \
  | tar -C "$LOCAL_OUT" -xf -
printf 'AFS=%s\n' "$AFS_OUT" | tee "$LOCAL_OUT/AFS_PATH.txt"
```

历史实测固定路径：

```text
/afs-a3-weight-share/yinjinrun.p/results/muxi-incast8-20260717_144014
/afs-a3-weight-share/yinjinrun.p/results/muxi-incast16-20260717_144048
```

## 3. 测量数据（表格：实验名、规模、消息大小、med/min/max、路径/文件）

### 3.1 AllReduce 规模曲线原始汇总

单位均为 `bus_bw_GBps`。`keep` 相对同轮 w8 中位数 258.587 GB/s。

| 实验名 | 规模 | 消息 | med | min | max | keep | 路径/文件 |
|---|---:|---:|---:|---:|---:|---:|---|
| AllReduce | w8（1 节点） | 256M | 258.587 | 258.214 | 259.919 | 100.0% | 本地 `20260717_103913-ib-rerun/RERUN_SUMMARY.json`；AFS `.../muxi-ib-rerun-20260717_103913/scale_8.jsonl` |
| AllReduce | w16（2 节点） | 256M | 78.006 | 77.858 | 78.167 | 30.2% | 同上，`scale_16.jsonl` |
| AllReduce | w32（4 节点） | 256M | 55.996 | 55.961 | 56.011 | 21.7% | 同上，`scale_32.jsonl` |
| AllReduce | w64（8 节点） | 256M | 37.990 | 37.972 | 38.371 | 14.7% | 同上，`scale_64.jsonl` |
| AllReduce | w128（16 节点） | 256M | 32.988 | 32.981 | 33.001 | 12.8% | 同上，`scale_128.jsonl` |
| AllReduce | w256（32 节点） | 256M | 13.043 | 13.039 | 13.072 | 5.0% | 同上，`scale_256.jsonl` |
| AllReduce | w512（64 节点） | 256M | 7.643 | 7.642 | 7.757 | 3.0% | 同上，`scale_512.jsonl` |

AR64+INFO 独立检查得到 med=36.550 GB/s，与规模曲线 w64=37.990 GB/s 接近；该次摘要没有单独落 min/max，不能凭空补值。路径：AFS `.../muxi-ar64-info-20260717_140105/`，本地 `20260717_140105-ar64-info/SUMMARY.txt`。

### 3.2 P2P 与并发 pair

| 实验名 | 规模 | 消息 | med | min | max | keep | 路径/文件 |
|---|---:|---:|---:|---:|---:|---:|---|
| P2P64 跨机边 | 16 条跨节点记录 | 16M | 23.563 | 22.341 | 23.677 | — | AFS `.../muxi-p2p64-link-20260717_135927/p2p_64.jsonl`；本地 `20260717_135927-p2p64/p2p_64.jsonl` |
| P2P64 机内边 | 112 条机内记录 | 16M | 33.567 | 0.631 | 34.670 | — | 同上；注意 min=0.631 是实测离群点 |
| 不相交 pair serial | 4 对/8 节点 | 16M | 23.846 | 20.755 | 23.947 | 基准 | AFS `.../muxi-congest-ab-20260717_143732/`；本地摘要 `AB_SUMMARY_20260717_143732.txt` |
| 不相交 pair concurrent | 同时 4 对 | 16M | 17.856 | 17.856 | 17.858 | 74.9% | 同上 |
| 不相交 pair serial | 8 对/16 节点 | 16M | 23.809 | 21.433 | 23.859 | 基准 | AFS `.../muxi-congest-ab16-20260717_143816/`；本地摘要 `AB16_SUMMARY_20260717_143816.txt` |
| 不相交 pair concurrent | 同时 8 对 | 16M | 23.753 | 23.744 | 23.757 | 99.8% | 同上 |
| 不相交 pair serial | 16 对/32 节点 | 256M | 24.396 | 12.881 | 24.407 | 基准 | AFS `.../muxi-congest-ab32-256M-20260717_143906/`；本地摘要 `AB32_256M_SUMMARY_20260717_143906.txt` |
| 不相交 pair concurrent | 同时 16 对 | 256M | 24.305 | 24.299 | 24.307 | 99.6% | 同上 |

### 3.3 Incast 原始数字

下表 `med/min/max` 对应本轮 rank0 记录中的“每发送者有效带宽”。每个 phase 只有一条聚合记录，因此 med=min=max；这不是多轮分布统计。

| 实验名 | 规模 | 每发送者消息 | phase | med=min=max | 汇总入口带宽 | keep | 路径/文件 |
|---|---:|---:|---|---:|---:|---:|---|
| Incast | 7→1（8 节点） | 16M | serial | 15.813 | 15.813 | 基准 | AFS `.../muxi-incast8-20260717_144014/inc.rank0.jsonl`；本地 `20260717_fabric-congestion-evidence/muxi-incast8-20260717_144014.inc.rank0.jsonl` |
| Incast | 7→1（8 节点） | 16M | concurrent | 3.468 | 24.276 | 21.9% | 同上 |
| Incast | 15→1（16 节点） | 16M | serial | 15.168 | 15.168 | 基准 | AFS `.../muxi-incast16-20260717_144048/inc.rank0.jsonl`；本地 `20260717_fabric-congestion-evidence/muxi-incast16-20260717_144048.inc.rank0.jsonl` |
| Incast | 15→1（16 节点） | 16M | concurrent | 1.615 | 24.230 | 10.6% | 同上 |

## 4. 现象与分析（对着数字说，指出奇怪点）

1. **第一次跨机就发生最大一级断崖。** AllReduce 从机内 w8 的 258.587 GB/s 降到 w16 的 78.006 GB/s，`8→16 keep=30.2%`，下降 69.8%。继续扩到 w64 只剩 37.990 GB/s（14.7%），w512 只剩 7.643 GB/s（3.0%）。

2. **同一 world 的 rank 分布极窄，不能用“一两张坏卡拖慢全局”解释。** 例如 w256 是 13.039～13.072 GB/s，跨度仅 0.033；w512 是 7.642～7.757 GB/s，跨度 0.115。下降随规模呈系统性，而非少量 rank 离群。

3. **静态链路健康但集合通信差。** 16 节点抽样的 64/64 个口都为 ACTIVE/LinkUp/200 Gb/s；主节点复抓 4/4 仍为 200 Gb/s。AR64 日志明确走 4×xscale RoCE/GDRDMA，但 med 只有 36.550 GB/s。这排除了“端口挂了”“协商成低速”“大数据误走 eth0”三种直接解释。

4. **不相交 P2P 并发大多能保住单流速度。** 8 对 @16M 从 serial 23.809 到 concurrent 23.753 GB/s，keep=99.8%；16 对 @256M 从 24.396 到 24.305，keep=99.6%。因此“只要跨机流一多就必然掉到 10% 以下”不成立。4 对 @16M 是异常较弱的一组，23.846→17.856，keep=74.9%，但仍远好于 incast 15→1 的 10.6%。

5. **共享接收端的 incast 能稳定复现严重下降。** 7→1 时，每发送者 15.813→3.468 GB/s，keep=21.9%；15→1 时，15.168→1.615 GB/s，keep=10.6%。发送者从 7 增至 15 后，并发汇总入口带宽几乎不变（24.276→24.230 GB/s），但人均带宽近似按发送者数继续摊薄（3.468→1.615）。这说明瓶颈集中在共享汇聚资源，而不是每个源端独立链路。

6. **奇怪点必须保留。** P2P64 的 112 条机内记录虽然中位数 33.567 GB/s，但 min=0.631 GB/s；4 对并发组和 16 对 serial 组也分别出现 20.755、12.881 GB/s 的低值。它们说明数据里存在瞬时离群或测量调度干扰，但不改变中位数对照：不相交并发约 24 GB/s，incast 人均降到 1.6～3.5 GB/s。后续若追单卡问题，应单独复测离群 pair，不能把它与规模性汇聚瓶颈混成一个问题。

现有证据与“交换网络汇聚方向出现拥塞或等价的共享资源限速”一致，候选机制包括 PFC pause 扩散、ECN/CNP、共享 buffer 打满、ECMP 多流碰撞或多 rail 未均匀利用。但容器内 xscale 没有可用的 sysfs counters 目录，本报告没有交换机队列、PFC、FEC、drop 计数，因此不能仅凭端到端带宽断言具体是哪一种机制。

## 5. 结论与给网络侧的问题清单

结论：端口静态状态和 MCCL 路径正常；单跨机流及不相交多流可达到约 23～24 GB/s；真正能复现数量级下降的是共享汇聚形态。15→1 incast 人均只剩 1.615 GB/s（10.6%），与 AllReduce 随规模扩展到 w512 后只剩机内基线 3.0% 的方向一致。优先排查 fabric/交换机汇聚路径，而不是先更换个别 GPU 或 xscale 口。

请网络侧针对上述固定作业、固定时间窗回答并导出原始计数：

1. 运行 7→1、15→1 incast 的 10～30 秒窗口内，接收节点所在 leaf 端口、leaf-spine 上行是否出现 PFC pause TX/RX、ECN mark、CNP、queue drop、buffer occupancy 峰值？
2. 4 个 rail（`xscale_0..3`）分别映射到哪些 ToR/leaf/spine 端口？是否真实独立，还是最终汇聚到同一上行或共享 buffer？
3. `MCCL_IB_TC=128` 在交换网络映射到哪个 DSCP/priority/PG？该 PG 的 PFC 阈值、headroom 和 ECN 阈值是多少？是否存在 pause 风暴？
4. 15→1 时汇总带宽稳定在约 24.23 GB/s，这是否对应某个单端口 200 Gb/s 的有效上限，还是某个交换机队列/策略限速？请给出该接收节点 4 个 xscale 口的逐口流量。
5. w256→w512 的 AllReduce med 从 13.043 再降到 7.643 GB/s。新增的 32 个节点跨越了哪些 leaf/spine 或故障域？是否出现 ECMP 哈希集中到少数链路？
6. 请从宿主机或厂商工具导出 xscale 的 CRC/FEC corrected/FEC uncorrected/symbol error/retry 计数；pod 内无法读取这些计数，当前只有 ACTIVE/LinkUp/200G 证据。
7. 请保留并对齐以下时间戳的交换机遥测：`20260717_135927`（P2P64）、`20260717_140105`（AR64+INFO）、`20260717_143816`（8 对并发）、`20260717_144014`（7→1）、`20260717_144048`（15→1）。

## 附录：关键文件路径

报告主文件：

```text
~/random-thing/project/lab-workspace/reports/rounds/muxi_link_topology_bandwidth_report_20260717.md
```

本报告备份与 15:23 复抓原文：

```text
~/Codespace/myportal/results/muxi-h3c/20260717_link-topo-bw-report/
  muxi_link_topology_bandwidth_report_20260717.md
  job_status.txt
  master_topo.txt
  master_xscale_rate.txt
```

本机已有原始数据/摘要：

```text
~/Codespace/myportal/results/muxi-h3c/20260717_103913-ib-rerun/RERUN_SUMMARY.json
~/Codespace/myportal/results/muxi-h3c/20260717_135116-link-sample/
~/Codespace/myportal/results/muxi-h3c/20260717_135927-p2p64/
~/Codespace/myportal/results/muxi-h3c/20260717_140105-ar64-info/
~/Codespace/myportal/results/muxi-h3c/20260717_fabric-congestion-evidence/
```

AFS 原始产物：

```text
/afs-a3-weight-share/yinjinrun.p/results/muxi-ib-rerun-20260717_103913/
/afs-a3-weight-share/yinjinrun.p/results/muxi-link-sample-20260717_135116/
/afs-a3-weight-share/yinjinrun.p/results/muxi-p2p64-link-20260717_135927/
/afs-a3-weight-share/yinjinrun.p/results/muxi-ar64-info-20260717_140105/
/afs-a3-weight-share/yinjinrun.p/results/muxi-congest-ab-20260717_143732/
/afs-a3-weight-share/yinjinrun.p/results/muxi-congest-ab16-20260717_143816/
/afs-a3-weight-share/yinjinrun.p/results/muxi-congest-ab32-256M-20260717_143906/
/afs-a3-weight-share/yinjinrun.p/results/muxi-incast8-20260717_144014/
/afs-a3-weight-share/yinjinrun.p/results/muxi-incast16-20260717_144048/
```

脚本：

```text
~/random-thing/project/lab-workspace/scripts/cluster/fire_nccl_scale_muxi.sh
~/random-thing/project/lab-workspace/scripts/cluster/nccl_torch_bench.py
~/random-thing/logs/muxi-ib-rerun-20260717_103913/fire512_parallel.sh
~/random-thing/logs/muxi-congest-ab-20260717/concurrent_pair_bw.py
~/random-thing/logs/muxi-congest-ab-20260717/incast_ab.py
```
