# 沐曦 512 卡 AllReduce 带宽异常：网络侧交接材料（2026-07-17）

作业：`yinjinrun-cs512-20260716-221823`（64 节点 × 8 卡，仍 Running）  
诉求：确认拓扑、复现 AllReduce@512 带宽过低，请网络侧查交换机/fabric 配置。

---

## 1. 拓扑结构（节点侧能确认的）

```text
每台机器
├─ 8× GPU（MXC550），机内两两 MetaXLink（mx-smi 全 MX）
│    NUMA：GPU0-1@0，GPU2-3@2，GPU4-5@4，GPU6-7@6
├─ 机间：4× xscale RoCE，每口 200Gb/s（ACTIVE / LinkUp）
│    生产 env：xscale_0..3，GID_INDEX=5，VSWITCH=1，IB_TC=128
│    控制面：eth0（OOB）；数据面 MCCL 日志确认 RoCE + GDRDMA
└─ 64 台经交换机互联（leaf/spine、超分比、PFC——节点上看不见，需网络侧提供）
```

主节点 topo 快照（本地）：  
`myportal/results/muxi-h3c/20260717_link-topo-bw-report/master_topo.txt`  
`master_xscale_rate.txt`

交换机侧我们**没有**端口级拓扑图；需要网络同事补：主机上联、ECMP、PFC/优先级、缓冲与超分。

---

## 2. 本次 512 卡 AllReduce 复测结果（刚跑完）

| 项 | 值 |
|----|-----|
| 算子 | `all_reduce` |
| 消息 | `256M`（268435456 B） |
| world | 512（64×8） |
| RoCE env | 四轨生产配置（见下） |
| **中位 bus_bw** | **5.025 GB/s** |
| min / max | 5.024 / 5.036（几乎无离群） |
| 相对机内 w8≈258.6 | **keep ≈ 1.9%** |

对照：今日早些时候同作业复现为 med≈**7.64 GB/s**（keep≈3.0%）；昨夜≈**4.47 GB/s**。  
→ **三次都在个位数 GB/s，形态稳定，不是偶发。**

产物：

- AFS：`/afs-a3-weight-share/yinjinrun.p/results/muxi-ar512-retest-20260717_155017/`
- 本地：`myportal/results/muxi-h3c/20260717_ar512-retest/RETEST_SUMMARY.json`

规模曲线（同日完整复现，供对照）：

| world | med GB/s | keep% |
|------:|---------:|------:|
| 8 | 258.6 | 100 |
| 16 | 78.0 | 30 |
| 64 | 38.0 | 15 |
| 256 | 13.0 | 5 |
| 512 | 7.64→**本次 5.02** | 3→**1.9** |

---

## 3. 复现命令（可复制）

```bash
cd ~/random-thing/project/lab-workspace
export HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897
export NO_PROXY=127.0.0.1,localhost
export KUBECONFIG=$HOME/.kube/config-vc-c550-h3c-test.yaml
JOB=yinjinrun-cs512-20260716-221823
MASTER_ADDR=$(kubectl get pod ${JOB}-master-0 -o jsonpath='{.status.podIP}')
STAMP=$(date +%Y%m%d_%H%M%S)
AFS=/afs-a3-weight-share/yinjinrun.p/results/muxi-ar512-retest-${STAMP}
PORT=29951   # 每次换新端口

kubectl exec ${JOB}-master-0 -- bash -lc "mkdir -p $AFS"
# 上传 bench（本机 kubectl，勿用坏掉的 ais vcctl）
kubectl exec -i ${JOB}-master-0 -- bash -c \
  'cat > /afs-a3-weight-share/yinjinrun.p/lab-workspace/scripts/cluster/nccl_torch_bench.py' \
  < scripts/cluster/nccl_torch_bench.py

# 各节点启动（生产 RoCE）：
#   NCCL/MCCL_IB_HCA=xscale_0,xscale_1,xscale_2,xscale_3
#   GID_INDEX=5  VSWITCH=1  IB_TC=128  SOCKET_IFNAME=eth0
#   torchrun --nnodes=64 --nproc_per_node=8 --ops all_reduce --sizes 256M

# 参考已验证 launcher：
#   AFS/.../muxi-ar512-retest-20260717_155017/launch_node.sh
```

完整 scale 复现与脚本说明另见：  
`reports/rounds/muxi_link_topology_bandwidth_report_20260717.md`

---

## 4. 请网络侧重点排查

1. **512 卡 AllReduce 时** leaf/spine 上行利用率、PFC pause、ECN/CNP、queue drop、共享缓冲。  
2. 主机 **4×200G** 上联是否对称、是否超分；ECMP 是否把多 rail 打到同一上行。  
3. TC=128 / 无损队列是否与交换机 PFC 优先级一致。  
4. 对比：不相交跨机 P2P 并发几乎不掉，但 **incast / AllReduce 汇聚** 很差——请按「汇聚拥塞」而不是「单口 LinkDown」查。

辅助数字（说明不是「线完全不通」）：

- 跨机单流 P2P@16M ≈ 15–24 GB/s  
- 四轨 AllReduce w64 ≈ 38 GB/s；单轨 w64 ≈ 21 GB/s（多轨有用，但救不了 512）  
- 多卡同时灌一台机总入口可达 ~77 GB/s，但 512 卡 AllReduce 仍只有 ~5 GB/s

---

## 5. 一句话结论（可直接转发）

> 节点侧 8 卡 MetaXLink + 4×200G RoCE 口状态正常，MCCL 确认走 RoCE；  
> **512 卡 AllReduce@256M 中位带宽约 5 GB/s（相对机内仅 ~2%），已多次复现。**  
> 请查交换机在大规模集合通信下的拥塞/PFC/上行/ECMP 配置，而不是单口故障。
