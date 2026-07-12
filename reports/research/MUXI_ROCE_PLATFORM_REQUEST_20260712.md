# 沐曦 RoCE 跨节点解锁 · 平台请求清单

**日期**：2026-07-12  
**现 Job**：`yushan-muxi-card-screen-128-cp-copy`（16×8，Running）  
**跳板**：`ais-cf3e61a5` / kubeconfig `config.muxi-mohe`  
**目标**：跨节点 AllReduce@256MB 相对机内保持率 **≥50%**，以便跑 Dense/MoE 多机 MFU。

## 现象（可复现）

1. 机内（world=8）`xscale` RoCE：AR@256M ≈ **202.7 GB/s**。
2. 跨节点（world=16）`xscale` + `GID_INDEX=4`：MCCL 选中 RoCE，但 **Proxy Connect / ncclInternalError**。
3. 容器内 net1–4 **已有**与 GID 一致的 172.23–26.x 地址；`rdma-training/roce:1` 已挂。
4. 从 master net1 ARP 解析 peer（如 worker net1 `172.23.168.110` / `172.23.171.187`）得到 **网关 MAC** `ac:5e:14:e5:93:f3`，**不是** peer 网卡 MAC（`5c:6a:ec:…`）。

## 请求（按优先级）

1. **RoCE 平面 L2/RDMA 直达**：peer IP 的 ARP 应落到真实 HCA/VF MAC；或提供平台认可的 RoCE CNI / SR-IOV / 设备插件配置。
2. **或** 提供可调度的 **hostNetwork（或等价）** 模板 job（2 节点即可先验证门禁），在不拆现网 128 卡作业的前提下可并行起小作业，或指导如何安全 clone。
3. 确认 Multus/`Network` 注解空值是否预期；当前 describe 中 `Network.Alpha.Kubernetes.Io/Network` 为空。
4. 若需换镜像：现镜像  
   `registry2.d.pjlab.org.cn/ccr-ailabdev/megatron-lm:0.12.0-maca.ai3.3.0.11-torch2.6-py312-ubuntu22.04-amd64-driver`  
   （我们认为主因是网络而非镜像。）

## 验收标准

- 2 节点 × 8 卡，`NCCL/MCCL_IB_HCA=xscale`，`GID_INDEX=4`，**无** `IB_DISABLE`  
- AR@256M bus_bw 中位 ≥ **~100 GB/s**（相对机内 202.7 的 50%）  
- MCCL INFO 可见 `Using network IB` 且训练/bench 跑完无 Proxy Connect

## 联系人侧已准备

- 门禁脚本：`scripts/cluster/fire_nccl_scale_muxi.sh`（已默认 GID=4）  
- 报告：`reports/rounds/muxi_ib_gate_20260712_gid4.md`  
- 取证：`reports/research/MUXI_MULTI_UNLOCK_EVIDENCE_20260712.md`
