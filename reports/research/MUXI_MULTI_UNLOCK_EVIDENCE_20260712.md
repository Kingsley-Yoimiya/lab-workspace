# Muxi 多机解锁取证 · 2026-07-12

## 结论更正（相对旧 IB 报告）

旧报告称「net1–4 无 IP」**不准确**：容器内无 `ip` 二进制导致误判。用 `netifaces` / `getifaddrs` 可见：

| 节点 | eth0 | net1 (xscale_0 GID4) | net2 | net3 | net4 |
|------|------|----------------------|------|------|------|
| master-0 | 10.120.29.180 | **172.23.168.210** | 172.24.168.208 | 172.25.171.215 | 172.26.170.149 |
| worker-1 | 10.120.29.142 | **172.23.168.110** | 172.24.168.158 | 172.25.171.179 | 172.26.170.110 |
| worker-2 | (略) | 172.23.168.21 | … | … | … |

- `ibv_devinfo`：`xscale_0..3` 均为 `PORT_ACTIVE`，`link_layer=Ethernet`（RoCE）
- GID index **4/5** = IPv4-mapped，与 net1–4 地址一致；6/7 = link-local
- Job 资源已含 `rdma-training/roce: 1`（`vcctl job describe`）
- `hostNetwork`：未见开启（默认 pod 网络 + Multus 附加 net1–4）

## 旧门禁失败更可能原因

w16 + `IB_HCA=xscale` 硬失败时，**未显式设置 `NCCL_IB_GID_INDEX` / `MCCL_IB_GID_INDEX`**。RoCE 若默认 GID 选错（如 link-local），会出现 Proxy Connect / no transport。

## 解锁试探顺序（本轮）

1. `NCCL/MCCL_IB_HCA=xscale` + `NCCL/MCCL_IB_GID_INDEX=4` + socket=`eth0`，**禁止** `IB_DISABLE=1`
2. 若仍失败：试 `GID_INDEX=5`；或 `NCCL_IB_HCA=xscale_0,xscale_1,xscale_2,xscale_3`
3. 仍失败：`vcctl job clone` 换注解 / 找平台（hostNetwork 或 RDMA CNI 策略）

## 本轮结果（更新）

- GID=4 已固化进 fire 脚本；**机内 w8 OK（~202.7 GB/s）**；**跨节点 w16 仍 Proxy Connect**。
- ARP：peer RoCE IP → **网关 MAC**（非 peer NIC）→ 判定为平台 RoCE L2/直达问题。
- `vcctl job clone -a hostNetwork=true -r 2` 注解成功但 **Pending**（128 卡作业占满）；试验 job 已删。
- 平台请求：`MUXI_ROCE_PLATFORM_REQUEST_20260712.md`；门禁报告：`muxi_ib_gate_20260712_gid4.md`。

## 产物

- `logs/muxi-multi-unlock-20260712/pod_net_evidence.txt`
- `logs/muxi-multi-unlock-20260712/vcctl_and_ifaces.txt`
- 本文件

## 单机基线收束

已停顺序长矩阵；队列收束为冒烟级（Dense 已入账 MFU≈11.85%；MoE 冒烟待补）。主战役改为 RoCE 门禁解锁。
