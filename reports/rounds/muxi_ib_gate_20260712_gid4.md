# Muxi IB/RoCE 门禁报告 · 2026-07-12（GID 解锁重试）

## 判定：**跨节点 RoCE 门禁仍未通过**（保持率门槛 ≥50% 未达）

| world | 配置 | AR@256M bus_bw 中位 | 相对 w8 | 结果 |
|------:|------|-------------------:|--------:|------|
| 8 | xscale + `GID_INDEX=4`（机内） | **202.7 GB/s** | 100% | OK |
| 16 | xscale + `GID_INDEX=4`（跨节点） | — | — | **FAIL**（Proxy Connect） |
| 16 | eth0 + `IB_DISABLE=1`（旧门禁） | ~0.26 GB/s | **0.13%** | 可跑但不可作 MFU 主路径 |

## 取证更正

1. **net1–4 有 IP**，且与 xscale GID index 4/5（IPv4-mapped）一致（旧报告「无 IP」因缺 `ip` 命令误判）。
2. Job 已申请 `rdma-training/roce: 1`；MCCL 日志确认 `Using network IB` / `xscale_*:RoCE`。
3. **真正阻塞**：跨节点 ARP 将 peer RoCE IP 解析到 **网关 MAC** `ac:5e:14:e5:93:f3`（与 `172.23.171.254` 相同），而不是 peer NIC MAC（`5c:6a:ec:…`）。RoCE 需要 L2/RDMA 直达；经网关则 Proxy Connect 失败。

## 本轮已做解锁尝试

| 动作 | 结果 |
|------|------|
| 固化 `NCCL/MCCL_IB_GID_INDEX=4` 进 `fire_nccl_scale_muxi.sh` | 脚本正确；机内 w8 OK；跨节点仍 FAIL |
| 校验 net1 L3/ARP | UDP 可达，但 ARP 指向网关 → 非 NIC 直连 |
| `vcctl job clone -r 2 -a hostNetwork=true` | 注解写入成功；因现网 16 节点占满 **Pending**，无法验证；已删除试验 job |

## 决策

- **不进** Dense/MoE 多机弱扩展主矩阵（避免 eth0 假扩展）。
- Phase2 多机：**阻塞于平台网络**（见平台请求清单）。
- 单机基线旁路继续（Dense 冒烟已入账；MoE 冒烟本轮补跑）。

## 产物

- AFS: `/afs-a3-weight-share/montyyin/results/muxi-ib-gate-gid4-20260712_090917/`（w8 OK）
- AFS: `/afs-a3-weight-share/montyyin/results/muxi-ib-gate-gid4b-20260712_091100/`（w16 FAIL）
- 取证: `reports/research/MUXI_MULTI_UNLOCK_EVIDENCE_20260712.md`
- 平台请求: `reports/research/MUXI_ROCE_PLATFORM_REQUEST_20260712.md`
