# Muxi MCCL / RoCE 环境变量（固化）

完整说明见 myportal：[`docs/muxi-mccl-roce-env.md`](../../../../Codespace/myportal/docs/muxi-mccl-roce-env.md)  
（本仓相对路径若不可用，打开 `/Users/yinjinrun/Codespace/myportal/docs/muxi-mccl-roce-env.md`）。

## 一页抄写版

```bash
# vcjob 资源：metax-tech.com/gpu:8 + rdma-training/roce:1

export NCCL_SOCKET_IFNAME=eth0
export MCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=xscale_0,xscale_1,xscale_2,xscale_3
export MCCL_IB_HCA=xscale_0,xscale_1,xscale_2,xscale_3
export NCCL_IB_GID_INDEX=5
export MCCL_IB_GID_INDEX=5
export MCCL_IB_TC=128
export MCCL_ENABLE_VSWITCH=1
export MCCL_PCIE_BUFFER_MODE=0
# 禁止默认 IB_DISABLE=1
```

默认已写入 `scripts/cluster/muxi.env` 与 `fire_nccl_scale_muxi.sh`。
