# Muxi H3C 机间带宽掉崖：可上报证据包（2026-07-17）

面向网络/平台同事：说明「不是偶发坏卡」，并给出指向 **交换机/fabric 在多对多、incast 并发下拥塞（或等价行为）** 的可复现实验。

## 一句话结论

1. **物理口没挂、也不是误走以太网**：抽样端口全 ACTIVE、200Gb/s；MCCL 日志明确走 `xscale_*/RoCE` + `NET/IB/*/GDRDMA`。  
2. **不相交的一对一对打（P2P）几乎不掉**；**多发送者同时打向同一接收者（incast）会陡降**。  
3. **512 卡 AllReduce@256M 带宽相对机内 w8 仅剩约 3%**，与「集合通信多对多/树形汇聚压交换机」一致，而不是少数坏链路。

---

## 证据 1：规模 AllReduce 系统性掉崖（已复现两次）

作业：`yinjinrun-cs512-20260716-221823`（64 节点 × 8 卡）  
算子：`all_reduce`，消息 `256M`，RoCE：`xscale_0..3`，GID=5，VSWITCH=1  

| world | 复现 med (GB/s) | keep%（相对 w8） | 昨夜 med |
|------:|----------------:|-----------------:|---------:|
| 8（机内） | 258.6 | 100 | 127.0 |
| 16 | 78.0 | 30.2 | 73.5 |
| 32 | 56.0 | 21.7 | 52.0 |
| 64 | 38.0 | 14.7 | 38.8 |
| 128 | 33.0 | 12.8 | 28.3 |
| 256 | 13.0 | 5.0 | 26.8 |
| 512 | **7.64** | **3.0** | **4.47** |

要点：

- 两次战役曲线同形态 → **非偶发**。  
- 同 world 内 min≈max → **不是 1～2 张病卡拖后腿**。  
- 最陡跌落在 **机内→跨机（8→16）** 与 **256→512**。

产物：`myportal/results/muxi-h3c/20260717_103913-ib-rerun/`  
报告：`reports/rounds/muxi_ib_scale_rerun_20260717.md`

---

## 证据 2：链路静态健康（排除口挂 / 降速）

抽样 16 节点 × 4×xscale = **64 口**：

| 项 | 结果 |
|----|------|
| state | 全部 PORT_ACTIVE |
| phys | 全部 LinkUp |
| rate | 全部 **200** Gb/s |
| GID index 5 | 均为 RoCEv2（IPv4-mapped），非全 0 |
| sysfs counters | **容器内 xscale 无 counters 目录**（无法在 pod 内做 FEC/symbol 差分） |

产物：`myportal/results/muxi-h3c/20260717_135116-link-sample/`

---

## 证据 3：数据面确认走 RoCE（排除 eth0 慢路径）

8 节点 AllReduce@256M + `MCCL_DEBUG=INFO`：

```
MCCL INFO Using network IB
MCCL INFO NET/IB : Using [0]xscale_0:1/RoCE ... [3]xscale_3:1/RoCE ; OOB eth0:...
MCCL INFO Channel ... [send] via NET/IB/*/GDRDMA
MCCL INFO Trees ...
```

`eth0` 仅作 OOB/控制面属正常。  
AR64 med≈**36.6 GB/s**（与 scale 曲线 w64≈38 一致）。

产物：`myportal/results/muxi-h3c/20260717_140105-ar64-info/`

---

## 证据 4：对照实验 —— 不相交 P2P 并发 vs Incast 并发

### 4a. 不相交跨机 pair（串行一对 vs 多对同时）

每节点 1 卡，消息 16M / 256M：

| 场景 | serial med | concurrent med | keep% |
|------|-----------:|---------------:|------:|
| 4 对 @16M | 23.8 | 17.9 | ~75%（轻度） |
| 8 对 @16M | 23.8 | 23.8 | ~100% |
| 16 对 @256M | 24.4 | 24.3 | ~100% |

说明：**单纯「多条互不抢同一接收端的跨机流」并不足以复现 AllReduce 那种数量级掉崖。**

### 4b. Incast（多发送者 → 同一接收者）★ 关键对照

每节点 1 卡，消息 16M；比较「轮流发」vs「同时发」时 **每个发送者有效带宽**：

| 规模 | serial per-sender | concurrent per-sender | keep% |
|------|------------------:|----------------------:|------:|
| 7→1（8 节点） | **15.8 GB/s** | **3.5 GB/s** | **21.9%** |
| 15→1（16 节点） | **15.2 GB/s** | **1.6 GB/s** | **10.6%** |

解读（上报可用表述）：

> 单流跨机 P2P 约 15～24 GB/s，口状态 200G ACTIVE；  
> 一旦 **多个节点同时打向同一汇聚点**，单流有效带宽立刻掉到约 1/5～1/10。  
> 这与交换机入向拥塞 / PFC 暂停 / 共享缓冲打满等「并发拥塞」现象一致，  
> 也与 AllReduce 树形/多通道汇聚时随规模掉带宽的形态同向。

产物：

- `AFS .../muxi-incast8-20260717_144014/`
- `AFS .../muxi-incast16-20260717_144048/`
- 日志摘要：`~/random-thing/logs/muxi-congest-ab-20260717/INCAST*_SUMMARY.txt`

---

## 建议网络侧一起看的点

1. **汇聚端口 / leaf-spine 上行**：incast、AllReduce 时是否出现 PFC pause、ECN/CNP、queue drop、buffer 打满。  
2. **ECMP 哈希与多 rail（4×xscale）**：多通道是否打到同一上行造成微突发。  
3. **无损队列（PFC）配置与 TC=128**：是否 pause 风暴导致有效吞吐塌陷。  
4. 宿主机侧 xscale/FEC/CRC 计数（**容器内无 counters**，需宿主机或厂商工具）。

## 不需要再争论的点

| 假说 | 结论 |
|------|------|
| 偶发抖动 | ❌ 两次 scale 同形态 |
| 少数坏卡/坏口 | ❌ ACTIVE+200G；同 world 无离群 |
| 数据走 eth0 | ❌ MCCL INFO 证明 RoCE/GDRDMA |
| 仅「链路带宽不够」的静态瓶颈 | ❌ 单流/不相交多流仍可达十余～二十 GB/s |

## 一页可贴会的数字

```
AllReduce@256M keep%(相对机内8卡):  16→30%  64→15%  256→5%  512→3%
跨机单流 P2P@16M:                   ~15–24 GB/s（健康）
Incast 7→1 并发 keep%:              21.9%   (15.8 → 3.5 GB/s/sender)
Incast 15→1 并发 keep%:             10.6%   (15.2 → 1.6 GB/s/sender)
MCCL: Using network IB + xscale_0..3/RoCE + GDRDMA
```

联系人可附：作业名、AFS 路径、本机备份 `myportal/results/muxi-h3c/` 下对应时间戳目录。
