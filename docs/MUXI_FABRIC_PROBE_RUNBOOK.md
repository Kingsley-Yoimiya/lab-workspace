# 沐曦 Fabric Probe 可复跑 runbook

## 1. 固定控制面

- 本地入口仓：myportal
- 实验落点：`~/random-thing/project/lab-workspace`
- 跳板：`ais-cf3e61a5`
- 跳板kubeconfig：`~/.kube/config-vc-c550-h3c-test.yaml`
- 身份：`yinjinrun.p`
- 作业：`yinjinrun-cs512-20260716-221823`
- AFS：`/afs-a3-weight-share/yinjinrun.p/{lab-workspace,results}`

任何日志不得打印kubeconfig、token、证书或私钥正文。

## 2. 默认回滚态

实验结束必须回到：

- HCA：`xscale_0,xscale_1,xscale_2,xscale_3`
- GID index：5
- `MCCL_IB_TC=128`
- `MCCL_ENABLE_VSWITCH=1`
- channel：unset `MCCL_MIN_NCHANNELS`、`MCCL_MAX_NCHANNELS`
- GPU可见性：unset `CUDA_VISIBLE_DEVICES`
- 不设置未知QP参数

这些是当前作业的默认实验态，不代表已证明线上DSCP/ECN或交换机QoS映射。

## 3. 实验前门禁

1. 本机执行 `python3 setup/check/verify_channels.py`
2. 跳板确认kubeconfig权限600和user=`yinjinrun.p`
3. `vcctl pod get --job ...` 必须64/64 Running
4. 目标pod无torchrun、benchmark、constitution、burn-in
5. GPU空闲
6. 预写run_id、端口、seed、顺序、假设、停止条件、回滚
7. 新建本机 `myportal/results/muxi-h3c/<run_id>/`

## 4. Collective入口

已验证入口：

`scripts/cluster/fire_nccl_scale_muxi_jump.sh`

约束：

- Mac只同步bundle、发起单条SSH和回拉
- launcher按nnodes全并发
- 每run新端口、新AFS目录
- 共享脚本经master上传并做SHA256校验
- 完成后校验rank数、iter数、schema、host/local_rank/physical_gpu映射
- `MCCL_DEBUG=INFO`时保留HCA、channel、算法/协议读取证据

可控且已验证：

- `WORLD`、`NNODES`、`NPROC_PER_NODE`
- `OPS`、`SIZES`、`WARMUP`、`ITERS`
- `NCCL_IB_HCA` / `MCCL_IB_HCA`
- `POD_ORDER`
- `CUDA_VISIBLE_DEVICES`
- `MCCL_MIN_NCHANNELS` / `MCCL_MAX_NCHANNELS`

`MCCL_ALGO` / `MCCL_PROTO`只证实“被接受/读取”，不能仅凭INFO断言实际kernel。

## 5. Pair与inventory入口

### inventory

- `fabric-probe/probes/inventory/run_muxi_inventory_jump.sh`
- parser：`parse_muxi_inventory.py`

### P2P矩阵

- schedule：`generate_round_robin.py`
- bench：`muxi_pair_bench.py`
- driver：`run_muxi_pair_rounds_jump.sh`
- parser：`parse_pair_rounds.py`
- 复检/复测：`analyze_full_pair_matrix.py`、
  `run_muxi_pair_retest_jump.sh`、`classify_pair_retest.py`

Pair任务使用跳板并发前台 `vcctl exec`，不能恢复为pod内detach。每pair必须
pair-local rendezvous、唯一端口、node_rank0/1与方向字段一致。

### QoS/counter可见性

- `fabric-probe/probes/qos/muxi_visibility_probe.py`
- `parse_muxi_visibility.py`
- `muxi_netdev_counter_snapshot.py`
- `parse_netdev_counter_delta.py`

这些脚本只读，不安装工具，不修改TC/QoS。

## 6. 停止与重试

- rank/case/marker或日志mtime连续180秒无增长：清理并停止
- 身份/权限错误：立即停止
- 清理失败：立即停止
- 单次SIGSEGV/极端stall：保留证据，新端口受控复测1～2次
- pair单次失败：清理双端进程组，新端口复测一次
- pair单轮持续失败超过25%：停止矩阵
- INVALID永久排除；即使JSON已写齐，只要fail非0或进程exit非0仍是INVALID
- 不把collective内rank当独立样本
- 不因后续成功覆盖、删除或改写旧INVALID结论

## 7. 每run必需产物

- `manifest.yaml`
- `run.log` / driver console
- `raw/*.jsonl`或逐pod/逐pair原始日志
- `SUMMARY.md`
- schema校验结果
- HCA/channel/GPU/rank mapping证据
- 失败marker和清理/postcheck证据
- 脚本与配置快照、SHA256

关键里程碑立即回拉：

`myportal/results/muxi-h3c/<run_id>/`

长矩阵每8轮回拉checkpoint和主run.log；大目录按round分片tar，单片失败最多
重试3次。

## 8. 报告解释边界

- `bus_bw`、P2P单向带宽、incast aggregate不能混为同一指标
- MCCL读取TC128不等于QP tclass已设置
- QP tclass不等于线上DSCP/ECN已确认
- 没有交换机ground truth时禁止命名leaf/spine
- 单次慢pair不能命名坏链路；必须原方向重复、反向重复和负对照
- 没有RNIC/交换机counter闭环时，不宣称PFC/ECN/ECMP机制已证实
