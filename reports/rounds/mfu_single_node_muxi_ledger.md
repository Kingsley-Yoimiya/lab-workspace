# Muxi 单机 8 卡 Dense/MoE MFU 账本（跨节点 IB 门禁未过 · 顺序执行）

> peak=279.9 · mock-data · local/unfused · 跳板 ais-cf3e61a5 · FORCE_POD=worker-1

| round | id | mode | pod | TP/PP/EP | GBS | SEQ | TFLOP med | MFU% | status | log | note |
|------:|----|------|-----|----------|----:|----:|----------:|-----:|--------|-----|------|
| 1 | d_tp4pp2_g64_s2k | dense | worker-1 | 4/2/1 | 64 | 2048 | 33.2 | 11.85 | ok | /Users/yinjinrun/random-thing/logs/muxi-mfu-sn-seq/d_tp4pp2_g64_s2k.train.log | smoke-ref |
| 2 | m_ep8_e8_k2 | moe | worker-1 | 1/1/8 | 64 | 2048 | 11.35 | 3.95 | ok | /Users/yinjinrun/random-thing/logs/muxi-mfu-sn-seq/m_ep8_e8_k2.train.log | baseline-moe-smoke |
