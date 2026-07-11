# jumphost MFU TP×PP scale ledger
| round | id | mode | scale | TP/PP/EP | DP | GBS | TFLOP | MFU% | status | note |
|------:|----|------|------:|----------|---:|----:|------:|-----:|--------|------|
| 0 | d_tp8pp1 | dense | 16 | 8/1/1 | 2 | 2048 | 98.80 | 33.74 | ok | GBS=2048 SEQ=4096 |
| 1 | d_tp8pp1 | dense | 32 | 8/1/1 | 4 | 2048 | 99.25 | 33.90 | ok | DONE detect bug fixed |
| 0 | d_tp8pp1 | dense | 16 | 8/1/1 | 2 | 2048 | 98.80 | 33.74 | leftover | 本机拉起后 jumphost 接管 |
| 2 | d_tp8pp1_rest | dense | 64 | 8/1/1 | 8 | 2048 | 86.00 | 29.37 | ok | P0 Dense TP8PP1 续 64/128（16/32已完成） |
| 2 | d_tp8pp1_rest | dense | 128 | 8/1/1 | 16 | 2048 | 76.15 | 26.01 | ok | P0 Dense TP8PP1 续 64/128（16/32已完成） |
| 3 | d_tp8pp2 | dense | 16 | 8/2/1 | 1 | 2048 | 93.20 | 31.83 | ok | P0 Dense TP8PP2 扩DP |
| 3 | d_tp8pp2 | dense | 32 | 8/2/1 | 2 | 2048 | 91.70 | 31.32 | ok | P0 Dense TP8PP2 扩DP |
| 3 | d_tp8pp2 | dense | 64 | 8/2/1 | 4 | 2048 | 81.35 | 27.78 | ok | P0 Dense TP8PP2 扩DP |
| 3 | d_tp8pp2 | dense | 128 | 8/2/1 | 8 | 2048 | 73.60 | 25.14 | ok | P0 Dense TP8PP2 扩DP |
| 4 | d_tp4pp2 | dense | 16 | 4/2/1 | 2 | 2048 | 125.50 | 42.86 | ok | P1 Dense TP4PP2 扩DP |
| 4 | d_tp4pp2 | dense | 32 | 4/2/1 | 4 | 2048 | 125.60 | 42.90 | ok | P1 Dense TP4PP2 扩DP |
| 4 | d_tp4pp2 | dense | 64 | 4/2/1 | 8 | 2048 | 107.05 | 36.56 | ok | P1 Dense TP4PP2 扩DP |
| 4 | d_tp4pp2 | dense | 128 | 4/2/1 | 16 | 2048 | 108.40 | 37.02 | ok | P1 Dense TP4PP2 扩DP |
| 5 | d_tp4pp4 | dense | 16 | 4/4/1 | 1 | 2048 | 104.30 | 35.62 | ok | P1 Dense TP4PP4 扩DP |
| 5 | d_tp4pp4 | dense | 32 | 4/4/1 | 2 | 2048 | 104.80 | 35.79 | ok | P1 Dense TP4PP4 扩DP |
| 5 | d_tp4pp4 | dense | 64 | 4/4/1 | 4 | 2048 | 100.35 | 34.27 | ok | P1 Dense TP4PP4 扩DP |
| 5 | d_tp4pp4 | dense | 128 | 4/4/1 | 8 | 2048 | 94.85 | 32.40 | ok | P1 Dense TP4PP4 扩DP |
| 6 | d_tp8pp4 | dense | 32 | 8/4/1 | 1 | 2048 | 79.75 | 27.24 | ok | P2 Dense TP8PP4 收尾后停 |
| 6 | d_tp8pp4 | dense | 64 | 8/4/1 | 2 | 2048 | 67.55 | 23.07 | ok | P2 Dense TP8PP4 收尾后停 |
| 6 | d_tp8pp4 | dense | 128 | 8/4/1 | 4 | 2048 | 68.00 | 23.22 | ok | P2 Dense TP8PP4 收尾后停 |
