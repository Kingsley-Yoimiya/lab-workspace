# MFU loop R6 · r1b_scale · scale=32

- 配置: mode=dense TP=1 PP=1 CP=1 EP=1 MBS=1 GBS=128 SEQ=4096 iters=5
- 稳态 TFLOP/s/GPU: 138.28 · MFU: 47.23% · status=ok
- 假设/备注: R1b: TP1PP1 MBS=1 扩 32/64（跳过 OOM 的 MBS=2）
- 日志: `/Users/yinjinrun/random-thing/logs/mfu-loop-r6-r1b_scale-20260711_004532`
- AFS: `/afs-a3-241ceshi-shared/montyyin/logs/train-loop-r6-r1b_scale-20260711_004532`
