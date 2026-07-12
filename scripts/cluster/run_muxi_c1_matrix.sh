#!/usr/bin/env bash
# C1：上传 c1_pair_matrix.py 到 master，跑 16×16，拉回
set +e
STAMP="${STAMP:-$(cat /tmp/muxi_day_stamp.txt 2>/dev/null)}"
AFS="${AFS_OUT:-$(cat /tmp/muxi_day_afs.txt 2>/dev/null)}"
DAY="${DAY_ROOT:-$(cat /tmp/muxi_day_root.txt 2>/dev/null)}"
JOB=yushan-muxi-card-screen-128-cp-copy
MASTER=${JOB}-master-0
OUT=$AFS/C1
LOCAL=${LOCAL_C1:-$DAY/results/C1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$LOCAL" "$DAY"

b64=$(base64 <"$SCRIPT_DIR/c1_pair_matrix.py" | tr -d '\n')
ssh -o BatchMode=yes -o ConnectTimeout=30 ais-cf3e61a5 \
  "KUBECONFIG=/root/.kube/config.muxi-mohe vcctl pod exec $MASTER -- bash -lc 'mkdir -p $OUT; echo $b64 | base64 -d > $OUT/c1_pair_matrix.py; C1_OUT=$OUT C1_JOB=$JOB python3 $OUT/c1_pair_matrix.py'" \
  2>&1 | tee "$DAY/c1_matrix.log" | grep -vE '^(Defaulted|Found)'

for f in pair_summary.json pair_matrix.json roce_note.json bad_links.json hosts_resolved.json; do
  raw=$(ssh -o BatchMode=yes ais-cf3e61a5 \
    "KUBECONFIG=/root/.kube/config.muxi-mohe vcctl pod exec $MASTER -- bash -lc 'cat $OUT/$f'" 2>/dev/null)
  # strip leading non-json noise
  python3 -c "import sys; t=sys.argv[1]; i=min([x for x in (t.find('{'),t.find('[')) if x>=0] or [0]); open(sys.argv[2],'w').write(t[i:])" \
    "$raw" "$LOCAL/$f"
  echo "pulled $f $(wc -c <"$LOCAL/$f") bytes"
done
touch "$DAY/C1.done"
echo C1_DONE
cat "$LOCAL/pair_summary.json"
