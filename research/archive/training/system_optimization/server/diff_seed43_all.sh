#!/bin/bash
# Re-run the frame-2 diffs (the first attempt used repo-relative script paths
# and failed; see closing_progress.log 21:03:40 note). Parallel wdiff+ulp for
# (a,c) and (c,d), then rel-L2 for both pairs.
set -u
LANE=/home/lyh/vlm-systems
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
L=$LANE/logs
A=$LANE/runs/s3/a_off/train/ckpt_final.pt
C=$LANE/runs/s3/c_off_seed43/train/ckpt_final.pt
D=$LANE/runs/s3/d_on_seed43/train/ckpt_final.pt
echo "$(date +'[%F %T]') frame-2 diffs (re-run) start" >> $L/closing_progress.log
$PY $LANE/scripts/wdiff_ckpt.py "$A" "$C" > $L/wdiff_seed43_ac.txt 2>&1 &
P1=$!
$PY $LANE/scripts/wdiff_ckpt.py "$C" "$D" > $L/wdiff_seed43_cd.txt 2>&1 &
P2=$!
$PY $LANE/scripts/ulp_drift.py "$A" "$C" --json $L/ulp_seed43_ac.json > $L/ulp_seed43_ac.txt 2>&1 &
P3=$!
$PY $LANE/scripts/ulp_drift.py "$C" "$D" --json $L/ulp_seed43_cd.json > $L/ulp_seed43_cd.txt 2>&1 &
P4=$!
wait $P1 $P2 $P3 $P4
$PY - "$A" "$C" "$D" > $L/rel_l2_seed43.txt 2>&1 <<'PYEOF'
import sys, torch
def state(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return ck["model"] if isinstance(ck, dict) and "model" in ck else ck
pa, pc, pd = sys.argv[1], sys.argv[2], sys.argv[3]
A, C, D = state(pa), state(pc), state(pd)
for name, X, Y in [("a-vs-c (order+init)", A, C), ("c-vs-d (cache @43)", C, D)]:
    t2a = t2d = 0.0
    rows = []
    for k in X:
        if not (torch.is_tensor(X[k]) and torch.is_tensor(Y[k])) or X[k].shape != Y[k].shape:
            continue
        x, y = X[k].double(), Y[k].double()
        d = x - y
        na, nd = float(x.pow(2).sum()), float(d.pow(2).sum())
        t2a += na; t2d += nd
        rows.append(((nd / na) ** 0.5 if na else 0.0, k))
    rows.sort(reverse=True)
    print(f"{name}: GLOBAL relL2={(t2d/t2a)**0.5:.6f}")
    for rel, k in rows[:5]:
        print(f"  {k}: relL2={rel:.4%}")
PYEOF
echo "$(date +'[%F %T]') frame-2 diffs (re-run) done" >> $L/closing_progress.log
