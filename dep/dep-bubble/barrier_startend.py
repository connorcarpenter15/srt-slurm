#!/usr/bin/env python3
"""Separate START-skew (compute arrival) from END-skew (ring completion) for the
EP collectives, to test whether FINDINGS-2's 'stable rank3 straggler' is a real
compute bubble or an NCCL RING completion-order artifact.

Method: ts-normalized UTC ns (all 4 ranks share node lyris0243 -> comparable clock).
For each ReduceScatter (and AllGather) kernel on the reference rank, find the
nearest-start kernel on each other rank (within THRESH). For clean 4-rank quartets:
  - start_spread = max(start)-min(start)   == compute-arrival skew
  - end_spread   = max(end)-min(end)       == completion skew
  - which rank is latest-to-START (compute straggler) and latest-to-END (ring last)
If compute is balanced: latest-START should be ~uniform across ranks; latest-END
should concentrate on a fixed rank (ring order) => artifact, not a compute bubble.
"""
import bisect
import sqlite3
from collections import Counter

BASE = "/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1977752/logs/profiles/agg"
RANKS = [0, 1, 2, 3]
THRESH_NS = 250_000  # 250us; inter-collective gap ~565us so a true match is well within

def load(rank, like):
    con = sqlite3.connect(f"file:{BASE}/lyris0243_agg_w0_rank{rank}_profile.sqlite?mode=ro", uri=True)
    rows = con.execute(
        "SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName=s.id WHERE s.value LIKE ? ORDER BY k.start", (like,)
    ).fetchall()
    con.close()
    return [r[0] for r in rows], [r[1] for r in rows]

def analyze(label, like, ref=0):
    starts, ends = {}, {}
    for r in RANKS:
        s, e = load(r, like)
        starts[r], ends[r] = s, e
    others = [r for r in RANKS if r != ref]
    start_spreads, end_spreads = [], []
    latest_start = Counter(); earliest_start = Counter()
    latest_end = Counter(); earliest_end = Counter()
    matched = 0
    for i, s0 in enumerate(starts[ref]):
        q_start = {ref: s0}; q_end = {ref: ends[ref][i]}
        ok = True
        for r in others:
            arr = starts[r]
            j = bisect.bisect_left(arr, s0)
            best = None
            for k in (j - 1, j):
                if 0 <= k < len(arr) and abs(arr[k] - s0) <= THRESH_NS:
                    if best is None or abs(arr[k] - s0) < abs(arr[best] - s0):
                        best = k
            if best is None:
                ok = False; break
            q_start[r] = starts[r][best]; q_end[r] = ends[r][best]
        if not ok:
            continue
        matched += 1
        ss = [q_start[r] for r in RANKS]; ee = [q_end[r] for r in RANKS]
        start_spreads.append(max(ss) - min(ss)); end_spreads.append(max(ee) - min(ee))
        latest_start[max(RANKS, key=lambda r: q_start[r])] += 1
        earliest_start[min(RANKS, key=lambda r: q_start[r])] += 1
        latest_end[max(RANKS, key=lambda r: q_end[r])] += 1
        earliest_end[min(RANKS, key=lambda r: q_end[r])] += 1

    def pct(c):
        return "  ".join(f"r{r}:{100*c[r]/matched:4.1f}%" for r in RANKS)
    def stats(xs):
        xs = sorted(xs); n = len(xs)
        return (f"mean {sum(xs)/n/1000:6.2f}us  p50 {xs[n//2]/1000:6.2f}us  "
                f"p90 {xs[int(n*0.9)]/1000:6.2f}us  p99 {xs[int(n*0.99)]/1000:6.2f}us")
    print(f"\n===== {label} =====  matched 4-rank quartets: {matched}")
    print(f"  START-skew (compute arrival): {stats(start_spreads)}")
    print(f"  END-skew   (ring completion): {stats(end_spreads)}")
    print(f"  latest-to-START (compute straggler): {pct(latest_start)}")
    print(f"  earliest-to-START (compute leader) : {pct(earliest_start)}")
    print(f"  latest-to-END   (finishes last)    : {pct(latest_end)}")
    print(f"  earliest-to-END (finishes first)   : {pct(earliest_end)}")

analyze("ReduceScatter (combine)", "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL%")
analyze("AllGather (dispatch)", "ncclDevKernel_AllGather_RING_LL%")
