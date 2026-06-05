#!/usr/bin/env python3
"""EP-barrier START/END-skew + kernel decomposition for the DRAIN-regime capture.

Parameterized re-run of barrier_startend.py + kernel_decomp.py for the
drainable-queue drain run (job 2188338, ptyche0162). Question for the drain
regime: when a rank is transiently underfed (Waiting=0, Running<cap; macro
running_skew up to 89 per FINDINGS-4-DRAIN), does it arrive EARLY at the
per-MoE-layer EP all-to-all and idle while the others catch up? That would make
the dipping rank the "earliest-to-START" with a larger START-skew than the
saturated runs ever showed.

Usage:
  python3 drain_barrier.py <sqlite_dir> <node_prefix>
e.g.
  python3 drain_barrier.py outputs/2188338/logs/profiles/agg ptyche0162
expects <sqlite_dir>/<node_prefix>_agg_w0_rank{0,1,2,3}_profile.sqlite
"""
import bisect
import sqlite3
import sys
from collections import Counter

BASE = sys.argv[1].rstrip("/")
NODE = sys.argv[2]
RANKS = [0, 1, 2, 3]
THRESH_NS = 250_000


def sqlite_path(rank: int) -> str:
    return f"file:{BASE}/{NODE}_agg_w0_rank{rank}_profile.sqlite?mode=ro"


def load(rank: int, like: str):
    con = sqlite3.connect(sqlite_path(rank), uri=True)
    rows = con.execute(
        "SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName=s.id WHERE s.value LIKE ? ORDER BY k.start",
        (like,),
    ).fetchall()
    con.close()
    return [r[0] for r in rows], [r[1] for r in rows]


def analyze(label: str, like: str, ref: int = 0) -> None:
    starts, ends = {}, {}
    for r in RANKS:
        s, e = load(r, like)
        starts[r], ends[r] = s, e
    counts = {r: len(starts[r]) for r in RANKS}
    if min(counts.values(), default=0) == 0:
        print(f"\n===== {label} =====  NO MATCHING KERNELS (per-rank counts {counts}) "
              f"-- check kernel-name pattern against StringIds")
        return
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

    if matched == 0:
        print(f"\n===== {label} =====  per-rank counts {counts} but 0 aligned quartets")
        return

    def pct(c):
        return "  ".join(f"r{r}:{100 * c[r] / matched:4.1f}%" for r in RANKS)

    def stats(xs):
        xs = sorted(xs); n = len(xs)
        return (f"mean {sum(xs) / n / 1000:6.2f}us  p50 {xs[n // 2] / 1000:6.2f}us  "
                f"p90 {xs[int(n * 0.9)] / 1000:6.2f}us  p99 {xs[int(n * 0.99)] / 1000:6.2f}us")

    print(f"\n===== {label} =====  matched 4-rank quartets: {matched}  (per-rank counts {counts})")
    print(f"  START-skew (compute arrival): {stats(start_spreads)}")
    print(f"  END-skew   (ring completion): {stats(end_spreads)}")
    print(f"  latest-to-START (compute straggler): {pct(latest_start)}")
    print(f"  earliest-to-START (compute leader) : {pct(earliest_start)}")
    print(f"  latest-to-END   (finishes last)    : {pct(latest_end)}")
    print(f"  earliest-to-END (finishes first)   : {pct(earliest_end)}")


# Candidate EP-collective kernel-name patterns (same Qwen3-235B NVFP4 EP config as
# the lyris saturated run). If counts are 0, dump StringIds to find the real names.
analyze("ReduceScatter (combine)", "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL%")
analyze("AllGather (dispatch)", "ncclDevKernel_AllGather_RING_LL%")


def dump_nccl_names() -> None:
    con = sqlite3.connect(sqlite_path(0), uri=True)
    print("\n=== distinct kernel names containing 'nccl' / 'AllGather' / 'ReduceScatter' (rank0) ===")
    for (v,) in con.execute(
        "SELECT DISTINCT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
        "ON k.demangledName=s.id WHERE s.value LIKE '%ccl%' OR s.value LIKE '%AllGather%' "
        "OR s.value LIKE '%ReduceScatter%' OR s.value LIKE '%AlltoAll%' OR s.value LIKE '%all_to_all%'"
    ):
        print(f"  {v}")
    con.close()


dump_nccl_names()
