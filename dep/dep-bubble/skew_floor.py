#!/usr/bin/env python3
"""Estimate a floor on EP-barrier bubble cost from the 1918158 temporal-skew CSV.

The CSV (max-min skew per ~10s window, parsed from vLLM engine logs) cannot
measure step-level barrier waits directly. It can, however, establish two
things for a go/no-go on profiling:

  1. The "equal work, unequal speed" fingerprint: windows where all four DP
     ranks sit at the same running-request count (running_skew == 0, pinned at
     the max-num-seqs=864 cap) yet report materially different per-engine
     generation throughput (gen_skew > 0). Queue imbalance cannot explain that.

  2. A *floor* on the throughput lost to transient inter-rank speed skew. With
     four ranks lock-stepped through the EP all-to-all, the step rate is gated
     by the momentarily slowest rank. From a window's max-min throughput spread
     S and a per-rank rate R, the average rank runs ~S/2 above the slowest, so
     gating everyone to the slowest costs >= (S/2)/R of aggregate throughput.

This is a FLOOR, not the bubble itself: interval-averaged logs wash out the
per-step laggard rotation, and the symmetric workload (fixed osl, even request
split) forces equal *total* tokens per rank, so the steady-state bubble shows
up as everyone running slower -- invisible to inter-rank skew. Only step- or
kernel-granular timing (nsys) can size the real cost.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

# Measured aggregate output throughput for job 1918158, divided across 4 ranks.
PER_RANK_RATE_TOK_S = 69122.16 / 4
SATURATED_RUNNING = 864


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "dep/dp-imbalance-repro/post-9915/temporal_skew_1918158.csv"
    )
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    # Only windows where all four ranks reported -> skew is meaningful.
    full = [r for r in rows if int(r["ranks"]) == 4]
    # Steady-state saturated windows: backend pinned near the running cap.
    sat = [r for r in full if int(r["running_max"]) >= SATURATED_RUNNING]
    # Of those, the ones with perfectly equal queue depth across ranks.
    equal_queue = [r for r in sat if int(r["running_skew"]) == 0]

    gen_sat = [float(r["gen_skew"]) for r in sat]
    gen_eq = [float(r["gen_skew"]) for r in equal_queue]

    # Floor on bubble cost from each window: (skew/2) / per-rank rate.
    floor_eq = [g / 2 / PER_RANK_RATE_TOK_S for g in gen_eq]

    print(f"csv: {csv_path}")
    print(f"per-rank rate assumed: {PER_RANK_RATE_TOK_S:,.0f} tok/s\n")

    print(f"windows (4 ranks reporting):        {len(full)}")
    print(f"  saturated (running_max>={SATURATED_RUNNING}):     {len(sat)}")
    print(f"  ...with equal queue (skew==0):    {len(equal_queue)}"
          f"  ({len(equal_queue)/len(sat)*100:.0f}% of saturated)\n")

    print("## Fingerprint: equal queue depth, unequal speed")
    for thresh in (100, 500, 1000):
        n = sum(1 for g in gen_eq if g > thresh)
        print(f"  equal-queue windows with gen_skew > {thresh:>4} tok/s: "
              f"{n:>3}  ({n/len(equal_queue)*100:.0f}%)")
    print()

    print("## gen_skew across saturated windows (tok/s, and % of per-rank rate)")
    for label, g in (("mean", statistics.fmean(gen_sat)),
                     ("p50", pct(gen_sat, 0.50)),
                     ("p95", pct(gen_sat, 0.95)),
                     ("max", max(gen_sat))):
        print(f"  {label:>4}: {g:8.1f} tok/s   ({g/PER_RANK_RATE_TOK_S*100:5.1f}% of rank rate)")
    print()

    print("## FLOOR on aggregate throughput lost to transient speed skew")
    print("   (equal-queue windows only; (skew/2)/rate)")
    for label, f in (("mean", statistics.fmean(floor_eq)),
                     ("p95", pct(floor_eq, 0.95)),
                     ("max", max(floor_eq))):
        print(f"  {label:>4}: {f*100:5.2f}%")


if __name__ == "__main__":
    main()
