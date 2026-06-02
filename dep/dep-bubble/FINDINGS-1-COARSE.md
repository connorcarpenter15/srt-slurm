# DEP Bubble — Step 1 (Coarse-Data) Findings

Date: 2026-06-01
Source data: `dep/dp-imbalance-repro/post-9915/temporal_skew_1918158.csv`
(per-window max−min skew parsed from vLLM engine logs of the *good* run,
job 1918158, post-9915 + `stream-interval: 50`).
Analysis: `skew_floor.py`.

## Question

Even after the throughput gap was closed, do DP ranks under DEP (DP attention +
EP experts) service work at **different speeds** — independent of load — in a
way consistent with bubbles waiting at the EP all-to-all barrier?

## Result: the fingerprint is present

In the saturated steady state, queue depth is essentially equalized across the
four ranks, yet per-rank generation throughput is not:

- 127 saturated windows (`running_max ≥ 864`); **95% have `running_skew == 0`**
  — all four ranks at the identical 864-request cap.
- Among those equal-queue windows:
  - **38%** show per-rank throughput spread > 100 tok/s
  - **18%** show spread > 500 tok/s
  - **13%** show spread > 1000 tok/s (> 6% of the ~17.3k tok/s per-rank rate)

Identical batch size but different token rate cannot be explained by load
imbalance, router policy, or queueing. It is the "same work, different speed"
signature the bubble hypothesis predicts.

## Result: magnitude (a floor, not the bubble)

`gen_skew` across saturated windows, as % of the per-rank rate (17,281 tok/s):

| | tok/s | % of rank rate |
|---|---:|---:|
| mean | 291 | 1.7% |
| p50 | 88 | 0.5% |
| p95 | 1209 | 7.0% |
| max | 1930 | 11.2% |

Floor on aggregate throughput lost to transient inter-rank speed skew
(`(skew/2)/rate`, equal-queue windows): **mean 0.86%, p95 3.50%, max 5.58%**.

## Why this is only a floor — and why coarse logs can't go further

Two structural reasons the interval logs **understate** the bubble:

1. **Interval averaging.** vLLM's "Avg generation throughput" is averaged over
   the logging interval (hundreds of decode steps). The EP barrier wait is
   per-step, and the laggard rank rotates step to step. Averaging washes out the
   per-step max, so most of the bubble disappears into the mean.

2. **Workload symmetry.** Fixed `osl=1024` + exactly even request distribution
   (20,480 per rank) forces **identical total output tokens per rank**. So the
   steady-state bubble cannot appear as one rank doing *less* than another — it
   appears as *every* rank running slower than an EP-barrier-free baseline. That
   common slowdown is invisible to any inter-rank skew metric by construction.

Corollary: pulling the raw per-rank logs from Lustre would let us compute an
exact `Σ(max−rate_i)` instead of approximating from max−min, but it inherits
both confounds above and would not change the go/no-go. The coarse data has hit
its ceiling.

## Verdict → proceed to Step 3 (nsys)

- The qualitative fingerprint (equal queue, unequal speed) is unambiguous and
  present in a large fraction of windows.
- The floor on cost is modest on average (~0.9%) but non-trivial in the tail
  (p95 3.5%, max 5.6%), and is a structural undercount.
- Only step-/kernel-granular timing can measure the actual barrier wait.

The numbers line up. Next: a targeted Nsight Systems profile of one saturated
window to capture ranks idling at the MoE dispatch/combine all-to-all directly.
Scope for that run is an open question for the user (see conversation).
