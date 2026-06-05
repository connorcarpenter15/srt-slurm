# DEP Bubble — Throughput-Knee Sweep + Final Synthesis

Date: 2026-06-05
Cluster: **ptyche** GB200, 1 node (`ptyche0267`).
Job: `2191166` — concurrency sweep `2048 → 3072 → 4096 → 6144`, canonical decode
shape (isl=2, osl=1024), DP=4 EP, identity permutation, `max-num-seqs=864`,
`stream-interval: 50`. Single job / single node → the throughput-vs-concurrency
curve and per-point skew are confound-free (same physical GPUs throughout).

Analysis: `sweep_skew.py` (segments the shared backend log by each point's
request-trace `wall_time_ns` window; reports backend aggregate decode throughput
+ per-window running/waiting skew), cross-checked against `benchmark-rollup.json`.

## Question

`FINDINGS-4-DRAIN.md` showed that a *drainable* queue (Waiting≈0, Running<cap)
surfaces macro temporal skew (`running_skew` up to 89 at conc 2048) and that the
EP-barrier skew is arrival-dominated and regime-driven (the saturated control
closed the node confound: drainage widens the barrier 2.6× on identical
hardware). But at conc 2048 the skew cost **no** throughput — 254 ms TTFT, ample
slack. The open performance question: **is there a concurrency where the queue
STILL drains (Waiting≈0) but the system is throughput-bound — so the temporal
skew actually costs tok/s?** Total running capacity = 4×864 = 3456, so the sweep
brackets that cap.

## Result: yes — the knee is at concurrency ≈ 3072

| conc | backend tok/s | rollup tok/s | req/s | TTFT mean | run/rank | wait/rank | run_skew p50/p95/max | wait_skew max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 55,706 | 55,257 | 54.0 | **1.18 s** | 485 | **0** | 0 / 1 / 1 | 0 |
| 3072 | 64,113 | 64,041 | 62.5 | **1.30 s** | 706 | **0** | 0 / **187** / 768 | 0 |
| 4096 | 65,198 | 65,315 | 63.8 | 9.81 s | 827 | 118 | 0 / 685 / 864 | 160 |
| 6144 | 65,836 | 65,723 | 64.2 | 37.95 s | 848 | 531 | 0 / 29 / 88 | 609 |

(backend = sum of per-rank mean gen tok/s; rollup matches within ~1 % at every
point — these points completed cleanly, no truncation.)

### Throughput plateaus after 3072 — the ceiling is hit while the queue still drains

```
backend decode throughput (tok/s), cap ≈ 66k:
  2048  ██████████████████████████████████        55,706
  3072  ███████████████████████████████████████   64,113   +15.1% vs 2048
  4096  ████████████████████████████████████████  65,198    +1.7% vs 3072
  6144  ████████████████████████████████████████  65,836    +1.0% vs 4096
                                                            (+2.7% total 3072→6144)
```

Throughput scales steeply 2048→3072 (**+15 %**) then flattens: doubling
concurrency from 3072 to 6144 buys only **+2.7 %** (64.1k → 65.8k). The system is
throughput-bound at ~66k tok/s from conc 3072 onward.

### The macro skew peaks exactly at the knee — while the queue still fully drains

```
run_skew p95 (per 10s window):
  2048  ▏                                            1     drains, slack absorbs
  3072  ███████████▏                               187     drains (wait/rank=0) ← KNEE
  4096  ███████████████████████████████████████   685     queue backing up (wait/rank=118)
  6144  ██                                          29     saturated — backlog refills, skew suppressed
```

The crucial point is **conc 3072**: `wait/rank = 0` on every rank (the queue
*fully drains*), yet `run_skew` p95 jumps from 1 → **187** (max 768) and
throughput has already reached 97.4 % of its ceiling. This is the regime
`FINDINGS-4` predicted but conc 2048 was too slack to show: the queue drains, the
ranks fall out of lockstep, **and** the system is throughput-bound at the same
time. At 4096 the queue starts backing up (wait/rank=118, TTFT 9.8 s); by 6144 it
is saturated (wait/rank=531, TTFT 38 s) and the macro skew is *suppressed*
(p95=29) because a dipped rank refills instantly from its deep backlog — the same
skew-suppression the saturated nsys control showed at the kernel level.

### How much does the temporal underfeed cost?

At the knee (3072, the largest fully-draining point) throughput is 64.1k =
**97.4 %** of the saturated ceiling (65.8k at conc 6144). So the temporal
underfeed leaves **~2.6 % of throughput on the table** — modest in absolute
terms, but it lands exactly where `run_skew` is largest, and it is the reason the
drained regime cannot quite reach the cap-pinned ceiling: with Waiting=0 a dipping
rank has nothing to refill from and idles at the EP barrier (the arrival skew
quantified in `FINDINGS-4`: ~203 µs p50 at the all-to-all). Past the knee, more
concurrency buys essentially no throughput (+2.7 %) and an order-of-magnitude
worse TTFT (1.3 s → 38 s).

## Final synthesis — cause, impact, fix

This closes the three-part goal (isolate cause · quantify impact · identify fix).

### 1. Cause (isolated)

**Temporal DP-rank underfeed under a drainable queue with round-robin routing.**
- It requires the drain precondition `Waiting≈0, Running<cap` — saturation
  (decode pinned at cap, or token-budget-bound prefill) structurally eliminates
  it (`FINDINGS-3`, `FINDINGS-4`).
- It is **not** static count imbalance (round-robin ends with equal per-rank
  counts), **not** expert-load skew, **not** GPU-speed skew, and **not** topology:
  the kernel-level EP barrier is **arrival-dominated** (START≈END), the
  leader/straggler role **rotates** (no fixed straggler), and compute (attention +
  expert GEMMs) is balanced to within ~2-10 % in every regime (`FINDINGS-4`).
- It is **regime-driven**: the saturated control (job 2191165, same node +
  identity perm + shape) showed the barrier at ~79 µs p50 vs the drain's ~203 µs —
  drainage causally widens the arrival skew **2.6×** on identical hardware. (A
  separate constant ~2× topology offset exists between ptyche-identity and
  lyris-identity, but it does not create the skew — drainage does.)

### 2. Performance impact (quantified)

**Small in throughput, large in latency headroom.** The throughput ceiling
(~66k tok/s on this node) is reached by concurrency ~3072. At the knee the
temporal underfeed costs **~2.6 %** throughput vs the saturated ceiling, peaking
exactly where `run_skew` peaks. The dominant practical effect is the
throughput/latency trade-off it defines: the knee (conc ~3072) delivers 97 % of
peak throughput at **1.3 s** TTFT, whereas chasing the last ~2.6 % by saturating
(conc 6144) costs **38 s** TTFT for +2.7 % tok/s. The underfeed is a real,
kernel-confirmed mechanism, but at the canonical decode shape it is a **minor
throughput tax**, not a major regression — consistent with the throughput-gap
thread already being closed by PR 9915 + `stream-interval: 50`.

### 3. Potential fix (directional)

The mechanism is "a dipping rank with an empty queue idles at the barrier." Three
levers, in order of how directly they target it:

1. **Least-loaded routing** (`frontend.args.router-mode: least-loaded`) instead of
   round-robin: send the next request to the rank with the fewest running, so a
   dip is refilled preferentially before it reaches the barrier. Directly testable
   — rerun conc **3072** (where run_skew peaks) round-robin vs least-loaded and
   measure whether run_skew p95 drops and throughput closes toward the 65.8k
   ceiling. (Note: `load-aware: true` is rejected by the CLI; use `least-loaded`.)
2. **Shallow per-rank prefetch backlog** (keep Waiting slightly >0, e.g. a small
   admission buffer): this reproduces saturation's skew-suppression (run_skew → 29
   at 6144) *without* saturation's 38 s TTFT — let a dipped rank refill instantly
   from a 1-2-deep local queue instead of waiting for the next round-robin assign.
3. **Operate at the knee.** Given the ceiling is reached by ~3072 and the underfeed
   tax is ~2.6 %, the simplest "fix" is to run at concurrency ~3072-4096 for the
   best throughput/latency point and accept the small tax, rather than over-driving
   to 6144+.

The highest-value next experiment is lever 1 at conc 3072 — it both validates the
mechanism (does refilling the dip collapse run_skew?) and quantifies the available
recovery.

## Artifacts

All paths on ptyche persistent Lustre,
`/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/<job>/`:

- **Sweep: `outputs/2191166/`** (node `ptyche0267`) — backend log
  `logs/ptyche0267_agg_w0.out`, per-point client traces + results
  `logs/sa-bench_isl_2_osl_1024/request_trace_concurrency_{2048,3072,4096,6144}_gpus_4.jsonl`
  and `results_concurrency_<C>_gpus_4.json`, rollup `logs/benchmark-rollup.json`.
- Node-confound control (saturated nsys): `outputs/2191165/` — see
  `FINDINGS-4-DRAIN.md` (closes the topology vs drainage question).
- Recipe (this repo, `dep/dep-bubble/`):
  `qwen3-235b-a22b-vllm-agg-ptyche-gb200-dp4-ep-round-robin-sweep-conc2048to6144.yaml`.
- Analysis script (this repo, `dep/dep-bubble/`): `sweep_skew.py`
  (`<job_log_dir> <node_prefix> <conc1,conc2,…> [--gpus 4]`).
