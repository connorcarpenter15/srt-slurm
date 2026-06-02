# DEP Bubble — Step 3 (Nsys Kernel-Level) Findings

Date: 2026-06-01
Source data: job **1977752** (Lyris GB200, 4×GB200, 1 node, aggregated).
Faithful clone of the post-9915 good run (1918158): Qwen3-235B-A22B-Instruct-2507-FP4,
vLLM `--data-parallel-size 4 --enable-expert-parallel`, FP8 KV, `max-num-seqs 864`,
`stream-interval 50`, SA-Bench `isl=2 osl=1024 concurrency=8192`. Dynamo @ `c70bdbe`.
Recipe: `qwen3-235b-a22b-vllm-agg-lyris-gb200-dp4-ep-round-robin-nsys.yaml`.
Capture: time-based Nsight Systems (`nsys profile -t cuda,nvtx --cuda-graph-trace=node`,
`--delay 1100 --duration 20`), one `.nsys-rep` per DP rank.
Analysis: ad-hoc sqlite queries over the four per-rank exports (see Artifacts).

> **⚠ CORRECTED 2026-06-02 (twice).** The expert-load-skew mechanism claimed
> below is **refuted**. Three independent kernel-level measures show per-rank
> compute is balanced, so the stable "rank3 straggler / rank2 leader" pattern is
> *not* expert imbalance. A GPU↔ring permutation control (**[job 1987608](#permutation-control--job-1987608-2026-06-02-the-ring-position-story-is-incomplete)**)
> then showed it is *not* a portable "fixed ring-position" effect either, and not
> a single bad GPU: it is a **ring↔NVLink-topology mapping artifact** — the stable
> ordering is a property of the default identity `dp_rank↔GPU` alignment, and
> decoupling them dissolves the leader and widens the barrier skew ~10×. The raw
> measurements below stand; their interpretation is superseded by the
> **[§ Correction](#correction-2026-06-02-the-straggler-is-an-nccl-ring-artifact)**
> and the **[permutation control](#permutation-control--job-1987608-2026-06-02-the-ring-position-story-is-incomplete)**.
> What also stands: EP collectives are ~25–28% of wall — genuine all-to-all comm
> cost. What does **not**: EPLB/expert-rebalancing would *not* recover it.

## Question

FINDINGS-1 (coarse engine logs) established the qualitative fingerprint — equal
queue depth, unequal per-rank speed — but interval averaging and workload
symmetry meant the logs could only put a *floor* on the cost (~0.86% mean) and
could not see the per-step EP-barrier wait. Step 3 measures that wait directly,
at GPU-kernel granularity, for one saturated window.

## Result: the EP barrier bubble is real and quantified

In DP-attention + EP-expert serving, every MoE layer dispatches/combines tokens
across the four DP ranks via a pair of NCCL collectives
(`ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` = combine,
`ncclDevKernel_AllGather_RING_LL` = dispatch). The ranks step in lockstep: each
launches its collective when its *local* attention+gating compute finishes, and
the RING_LL kernel then spin-waits until the **slowest** rank arrives. The early
arriver's extra spin-wait is the bubble.

### EP collectives dominate the decode step

Per rank, over the 20s capture (~376 decode steps, ~94 MoE layers each):

| rank | wall (s) | GPU idle | in EP collectives | #ReduceScatter | #AllGather |
|---|---:|---:|---:|---:|---:|
| 0 | 19.93 | 6.7% | **25.8%** | 35 405 | 35 406 |
| 1 | 19.90 | 4.3% | **28.1%** | 36 605 | 36 605 |
| 2 | 19.89 | 9.5% | **25.0%** | 34 050 | 34 050 |
| 3 | 19.91 | 4.5% | **26.4%** | 36 627 | 36 628 |

A **quarter of every rank's wall-clock is inside the EP all-to-all.** That is the
unavoidable-transfer + barrier-wait cost combined; the wait portion is isolated
below.

### Barrier arrival skew (the bubble itself)

Matching collectives across ranks by absolute UTC timestamp (`--ts-normalize`),
30 854 cleanly-aligned 4-rank ReduceScatter barriers in the 17.1s overlap region:

```
per-barrier arrival skew (latest − earliest rank), ReduceScatter:
  mean 38.7us   p50 37.1us   p90 59.3us   p99 82.0us
per-barrier arrival skew, AllGather:
  mean 24.1us   p50 19.0us   p90 27.4us   p99 120.8us
```

Aggregated to per-rank idle spin-wait at the barrier (RS + AG combined):

```
EP-barrier idle as % of wall (fast ranks wait for slow ranks):
 rank0  5.4% ############
 rank1  3.7% ########
 rank2  8.9% ####################   <- fastest rank, idles most
 rank3  1.6% ###                    <- straggler, others wait for it
        mean 4.9% lost to the EP barrier
```

### The straggler is *stable*, not rotating

FINDINGS-1 guessed the laggard rotates step-to-step. It does not. Who arrives
first vs last is consistent across the whole window:

| rank | earliest-arriver (RS) | straggler (RS) | straggler (AG) |
|---|---:|---:|---:|
| 0 | 10.2% | 23.3% | 2.3% |
| 1 | 10.8% | 24.5% | 6.3% |
| 2 | **78.4%** | 1.3% | 2.0% |
| 3 | 0.7% | **50.8%** | **89.5%** |

Binned into ten ~1.7s slices, **rank2 is the earliest arriver in every bin**
(70–90%) and **rank3 is the dominant straggler in every bin**. The speed
ordering rank2 > rank0 ≈ rank1 > rank3 is fixed for the entire capture.

## Interpretation

- **PR-9915 + `stream-interval` balanced request *routing*** (1918158 recovered to
  69.1k tok/s; counts per rank equalize). What remains is a deeper, separate
  layer: **EP expert-load imbalance**. Each rank hosts a fixed shard of experts;
  if the hot experts of this workload happen to live on rank3, rank3's expert
  GEMM is consistently the largest, so rank3 is the perpetual straggler and the
  other ranks idle at the all-to-all waiting for it.
- This is the mechanism FINDINGS-1's symmetry argument pointed at: equal *total*
  tokens per rank (osl=1024, even request split) coexists with unequal *per-step*
  expert load, because token→expert routing is not uniform. The interval logs
  averaged that away; the per-barrier view exposes it.
- **Cost.** The fastest rank (rank2) wastes ~8.9% of wall purely spin-waiting at
  EP barriers; mean across ranks ~4.9%. System decode throughput is gated by the
  straggler, so on the order of ~5% of decode capacity is lost to the EP bubble
  here — ~5× the 0.86% coarse-log floor, confirming FINDINGS-1's "structural
  undercount" call. This is *additive* to (not the same as) any routing
  imbalance, and it is intrinsic to static expert sharding under skewed expert
  popularity.

## Caveats

- **nsys overhead.** This run measured 58.5k tok/s vs 1918158's 69.1k (~15%
  lower) because `--cuda-graph-trace=node` traces every CUDA-graph node. Absolute
  step times are inflated; the *relative* per-rank skew and straggler structure
  are unaffected (all four ranks pay the same tracing tax).
- **Capture window = backlog drain.** The client finished its measured requests
  at ~17:24; with mean TTFT 63s and concurrency 8192 the server kept decoding its
  queue. The capture (~17:27) landed in that saturated decode backlog — confirmed
  saturated: 35 405 RS / 20s ÷ ~94 layers ≈ 53 ms/step, matching the 51.6 ms
  TPOT. Pure decode, no competing prefills — a clean steady-state, if not
  mid-benchmark.
- **Per-rank window offsets.** Each rank's `--delay 1100` ran from its own worker
  launch; the four launched ~2.8s apart, so the windows are offset by up to
  ±1.4s. Index-aligned cross-rank matching is therefore invalid (it yielded
  multi-second nonsense skews); all barrier-skew numbers above use
  absolute-timestamp matching over the 17.1s overlap region.
- `deviceId` reads 0 in every export because each DP rank is a separate process
  with its own `CUDA_VISIBLE_DEVICES`; rank identity comes from the per-process
  file name (`..._rank{N}_profile`), not `deviceId`.

## Correction (2026-06-02): the straggler is an NCCL RING artifact

The original *Interpretation* above attributes the stable straggler to
**expert-load skew** — the hot experts of this workload supposedly living on
rank3, making rank3's expert GEMM perpetually largest. Three kernel-level
measures over the same four profiles refute that. The arrival skew is real, but
its cause is the fixed NCCL ring traversal order, not unequal expert load.

### Test 1 — expert load is uniform (mean gridY per rank)

For the dynamic grouped-GEMM kernels, `gridY` is the token-tile dimension: it
scales with the number of tokens routed to a rank's *local* experts. If rank3
hosted the hot experts, its expert-GEMM `gridY` would be larger. It is not —
identical to 0.01% across all four ranks:

```
bucket            r0          r1          r2          r3      max/min
expert_up      247.499     247.483     247.519     247.482    1.0001x
expert_down    247.499     247.483     247.519     247.482    1.0001x
fi_fp4_gemm     33.979      33.979      33.978      51.933    1.5284x   (*)
moe_finalize     1.000       1.000       1.000       1.000    1.0000x
topk_gating      1.000       1.000       1.000       1.000    1.0000x
```

`max/min ≈ 1.0001` for both expert GEMMs ⇒ **no expert-load skew.** Token→local-
expert routing is uniform across ranks. (*) `fi_fp4_gemm` shows a rank3 1.53×
`gridY` anomaly — a separate, smaller effect on one fused-MoE GEMM, not the
combine/dispatch GEMMs that dominate; noted, not load-bearing for this verdict.

### Test 2 — per-rank compute is balanced, and the "leader" is not the fastest GPU

Per-kernel mean durations (slow/fast ratio) and grid shapes:

```
bucket          r0      r1      r2      r3    slow/fast
attn_decode    98.58   94.11  102.70   93.81   1.095x   <- rank2 is the SLOWEST here
expert_up      86.31   80.69   79.83   84.15   1.081x
expert_down    45.83   45.44   44.23   45.83   1.036x
combine_rs     80.11   86.42   87.09   79.00   1.102x
dispatch_ag    65.36   66.58   59.16   64.36   1.126x
```

No rank is uniformly slow; spreads are 3–13%. Expert-GEMM grids are **identical**
across ranks (`expert_up` gridY[215,287], `expert_down` gridY[215,287] on all
four) — confirming Test 1's "no token skew." Crucially, **rank2 has the slowest
attention** (102.7µs) yet (Test 3) finishes every collective first. A
compute-bound straggler story cannot survive that.

### Test 3 — START-skew (compute arrival) vs END-skew (ring completion)

Matching 4-rank collective quartets by absolute UTC timestamp and separating the
spread in *start* times (compute arrival) from the spread in *end* times (ring
completion):

```
ReduceScatter (combine), 30 854 quartets:
  START-skew (compute arrival): mean 38.7us  p50 37.1us  p90 59.3us  p99 82.0us   (loose)
  END-skew   (ring completion): mean 22.9us  p50 22.8us  p90 24.4us  p99 25.1us   (tight)
  earliest-to-END (finishes first): r0:0.0%  r1:0.0%  r2:100.0%  r3:0.0%
  latest-to-END   (finishes last) : r0:0.0%  r1:34.7%  r2: 0.0%  r3:65.3%
AllGather (dispatch), 30 821 quartets:
  earliest-to-END (finishes first): r0:0.0%  r1:0.0%  r2:100.0%  r3:0.0%
  latest-to-END   (finishes last) : r0:0.0%  r1: 5.8%  r2: 0.0%  r3:94.2%
```

The END distribution is razor-tight (p99−p50 ≈ 2µs) and **100% rank-pinned**:
rank2 completes the collective first in *every* quartet, rank3 last. A load-driven
straggler would jitter step-to-step; perfect determinism is the signature of a
fixed traversal order. In a RING collective, completion order is set by ring
position (= NCCL rank = `--data-parallel-rank`), not by who arrived first.

```
who finishes the EP collective first (earliest-to-END), ReduceScatter:
 rank0    0% 
 rank1    0% 
 rank2  100% ####################   <- ring head, every single step (yet slowest at attention)
 rank3    0% 
```

The 37µs START-skew (the "bubble") is real but **inherited**: the rank the ring
hands data to first (rank2, earliest-to-START 78%) starts the next layer's
compute first and so arrives at the next barrier first — self-perpetuating ring
serialization, on top of balanced compute (Tests 1–2), not expert imbalance.

### What we got wrong

- **"Driver is expert-load skew across the fixed EP shard."** Refuted — mean
  expert-GEMM gridY is identical to 0.01% across ranks; no rank hosts hotter
  experts.
- **"rank3 is the perpetual straggler because its expert GEMM is consistently
  largest."** Refuted — rank3's expert GEMM is not the largest; per-rank compute
  is balanced. rank3 is *latest-to-END* because of its ring position, not its
  load.
- **"~5% recoverable; EPLB would help."** Unsupported — EPLB rebalances experts
  across ranks; it cannot change a fixed ring completion order, and there is no
  expert imbalance here for it to fix.
- The "stable straggler" framing conflated barrier-arrival order with a compute
  bubble. The stability is deterministic ring order, not a slow computer.

### What still stands

- **EP collectives are ~25–28% of every rank's wall-clock** (Result table) —
  genuine, largely-unavoidable all-to-all communication cost intrinsic to
  DP-attention + EP-experts at this scale. Not recoverable by load balancing.
- The per-barrier arrival skew (~37µs p50) is real and measured; only its *cause*
  is reattributed (ring serialization, not expert skew).
- All capture caveats (nsys overhead, saturated backlog window, per-rank window
  offsets, `deviceId=0`) are unchanged.

### Permutation control — job 1987608 (2026-06-02): the ring-position story is *incomplete*

To isolate ring position from physical GPU I ran **job 1987608**, identical to
baseline 1977752 except `SRT_DP_GPU_PERMUTATION=reverse`, which reverses the
`dp_rank → physical-GPU` mapping while keeping `--data-parallel-rank` =
profile-`rank{N}` sequential. The reversed mapping is confirmed in
`sweep_1987608.log`: dp_rank 0→GPU3, 1→GPU2, 2→GPU1, 3→GPU0.
Recipe: `qwen3-235b-a22b-vllm-agg-lyris-gb200-dp4-ep-round-robin-nsys-gpuperm-rev.yaml`.

**Prediction (binary):** profile `rank2` (ring position 2, now on GPU 1) stays
the 100% earliest-to-END leader ⇒ pure ring-position artifact; *or* the leader
follows physical GPU 2 (now dp_rank 1) ⇒ residual hardware effect.

**Neither happened.** Same `barrier_startend.py`, same THRESH, both runs:

| measure (ReduceScatter) | baseline 1977752 (identity) | permuted 1987608 (reverse) |
|---|---|---|
| earliest-to-END | **r2 = 100%** (r0/r1/r3 = 0%) | r0:14% r1:28% r2:30% r3:29% (diffuse) |
| latest-to-END   | r3:65% r1:35% (r0/r2 = 0%)  | r0:12% r1:29% r2:29% r3:30% (diffuse) |
| END-skew p50 / p99 | **22.8 / 25.1 µs** (razor-tight) | **247.6 / 470.9 µs** (~10× wider) |
| START-skew p50 / p99 | 37.1 / 82.0 µs | 244.3 / 465.4 µs |
| START vs END | END ≪ START (tight completion) | END ≈ START |

```
earliest-to-END (ReduceScatter), who leads the ring:
                baseline (identity)         permuted (reverse)
 rank0    0%                          14%  ###
 rank1    0%                          28%  ######
 rank2  100%  ####################    30%  #######
 rank3    0%                          29%  ######
           one fixed leader             no fixed leader
```

Three consequences:

1. **The strong "completion order is fixed by ring position, independent of
   hardware" claim (stated in the Correction above and the first Verdict) is
   over-stated.** If it were true, `rank2` — still ring position 2 — would remain
   the 100% leader. It does not; leadership dissolves into a near-uniform spread.
   Ring *position* alone does not pin the leader.
2. **It is not a single fast/slow GPU either.** The leader does not migrate to a
   predictable rank when GPUs are permuted; it simply de-concentrates. Combined
   with Tests 1–2 (balanced grids/durations), **expert-load skew *and* a bad-GPU
   explanation are both refuted.**
3. **The real driver is the ring↔topology *mapping*.** The baseline's tight,
   100%-pinned completion order is a property of the *natural identity alignment*
   between NCCL's ring and the GB200 NVLink topology. Break that alignment and the
   order destabilises and the skew grows ~10× (and START≈END) — a topological
   serialization artifact, not load and not an abstract ring index.

One ring-position fingerprint *did* survive: profile `rank0` (ring position 0,
the `--data-parallel-address` coordinator) is the "always-middle" rank in **both**
runs — 0% earliest-END and 0% latest-END in baseline, and the least-extreme rank
(~12–14%) under permutation.

**Confound (honest):** the two captures are on *different physical nodes*
(lyris0243 vs lyris0153), so node effects are not fully separable from the mapping
change. A same-node identity-vs-reverse pair would be the clean control. The
*qualitative* change — a single 100% leader → no leader, 23µs → 248µs END-skew —
is far larger than run-to-run node noise alone would explain, but this caveat is
not eliminated by the present data.

## Verdict (revised 2026-06-02, refined after permutation control)

Under DEP MoE serving the ranks reach the EP all-to-all at staggered times, with a
stable, 100%-pinned completion order **in the baseline identity mapping** (rank2
leads, rank3 trails for the whole capture). **What this is NOT:** it is not
expert-load skew (Test 1: mean expert-GEMM gridY identical to 0.01% across ranks),
and not a single slow/fast GPU (Test 2: per-kernel durations within 3–13% with no
uniformly-slow rank; the permutation control does not migrate the leader to a
fixed GPU). **EPLB/expert rebalancing would not recover anything** — there is no
expert imbalance to fix.

**What it IS:** a ring↔NVLink-topology serialization artifact. The stable
single-leader ordering is a product of the default identity `dp_rank↔GPU`
alignment; the permutation control (job 1987608) shows that decoupling ring
position from physical GPU *destroys* the fixed ordering and widens the barrier
skew ~10×, rather than moving the leader to a new rank or GPU. Completion order is
therefore **not a portable function of ring index** (the earlier "ring-position
artifact" wording over-claimed) — it is a function of how the NCCL ring is laid
onto the physical topology.

The ~25–28% of wall spent in EP collectives remains genuine, largely-unavoidable
all-to-all communication cost and stands. Both the original expert-skew verdict
and the intermediate "pure ring-position" wording are withdrawn in favor of the
topology-mapping account above.

## Artifacts

All on Lyris Lustre under job 1977752:

- Profiles (4 per-rank, raw + converted):
  `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1977752/logs/profiles/agg/lyris0243_agg_w0_rank{0,1,2,3}_profile.{qdstrm,nsys-rep,sqlite}`
- Sweep log: `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1977752/logs/sweep_1977752.log`
- Benchmark result:
  `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1977752/logs/sa-bench_isl_2_osl_1024/results_concurrency_8192_gpus_4.json`
  (58.5k tok/s, mean TTFT 63.2s, mean TPOT 51.6 ms, 38 016 completed in 619s)
- Staged aarch64 Nsight Systems 2025.3.2 (bind-mounted over the container's
  dangling symlink; also used to convert `.qdstrm`→`.nsys-rep`→`.sqlite`):
  `/lustre/fsw/coreai_dlfw_dev/connorc/tools/nsight-systems/2025.3.2/`

Permutation control — job **1987608** (`SRT_DP_GPU_PERMUTATION=reverse`, node
lyris0153):

- Profiles (4 per-rank, raw + converted):
  `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1987608/logs/profiles/agg/lyris0153_agg_w0_rank{0,1,2,3}_profile.{qdstrm,nsys-rep,sqlite}`
- Sweep log (records reversed dp_rank→GPU mapping):
  `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1987608/logs/sweep_1987608.log`
- On-cluster `.qdstrm`→`.sqlite` conversion driver (the in-job nsys auto-import
  was killed by benchmark cleanup, leaving only `.qdstrm`):
  `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1987608/convert.sh`
- Generalized analysis copy (BASE as `argv[1]`, globs per-rank sqlite):
  `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1987608/barrier_startend.py`

Correction (2026-06-02) analysis scripts — stdlib-only, run against the four
per-rank `*.sqlite` above (edit `BASE` for other jobs):

- `dep/dep-bubble/gridy_skew.py` — Test 1: mean `gridY` (token-tile dim) per
  rank for the grouped-GEMM kernels; the sharp expert-load-skew test.
- `dep/dep-bubble/kernel_decomp.py` — Test 2: per-rank kernel bucketing (AVG/TOTAL
  duration + grid min/max) — attention as the balanced control, expert GEMM as
  treatment.
- `dep/dep-bubble/barrier_startend.py` — Test 3: timestamp-matched 4-rank
  collective quartets, separating START-skew (compute arrival) from END-skew
  (ring completion) and tallying earliest/latest-to-START/END per rank.
