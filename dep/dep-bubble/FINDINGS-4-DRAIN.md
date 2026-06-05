# DEP Bubble — Drainable-Queue Probe Findings (3-shape campaign)

Date: 2026-06-04
Cluster: **ptyche** GB200, 1 node each.
Jobs:
- `2188160` — **drain test** (isl=2, osl=1024, **concurrency=2048**) — node `ptyche0056`
- `2188157` — **mixed** (isl=1024, osl=1024, concurrency=8192) — node `ptyche0072`
- `2188156` — **higher-ISL prefill confirm** (isl=4096, osl=1, concurrency=8192) — node `ptyche0108` — **CRASHED (CUDA OOM at cudagraph capture)**

Analysis: `backend_log_summary.py`, `trace_summary.py`.

## Question

`FINDINGS-3-PREFILL.md` concluded the underfeed bubble is a **decode-regime**
phenomenon that needs a **drainable queue**: a rank reaches `Waiting: 0` with
`Running < cap`, starves, and the other ranks wait for it at the per-MoE-layer EP
all-to-all. Every run so far was *saturated* — the queue never drained, so that
precondition was never realized and the macro metrics showed the ranks perfectly
matched.

This campaign was designed to **re-create the precondition**. Three shapes:

1. **Drain (the key test):** canonical decode shape (isl=2/osl=1024) but at
   concurrency **2048** instead of 8192. With ~512 requests/rank under a
   `max-num-seqs=864` cap, steady state should hold `Running ~512 < 864` with
   `Waiting ~0` — the queue drains. Does the system fall out of lockstep when it
   can?
2. **Mixed:** isl=1024/osl=1024 at concurrency 8192 — interleaves substantial
   prefill and decode. Does a mixed shape drain or stay saturated?
3. **Higher-ISL prefill confirm:** isl=4096/osl=1 — confirm the FINDINGS-3
   prefill-saturation result generalizes to a 4× larger prefill.

## Result: the drainable queue DOES surface temporal running-skew

Only the drain run achieves the precondition (`Waiting=0` with `Running<cap`), and
it is the **only** run in the whole investigation where macro per-window running
counts fall out of lockstep — `running_skew` up to **89** — even though the final
per-rank counts and per-rank average durations stay essentially identical.

### Headline benchmark numbers

| Metric | Drain (2188160) | Mixed (2188157) |
|---|---:|---:|
| Concurrency | 2048 | 8192 |
| Shape (isl/osl) | 2 / 1024 | 1024 / 1024 |
| Total token throughput (tok/s) | 56,709 | 45,676 |
| Output token throughput (tok/s) | 56,599 | 20,880 |
| Request throughput (req/s) | 55.27 | 24.21 |
| Median TTFT | **254 ms** | 130.3 s |
| Mean TTFT | **619 ms** | 109.5 s |
| Mean TPOT | 35.3 ms | 117.2 ms |
| Mean E2EL | 36.7 s | 182.5 s |
| Completed requests | 20,480 (all) | 10,338 (window-limited; 426.9 s) |

The drain run's **254 ms median TTFT** (vs. tens of seconds when saturated) is the
direct signature of de-saturation: requests are admitted almost immediately
because the queue is empty. The mixed run is the opposite — a 130 s median TTFT
and a queue that never drains.

## Drain (2188160): Waiting=0, Running<cap — the precondition is realized

`Engine NNN:` interval logs, 192 samples (~8 min), 10 s bins:

```
Running per rank (steady state, cap = 864):
  dp0  ███████████████████░  452.7 mean   (p50 512)
  dp1  ███████████████████▌  462.2 mean   (p50 512)
  dp2  ████████████████████  469.7 mean   (p50 512)
  dp3  ███████████████████▌  460.4 mean   (p50 512)
                              ^ all well below the 864 cap — DE-SATURATED

Waiting per rank (mean):
  dp0  0.0      dp1  0.0      dp2  0.0      dp3  0.0
                              ^ queue fully drained on EVERY rank, EVERY window
```

This is the exact `Running < cap, Waiting = 0` condition that no saturated run
ever reached. And under it, the ranks do **not** stay in lockstep:

| skew metric | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| running_skew | 3.3 | 0.0 | 23.0 | **89.0** |
| waiting_skew | 0.0 | 0.0 | 0.0 | 0.0 |
| gen_skew | 33.1 | 3.3 | 154.8 | 242.0 |

Worst window (t+160 s): one rank fell to `running ≈ 403` while another held ~492
— a `running_skew` of **89 (~17 % of the ~512 mean)**. With `Waiting = 0`, a rank
that dips cannot immediately refill from its queue; it must wait for the next
round-robin assignment, so its running count stays low relative to peers until new
work arrives. That is the temporal-underfeed mechanism, made macro-visible for the
first time.

Contrast with the two regimes that **cannot** show this:
```
running_skew (max, per 10s window):
  saturated decode (osl=1024, conc=8192)  ~0    running pinned at 864 cap
  prefill-heavy   (isl=1024, osl=1)       ~0    running pinned at ~3 (token-budget)
  mixed           (isl/osl=1024, conc=8192) 63  running floats but Waiting>0 (refills)
  DRAIN           (isl=2/osl=1024, conc=2048) 89  Waiting=0, Running<cap  ← precondition met
```

### But final counts and per-rank averages are still even

The skew is **transient** (p50 = 0): averaged over the whole run the ranks are
indistinguishable. `trace_summary` `backend_dp_enter → backend_dp_done`:

| DP rank (source-log proxy) | joined reqs | backend_duration mean (s) | p95 (s) |
|---|---:|---:|---:|
| dp0 (…1801651) | 5,120 | 36.328 | 37.449 |
| dp1 (…1801675) | 5,120 | 36.346 | 37.455 |
| dp2 (…1801836) | 5,120 | 36.325 | 37.426 |
| dp3 (…1801862) | 5,120 | 36.341 | 37.447 |

Per-rank mean duration spread = **21 ms on a 36.3 s base = 0.058 %**; requests
split exactly 5,120/rank. **This is precisely the dep/CLAUDE.md warning in
action:** balanced final counts and balanced averages prove nothing — the failure
mode lives in the short windows, where `running_skew` reaches 89 while the run
average is flat.

Caveat on magnitude: at concurrency 2048 the system has ample slack (median TTFT
254 ms), so this transient skew does **not** cost throughput here — it is a
*visibility* result (the precondition produces macro skew), not a demonstrated
throughput regression.

## Mixed (2188157): saturated — precondition never met

192→320 samples, 8192 concurrency. Running floats but pins to the cap in the upper
tail; the queue never drains:

```
Running per rank: mean ~675-691, p95 = 864 (hits cap), max 864
Waiting per rank: mean ~1000-1024, p95 ~1440, max ~1800   ← never near 0
```

| skew metric | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| running_skew | 4.3 | 2.0 | 17.0 | 63.0 |
| waiting_skew | 4.9 | 3.0 | 15.0 | 21.0 |
| gen_skew | 236.3 | 89.0 | 933.5 | 1437.7 |

`running_skew` reaches 63, but because `Waiting` stays ~1000 on every rank a
dipping rank refills instantly from its backlog — this is saturation jitter, not
the underfeed condition. The mixed shape behaves like the canonical saturated
decode baseline, not like the drain run. (It was also window-limited: the 426.9 s
sa-bench measurement window completed only 10,338 of the 81,920 requested, because
each request takes ~182 s e2e.)

## Higher-ISL prefill confirm (2188156): CUDA OOM, no data

The job never served. All four DP workers hit **`CUDA out of memory`** during
`capture_model()` (cudagraph capture): "Tried to allocate 1.44 GiB. GPU 0 has a
total capacity of 184.31 GiB of which 1.43 GiB is free … 4.95 GiB allocated in
private pools (e.g., CUDA Graphs)." The model then failed the 1800 s health gate.

Root cause: the recipe bumped `max-model-len`, `max-num-batched-tokens`, and
`max-cudagraph-capture-size` all to **8192** so the 4096-token prompt fits. At
`gpu-memory-utilization: 0.90` the 8192-ctx FP8-KV reservation plus a cudagraph
capture list running all the way to size 8192 overcommitted the 184 GiB GB200.
Prefill is token-budget bound (`running ≈ 2`), so large cudagraphs buy nothing.
Fix for the rerun: set `enforce-eager: true` (or `max-cudagraph-capture-size`
small, e.g. 512) and/or lower `gpu-memory-utilization` to ~0.80. This run only
*confirms* the already-established prefill result, so it is the lowest-priority
resubmit.

## nsys: captures failed this round (mechanical, not analytical)

None of the three jobs produced usable `.nsys-rep`:

- **Drain (2188160): nothing.** `delay_secs=800` (from worker launch) fired at
  ~15:07:41, but the de-saturated run is short and finished at 15:08:02 — the
  capture window opened ~20 s before teardown and never wrote. The traffic window
  was ~6 min starting ~460 s after worker launch; the correct delay is ~**560 s**
  (≈2 min into steady state), not 800.
- **Mixed (2188157): four `.qdstrm`, no `.nsys-rep`.** The time-based capture
  streamed all four ranks but the raw `.qdstrm` was never finalized to
  `.nsys-rep` — the container's Nsight tree has only the **target** subtree
  (the bind-mount fix for the dangling symlink); the host `QdstrmImporter` used
  for finalization is absent. The `.qdstrm` files are recoverable offline with a
  full Nsight host install.
- **Higher-ISL (2188156): nothing** (OOM before serving).

Consequence: the **kernel-level EP-barrier confirmation** (does the dipping rank
actually idle at the all-to-all?) was missing for the drain regime after this
round. The macro signal (running_skew=89 under Waiting=0) was in hand; tying it to
barrier idle time needed a retimed drain-run nsys capture — **job 2188338**, the
next section.

## nsys kernel-level EP-barrier (job 2188338 — retimed drain capture)

The drain shape was resubmitted with `delay_secs=560` (job `2188338`, node
**ptyche0162**, default/identity GPU permutation). It captured four ~43 MB
`.qdstrm` in steady state; finalized offline with an x86 Nsight host install
(`QdstrmImporter` → `.nsys-rep` → `nsys export --type sqlite`, ~165 MB / ~2.5 M
events per rank). Analysis: `drain_barrier.py` (START- vs END-skew per collective
quartet) and `drain_kernel_decomp.py` (per-rank kernel-shape decomposition).

### Barrier skew is arrival-dominated and rotates across ranks

`drain_barrier.py`, 43,162 aligned 4-rank quartets (ReduceScatter combine;
AllGather dispatch is within ~1 µs on every figure):

```
START-skew (compute arrival): mean 214.7us  p50 203.2us  p90 323.2us  p99 375.8us
END-skew   (ring completion): mean 214.9us  p50 204.6us  p90 323.6us  p99 376.5us

latest-to-START (straggler): r0:13.9%  r1:27.7%  r2:29.4%  r3:29.0%
earliest-to-START (leader) : r0:11.2%  r1:29.9%  r2:31.2%  r3:27.7%
latest-to-END (finish last): r0:13.0%  r1:28.1%  r2:30.6%  r3:28.3%
earliest-to-END (finish 1st): r0:12.1%  r1:29.4%  r2:30.2%  r3:28.3%
```

Two structural facts:

1. **START-skew ≈ END-skew** (both ~203 µs p50, ~376 µs p99). The barrier spread
   is set by *when ranks arrive* — the ring transit adds essentially zero extra
   spread. This is the kernel-level signature of genuine arrival skew: ranks
   reach the per-MoE-layer all-to-all at different wall-clock times (~203 µs apart
   at the median, ~376 µs in the p99 tail) and whoever is ahead idles in the
   collective until the laggard shows up.
2. **No fixed straggler — the lead rotates.** Leader/straggler roles spread
   roughly uniformly over r1/r2/r3 (~28-31 % each), with r0 slightly under
   (~11-14 %). No rank is structurally early or late. This matches the macro
   picture exactly: the `running_skew=89` dip is transient (p50=0) and rotates
   among ranks, so at the kernel level the early-arriver rotates too.

This is qualitatively unlike the **lyris identity saturated baseline**
(`FINDINGS-2-NSYS.md`), where START (37 µs p50) ≫ END (23 µs p50), END was
razor-tight, and one rank (r2) was the earliest-to-END **100 %** of the time — a
pure ring-position artifact. Here the skew has moved out of the ring and into
arrival timing, and de-concentrated.

### The arrival skew is NOT compute-load skew

`drain_kernel_decomp.py`, per-rank kernel AVG-duration slow/fast ratios:

| bucket | r0 | r1 | r2 | r3 | slow/fast |
|---|---:|---:|---:|---:|---:|
| attn_decode (GPU-speed control) | 65.95 | 66.02 | 65.83 | 65.04 | **1.015×** |
| expert_up (EP-sharded GEMM) | 61.88 | 58.88 | 57.64 | 60.49 | 1.074× |
| expert_down (EP-sharded GEMM) | 37.25 | 35.41 | 33.98 | 36.25 | 1.096× |
| moe_finalize | 17.46 | 17.24 | 16.59 | 17.08 | 1.052× |
| topk_gating | 9.04 | 8.64 | 8.77 | 8.68 | 1.047× |
| combine_rs (ReduceScatter) | 61.33 | 66.14 | 68.17 | 63.02 | 1.111× |
| dispatch_ag (AllGather) | 37.06 | 37.59 | 33.69 | 42.48 | 1.261× |

- **attn_decode is balanced to 1.015×** (gridY=4 constant, n≈48 k each) → raw GPU
  speed is uniform; no hardware asymmetry.
- **Expert GEMMs are balanced to 1.07-1.10×** with matching gridY ranges
  (135-167) → token→expert routing is even; **no expert-load skew**, exactly as
  `FINDINGS-2-NSYS` Test 1 found for the saturated regime.
- The only buckets carrying wider ratios are the **NCCL collectives** (combine_rs
  1.11×, dispatch_ag 1.26×) — because the collective kernel duration *absorbs the
  barrier wait* for the laggard. The variance lives in the wait, not the compute.

So the ~203 µs arrival skew is **queue/admission timing**, not compute imbalance:
all real compute (attention + expert GEMMs, the two dominant buckets at ~3,180 ms
and ~2,800-2,990 ms total per rank) is balanced to within 1.5-10 %, and the spread
shows up only in the collective wait. This is the mechanism FINDINGS-3/4 predicted
— a drained queue lets ranks fall out of lockstep on *when* they start each layer,
and they idle on each other at the EP all-to-all.

### Caveat: the magnitude carries the same node confound as FINDINGS-2

The drain ran **identity** permutation, yet its ~203 µs p50 / ~376 µs p99 START≈END
diffuse pattern matches the lyris **reverse-permutation** control
(`FINDINGS-2-NSYS`: START 244 µs / END 248 µs p50, diffuse roles) far more closely
than the razor-tight lyris **identity** baseline (37 µs). But this run is on
**ptyche0162**, a different node, and there is **no saturated kernel-level baseline
on ptyche** (FINDINGS-3 prefill was macro-only; the other FINDINGS-4 nsys captures
failed). So the ~5× widening over the lyris-identity baseline cannot be attributed
cleanly to *drainage* vs. ptyche0162's GPU↔NVLink ring mapping — "identity on
ptyche" may simply correspond to a non-trivial ring order, like "reverse on lyris."
What *is* node-independent and solid: the skew is arrival-dominated, it rotates
(no fixed straggler), and it is **not** expert-load or GPU-speed skew. Separating
the magnitude needs the same-node saturated-vs-drain pair on a single ptyche node
— **job 2191165, the next section, which closes this.**

## nsys saturated control (job 2191165 — closes the node confound)

To split *drainage* from *ptyche ring mapping*, the canonical decode shape was
re-run **saturated** (concurrency **8192**, identity permutation, same
isl=2/osl=1024) on node **ptyche0287**. This holds node-class, shape, and
permutation FIXED relative to the drain capture (job 2188338, ptyche0162,
identity) and varies **only the regime**: saturated (`Running` pinned at the
864 cap, `Waiting ~658-702/rank, p95 1184`) vs. drained (`Running ≈ 512 < 864`,
`Waiting = 0`). nsys `delay_secs=640` landed ~180 s into steady-state saturation;
finalized offline to four ~300 MB sqlite (~4.1 M events each).

### Drainage widens the barrier ~2.6× on the *identical* node + permutation

`drain_barrier.py`, 104,815 aligned 4-rank quartets (ReduceScatter combine;
AllGather within ~1 µs):

```
                              START-skew (compute arrival)
regime (ptyche, identity, isl=2/osl=1024)   mean    p50    p90    p99
  SATURATED  (conc 8192, 2191165)          88.9us  78.9us 142us  250us
  DRAIN      (conc 2048, 2188338)         214.7us 203.2us 323us  376us
                                          ─────── ─────── ────── ──────
  drain / saturated                         2.4×    2.6×   2.3×   1.5×
```

Both regimes are **START ≈ END** (arrival-dominated, ring adds ~0 spread).
The only variable between them is concurrency, so the **2.6× median widening
(79 µs → 203 µs) is caused by drainage**, not by ptyche0xxx's GPU↔NVLink ring
order. The node confound from `FINDINGS-2`/the caveat above is closed: the
temporal-underfeed bubble is **real and regime-driven**.

```
START-skew p50 (µs), per regime:
  lyris identity SATURATED   ███▋                 37   (ring-pinned: START≫END, r2 leads 100%)
  ptyche identity SATURATED  ████████             79   (arrival-dominated, rotating)
  ptyche identity DRAIN      ████████████████████ 203  (arrival-dominated, rotating)
```

Two layers separate cleanly:

1. **Regime (the mechanism, ~2.6×):** drainage widens arrival skew 79 → 203 µs
   on fixed ptyche/identity/shape. This is the underfeed bubble.
2. **Node/topology (a constant offset, ~2×):** ptyche identity at *saturation*
   (79 µs) is already ~2× the lyris identity baseline (37 µs) — **and** it is
   arrival-dominated (START≈END, rotating roles) whereas lyris identity was
   ring-pinned (START≫END, r2 earliest-to-END 100 %). So "identity on ptyche"
   does behave like a non-trivial ring order, exactly as the caveat suspected —
   but that is a fixed ~2× baseline, and drainage stacks its 2.6× *on top* of it.

Roles in the saturated control rotate as in the drain (straggler r2/r3 ~31-32 %,
r0 lowest ~16-17 %; leader r1 ~42 %) — biased but not pinned, and START≈END, so
even saturated ptyche is arrival-skewed, just far less than when drained.

### Compute stays balanced in both regimes — the wait shrinks, not the skew source

`drain_kernel_decomp.py` slow/fast AVG-duration ratios, saturated vs drain:

| bucket | saturated slow/fast | drain slow/fast |
|---|---:|---:|
| attn_decode (GPU-speed control) | 1.094× | 1.015× |
| expert_up (EP GEMM) | 1.033× | 1.074× |
| expert_down (EP GEMM) | 1.024× | 1.096× |
| combine_rs (ReduceScatter) | 1.165× | 1.111× |
| dispatch_ag (AllGather) | 1.303× | 1.261× |

Same signature in both: real compute (attention + expert GEMMs) balanced to
within ~2-10 %, and the wider ratios live only in the collectives (which absorb
the barrier wait). Saturation does not change *what* is skewed — compute is even
either way — it only shrinks the arrival-wait the collective has to absorb
(79 µs vs 203 µs). The expert GEMM gridY ranges confirm even token→expert
routing in both (no expert-load skew).

## Verdict

- **The drain test succeeded at its goal:** concurrency 2048 realizes the
  `Running ≈ 512 < 864`, `Waiting = 0` precondition that every saturated run
  missed. Under it, macro per-window `running_skew` reaches **89** — the ranks
  fall out of lockstep — while final counts (5,120 each) and per-rank average
  durations (0.058 % spread) stay perfectly even. This is the clearest macro
  confirmation yet of *temporal* underfeed that averaged metrics hide.
- **Saturation suppresses the signal**, both ways: pinned-at-cap decode and
  token-budget-bound prefill both show `running_skew ≈ 0`; even the mixed shape,
  with a deep ~1000-request queue, refills instantly and never drains. Only a
  drainable queue exposes the skew.
- **Magnitude caveat:** at conc 2048 the transient skew costs no throughput
  (254 ms median TTFT — lots of slack). This is a visibility/mechanism result.
- **Kernel-level confirmation obtained** (job 2188338, retimed `delay_secs=560`,
  finalized offline): the per-MoE-layer EP barrier shows an **arrival-dominated**
  skew (START≈END, ~203 µs p50 / ~376 µs p99) that **rotates across ranks** (no
  fixed straggler) and is **not** compute-load skew (attn 1.015×, expert GEMM
  1.07-1.10× — balanced; the variance lives in the collective wait). Ranks reach
  the all-to-all at different times and idle on each other — the temporal-underfeed
  mechanism, now visible at the kernel level.
- **Node confound CLOSED** (job 2191165, saturated control, same ptyche node-class
  + identity perm + shape, only the regime varies): drainage widens the
  arrival-dominated barrier **2.6× at the median (79 µs → 203 µs p50)**, so the
  widening is the **drain regime, not ptyche ring mapping**. A residual ~2× node
  offset (ptyche identity 79 µs vs lyris identity 37 µs, and ptyche identity being
  arrival-dominated where lyris identity was ring-pinned) is a *constant* topology
  baseline; drainage stacks its 2.6× on top. Compute stays balanced in both
  regimes — saturation shrinks the wait the collective absorbs, not the skew
  source. (The higher-ISL OOM remains a config fix: enforce-eager / lower util.)

## Artifacts

All paths on ptyche persistent Lustre,
`/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/<job>/`:

- Drain (key): `outputs/2188160/` — `logs/benchmark-rollup.{json,csv}`,
  backend log `logs/ptyche0056_agg_w0.out`, per-DP traces
  `logs/dynamo_request_trace_vllm_{1801651,1801675,1801836,1801862}.jsonl`,
  client trace `logs/sa-bench_isl_2_osl_1024/request_trace_concurrency_2048_gpus_4.jsonl`,
  joined CSVs `logs/dp_trace_join/`.
- Mixed: `outputs/2188157/` — backend log `logs/ptyche0072_agg_w0.out`,
  per-DP traces `logs/dynamo_request_trace_vllm_{1112316,1112512,1112633,1112727}.jsonl`,
  raw nsys `logs/profiles/agg/ptyche0072_agg_w0_rank{0,1,2,3}_profile.qdstrm`
  (not finalized).
- Higher-ISL (crashed): `outputs/2188156/` — backend log `logs/ptyche0108_agg_w0.out`
  (CUDA OOM at `multiproc_executor.py:949`), no benchmark output.
- **Drain nsys (kernel-level): `outputs/2188338/`** (node `ptyche0162`,
  `delay_secs=560`) — raw `logs/profiles/agg/ptyche0162_agg_w0_rank{0,1,2,3}_profile.qdstrm`,
  finalized `…_profile.nsys-rep`, exported `…_profile.sqlite` (~165 MB / ~2.5 M
  events each). Offline finalizer: x86 Nsight 2025.3.1 host install at
  `/lustre/fsw/coreai_dlfw_dev/connorc/tools/nsight-host-x64/` (`QdstrmImporter`
  + `target-linux-x64/nsys export`).
- **Saturated control (closes the node confound): `outputs/2191165/`** (node
  `ptyche0287`, conc 8192, identity perm, `delay_secs=640`) — backend log
  `logs/ptyche0287_agg_w0.out`, raw
  `logs/profiles/agg/ptyche0287_agg_w0_rank{0,1,2,3}_profile.qdstrm`, finalized
  `…_profile.nsys-rep`, exported `…_profile.sqlite` (~300 MB / ~4.1 M events each).
- Recipes (this repo, `dep/dep-bubble/`):
  `qwen3-235b-a22b-vllm-agg-ptyche-gb200-dp4-ep-round-robin-drain-conc2048-nsys.yaml`,
  `…-mixed-isl1024-osl1024-nsys.yaml`,
  `…-prefill-isl4096-nsys.yaml`,
  `…-saturated-conc8192-nsys.yaml`.
- Offline finalizer: `finalize_nsys.sh <profile_dir> <node_prefix> [ranks…]`
  (x86 Nsight 2025.3.1 host at
  `/lustre/fsw/coreai_dlfw_dev/connorc/tools/nsight-host-x64/`).
- Analysis scripts (this repo, `dep/dep-bubble/`): `drain_barrier.py`
  (START/END-skew per collective quartet), `drain_kernel_decomp.py` (per-rank
  kernel-shape decomposition) — parameterized as `<sqlite_dir> <node_prefix>`.
