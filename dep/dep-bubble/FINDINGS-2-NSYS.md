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

## Verdict

The bubble hypothesis is confirmed at kernel granularity. Under DEP MoE serving,
DP ranks reach the EP all-to-all at different times (median 37µs spread per
ReduceScatter), a stable straggler (rank3) gates the barrier, and the fastest
rank (rank2) idles ~9% of wall waiting — costing ~5% of decode throughput on top
of any routing effects. The driver is expert-load skew across the fixed EP shard,
not request distribution.

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
