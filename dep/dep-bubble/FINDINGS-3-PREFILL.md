# DEP Bubble — Prefill-Heavy Probe Findings

Date: 2026-06-04
Cluster/job: **ptyche** GB200, 1 node, job `2187958`
Recipe: `dep/dep-bubble/qwen3-235b-a22b-vllm-agg-ptyche-gb200-dp4-ep-round-robin-prefill-isl1024.yaml`
Analysis: `backend_log_summary.py`, `trace_summary.py`.

## Question

The canonical DEP probe is decode-heavy (isl=2, osl=1024). There the temporal
"underfeed bubble" fingerprint shows up: equal final per-rank request counts, yet
short windows where a rank drains its queue and others idle at the per-MoE-layer
EP all-to-all (see `FINDINGS-1-COARSE.md`, `FINDINGS-2-NSYS.md`).

This run **inverts the shape** (isl=1024, osl=1, random_range_ratio=1.0 → every
request is a 1024-token prefill + exactly one decode token). Question: under a
**prefill-dominant** workload, do the DP=4 ranks still service requests at
different speeds / show temporal underfeed?

## Result: no underfeed, ranks are uniformly saturated and balanced

Headline benchmark (concurrency 8192, 81920 measured requests):

| Metric | Value |
|---|---:|
| Total token throughput | 99,038 tok/s |
| Total token throughput / GPU | 24,759 tok/s |
| Request throughput | 96.6 req/s |
| Output token throughput | 96.6 tok/s |
| Median TTFT | 83.4 s |
| Mean TTFT | 80.5 s |
| TPOT / ITL | 0 (single output token) |
| Completed requests | 81,920 |

The TTFT is large because the queue is ~2000 deep per rank: prefill is
compute-bound and only ~2–3 requests fit per step, so most requests wait a long
time before their prefill runs. This is the expected prefill-saturation regime,
not a defect.

### Final per-rank request distribution: perfectly even

Per DP process (source-log proxy — backend trace events carry `dp_rank=None`,
so file identity is the per-rank proxy; see `dep/CLAUDE.md`):

```
Measured requests handled per DP rank (of 81,920):
  dp0  ████████████████████  20,480
  dp1  ████████████████████  20,480
  dp2  ████████████████████  20,480
  dp3  ████████████████████  20,480
```

Round-robin split the work exactly 1/4 each — same as the decode-heavy run.
(Per `dep/CLAUDE.md`, balanced final counts prove nothing on their own; the
decode-heavy run was *also* balanced at the end yet still bubbled. The temporal
signal below is what matters.)

### Per-rank speed: near-identical (spread 0.13%)

`backend_dp_enter → backend_dp_first_token` (i.e. queue-wait + prefill compute
per request), by DP process:

| DP rank | count | mean (s) | p50 (s) |
|---|---:|---:|---:|
| dp0 (…2237583) | 24,577 | 73.285 | 83.211 |
| dp1 (…2237670) | 24,577 | 73.345 | 83.286 |
| dp2 (…2237778) | 24,576 | 73.378 | 83.326 |
| dp3 (…2237873) | 24,577 | 73.322 | 83.227 |

Mean spread across ranks = **93 ms on a 73 s base = 0.13%**; p50 spread 115 ms.
The four ranks service requests at the same speed.

### Temporal skew: flat in steady state

From the vLLM `Engine NNN:` interval logs (415 samples, ~18 min, 10 s bins):

```
Concurrent requests per rank (steady state):
  prefill-heavy (isl=1024)   running ≈ 3    ▏                     (token-budget bound: 2048 / 1024)
  decode-heavy  (osl=1024)   running = 864  ████████████████████  (max-num-seqs bound, for contrast)

Waiting queue depth per rank (mean):
  dp0  ███████████████████████████████████  1,745
  dp1  ██████████████████████████████████░  1,712
  dp2  ███████████████████████████████████  1,742
  dp3  ██████████████████████████████████░  1,729
```

Per-window max−min skew across the four ranks:

| skew metric | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| running_skew | 0.0 | 0.0 | 0.0 | 1.0 |
| waiting_skew | 8.4 | 3.0 | 10.0 | 490.0 |
| gen_skew | 0.2 | 0.2 | 0.3 | 1.2 |

`running_skew` is essentially zero at all times — all four ranks hold the same
~3 concurrent prefills every window. The single large `waiting_skew` (490) is at
**t+240 s only**, during queue ramp-up, not steady state.

## Why prefill-heavy does not reproduce the bubble

The decode-heavy underfeed bubble requires a **drainable queue**: a rank reaches
`Waiting: 0` with `Running < cap`, starves, and the others wait for it at the EP
barrier. The prefill-heavy shape removes that precondition by construction:

1. **Token-budget bound, not seq bound.** `max-num-batched-tokens=2048` with
   isl=1024 admits only ~2 prefills per step (`running` pinned at ~3), versus
   the decode regime's `max-num-seqs=864` cap. The admission cap is tiny and
   identical on every rank.

2. **The queue never drains.** `Waiting` is ≥ ~1700 on every rank for the entire
   measured run. No rank ever hits `Waiting: 0`, so the underfeed/starvation
   mode that produces the bubble simply cannot occur. Every rank is uniformly
   backlogged and uniformly busy.

Consequently the request-granular and 10 s-bin metrics show the ranks perfectly
matched. As with the decode-heavy Step 1 analysis, these coarse metrics still
cannot see the *per-MoE-layer* (microsecond) barrier wait — that needs nsys —
but the macro-level underfeed signal that motivated the DEP-routing concern is
**absent** here.

## Verdict

- Under a prefill-dominant workload the DP=4 ranks are uniformly saturated,
  evenly loaded (20,480 req each), and equally fast (per-rank duration spread
  0.13%). **No temporal underfeed.**
- The DEP underfeed bubble is a **decode-regime phenomenon**: it depends on a
  queue that can drain to zero on a single rank, which prefill saturation
  prevents. Prefill-heavy is therefore a poor probe for the underfeed hypothesis
  — it eliminates the very condition that creates the bubble.
- No nsys was run here (the decode-tuned `delay_secs=1100` capture window does
  not transfer to the OSL=1 timeline). A retuned prefill-window nsys capture
  could still look for per-step barrier skew, but there is no macro signal to
  chase, so it is low priority.

## Artifacts

All paths on ptyche persistent Lustre:

- Run outputs: `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/2187958/`
- Benchmark rollup: `outputs/2187958/logs/benchmark-rollup.json`, `…/benchmark-rollup.csv`
- Backend worker log (per-DP `Engine NNN:` lines):
  `outputs/2187958/logs/ptyche0286_agg_w0.out`
- Per-DP server traces (DP-rank proxy by file identity):
  `outputs/2187958/logs/dynamo_request_trace_vllm_{2237583,2237670,2237778,2237873}.jsonl`
- SA-Bench client trace:
  `outputs/2187958/logs/sa-bench_isl_1024_osl_1/request_trace_concurrency_8192_gpus_4.jsonl`
- Joined CSVs: `outputs/2187958/logs/dp_trace_join/`
- Recipe (also in this repo):
  `dep/dep-bubble/qwen3-235b-a22b-vllm-agg-ptyche-gb200-dp4-ep-round-robin-prefill-isl1024.yaml`
