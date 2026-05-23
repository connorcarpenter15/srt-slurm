# Dynamo DP Imbalance and Routing Performance Gap Report

Date: 2026-05-22

This report summarizes the Lyris reproduction of the Dynamo DP imbalance /
routing-performance gap discussed in Slack thread
`C093RGPF84E/p1778252809274269`. The goal was to compare Dynamo against direct
vLLM on a high-concurrency DP=4 EP workload and determine whether Dynamo
underfeeds or imbalances backend DP ranks.

## Executive Summary

The original gap reproduces directionally on the accessible Lyris GB200 stack.
The best Dynamo result was the `least-loaded` router variant at 49.31k output
tokens/s. Direct vLLM reached 58.23k output tokens/s with the default API server
count, which vLLM 0.19.0 logs as defaulting to `data_parallel_size`, so it is
effectively `--api-server-count 4`.

Compared with the best direct-vLLM throughput, Dynamo is about 15.3% lower.
That is smaller than the original report's roughly 19% gap, but the shape is
the same: Dynamo is materially slower on the same DP=4 EP stress pattern.

The TTFT gap is also reproduced. The best Dynamo mean TTFT was 98.94 s. Direct
vLLM with default/api4 had 75.42 s mean TTFT, a 23.5 s advantage. Direct vLLM
with `--api-server-count 8` had 61.15 s mean TTFT, a 37.8 s advantage, close to
the original report's roughly 40 s TTFT gap.

Even final request distribution did not fix the gap. Dynamo round robin drained
with effectively perfect per-backend request counts:

```text
24576, 24577, 24576, 24578
```

Despite that, round robin still produced only 49.26k output tokens/s and
99.92 s mean TTFT. This means the issue is not explained by simple final count
imbalance across DP ranks. The evidence points more toward temporal imbalance,
admission/backpressure behavior, stale or insufficient router load feedback, or
some combination of those effects.

## Scope and Deviations

The completed runs used Lyris GB200 nodes rather than the originally requested
GB300 environment. The original model path was present on Lyris and was used:

```text
/lustre/share/coreai_dlfw_dev/models/Qwen3-235B-A22B-Instruct-2507-FP4
```

The completed Lyris runs used:

```text
nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0
```

The direct vLLM logs identify vLLM as `v0.19.0`. The staged generic GB300
recipes use the newest registry-indexed image found during the reproduction
attempt, but those recipes were not executed because the completed repro was on
Lyris.

The current Dynamo frontend did not expose a `token-dp-balance` router mode.
Available modes found locally were:

```text
round-robin, random, power-of-two, kv, direct, least-loaded,
device-aware-weighted
```

The old `frontend.args.load-aware: true` form was rejected by the current
frontend CLI. The load-aware variant in this bundle uses:

```yaml
frontend:
  args:
    router-mode: "least-loaded"
```

No broad `/home` filesystem discovery was used for the final work. The repro
bundle documents the earlier issue and keeps future discovery targeted to known
scratch/model roots.

## Test Matrix

| Test | Backend | Router / frontend | API servers | Status |
|---|---|---|---:|---|
| Dynamo round robin | vLLM DP=4 EP | Multi-frontend round robin | Dynamo managed | Completed |
| Dynamo least loaded | vLLM DP=4 EP | Multi-frontend `least-loaded` | Dynamo managed | Completed |
| Dynamo dedicated KV router | vLLM DP=4 EP | Single frontend, KV router, requested knobs | Dynamo managed | Completed |
| Direct vLLM default/api4 | vLLM DP=4 EP | vLLM internal DP routing | 4 effective | Completed |
| Direct vLLM api8 | vLLM DP=4 EP | vLLM internal DP routing | 8 explicit | Completed |
| Dynamo token-DP-balance | vLLM DP=4 EP | Not available in this stack | N/A | Not run |

## Current-Version Instrumented Rerun

A May 23, 2026 rerun completed on Lyris job `1873079` using the current
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0` container and the same DP=4 EP
configuration. This run uses the patched SA-Bench peak-throughput calculation,
Prometheus time-series scraping, and client request-trace output.

Results:

| Job | Phase | Output tok/s | Req/s | Mean TTFT | Median TTFT | P99 TTFT | Mean TPOT | Mean ITL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1873079 | warmup, 16,384 requests at request-rate 250 | 33,727.73 | 32.94 | 479.18 ms | 312.62 ms | 1,839.71 ms | 27.93 ms | 30.97 ms |
| 1873079 | measured, 81,920 requests at concurrency 8,192 | 38,333.23 | 37.43 | 147.10 s | 148.75 s | 258.15 s | 62.15 ms | 89.87 ms |
| 1880624 | warmup, 16,384 requests at request-rate 250 | 35,794.91 | 34.96 | 620.16 ms | 369.18 ms | 3,458.67 ms | 29.07 ms | 33.43 ms |
| 1880624 | measured, 81,920 requests at concurrency 8,192 | 38,111.10 | 37.22 | 146.46 s | 146.67 s | 274.66 s | 61.07 ms | 89.82 ms |

The measured pass started at `2026-05-23 00:10:05 PDT` with
`--request-rate inf`, `--num-prompts 81920`, and `--max-concurrency 8192`.
Backend vLLM logs showed all DP ranks reaching
`Running: 864` and `Waiting: 1184`. Per-rank averages were close, but
time-window skew was large:

| Samples | Running mean range | Waiting mean range | Max running skew | Max waiting skew | Max generation-throughput skew |
|---:|---:|---:|---:|---:|---:|
| 1,065 | 521.0-524.8 | 76.7-81.7 | 603 requests | 938 requests | 38,585 tok/s |

This strengthens the temporal-imbalance interpretation: final or aggregate
rank balance can look even while short windows show very different rank drain
states. During the same measured pass, Dynamo frontend metrics showed inflight
requests pinned near 8,192 and queued requests in the thousands before both
drained to zero. The final frontend metrics summary had 2,204 scrapes:
`dynamo_frontend_inflight_requests` mean 7,513.68, p50 8,145, p95/max 8,192,
last 0; `dynamo_frontend_queued_requests` mean 5,197.97, p50 5,516, p95 6,622,
max 7,105, last 0. Request-plane send time stayed small (30.9 ms window mean),
while request-plane roundtrip TTFT was 143.02 s in the active window.
`dynamo_router_overhead_scheduling_ms` still had zero samples, and no explicit
forward-pass state freshness / slot-age metric was exposed by the collected
Prometheus names.

The client request trace contained all 81,920 measured requests, with TTFT mean
147.095776 s, p50 148.747074 s, p95 199.332955 s, and max 268.157778 s.

The direct backend JSONL hook from Dynamo commit
`37c2ce5bc42bc1442d99c0fe3eb2e8fe57be61ef` did not appear in job `1873079`
because the live Lyris path uses `components/src/dynamo/vllm/handlers.py`.
Follow-up recipes now point at Dynamo
`732e31b751c1ea70c9992a3b392937baa802431f`, which adds backend lifecycle
events to the active handler path.

The handler-trace rerun `1880624` completed successfully and confirms that the
lower-throughput/high-TTFT behavior persists even when backend request
lifecycle events are captured. Final backend trace enter/done counts by backend
trace file were effectively even:

```text
24577, 24577, 24577, 24576
```

The trace JSON carried `dp_rank=None`, so these file-level counts are the
reliable DP-process proxy for this run. The joined trace summary saw 98,307
backend enter/first-token/done lifecycles across probe, warmup, and measured
traffic. Backend enter-to-first-token mean was 117.619 s and backend duration
mean was 174.780 s. The final backend queue summary again showed close
aggregate rank means, but large temporal skew: running skew mean 180.3,
p95 521, max 794 requests; waiting skew max 864; and generation-throughput skew
p95 23,826.8 tok/s, max 42,954.8 tok/s.

## Primary Results

| Variant | Output tok/s | Req/s | Mean TTFT | Median TTFT | P99 TTFT | Mean TPOT | Mean ITL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dynamo round robin | 49,263.13 | 48.11 | 99.92 s | 98.24 s | 188.38 s | 57.48 ms | 71.83 ms |
| Dynamo instrumented round robin rerun | 38,333.23 | 37.43 | 147.10 s | 148.75 s | 258.15 s | 62.15 ms | 89.87 ms |
| Dynamo handler-trace round robin rerun | 38,111.10 | 37.22 | 146.46 s | 146.67 s | 274.66 s | 61.07 ms | 89.82 ms |
| Dynamo dedicated KV router | 48,990.56 | 47.84 | 103.81 s | 108.88 s | 187.54 s | 57.64 ms | 66.44 ms |
| Dynamo least loaded | 49,309.21 | 48.15 | 98.94 s | 97.62 s | 156.63 s | 62.74 ms | 70.10 ms |
| Direct vLLM default/api4 | 58,227.52 | 56.86 | 75.42 s | 72.09 s | 373.39 s | 62.25 ms | 71.20 ms |
| Direct vLLM api8 | 58,043.19 | 56.68 | 61.15 s | 59.37 s | 250.42 s | 68.00 ms | 69.05 ms |

The best Dynamo variant by throughput and mean TTFT was `least-loaded`, but it
was only marginally better than round robin. The dedicated KV router was worse
than both round robin and least-loaded for this workload.

Direct vLLM default/api4 had the best throughput. Direct vLLM api8 had nearly
the same throughput but significantly better mean TTFT.

The instrumented rerun is lower than the earlier round-robin result and should
be treated as an observability run, not a clean apples-to-apples throughput
replacement. It enabled client request tracing, one-second metrics scraping,
and `DYN_REQUEST_TRACE_LOGGING`, and it used a source-installed Dynamo hash
rather than the original completed-run package.

## Output Locations

| Run | Job / session | Output directory |
|---|---|---|
| Dynamo round robin | Slurm job 1859591 | `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1859591` |
| Dynamo instrumented round robin rerun | Slurm job 1873079 | `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1873079` |
| Dynamo handler-trace round robin rerun | Slurm job 1880624 | `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1880624` |
| Dynamo dedicated KV router | Slurm job 1859688 | `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1859688` |
| Dynamo least loaded | Slurm job 1859711 | `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1859711` |
| Direct vLLM default/api4 | Session `dp-imbalance-vllm-direct-lyris`, Slurm job 1859427 | `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/direct-vllm/default` |
| Direct vLLM api8 | Session `dp-imbalance-vllm-direct-api8-lyris`, Slurm job 1864796 | `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/direct-vllm/api8-final` |

Both direct-vLLM compute sessions were released after result collection.
`list_sessions` reported no active compute sessions after cleanup.

## DP-Rank Request Distribution

Dynamo backend request counts were extracted from `*agg_w0.out` logs.

| Dynamo variant | Per-backend received/completed |
|---|---|
| Round robin | `24576, 24577, 24576, 24578` |
| Dedicated KV router | `24688, 24629, 24607, 24383` |
| Least loaded | `25270, 24255, 24447, 24335` |

Direct vLLM per-engine success counts came from `/metrics`.

| Direct variant | Per-engine length-success counts |
|---|---|
| Default/api4 | `24579, 24576, 24576, 24576` |
| Api8 | `24579, 24568, 24582, 24578` |

The important result is that Dynamo round robin had final distribution at least
as even as direct vLLM, but still underperformed. Count balance alone is
therefore not sufficient.

## Queue and Request-Plane Metrics

Final Dynamo Prometheus scrape summaries:

| Variant | Request-plane queue mean | Request-plane send mean | Roundtrip TTFT mean | Frontend queued max / mean | Frontend inflight max / mean |
|---|---:|---:|---:|---:|---:|
| Round robin | 0.000025729 s | 0.060338 s | 70.199 s | 5330 / 3093.0 | 8192 / 5440.4 |
| Dedicated KV router | 0.000016933 s | 0.055285 s | 78.464 s | 6039 / 3641.6 | 8192 / 6014.1 |
| Least loaded | 0.000026937 s | 0.051928 s | 76.409 s | 6574 / 3642.4 | 8192 / 6322.3 |

The local request-plane queue time was tiny, on the order of tens of
microseconds. Request-plane send time was roughly 52 to 60 ms. These are not
large enough to directly explain 23 to 38 s of mean TTFT gap.

At the same time, Dynamo frontend queued and inflight request gauges stayed
large, and `dynamo_request_plane_roundtrip_ttft_seconds` averaged tens of
seconds. That combination suggests the backlog is not in the local
request-plane enqueue path. It is more likely in frontend admission,
backend-slot replenishment, backend scheduling, router feedback delay, or the
interaction between those pieces.

## Router Scheduling Overhead

`dynamo_router_overhead_scheduling_ms` was non-zero only for the dedicated KV
router run:

```text
round robin:      count 0
dedicated router: sum 4215602.107892993 ms, count 98307, mean 42.882 ms
least loaded:     count 0
```

The dedicated KV router also had the lowest throughput and worst mean TTFT
among the Dynamo variants. For this workload, the requested KV router knobs did
not improve DP balance or throughput, and they added measurable per-request
scheduling cost. That is not surprising for this stress shape: `isl=2`,
`osl=1024`, no prefix caching, and random range ratio 1.0 do not create much
useful KV locality for a KV-aware router to exploit.

## Backend Queue Observability

Direct vLLM exposed per-engine queue gauges:

```text
num_requests_running max:
engine 0: 864
engine 1: 864
engine 2: 864
engine 3: 864

num_requests_waiting max:
default/api4: 1184, 1184, 1184, 1184
api8:         1185, 1184, 1184, 1185
```

During the direct api8 measured run, a transient sample showed one engine with
substantial waiting work while the others had no waiting work:

```text
engine 0: running 735, waiting 0
engine 1: running 730, waiting 0
engine 2: running 736, waiting 0
engine 3: running 860, waiting 1001
```

Later direct-vLLM samples also showed waiting draining to zero while requests
were still running. This matters because "backend waiting requests may drop to
zero despite concurrency=8192" is not, by itself, proof that Dynamo is uniquely
starving the backend. Waiting can be zero in direct vLLM as well. The stronger
question is whether running slots are replenished evenly and promptly over time.

The Dynamo scrapes collected in this repro did not expose direct vLLM-style
backend `num_requests_running` / `num_requests_waiting` gauges. They also did
not expose metrics matching `forward`, `fresh`, `stale`, `age`, `slot`, or
`state`, so forward-pass metric age and router state freshness were not directly
observable.

## Does Even Request Distribution Fix TTFT Skew?

No. The round-robin Dynamo run was essentially perfectly balanced by final
request count, yet it was still about 15% slower than direct vLLM and had much
worse mean TTFT.

That does not prove that every DP rank had identical TTFT or identical service
time. Final request count is a coarse metric. It does not capture:

- when each request was admitted to a rank,
- how many active sequences each rank had at each scheduling point,
- how quickly each rank's running slots were refreshed,
- whether one rank received a burst of long-running work earlier than others,
- MoE/EP service-time variance,
- time spent upstream of the backend before the request is visible to vLLM.

Per-DP-rank TTFT was not directly available from the emitted metrics/logs in
this run, so TTFT skew across ranks remains an inferred risk rather than a
directly proven fact.

## Does Dynamo State Appear Stale Versus vLLM Internal LB?

The evidence is consistent with stale or insufficient Dynamo backend-state
feedback, but it is not conclusive because the needed state-age metrics were not
emitted in the scrapes.

Reasons the stale-state hypothesis fits the data:

- Direct vLLM can route using internal engine state in the same serving stack.
  Dynamo routes through external frontend/request-plane/backend feedback.
- The `least-loaded` router did not materially improve throughput over round
  robin, even though a current-load-aware router should help if it has fresh,
  accurate, and useful load signals.
- Dynamo frontend queued and inflight gauges stayed large while the local
  request-plane queue time stayed near zero, which suggests backlog and pacing
  are happening outside the local enqueue path.
- Round-robin final counts were even, so a simple "wrong rank got too many
  requests" explanation is insufficient. Temporal slot refresh and feedback
  timing are more plausible.

Reasons this is not yet proven:

- Dynamo did not expose per-rank running/waiting queue depth in the collected
  scrapes.
- Dynamo did not expose router load-state age or forward-pass metric freshness.
- Per-rank TTFT was not directly collected.
- The direct vLLM api4 and api8 runs were not exact same-process comparisons
  with Dynamo; they use different frontend architecture by design.

## Likely Causes

The most likely cause is not static DP assignment imbalance. It is a dynamic
load and admission problem. The current evidence supports these contributing
mechanisms:

1. **Frontend admission and backend-slot refresh may be pacing requests less
   efficiently than vLLM internal routing.**
   Dynamo shows large frontend queued/inflight counts and high roundtrip TTFT,
   but negligible local request-plane queue time. That points to release timing,
   backend-slot availability, or response feedback rather than a simple
   request-plane enqueue bottleneck.

2. **Router load signals may be stale, too coarse, or not the right signal for
   this workload.**
   The `least-loaded` router was only marginally better than round robin and
   did not close the gap. If the load signal lags the backend state, the router
   can keep dispatching according to an already-obsolete view of each DP rank.

3. **Final request-count balance hides temporal imbalance.**
   The round-robin run proves final counts can be balanced while TTFT remains
   poor. High-concurrency decode-heavy workloads care about when slots refill,
   not just how many total requests each rank eventually receives.

4. **KV-router logic is a poor fit for this benchmark shape.**
   With `isl=2`, `osl=1024`, no prefix caching, and high randomness, there is
   little useful KV overlap to exploit. The dedicated KV router added about
   42.9 ms of mean scheduling overhead per routed request and did not improve
   throughput.

5. **API server parallelism affects TTFT independently of raw backend
   throughput.**
   Direct vLLM api8 and api4 had similar throughput, but api8 reduced mean TTFT
   by about 14.3 s. That suggests frontend/API-server concurrency and admission
   path parallelism materially influence TTFT under this workload.

6. **Backend waiting depth is not enough to diagnose starvation.**
   Waiting can drop to zero in direct vLLM too. The useful starvation signal is
   a time series of per-rank running slots, waiting queue, scheduler admissions,
   and completed requests, correlated with frontend queued/inflight state.

## What Performed Best

Best direct result:

```text
Direct vLLM default/api4
Output throughput: 58,227.52 tok/s
Mean TTFT:         75.42 s
```

Best direct TTFT result:

```text
Direct vLLM api8
Output throughput: 58,043.19 tok/s
Mean TTFT:         61.15 s
```

Best Dynamo result:

```text
Dynamo least-loaded
Output throughput: 49,309.21 tok/s
Mean TTFT:         98.94 s
```

In practice, round robin and least-loaded were close enough that the Dynamo
variant result should be treated as a tie unless confirmed by repeated runs.
The dedicated KV router should not be preferred for this no-prefix, short-ISL
workload.

## Recommended Next Steps

1. Add or expose Dynamo-side per-backend vLLM gauges for
   `num_requests_running`, `num_requests_waiting`, completed requests, and
   scheduled tokens by DP rank.

2. Add router state freshness metrics:
   last backend update timestamp, age of load signal at scheduling time,
   observed running/waiting at scheduling time, and selected backend.

3. Add per-request backend rank annotation so TTFT and queueing can be grouped
   by DP rank after the run.

   The srt-slurm fork now has the client-side half of this instrumentation:
   `benchmark.request_trace: true` makes SA-Bench send stable UUID request IDs
   in `X-Request-Id`, `X-Dynamo-Request-Id`, and `X-Client-Request-Id` headers
   and write `client_submit`, `client_first_token`, and `client_done` JSONL
   events for measured requests. Server-side Dynamo/router/backend events can
   join against the same request id to identify exact DP-rank ingress and
   egress.

   The staged repro recipes now set `DYN_REQUEST_TRACE_LOGGING=1` on the
   Dynamo frontend/router and vLLM backend. With the matching Dynamo worktree
   patch, logs include `dynamo_request_trace` events for router assignment,
   router request tracking/freeing, backend DP entry, backend first output,
   backend first token, and backend completion. Router assignment events also
   include the selected worker's observed decode blocks, observed prefill
   tokens, pending queue depth at admission, pending ISL tokens at admission,
   and scheduler queue delay when the request had been parked.

   Use `dep/dp-imbalance-repro/trace_summary.py` to join the SA-Bench JSONL
   with frontend/backend logs and report per-DP-rank counts plus timing deltas
   such as router-to-backend-enter and backend-enter-to-first-token.

4. Capture time-series scrapes during the full run rather than relying mainly
   on final scrapes. The suspected issue is temporal, so final counts are not
   enough.

   The srt-slurm fork now supports this for SA-Bench via
   `benchmark.metrics_scrape: true`. Measured runs write
   `metrics_trace_concurrency_*/index.jsonl` plus raw Prometheus `.prom`
   snapshots for the frontend and discovered backend worker metrics endpoints.

   While rerunning with client request tracing, SA-Bench exposed an additional
   client-side bottleneck: after all backend ranks drained, the benchmark client
   stayed CPU-bound computing peak output throughput by tokenizing every
   streamed text chunk. The srt-slurm fork now caps exact per-chunk tokenization
   for this metric and uses a proportional output-token approximation for large
   runs. This keeps the high-concurrency trace runs from holding GPU
   allocations after serving has completed.

5. Repeat the same matrix on the same node class with at least two seeds or
   repeated runs:
   round robin, least-loaded, power-of-two, direct/api4, direct/api8.

6. Sweep concurrency below 8192, for example 1024, 2048, 4096, and 8192, to
   find where Dynamo diverges from direct vLLM.

7. If token-DP-balance exists in a newer Dynamo build, rerun the same matrix and
   compare it against round robin and least-loaded.

8. For the dedicated router path, avoid KV mode for this benchmark unless the
   workload is changed to include meaningful prefix reuse.

## Artifact Index

The reproduction bundle is under:

```text
dep/dp-imbalance-repro/
```

Key files:

```text
README.md
REPORT.md
qwen3-235b-a22b-vllm-agg-lyris-gb200-dp4-ep-round-robin.yaml
qwen3-235b-a22b-vllm-agg-lyris-gb200-dp4-ep-load-aware.yaml
qwen3-235b-a22b-vllm-agg-lyris-gb200-dp4-ep-dedicated-router.yaml
qwen3-235b-a22b-vllm-agg-gb300-dp4-ep-round-robin.yaml
qwen3-235b-a22b-vllm-agg-gb300-dp4-ep-load-aware.yaml
qwen3-235b-a22b-vllm-agg-gb300-dp4-ep-dedicated-router.yaml
```

The README contains the full run log, direct-vLLM commands, raw benchmark
summaries, metric extraction notes, and cleanup state. This report contains the
interpretation and recommended follow-up instrumentation.
