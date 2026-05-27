# Instrumented Run 1918158 Data Report

Date: 2026-05-27

## Summary

The corrected post-PR 9915 instrumented run, job `1918158`, did **not**
reproduce the original Dynamo underfeed / DP-rank imbalance failure mode. With
`stream-interval: 50` applied, Dynamo reached `69,122.16` output tok/s,
distributed measured requests exactly evenly across the four backend DP process
proxies, and kept backend ranks near the `max-num-seqs=864` running-request cap
for most of the measured run.

Temporal skew was still present, but it was bounded: running-request skew was
`1.9` requests on average, `21` at p95, and `64` max. Waiting-queue skew reached
`518` requests in the worst sampled window, but the trace and queue summaries
do not show sustained backend starvation.

## Run Configuration

| Field | Value |
|---|---|
| Job | `1918158` |
| Cluster | Lyris GB200, 1 node, 4 GPUs |
| Output directory | `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1918158` |
| Recipe | `post-9915/instrumented/qwen3-235b-a22b-vllm-agg-lyris-gb200-dp4-ep-round-robin-instrumented.yaml` |
| Runtime image | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0` |
| Workload | Qwen3-235B FP4, vLLM DP=4, EP enabled |
| Benchmark shape | `isl=2`, `osl=1024`, concurrency `8192`, `max-num-seqs=864` |
| Critical setting | `stream-interval: 50` |

The generated vLLM launch commands in `sweep_1918158.log` included
`--stream-interval 50` for all four data-parallel ranks.

## Instrumentation Captured

| Signal | Captured? | Source |
|---|---|---|
| Client submit / first token / done timestamps | Yes | SA-Bench request trace |
| Request-plane enqueue / send / first response timestamps | Yes | Dynamo request trace |
| Backend DP enter / first token / done timestamps | Yes | Dynamo backend trace |
| Per-DP request counts | Yes | Backend trace files, used as DP-process proxies |
| Backend running / waiting queue depth | Yes | vLLM engine log parser |
| Frontend queued / inflight requests | Yes | Frontend Prometheus scrape |
| Request-plane queue / send / roundtrip TTFT histograms | Yes | Frontend Prometheus scrape |
| Backend DP Prometheus gauges | No | Scrape captured frontend files only |
| Router scheduling overhead histogram | Not populated | Metric present with zero count |
| Forward-pass metric age / state freshness | Not directly captured | Only generic frontend stage metrics present |
| Explicit `dp_rank` label in backend trace | No | Process log file is used as the rank proxy |

## SA-Bench Results

| Phase | Successful requests | Duration | Req/s | Output tok/s | Peak output tok/s | Mean TTFT | P99 TTFT | Mean TPOT | Mean ITL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Warmup | 16,384 | 256.58 s | 63.86 | 65,388.36 | 77,685.20 | 46.55 s | 90.28 s | 48.57 ms | 2,368.35 ms |
| Measured | 81,920 | 1,213.59 s | 67.50 | 69,122.16 | 95,496.00 | 66.45 s | 106.28 s | 48.93 ms | 2,385.48 ms |

Measured-run latency detail:

| Metric | Mean | Median | P99 |
|---|---:|---:|---:|
| TTFT | 66.45 s | 53.78 s | 106.28 s |
| TPOT | 48.93 ms | 48.85 ms | 52.07 ms |
| ITL | 2,385.48 ms | 2,388.32 ms | 3,880.67 ms |
| E2EL | 116.51 s | n/a | 155.67 s |

`ITL` is unusually large relative to `TPOT`; treat it as SA-Bench streaming
accounting under the high-concurrency streaming workload, and read it alongside
output throughput, TTFT, and TPOT rather than as a standalone decode-speed
indicator.

## Backend Request Distribution

The measured run had exactly even request distribution by backend trace file.
The trace files lack explicit `dp_rank`, so each file is treated as a backend
DP process proxy.

| Backend trace file | Measured requests |
|---|---:|
| `dynamo_request_trace_vllm_2932414.jsonl` | 20,480 |
| `dynamo_request_trace_vllm_2932501.jsonl` | 20,480 |
| `dynamo_request_trace_vllm_2932588.jsonl` | 20,480 |
| `dynamo_request_trace_vllm_2932683.jsonl` | 20,480 |

Distribution chart:

```text
2932414  20,480 | ####################
2932501  20,480 | ####################
2932588  20,480 | ####################
2932683  20,480 | ####################
```

Including warmup, the same backend trace files saw `24,577`, `24,577`,
`24,577`, and `24,576` backend-enter events.

## Backend Queue Depth

Backend queue summary was parsed from 598 vLLM engine log samples covering
`2026-05-27T02:15:55.883723Z` to `2026-05-27T02:41:06.125057Z`.

| DP rank | Samples | Running mean | Running p50 | Running p95 | Running max | Waiting mean | Waiting p95 | Waiting max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 150 | 820.0 | 864 | 864 | 864 | 972.6 | 1184 | 1184 |
| 1 | 150 | 818.4 | 864 | 864 | 864 | 969.7 | 1184 | 1184 |
| 2 | 150 | 819.0 | 864 | 864 | 864 | 972.8 | 1184 | 1184 |
| 3 | 148 | 830.4 | 864 | 864 | 864 | 984.7 | 1184 | 1184 |

The ranks spent most sampled windows at the running-request cap:

```text
Running mean by rank, cap 864
rank 0  820.0 | ###############################################
rank 1  818.4 | ###############################################
rank 2  819.0 | ###############################################
rank 3  830.4 | ################################################
```

## Temporal Skew

| Skew metric | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| Running requests | 1.9 | 0.0 | 21.0 | 64.0 |
| Waiting requests | 24.8 | 0.0 | 177.0 | 518.0 |
| Generation throughput | 272.3 tok/s | 88.1 tok/s | 1,209.0 tok/s | 1,929.8 tok/s |

Largest waiting-skew windows:

| Approx offset | Waiting skew | Running skew | Waiting max | Running max | Generation skew |
|---:|---:|---:|---:|---:|---:|
| t+1110s | 518 | 0 | 1131 | 864 | 361.2 tok/s |
| t+1010s | 495 | 0 | 1184 | 864 | 1,319.9 tok/s |
| t+900s | 407 | 29 | 1107 | 864 | 130.0 tok/s |

This confirms temporal skew still exists, but the more important signal is that
`running_skew` stayed small while each rank remained close to saturation. The
earlier failure shape was backend underfeed, where waiting requests dropped to
zero while running requests also fell well below the `864` cap.

## Request-Plane Timing

Joined request-trace timings:

| Timing | Count | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| Client TTFT | 81,920 | 66.45 s | 53.78 s | 99.44 s | 149.48 s |
| Backend enter to first token | 98,307 | 59.48 s | 51.43 s | 95.32 s | 97.73 s |
| Backend duration | 98,307 | 109.57 s | 101.70 s | 145.52 s | 147.98 s |
| Request-plane enqueue to send | 98,307 | 0.160 s | 0.029 s | 0.649 s | 1.905 s |
| Request-plane send wall time | 98,307 | 0.000575 s | 0.000558 s | 0.000709 s | 0.0151 s |
| Request-plane roundtrip first response | 98,307 | 59.53 s | 51.45 s | 95.39 s | 97.74 s |

The request-plane send path itself is not the bottleneck. Queue-to-send latency
is sub-second at p95, while TTFT is dominated by the backend time to first
token under the fully loaded DP=4 workload.

Per-backend-process timing was nearly identical:

| Backend trace file | Backend duration mean | Enter-to-first-token mean |
|---|---:|---:|
| `2932414` | 109.58 s | 59.49 s |
| `2932501` | 109.57 s | 59.48 s |
| `2932588` | 109.54 s | 59.45 s |
| `2932683` | 109.60 s | 59.51 s |

## Prometheus Metrics

Metrics were scraped from
`/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1918158/logs/sa-bench_isl_2_osl_1024/metrics_trace_concurrency_8192_gpus_4`.

Frontend gauges:

| Metric | Samples | Mean | P50 | P95 | Max | Last |
|---|---:|---:|---:|---:|---:|---:|
| `dynamo_frontend_inflight_requests` | 1,215 | 7,444.81 | 8,192 | 8,192 | 8,192 | 0 |
| `dynamo_frontend_queued_requests` | 1,215 | 4,127.58 | 4,736 | 4,737 | 6,158 | 0 |

Histograms:

| Metric | Final count | Final sum | Window mean |
|---|---:|---:|---:|
| `dynamo_request_plane_queue_seconds` | 98,307 | 15,715.5 | 0.1916 s |
| `dynamo_request_plane_send_seconds` | 98,307 | 56.5367 | 0.000575 s |
| `dynamo_request_plane_roundtrip_ttft_seconds` | 98,307 | 5,852,630 | 62.14 s |
| `dynamo_router_overhead_scheduling_ms` | 0 | 0 | n/a |

The scrape directory contained frontend Prometheus snapshots only
(`frontend__*.prom`). Backend DP gauges such as
`dynamo_component_vllm_dp_requests_running` and
`dynamo_component_vllm_dp_requests_waiting` were absent from the captured
snapshots, so backend queue analysis relies on vLLM logs and trace files.

## Interpretation

The instrumented data supports these conclusions:

1. The original Dynamo-vs-vLLM throughput gap did not reproduce in this
   corrected post-PR 9915 run. Dynamo exceeded the prior direct-vLLM controls
   available in this artifact set.
2. Even final request distribution is confirmed: the measured 81,920 requests
   split exactly `20,480` per backend process proxy.
3. Temporal skew still exists, especially in waiting queues, but it did not
   translate into sustained underfeed. Running-request skew was small and the
   backend stayed near the configured running-request cap.
4. Request-plane send latency is too small to explain TTFT. The dominant
   latency is backend roundtrip to first token under the saturated DP workload.
5. The main remaining observability gap is backend Prometheus coverage and
   explicit `dp_rank` labels in backend traces. Those would make future runs
   easier to interpret without relying on log-file identity as the rank proxy.

## Artifact Paths

- SA-Bench result:
  `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1918158/logs/sa-bench_isl_2_osl_1024/results_concurrency_8192_gpus_4.json`
- Client request trace:
  `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1918158/logs/sa-bench_isl_2_osl_1024/request_trace_concurrency_8192_gpus_4.jsonl`
- Metrics scrape:
  `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1918158/logs/sa-bench_isl_2_osl_1024/metrics_trace_concurrency_8192_gpus_4`
- Backend queue summary:
  `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1918158/logs/analysis/backend_queue_summary.md`
- Metrics summary:
  `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1918158/logs/analysis/metrics_summary.md`
- Joined trace summary:
  `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1918158/logs/analysis/trace_with_frontend_summary.md`
- Joined trace CSV output:
  `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1918158/logs/analysis/trace_with_frontend_csv`
