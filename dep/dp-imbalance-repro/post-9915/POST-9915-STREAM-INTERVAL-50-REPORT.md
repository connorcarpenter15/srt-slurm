# Post-PR 9915 Stream-Interval 50 Report

Date: 2026-05-26

## Summary

The corrected post-PR 9915 Dynamo rerun with `stream-interval: 50` does **not**
reproduce the original Dynamo underfeed/performance gap. Job `1918158`
completed successfully at 69.12k output tok/s and 66.45 s mean TTFT, which is
materially better than the earlier post-9915 jobs that omitted stream interval
and higher throughput than the prior direct-vLLM controls in this artifact set.

The previous post-9915 conclusion was invalid for PR 9915 validation because
the recipes did not pass the stream interval setting used by the PR-side vLLM
path. With the setting present, backend DP request distribution was exactly
even and the backend logs did not show the `Waiting: 0` starvation pattern that
appeared in earlier runs.

## Test

- Job: `1918158`
- Output path:
  `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1918158`
- Recipe:
  `post-9915/instrumented/qwen3-235b-a22b-vllm-agg-lyris-gb200-dp4-ep-round-robin-instrumented.yaml`
- Cluster: Lyris GB200, 1 node, 4 GPUs
- Runtime image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0`
- Dynamo source: post-PR 9915, with additional request/DP instrumentation
- Workload: Qwen3-235B FP4, vLLM DP=4, EP enabled, `isl=2`, `osl=1024`,
  concurrency `8192`
- Critical setting: `stream-interval: 50`

Generated launch commands in `sweep_1918158.log` included
`--stream-interval 50` for all four data-parallel ranks.

## Results

| Run | Output tok/s | Req/s | Mean TTFT | P99 TTFT | Mean TPOT | Mean ITL |
|---|---:|---:|---:|---:|---:|---:|
| Dynamo post-9915 + stream interval 50 | 69,122.16 | 67.50 | 66.45 s | 106.28 s | 48.93 ms | 2,385.48 ms |
| Dynamo post-9915 round robin, no stream interval | 41,237.16 | 40.27 | 141.56 s | 304.86 s | 46.65 ms | 64.93 ms |
| Direct vLLM api4/default, prior control | 58,227.52 | 56.86 | 75.42 s | 373.39 s | 62.25 ms | 71.20 ms |
| Direct vLLM api8, prior control | 58,043.19 | 56.68 | 61.15 s | 250.42 s | 68.00 ms | 69.05 ms |

Throughput:

```text
Dynamo post-9915 stream50  69.12k | ##################################################
Direct api4 prior          58.23k | ##########################################
Direct api8 prior          58.04k | ##########################################
Dynamo no-stream post9915  41.24k | ##############################
```

Mean TTFT:

```text
Direct api8 prior          61.15s | ####################
Dynamo post-9915 stream50  66.45s | ######################
Direct api4 prior          75.42s | #########################
Dynamo no-stream post9915 141.56s | ###############################################
```

## DP-Rank Evidence

The backend per-request trace files do not carry an explicit `dp_rank` label,
so the four `dynamo_request_trace_vllm_*.jsonl` files are used as the DP-process
proxy.

Measured-run backend request counts:

| Backend trace file | Requests |
|---|---:|
| `dynamo_request_trace_vllm_2932414.jsonl` | 20,480 |
| `dynamo_request_trace_vllm_2932501.jsonl` | 20,480 |
| `dynamo_request_trace_vllm_2932588.jsonl` | 20,480 |
| `dynamo_request_trace_vllm_2932683.jsonl` | 20,480 |

Backend queue summary from vLLM engine logs:

| Rank | Running mean | Running p50 | Running p95 | Waiting mean | Waiting p95 |
|---:|---:|---:|---:|---:|---:|
| 0 | 820.0 | 864 | 864 | 972.6 | 1184 |
| 1 | 818.4 | 864 | 864 | 969.7 | 1184 |
| 2 | 819.0 | 864 | 864 | 972.8 | 1184 |
| 3 | 830.4 | 864 | 864 | 984.7 | 1184 |

Temporal skew remained but was bounded:

| Metric | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| Running skew | 1.9 | 0.0 | 21.0 | 64.0 |
| Waiting skew | 24.8 | 0.0 | 177.0 | 518.0 |
| Generation throughput skew | 272.3 tok/s | 88.1 tok/s | 1,209.0 tok/s | 1,929.8 tok/s |

Interpretation: final distribution was perfectly even, and the queue samples
showed all ranks staying near the `max-num-seqs=864` ceiling for most of the
measured run. This is not the earlier failure mode where ranks repeatedly
drained to `Waiting: 0` with `Running` far below 864.

## Request-Plane Metrics

| Metric | Value |
|---|---:|
| `dynamo_frontend_inflight_requests` p50 / p95 / max | 8192 / 8192 / 8192 |
| `dynamo_frontend_queued_requests` p50 / p95 / max | 4736 / 4737 / 6158 |
| `dynamo_request_plane_queue_seconds` window mean | 0.1916 s |
| `dynamo_request_plane_send_seconds` window mean | 0.000575 s |
| `dynamo_request_plane_roundtrip_ttft_seconds` window mean | 62.14 s |

Joined request-trace timings:

| Timing | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| Client TTFT | 66.45 s | 53.78 s | 99.44 s | 149.48 s |
| Request-plane enqueue to send | 0.160 s | 0.029 s | 0.649 s | 1.905 s |
| Request-plane send wall time | 0.000575 s | 0.000558 s | 0.000709 s | 0.0151 s |
| Request-plane roundtrip first response | 59.53 s | 51.45 s | 95.39 s | 97.74 s |
| Backend enter to first token | 59.48 s | 51.43 s | 95.32 s | 97.73 s |

The request-plane send path is not the bottleneck. The dominant time is still
backend roundtrip to first token, but with `stream-interval: 50` the backend
stays fed and the overall throughput/TTFT move into the expected range.

## Conclusion

PR 9915 appears to fix the reproduced performance issue when the intended
`stream-interval: 50` setting is actually applied. The earlier no-stream
post-9915 results should not be used to judge the PR.

Remaining caveats:

- The direct-vLLM controls were not rerun in the corrected pass; they are prior
  controls from the same reproduction effort.
- Backend DP gauges were not present in the Prometheus scrape output; DP queue
  analysis came from vLLM engine logs and per-process backend trace files.
- The vLLM backend trace files did not include explicit `dp_rank`, only process
  identity, so process files are used as the DP-rank proxy.

## Artifacts

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
