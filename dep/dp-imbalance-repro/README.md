# Dynamo DP Imbalance Repro Log

This directory stages and records the current-stack reproduction requested from
Slack thread `C093RGPF84E/p1778252809274269`: compare Dynamo against direct
vLLM for a high-concurrency DP=4 EP workload, and check whether Dynamo
underfeeds or imbalances backend DP ranks.

Date: 2026-05-22
Cluster used for the completed repro work: Lyris, GB200, 4 GPUs/node.

## Artifact Index

- `README.md`: run log, commands, raw benchmark summaries, metric extraction
  notes, and cleanup notes.
- `REPORT.md`: findings report with interpretation and likely causes.
- `RERUN-2026-05-22.md`: live rerun log for the instrumented SA-Bench metrics
  scrape and Dynamo router admission trace changes.
- `backend_log_summary.py`: parses backend `Engine NNN` log lines into
  per-rank running/waiting queue and temporal skew summaries.
- `qwen3-235b-a22b-vllm-agg-lyris-gb200-dp4-ep-*.yaml`: Lyris recipes used
  for the completed Dynamo variants.
- `qwen3-235b-a22b-vllm-agg-gb300-dp4-ep-*.yaml`: GB300 recipes staged for a
  later run on the originally requested hardware class.

## Benchmark Tests

The reproduction tests in this bundle are:

- Dynamo DP=4 EP, multi-frontend round robin.
- Dynamo DP=4 EP, multi-frontend least-loaded router.
- Dynamo DP=4 EP, dedicated frontend/router with KV mode and the requested
  router knobs.
- Direct vLLM DP=4 EP with default API server count. vLLM 0.19.0 logs this as
  defaulting to `data_parallel_size`, so this is effectively
  `--api-server-count 4`.
- Direct vLLM DP=4 EP with `--api-server-count 8`.

The staged recipes now set `benchmark.request_trace: true`. Future measured
SA-Bench runs will write per-request client timeline JSONL files next to the
normal `results_*.json` files:

```text
/logs/sa-bench_isl_<ISL>_osl_<OSL>/request_trace_concurrency_<C>_gpus_<N>.jsonl
```

Each measured request gets a deterministic UUID request id derived from a
human-readable request tag and sends it as `X-Request-Id`,
`X-Dynamo-Request-Id`, and `X-Client-Request-Id`. The JSONL trace contains
`client_submit`, `client_first_token`, and `client_done` events. These client
events are intended to be joined with Dynamo/router/backend events by request
id when server-side tracing is available.

The repro recipes also set `DYN_REQUEST_TRACE_LOGGING=1` on the Dynamo
frontend/router and vLLM backend environments. With the matching Dynamo
instrumentation worktree, the server writes `dynamo_request_trace_*.jsonl`
files under `/logs` and also emits `dynamo_request_trace` log lines when the
logger target allows them. Events cover router assignment, router slot
tracking/freeing, backend DP-rank entry, backend first token, and backend
completion. Router assignment events include selected-rank load and scheduler
admission fields such as `selected_decode_blocks`, `selected_prefill_tokens`,
`pending_count_at_admit`, `pending_isl_tokens_at_admit`, and
`scheduler_queue_delay_ms`.

After a run, summarize the joined trace with:

```bash
python dep/dp-imbalance-repro/trace_summary.py \
  --client-trace /logs/sa-bench_isl_2_osl_1024/request_trace_concurrency_8192_gpus_4.jsonl \
  --server-log /path/to/frontend.log \
  --server-log /path/to/backend.log
```

Summarize backend DP queue skew from vLLM engine logs with:

```bash
python dep/dp-imbalance-repro/backend_log_summary.py /path/to/lyris0213_agg_w0.out
```

Summarize selected Dynamo `/metrics` scrape snapshots with:

```bash
python dep/dp-imbalance-repro/metrics_summary.py /path/to/metrics_trace_concurrency_8192_gpus_4
```

The recipes also set `benchmark.metrics_scrape: true`. Each measured
concurrency run writes a metrics scrape index and raw Prometheus snapshots:

```text
/logs/sa-bench_isl_<ISL>_osl_<OSL>/metrics_trace_concurrency_<C>_gpus_<N>/index.jsonl
/logs/sa-bench_isl_<ISL>_osl_<OSL>/metrics_trace_concurrency_<C>_gpus_<N>/*.prom
```

Targets include the frontend `/metrics` endpoint and backend worker system
endpoints discovered by srt-slurm. This is intended to preserve temporal queue,
request-plane, and backend scheduler signals during the high-concurrency run.

For very large streaming runs, SA-Bench now avoids exact per-SSE-chunk
tokenization while computing `Peak output token throughput`. Exact chunk
tokenization is still used for small runs, but large runs use the server's
output-token count distributed across observed chunks. Override with
`SA_BENCH_PEAK_TOKENIZE_MAX_CHUNKS=-1` to force exact mode, or set a higher
positive chunk threshold if exact peak throughput is more important than fast
post-processing.

## Discovery Policy

Do not use broad filesystem discovery for model paths. In particular, do not
walk `/home` or other NFS fanout roots with `find`.

Prefer one of these instead:

- Pull a public HF model into explicit scratch/cache with `HF_HOME` under
  `/lustre/fsw/<account>/<user>/...`.
- Use a known shared model root directly, such as
  `/lustre/share/coreai_dlfw_dev/models`.
- Probe only an explicit candidate path with `stat` or `ls`.

For this run, the original local checkpoint was available on Lyris:

```text
/lustre/share/coreai_dlfw_dev/models/Qwen3-235B-A22B-Instruct-2507-FP4
```

For a looser future repro, use a public HF MoE model and keep the same stress
shape: `--data-parallel-size 4`, `--enable-expert-parallel`, high concurrency,
and long output length.

## Recipes

- Lyris GB200:
  - `qwen3-235b-a22b-vllm-agg-lyris-gb200-dp4-ep-round-robin.yaml`
  - `qwen3-235b-a22b-vllm-agg-lyris-gb200-dp4-ep-load-aware.yaml`
  - `qwen3-235b-a22b-vllm-agg-lyris-gb200-dp4-ep-dedicated-router.yaml`
- GB300 variants, kept for later:
  - `qwen3-235b-a22b-vllm-agg-gb300-dp4-ep-round-robin.yaml`
  - `qwen3-235b-a22b-vllm-agg-gb300-dp4-ep-load-aware.yaml`
  - `qwen3-235b-a22b-vllm-agg-gb300-dp4-ep-dedicated-router.yaml`

The generic recipes currently use the latest registry-indexed vLLM runtime found
during the May 22, 2026 repro attempt:

```text
210086341041.dkr.ecr.us-west-2.amazonaws.com/ai-dynamo/dynamo:959364f561c555aa48b48717406fb056e2cdfaf6-vllm-runtime-cuda13
```

That image corresponds to Dynamo commit `959364f56` from 2026-05-22. For
instrumented reruns, the recipes set `dynamo.install: true` with
`dynamo.hash: 732e31b751c1ea70c9992a3b392937baa802431f` so the job builds and
installs the patched Dynamo router and vLLM-handler trace code instead of using
only the image's bundled Dynamo package.

The Lyris GB200 recipes use the newest public arm64-capable runtime found during
the same attempt:

```text
nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0
```

The newer ECR CI image was preferred but local AWS SSO refresh was blocked; the
public 1.1.0 manifest includes both amd64 and arm64 platforms and does not need
private ECR credentials.

`token-dp-balance` is not staged because the local Dynamo checkout exposes these
frontend router modes only: `round-robin`, `random`, `power-of-two`, `kv`,
`direct`, `least-loaded`, and `device-aware-weighted`.

The load-aware variant uses `router-mode: least-loaded`; the older
`load-aware: true` frontend arg is rejected by the current frontend CLI.

## Lyris Dynamo Runs

Submitted via `srtctl_apply(..., cluster="lyris")` after MFA socket login.

| Variant | Job | Output directory | Status / result |
|---|---:|---|---|
| Round robin | 1859591 | `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1859591` | Completed |
| Dedicated router KV config | 1859688 | `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1859688` | Completed |
| Load aware / least loaded | 1859711 | `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1859711` | Completed |
| Instrumented round robin rerun | 1873079 | `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1873079` | Completed |
| Handler-trace round robin rerun | 1876728 | `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1876728` | Cancelled at user request before final measured result |
| Handler-trace round robin rerun | 1880624 | `/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1880624` | Completed: 38,111.10 output tok/s, 37.22 req/s, 146.46 s mean TTFT |

The instrumented rerun uses patched SA-Bench metrics scraping and request-trace
support. The warmup pass completed at 33,727.73 output tok/s and 479.18 ms mean
TTFT. The measured `concurrency=8192` pass completed at 38,333.23 output tok/s,
37.43 req/s, 147.10 s mean TTFT, 62.15 ms mean TPOT, and 89.87 ms mean ITL.
The run captured a complete client request trace, frontend metrics scrapes, and
backend log-derived temporal skew up to 603 running requests / 938 waiting
requests across DP ranks in sampled windows. See `RERUN-2026-05-22.md` for the
full step log.

The handler-trace rerun `1876728` used the newer Dynamo handler lifecycle trace
patch and confirmed server-side backend trace files were being written, but it
was intentionally cancelled on May 23, 2026 at 01:42 PT before final measured
SA-Bench output was produced. Use its partial metrics and trace evidence only
for live temporal-skew observations, not for final throughput comparison. After
cancellation, `squeue -u $USER` on Lyris returned no active allocations.

The handler-trace rerun `1880624` was submitted on May 23, 2026 at 10:34 PT
after verifying that the Lyris queue was empty. It started on `lyris0251`, loaded
the same corrected Dynamo source commit
`732e31b751c1ea70c9992a3b392937baa802431f`, reached health at 10:41 PT, and
completed successfully at 11:29:17 PT. Raw artifacts are under
`/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1880624/logs`, including
`benchmark-rollup.json`, `benchmark-rollup.csv`,
`sa-bench_isl_2_osl_1024/results_concurrency_8192_gpus_4.json`,
`sa-bench_isl_2_osl_1024/request_trace_concurrency_8192_gpus_4.jsonl`,
`sa-bench_isl_2_osl_1024/metrics_trace_concurrency_8192_gpus_4`, and four
`dynamo_request_trace_vllm_*.jsonl` backend lifecycle traces. Post-run summary
files are saved as `metrics_summary_final.md`,
`backend_queue_summary_final.md`, and `request_trace_summary_final.md`.

Handler-trace round-robin measured run `1880624`:

```text
Successful requests:              81920
Benchmark duration:               2201.09 s
Request throughput:               37.22 req/s
Output token throughput:          38111.10 tok/s
Mean TTFT:                        146461.12 ms
Median TTFT:                      146665.01 ms
P99 TTFT:                         274655.47 ms
Mean TPOT:                        61.07 ms
Mean ITL:                         89.82 ms
```

Backend trace enter/done counts by backend trace file were effectively even:
`24577, 24577, 24577, 24576`. The handler trace records `dp_rank=None`, so the
file-level counts are the reliable DP-process proxy for this run.

Round-robin measured run:

```text
Successful requests:              81920
Benchmark duration:               1702.82 s
Request throughput:               48.11 req/s
Output token throughput:          49263.13 tok/s
Mean TTFT:                        99917.05 ms
Median TTFT:                      98243.02 ms
P99 TTFT:                         188380.35 ms
Mean TPOT:                        57.48 ms
Mean ITL:                         71.83 ms
```

Dedicated-router measured run:

```text
Successful requests:              81920
Benchmark duration:               1712.29 s
Request throughput:               47.84 req/s
Output token throughput:          48990.56 tok/s
Mean TTFT:                        103814.50 ms
Median TTFT:                      108879.18 ms
P99 TTFT:                         187541.97 ms
Mean TPOT:                        57.64 ms
Mean ITL:                         66.44 ms
```

Least-loaded measured run:

```text
Successful requests:              81920
Benchmark duration:               1701.23 s
Request throughput:               48.15 req/s
Output token throughput:          49309.21 tok/s
Mean TTFT:                        98940.79 ms
Median TTFT:                      97619.37 ms
P99 TTFT:                         156626.58 ms
Mean TPOT:                        62.74 ms
Mean ITL:                         70.10 ms
```

Instrumented round-robin rerun measured pass:

```text
Successful requests:              81920
Benchmark duration:               2188.34 s
Request throughput:               37.43 req/s
Output token throughput:          38333.23 tok/s
Mean TTFT:                        147095.77 ms
Median TTFT:                      148747.01 ms
P99 TTFT:                         258153.19 ms
Mean TPOT:                        62.15 ms
Mean ITL:                         89.87 ms
```

The original `load-aware: true` recipe failed because current Dynamo rejects
that CLI flag. The staged load-aware recipe now uses:

```yaml
frontend:
  args:
    router-mode: "least-loaded"
```

## Direct vLLM Matrix

The direct vLLM run used a dedicated Lyris allocation:

```text
session: dp-imbalance-vllm-direct-lyris
slurm job: 1859427
node: lyris0150
```

Because the session host shell did not include `vllm`, every direct command was
run with nested `srun --jobid=1859427 --overlap` and the same public runtime
image:

```text
nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0
```

Base server command:

```bash
vllm serve /model \
  --served-model-name Qwen3-235B-A22B-Instruct-2507-FP4 \
  --host 0.0.0.0 --port 8000 \
  --data-parallel-size 4 \
  --data-parallel-rpc-port 13345 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.9 \
  --kv-cache-dtype fp8 \
  --max-cudagraph-capture-size 2048 \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 864 \
  --no-enable-prefix-caching \
  --quantization modelopt
```

The `api-server-count=8` variant appends:

```bash
--api-server-count 8
```

Direct vLLM default measured run:

```text
Output directory:                  /lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/direct-vllm/default
Successful requests:              81920
Benchmark duration:               1440.66 s
Request throughput:               56.86 req/s
Output token throughput:          58227.52 tok/s
Mean TTFT:                        75423.15 ms
Median TTFT:                      72090.72 ms
P99 TTFT:                         373394.98 ms
Mean TPOT:                        62.25 ms
Mean ITL:                         71.20 ms
```

In vLLM 0.19.0, omitting `--api-server-count` logs that it defaults to
`data_parallel_size`, so this default direct run is effectively
`--api-server-count 4`.

The first `api-server-count=8` attempt failed because stale worker PIDs from the
default direct run were still holding GPU memory. After confirming the PIDs were
the previous `VLLM::Worker_DP*_EP*` processes, those stale PIDs were killed and
the api8 retry was started in:

```text
/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/direct-vllm/api8-retry
```

That api8 retry reached server readiness at 2026-05-22 12:42:17 and responded
to `/metrics`, but no benchmark result was produced before the original direct
allocation timed out:

```text
direct session: dp-imbalance-vllm-direct-lyris
slurm job:      1859427
state:          TIMEOUT at 2026-05-22 14:23:34
server log:     /lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/direct-vllm/api8-retry/server.log
```

Continuation step for the missing direct-vLLM api8 result:

```text
new direct session: dp-imbalance-vllm-direct-api8-lyris
slurm job:          1864796
purpose:            direct vLLM with --api-server-count 8 only
node:               lyris0283
status:             completed; measured benchmark finished
output directory:   /lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/direct-vllm/api8-final
```

The continuation server command is the base direct vLLM command above plus
`--api-server-count 8`, launched through:

```bash
srun --jobid=1864796 --overlap --nodes=1 --ntasks=1 \
  --container-image=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0 \
  --no-container-entrypoint --no-container-mount-home \
  --container-mounts=/lustre/share/coreai_dlfw_dev/models/Qwen3-235B-A22B-Instruct-2507-FP4:/model,/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm:/srt-slurm \
  bash -lc 'exec vllm serve /model ... --api-server-count 8'
```

Server readiness was confirmed with `curl -sS http://localhost:8000/health`.
The api8 benchmark and metrics scrape were then launched:

```text
benchmark log: /lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/direct-vllm/api8-final/benchmark.out
metrics log:   /lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/direct-vllm/api8-final/metrics_scrape.prom
results dir:   /lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/direct-vllm/api8-final/logs/sa-bench_isl_2_osl_1024
```

Benchmark command:

```bash
srun --jobid=1864796 --overlap --nodes=1 --ntasks=1 \
  --container-image=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0 \
  --no-container-entrypoint --no-container-mount-home \
  --container-mounts=/lustre/share/coreai_dlfw_dev/models/Qwen3-235B-A22B-Instruct-2507-FP4:/model,/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm:/srt-slurm,/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/direct-vllm/api8-final/logs:/logs \
  bash -lc 'export SRTCTL_FRONTEND_TYPE=vllm_direct; exec bash /srt-slurm/src/srtctl/benchmarks/scripts/sa-bench/bench.sh http://localhost:8000 2 1024 8192 inf /model Qwen3-235B-A22B-Instruct-2507-FP4 false 4 0 0 1.0 10 2'
```

Api8 warmup completed successfully:

```text
Successful requests:              16384
Benchmark duration:               437.18 s
Request throughput:               37.48 req/s
Output token throughput:          38376.12 tok/s
Mean TTFT:                        315.37 ms
Mean TPOT:                        29.57 ms
Mean ITL:                         29.59 ms
```

The api8 measured run started at 2026-05-22 19:57:16 PDT and completed at
2026-05-22 20:23:12 PDT.

Direct vLLM api8 measured run:

```text
Output directory:                  /lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/direct-vllm/api8-final
Successful requests:              81920
Benchmark duration:               1445.24 s
Request throughput:               56.68 req/s
Output token throughput:          58043.19 tok/s
Mean TTFT:                        61149.62 ms
Median TTFT:                      59371.98 ms
P99 TTFT:                         250422.32 ms
Mean TPOT:                        68.00 ms
Mean ITL:                         69.05 ms
```

In-run api8 `/metrics` sample while the measured run was active:

```text
length_success_max_so_far:
engine 0: 9273
engine 1: 9284
engine 2: 9293
engine 3: 9305

last running/waiting:
engine 0: running 735, waiting 0
engine 1: running 730, waiting 0
engine 2: running 736, waiting 0
engine 3: running 860, waiting 1001
```

This is a transient sample, not the final result, but it shows that direct vLLM
can also have momentary per-engine queue skew during the run.

## Extracted Metrics

Dynamo request-plane and frontend metrics at final scrape:

| Variant | Queue mean | Send mean | Roundtrip TTFT mean | Frontend queued max / mean | Frontend inflight max / mean |
|---|---:|---:|---:|---:|---:|
| Round robin | 0.000025729 s | 0.060338 s | 70.199 s | 5330 / 3093.0 | 8192 / 5440.4 |
| Dedicated router | 0.000016933 s | 0.055285 s | 78.464 s | 6039 / 3641.6 | 8192 / 6014.1 |
| Least loaded | 0.000026937 s | 0.051928 s | 76.409 s | 6574 / 3642.4 | 8192 / 6322.3 |

`dynamo_router_overhead_scheduling_ms` was only non-zero for the dedicated KV
router run:

```text
round robin:      count 0
dedicated router: sum 4215602.107892993 ms, count 98307, mean 42.882 ms
least loaded:     count 0
```

Backend request distribution from `*agg_w0.out`:

```text
round robin received/completed:
24576, 24577, 24576, 24578

dedicated router received/completed:
24688, 24629, 24607, 24383

least loaded received/completed:
25270, 24255, 24447, 24335
```

Direct vLLM default/api4 per-engine metric maxima from `/metrics`:

```text
request_success_total{finished_reason="length"}:
engine 0: 24579
engine 1: 24576
engine 2: 24576
engine 3: 24576

num_requests_running max:
engine 0: 864
engine 1: 864
engine 2: 864
engine 3: 864

num_requests_waiting max:
engine 0: 1184
engine 1: 1184
engine 2: 1184
engine 3: 1184
```

Direct vLLM api8 per-engine metric maxima from `/metrics`:

```text
request_success_total{finished_reason="length"}:
engine 0: 24579
engine 1: 24568
engine 2: 24582
engine 3: 24578

num_requests_running max:
engine 0: 864
engine 1: 864
engine 2: 864
engine 3: 864

num_requests_waiting max:
engine 0: 1185
engine 1: 1184
engine 2: 1184
engine 3: 1185

final num_requests_running:
engine 0: 0
engine 1: 0
engine 2: 0
engine 3: 0
```

No metrics matching `forward`, `fresh`, `stale`, `age`, `slot`, or `state` were
found in the Dynamo Prometheus scrapes for these runs, so forward-pass metric
age / state freshness is not directly observable from the emitted metrics in
this setup.

The Dynamo scrapes also did not expose vLLM-style
`num_requests_running` / `num_requests_waiting` backend gauges. Those gauges
were available on direct vLLM only.

## Current Signal

The original gap reproduces directionally on the current Lyris stack:

```text
Dynamo round robin:       49.26k output tok/s, mean TTFT 99.9 s
Dynamo dedicated router:  48.99k output tok/s, mean TTFT 103.8 s
Dynamo least loaded:      49.31k output tok/s, mean TTFT 98.9 s
Direct vLLM default/api4: 58.23k output tok/s, mean TTFT 75.4 s
Direct vLLM api8:         58.04k output tok/s, mean TTFT 61.1 s
Gap vs best direct tput:  Dynamo is about 15.3% lower throughput and 23.5 s worse mean TTFT
Gap vs direct api8 TTFT:  Dynamo is about 15.0% lower throughput and 37.8 s worse mean TTFT
```

Round-robin Dynamo request assignment was effectively even by backend instance
at drain:

```text
24576, 24577, 24576, 24578 requests received/completed per backend instance
```

That means even request distribution alone did not fix the TTFT/throughput gap.
During the run, completion counts were temporarily skewed even while received
counts stayed even, which points more toward transient backend service/drain
skew or stale state than static request assignment skew.

## Metrics To Collect

Collection status:

- Output tok/s, req/s, TTFT, TPOT/ITL: collected from SA-Bench logs.
- Per-DP-rank request counts: collected from Dynamo backend logs and direct
  vLLM `/metrics`.
- Per-DP-rank TTFT: not directly available from current emitted metrics/logs.
- Backend waiting/running queue depth: collected for direct vLLM from
  `/metrics`; not exposed in Dynamo scrapes.
- `dynamo_request_plane_queue_seconds`: collected.
- `dynamo_request_plane_send_seconds`: collected.
- `dynamo_request_plane_roundtrip_ttft_seconds`: collected.
- `dynamo_frontend_queued_requests`: collected.
- `dynamo_frontend_inflight_requests`: collected.
- `dynamo_router_overhead_scheduling_ms`: collected where non-zero.
- Forward-pass metric age / state freshness: not found in current scrapes.

## Cleanup

Released direct-vLLM compute sessions after collecting results:

```text
dp-imbalance-vllm-direct-api8-lyris
dp-imbalance-vllm-direct-lyris
```

`list_sessions` reported no active compute sessions after cleanup.
