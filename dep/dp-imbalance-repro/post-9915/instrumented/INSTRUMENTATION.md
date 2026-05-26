# DP Imbalance Instrumentation

This directory contains the trace-enabled follow-up recipes for the post-PR
9915 Dynamo/vLLM DP imbalance reproduction.

## Dynamo Source

- Branch: `dev/connorc/dp-instrumentation-post9915`
- Commit: `c70bdbe76083e0039dccf68bb3c479ab3994a053`
- Commit title: `feat(dp-trace): add request lifecycle instrumentation`

The recipes pin `dynamo.hash` to that commit so the cluster source install uses
the same instrumentation revision.

## What It Emits

Enablement is controlled by:

```bash
DYN_REQUEST_TRACE_LOGGING=1
DYN_REQUEST_TRACE_DIR=/logs
```

Trace events are emitted as `dynamo_request_trace {json}` log lines. Python
vLLM handlers also write direct `/logs/dynamo_request_trace_vllm_<pid>.jsonl`
sidecar files.

Collected events:

- `router_enqueued`: router queue admission, pending request/token counts.
- `router_assigned`: selected worker/DP rank, selected load snapshot, per-rank
  load snapshot map, queue delay, snapshot age.
- `request_plane_enqueue`, `request_plane_send_start`,
  `request_plane_send_done`, `request_plane_first_response`: frontend
  request-plane timing.
- `backend_dp_enter`, `backend_dp_first_token`, `backend_dp_done`,
  `backend_dp_no_output`, `backend_dp_error`: backend vLLM lifecycle by DP
  rank.

New Prometheus gauges:

- `dynamo_component_vllm_dp_requests_running{dp_rank=...}`
- `dynamo_component_vllm_dp_requests_waiting{dp_rank=...}`

## Recipes

- `qwen3-235b-a22b-vllm-agg-lyris-gb200-dp4-ep-round-robin-instrumented.yaml`
- `qwen3-235b-a22b-vllm-agg-lyris-gb200-dp4-ep-load-aware-instrumented.yaml`
- `qwen3-235b-a22b-vllm-agg-lyris-gb200-dp4-ep-dedicated-router-instrumented.yaml`

## Analysis

Join client, router, request-plane, and backend events:

```bash
python dep/dp-imbalance-repro/trace_summary.py \
  --client-trace /logs/sa-bench_isl_2_osl_1024/request_trace_concurrency_8192_gpus_4.jsonl \
  --server-log /logs/dynamo_request_trace_vllm_*.jsonl \
  --server-log <frontend-or-backend-log-containing-dynamo_request_trace> \
  --csv-dir /logs/dp_trace_join
```

Summarize frontend/backend scrape snapshots and estimate underfeed area:

```bash
python dep/dp-imbalance-repro/metrics_summary.py /logs/metrics \
  --glob 'frontend__*.prom' \
  --backend-glob 'backend__*.prom' \
  --max-num-seqs 864 \
  --interval-s 1.0
```

The main follow-up plots should use `request_join.csv` and
`admissions_per_second.csv` for per-rank TTFT, rank admission rate, router
snapshot age, and temporal underfeed analysis.

## Local Checks

- `python -m py_compile` on touched Dynamo Python files.
- `python -m py_compile dep/dp-imbalance-repro/trace_summary.py dep/dp-imbalance-repro/metrics_summary.py`.
- `cargo fmt --check` in `dynamo:latest-vllm-local-dev`.
- `cargo check -p dynamo-kv-router -p dynamo-runtime` in
  `dynamo:latest-vllm-local-dev`.

Local `pytest` was not available in the base shell, so the added Python unit
test was syntax-checked but not executed locally.
