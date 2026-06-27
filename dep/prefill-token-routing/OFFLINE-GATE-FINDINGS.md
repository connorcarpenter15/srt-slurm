# Prefill Token Routing Offline Gate Findings

Date: 2026-06-26

## Decision

The offline gate failed. KV-default and `prefill-token-balance` made zero
different non-tied worker choices on both prefix-reuse replays, far below the
pre-registered 5% threshold. No GPU campaign was run.

The policy, its CLI/config/environment surface, its documentation, the shadow
replay option, and policy-specific tests were retired. The policy-independent
frontend assignment counters and their analyzer support were retained.

## Gate Results

Each trace contained 10,240 requests, used OSL 1 and block size 64, and was
replayed with 6 prefill and 2 decode workers at concurrency 1,024. Both policy
directions evaluated the same scheduler state before primary admission mutated
it. A request was comparable only when both policies had exactly one minimum.

| Trace | Primary | Comparable unique choices | Different workers | Divergence |
|---|---|---:|---:|---:|
| No cache | KV-default | 10,030 | 0 | 0.000% |
| No cache | Token-aware | 10,030 | 0 | 0.000% |
| Prefix reuse | KV-default | 4,557 | 0 | 0.000% |
| Prefix reuse | Token-aware | 4,554 | 0 | 0.000% |

The no-cache runs each excluded 210 tied requests. The prefix-reuse runs
excluded 5,683 and 5,686 tied requests, respectively. The replay-reported
prefix-cache reuse ratio was 0 for the control and approximately 0.72 for the
contrast. The generated contrast assigns shared prefixes to 32 groups and
places 73.333% of aggregate input tokens in the shared portions.

## Artifacts and Reproduction Record

- Offline host: `aflowers-workstation`
- Trace seed: `20260626`
- GPU trial seeds reserved but not used: `17`, `29`, `43`
- srt-slurm source before these changes:
  `6cea0a5be74221b679b4de655212d70cee9acd61`
- Initial experimental Dynamo policy:
  `65a074cde49ae4ed5034d4f12d11ecad4de23598`
- Same-state shadow replay implementation used for the gate:
  `b5a99f3d92`
- Gate-driven policy retirement:
  `0647cdbcdf`
- Final Dynamo observability branch:
  `92430a54a993b66935fe2c5c3e4cd11a832c53c4`
- A transient validation allocation, job `2840535` on `ipp2-0041`, was released
  without running a benchmark. No GPU node contributed measurements.

The exact traces are in `traces/`. Full per-request replay JSONL is retained as
four zstd archives in `shadow-gate-results/`; decompress with `zstd -d`. The
uncompressed JSONL SHA-256 values were:

| Replay | Uncompressed JSONL SHA-256 |
|---|---|
| No cache, KV-default primary | `dd9da6d233ee8f9767342cc4b782636ee76ba273c7699a76d14fa5625e270f19` |
| No cache, token-aware primary | `5f1f21641d015e310a3b7e0daadf2bfece5702f6832ea72600343ba5b5545a6c` |
| Prefix reuse, KV-default primary | `4fe406c508ffb8be7d429aa4a72b0d7e44969e4e212f07a6ac8287f4624e4c6e` |
| Prefix reuse, token-aware primary | `fbc2dc3a579a536b306787980f882336d65ba627a0f8db944d9b95e2ad2855cd` |

Current artifact hashes are recorded in `CHECKSUMS.sha256`.

## Retained Implementation

The final Dynamo branch exposes these frontend counters with `worker_id` and
`dp_rank` labels:

```text
dynamo_frontend_prefill_worker_assigned_requests_total
dynamo_frontend_prefill_worker_assigned_input_tokens_total
```

They increment in the shared prefill dispatch preparation path after worker
selection, cover round-robin and KV routing, preserve concrete DP ranks (or
`none`), count actual tokenized input length, and exclude advisory query-only
selection.

The srt-slurm analyzer reports per-worker deltas, normalized range, coefficient
of variation, max/mean, one-snapshot-window median and p95 imbalance, missing
workers, resets, exact SA-Bench reconciliation, and sustained-load active-token
skew. SA-Bench now accepts `benchmark.seed` and applies it to warmup and measured
invocations. Explicit baseline and final metric scrapes bound the measured
counter interval.
