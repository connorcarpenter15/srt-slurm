# DEP Routing Investigation — Final Closure

Date: 2026-06-27

## Decision

Close the Dynamo DEP routing issue.

The evidence does not support request routing or DP-rank load imbalance as a
material throughput bottleneck for the investigated Qwen3-235B DP=4/EP workload.
Temporal rank underfeed exists in drainable-queue regimes, but changing the
routing behavior enough to substantially reduce the observed skew did not
improve throughput. Further routing-policy, token-balancing, or EPLB experiments
on this workload are not justified.

This decision closes the routing hypothesis only. A separate Dynamo output-
streaming question and the vLLM/NCCL EP collective and GEMM costs remain outside
this issue.

## Evidence

| Question | Result | Evidence |
|---|---|---|
| Did static DP-rank assignment imbalance cause the original gap? | No. Round robin ended with nearly exact per-rank request counts while still trailing direct vLLM by 15.3%. | [`dp-imbalance-repro/FINAL-REPORT.md`](dp-imbalance-repro/FINAL-REPORT.md) |
| Was Dynamo's request-plane queue or send path the bottleneck? | No. Local queue and send costs were milliseconds, versus tens of seconds of backend TTFT. | [`dp-imbalance-repro/FINAL-REPORT.md`](dp-imbalance-repro/FINAL-REPORT.md) |
| Does temporal DP-rank underfeed exist? | Yes. With a drainable queue, short-window running skew appeared despite equal final counts and only 0.058% per-rank mean-duration spread. | [`dep-bubble/FINDINGS-4-DRAIN.md`](dep-bubble/FINDINGS-4-DRAIN.md) |
| Does that temporal skew materially limit throughput? | No demonstrated effect. At concurrency 3072 and 4096, the least-loaded run reduced p95 running skew by 37% and 85%, while throughput changed by only +0.5% and +0.1%. Across the sweep, throughput remained within +/-1.4%. | [`dep-bubble/FINDINGS-5-SWEEP.md`](dep-bubble/FINDINGS-5-SWEEP.md) |
| Does prefill-heavy DEP show the same underfeed? | No. All ranks handled exactly 20,480 requests, running skew was effectively zero, and service-duration spread was 0.13%. | [`dep-bubble/FINDINGS-3-PREFILL.md`](dep-bubble/FINDINGS-3-PREFILL.md) |
| Would expert rebalancing or EPLB recover the loss? | No evidence supports it. Dominant expert-GEMM grid dimensions matched within 0.01%, and no GPU was uniformly slower. | [`dep-bubble/FINDINGS-2-NSYS.md`](dep-bubble/FINDINGS-2-NSYS.md) |
| Is a new prefill-token routing policy needed? | No. Same-state replay found 0% divergent non-tied choices from KV-default under both no-cache and approximately 72% prefix-reuse traces. | [`prefill-token-routing/OFFLINE-GATE-FINDINGS.md`](prefill-token-routing/OFFLINE-GATE-FINDINGS.md) |

## Interpretation

The investigation isolated a real mechanism: when a rank's queue drains, ranks
can fall out of lockstep and arrive at the per-layer EP collective at different
times. That mechanism is visible in backend queue data and GPU profiles. It is
not, however, the throughput ceiling in the canonical workload.

The decisive observation is that large changes in rank skew did not produce a
corresponding throughput change. Once concurrency reached approximately 3072,
throughput plateaued near 66k output tokens/s even as queue depth and temporal
skew changed substantially. GPU profiles instead placed approximately 25-28% of
wall time inside EP collectives, with the remainder of the limiting step
dominated by balanced attention and expert GEMM work. The recoverable performance
work is therefore in backend execution, not request placement.

The current Dynamo router also already has the relevant mechanisms: KV routing
can select an exact worker and DP rank using cache overlap plus active prefill and
decode load, and it passes that rank to vLLM. The retired token-balance policy
duplicated the existing cost model in the tested regimes rather than filling a
missing routing capability.

## Evidence Limitations

Some earlier reports described controls as same-node when they were actually on
different physical nodes of the same class. The round-robin and least-loaded
sweeps were also single-trial, cross-node comparisons. In addition, non-KV
`least-loaded` selects Dynamo endpoint instances, not individual vLLM DP ranks;
with one aggregated endpoint its influence on vLLM's internal rank selection was
indirect.

These limitations could hide a small run-to-run effect, but they do not support a
material routing opportunity. Repeating the campaign could improve the precision
of a sub-percent estimate; it would not change the engineering decision unless a
new workload first demonstrates a routing-correlated performance regression.

## Remaining Work Is Separate

Two performance areas remain, neither of which should keep the routing issue
open:

1. **Dynamo output streaming:** the corrected high-throughput run used
   `stream-interval: 50`, reaching 69.1k output tokens/s while client-visible
   mean ITL increased to 2.385 seconds. A matched current-code direct-vLLM versus
   Dynamo comparison at equal stream intervals is still warranted as a separate
   Python/Rust bridge and output-processing issue.
2. **vLLM/NCCL execution:** all-to-all backend selection, collective/computation
   overlap, expert GEMM efficiency, communication precision, and parallelism
   topology are the credible paths to raising the approximately 66k-token/s
   execution ceiling.

## Final Disposition

- Close the DEP routing/DP-rank imbalance issue as **not a material throughput
  cause**.
- Retain the assignment and per-rank observability as regression infrastructure.
- Do not run more round-robin, least-loaded, KV-threshold, prefill-token-balance,
  or EPLB campaigns for this fixed workload.
- Track Dynamo streaming overhead separately.
- Track EP collective/GEMM optimization with the vLLM/NCCL execution work.

Reopen routing only if a new workload shows both a reproducible routing-dependent
imbalance and a material performance response when that imbalance is removed.
