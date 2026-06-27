# Experimental Prefill Token-Balance Routing - Initial GPU Results

Date: 2026-06-26

Cluster: **computelab**, 1 H100 NVL node (`4u8g-gen-0033`) for all usable runs.

Model: `Qwen/Qwen3-32B-FP8`, 6 single-GPU prefill workers + 2 single-GPU
decode workers.

Workload: SA-Bench random prompts, nominal ISL 1024 with
`random-range-ratio=0.8` (approximately 819-1024 tokens), OSL 1, no prefix
caching, concurrency sweep 512/1024/2048/4096.

## Verdict

**The experimental `prefill-token-balance` policy has not demonstrated a
material improvement over the existing load-aware KV policy.** At the two
completed points, KV-default and token-aware routing produced essentially the
same sustained active-prefill-token balance. At concurrency 1024, the average
max-minus-min load across the six prefill workers was 656 tokens for KV-default
and 686 tokens for token-aware, or about 2.0% of the per-worker mean in both
cases.

Token-aware throughput was 2.6% higher than KV-default at concurrency 512 and
0.6% higher at 1024. Mean TTFT improved 4.2% and 0.7%, respectively. There were
no repeated trials, and the token-skew metric did not improve consistently, so
these small performance deltas should be treated as run-to-run variation rather
than a policy win.

The existing KV/load-aware router is the important result in this experiment:
it delivered 82% more requests/s than round robin at concurrency 512 and 145%
more at 1024, while avoiding round robin's extreme TTFT tail growth. Under this
no-cache, narrow-ISL workload, the default KV cost model already behaves like an
effective prefill-load balancer. The new policy mostly reproduces that behavior.

The original success criterion cannot yet be declared met or failed. Round-robin
mode did not export `dynamo_frontend_worker_active_prefill_tokens`, so this run
cannot directly show whether token-aware routing reduced token skew relative to
round robin. The token-aware run also exhausted `/home` space while writing the
concurrency-2048 request trace, leaving no measured 2048 or 4096 result.

## Jobs

| Job | Policy | Node | Result |
|---:|---|---|---|
| `2835756` | Round robin | `4u8g-gen-0033` | Complete: 512, 1024, 2048, 4096 |
| `2836045` | KV-default, load-aware | `4u8g-gen-0033` | Complete: 512, 1024, 2048, 4096 |
| `2836053` | Prefill token balance, first attempt | `4u8g-gen-0289` | No benchmark data; node Slurm plugin could not load `libpython3.12.so.1.0` |
| `2838557` | Prefill token balance, retry | `4u8g-gen-0033` | Complete: 512, 1024; failed during 2048 trace write with `ENOSPC` |

The failed jobs are infrastructure/storage failures, not evidence of router or
engine failure. The usable token-aware points are directly comparable with the
baselines because all three usable jobs ran on the same node and deployment
shape.

## Performance Results

These are the full measured runs, excluding each concurrency point's warmup.
Total token throughput includes input and output tokens; with OSL 1, request
throughput is the cleaner measure of prefill completion rate.

| Policy | Conc. | Req/s | Total tok/s | TTFT mean | TTFT median | TTFT p99 |
|---|---:|---:|---:|---:|---:|---:|
| Round robin | 512 | 22.60 | 20,845 | 16.00 s | 3.51 s | 106.27 s |
| KV-default | 512 | 41.10 | 37,912 | 11.43 s | 7.44 s | 58.03 s |
| Token-aware | 512 | **42.17** | **38,899** | **10.95 s** | **7.26 s** | **49.22 s** |
| Round robin | 1024 | 20.49 | 18,906 | 35.08 s | 9.71 s | 299.83 s |
| KV-default | 1024 | 50.09 | 46,229 | 18.81 s | 19.40 s | 26.87 s |
| Token-aware | 1024 | **50.38** | **46,490** | **18.69 s** | **19.23 s** | **26.49 s** |
| Round robin | 2048 | 20.39 | 18,818 | 70.61 s | 22.38 s | 587.93 s |
| KV-default | 2048 | **49.91** | **46,060** | **38.34 s** | **40.14 s** | **47.62 s** |
| Token-aware | 2048 | - | - | - | - | - |
| Round robin | 4096 | 20.33 | 18,767 | 142.04 s | 46.33 s | 1,157.22 s |
| KV-default | 4096 | **50.00** | **46,155** | **77.16 s** | **81.02 s** | **88.54 s** |
| Token-aware | 4096 | - | - | - | - | - |

Request throughput at the two directly comparable points:

```text
concurrency 512 (req/s)
  round robin  #####################                     22.60
  KV-default   #######################################   41.10
  token-aware  ########################################  42.17

concurrency 1024 (req/s)
  round robin  ################                        20.49
  KV-default   ######################################## 50.09
  token-aware  ######################################## 50.38
```

Token-aware relative to KV-default:

| Conc. | Req/s | Total tok/s | Mean TTFT | Median TTFT | p99 TTFT |
|---:|---:|---:|---:|---:|---:|
| 512 | +2.60% | +2.60% | -4.21% | -2.42% | -15.18% |
| 1024 | +0.57% | +0.56% | -0.65% | -0.88% | -1.43% |

The p99 improvement at 512 is interesting but not sufficient to claim a policy
effect. It does not persist at the same magnitude at 1024, and there is only one
trial per configuration.

## Active Prefill Token Balance

The frontend was scraped once per second. For every snapshot with total active
prefill tokens at least 90% of that run's peak, the analysis computed:

```text
token skew = max(active_prefill_tokens across 6 workers)
           - min(active_prefill_tokens across 6 workers)

normalized skew = token skew / mean(active_prefill_tokens across 6 workers)
```

The 90%-of-peak filter isolates the sustained high-load plateau and removes
startup and drain snapshots, where one remaining request can produce a large
normalized skew without representing an overloaded rank.

| Conc. | Policy | Snapshots | Mean skew | p95 skew | Mean normalized | p95 normalized |
|---:|---|---:|---:|---:|---:|---:|
| 512 | KV-default | 61 | 780 tokens | 1,925 tokens | 2.36% | 6.17% |
| 512 | Token-aware | 63 | 778 tokens | **1,269 tokens** | 2.36% | **3.83%** |
| 1024 | KV-default | 185 | **656 tokens** | **886 tokens** | **1.97%** | **2.66%** |
| 1024 | Token-aware | 184 | 686 tokens | 899 tokens | 2.07% | 2.70% |

At concurrency 512, token-aware lowers the p95 transient but leaves mean skew
unchanged. At 1024, all skew measures are effectively tied, with KV-default
slightly lower on the point estimates. This is not a material or consistent
reduction beyond the current policy.

Round robin emitted no worker active-prefill-token gauge in these captures, even
though the recipe requested prefill-token tracking. Therefore the most important
planned comparison, token-load skew versus round robin, is missing. Final
per-worker completed request and input-token counters were also not exposed, so
those planned balance checks cannot be reconstructed from these artifacts.

## Interpretation

1. **Load-aware routing solves the large performance problem.** Both KV policies
   more than double round-robin throughput at concurrency 1024 and sharply bound
   p99 TTFT. Round robin's throughput plateaus near 20 req/s while its queueing
   latency grows with concurrency.
2. **The experimental scoring rule adds no demonstrated value here.** With prefix
   caching disabled, cache overlap is always zero. The workload's prompt lengths
   also occupy a relatively narrow 819-1024-token band. In that regime,
   KV-default's active-load behavior and pure projected-token balancing make
   nearly identical choices.
3. **This is not a broad rejection of token-aware routing.** A wider ISL
   distribution, heterogeneous workers, or cache-bearing traffic creates more
   opportunity for projected token work to differ from the default cost. None of
   those conditions were exercised here.
4. **No upstream policy claim should be made from this campaign yet.** The run is
   partial, has no repeated seeds, and lacks the round-robin token-skew and
   per-worker completion metrics required by the experiment plan.

## Recommended Next Experiment

1. Move request traces and metric snapshots off quota-limited `/home`, then run
   token-aware at concurrency 2048 and 4096 to complete the original matrix.
2. Add a policy-independent assignment counter with worker ID, request count,
   and assigned input tokens so round robin and KV routing expose the same
   per-worker measurements.
3. Repeat each configuration at least three times and report confidence ranges.
4. Use a deliberately broad ISL distribution, such as several prompt-length
   buckets spanning at least 8x, while retaining OSL 1. This is the workload most
   likely to distinguish request-aware from token-aware balancing.
5. After the no-cache isolation test, add a prefix-reuse workload to measure the
   intended trade-off between cache affinity and projected prefill-token balance.

The immediate engineering conclusion is to keep `prefill-token-balance`
experimental. The next run should focus on observability and workload contrast,
not another identical single-seed comparison against KV-default.

## Measurement Gaps

- SA-Bench was configured with percentile `99`, so p95 TTFT was not recorded.
- Round-robin mode did not expose the active-prefill-token worker gauge.
- No per-worker completed-request or completed-input-token counters were present.
- Token-aware results at concurrency 2048 and 4096 are absent due to storage
  exhaustion, not policy behavior.
- One trial per policy/concurrency is insufficient to separate small deltas from
  run-to-run variance.

## Artifacts

Cluster outputs:

- Round robin: `/home/connorc/work/srt-slurm/outputs/2835756/`
- KV-default: `/home/connorc/work/srt-slurm/outputs/2836045/`
- Token-aware first attempt: `/home/connorc/work/srt-slurm/outputs/2836053/`
- Token-aware retry: `/home/connorc/work/srt-slurm/outputs/2838557/`

Local recipes and analysis:

- `dep/prefill-token-routing/qwen3-32b-h100-pcie-1node-prefill-round-robin.yaml`
- `dep/prefill-token-routing/qwen3-32b-h100-pcie-1node-prefill-kv-default.yaml`
- `dep/prefill-token-routing/qwen3-32b-h100-pcie-1node-prefill-token-balance.yaml`
- `dep/prefill-token-routing/analyze_prefill_outputs.py`
