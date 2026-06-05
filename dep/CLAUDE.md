# CLAUDE.md — DEP Routing / DP Imbalance Investigation

Working guide for the `dep/` investigation. This is **not** application code — it
is a benchmark/analysis workspace for diagnosing why **Dynamo** underperforms
**direct vLLM** on a high-concurrency data-parallel (DP=4) expert-parallel (EP)
workload, and whether the gap is caused by backend DP-rank underfeed/imbalance
("DEP routing").

It is a sibling project to the `srtctl` codebase one level up (see
`../CLAUDE.md` for srt-slurm itself). Recipes here are submitted *through*
srtctl; the analysis scripts here post-process the resulting cluster logs.

## What This Investigation Is

Origin: Slack thread `C093RGPF84E/p1778252809274269`. The ask was to compare
Dynamo against direct vLLM for a high-concurrency DP=4 EP workload and check
whether Dynamo underfeeds or imbalances backend DP ranks.

The canonical stress workload (keep this shape fixed when reproducing):

- Model: `Qwen3-235B-A22B-Instruct-2507-FP4` (MoE), FP4 / modelopt quant
- vLLM `--data-parallel-size 4 --enable-expert-parallel`, FP8 KV cache
- Backend limits: `max-num-seqs=864`, `max-num-batched-tokens=2048`,
  `max-model-len=2048`, **no prefix caching**
- SA-Bench: `isl=2`, `osl=1024`, `concurrency=8192`, 81920 measured requests
- Cluster: Lyris GB200, 1 node, 4 GPUs/node (GB300 recipes staged for later)

The decode-heavy, no-locality shape is deliberate: it maximizes pressure on
backend admission/refill and gives the KV router nothing to exploit.

## Current State / Conclusion

Read these in order; later supersedes earlier:

1. `dp-imbalance-repro/FINAL-REPORT.md` — clean-stack repro: gap reproduces
   (~15% lower throughput, +24–38s mean TTFT). Root-cause signal is **temporal
   DP-rank underfeed**, not static final request-count imbalance (round-robin
   ends with near-perfect per-rank counts yet still loses).
2. `dp-imbalance-repro/post-9915/POST-9915-COMPREHENSIVE-REPORT.md` — first
   post-PR-9915 rerun. **Superseded / invalid**: recipes omitted
   `stream-interval`, so PR 9915's code path was not actually exercised.
3. `dp-imbalance-repro/post-9915/POST-9915-STREAM-INTERVAL-50-REPORT.md` —
   **current authoritative conclusion (throughput-gap thread).** With
   `stream-interval: 50` set, job `1918158` reached 69.1k tok/s / 66.5s mean
   TTFT — the gap does **not** reproduce. PR 9915 fixes it *when the stream
   interval is applied.*
4. `dep-bubble/FINDINGS-2-NSYS.md` — **separate, still-open thread:** the EP-phase
   "bubble" investigation. After the gap was fixed, the question became whether DP
   ranks still enter the per-MoE-layer EP all-to-all at different times (temporal
   skew within the barrier). nsys profiling of the agg run shows the END-skew is a
   **ring↔NVLink-topology mapping artifact** (not expert-load skew, not one bad
   GPU): the `SRT_DP_GPU_PERMUTATION=reverse` control (job 1987608) refuted the
   simpler "fixed ring-position leader" story. Verdict is revised/refined twice in
   that file; a same-node identity-vs-reverse control is still pending to remove a
   node confound.
5. `dep-bubble/FINDINGS-3-PREFILL.md` — **prefill-heavy control (ptyche, job
   2187958).** Inverts the shape to isl=1024/osl=1. Result: **no underfeed.** The
   DP=4 ranks are uniformly saturated (running pinned at ~3, token-budget bound),
   evenly loaded (20,480 req each), and equally fast (per-rank duration spread
   0.13%); `Waiting` never drains below ~1700 so the starvation mode that creates
   the bubble can't occur. Conclusion: the underfeed bubble is a *decode-regime*
   phenomenon — prefill saturation removes the drainable-queue precondition.
6. `dep-bubble/FINDINGS-4-DRAIN.md` — **drainable-queue probe (ptyche, 3-shape
   campaign).** Drains the decode queue by dropping concurrency 8192→2048 (job
   `2188160`): steady state holds `Running ≈ 512 < 864` with `Waiting = 0` on
   every rank — the precondition FINDINGS-3 said was missing. Under it, macro
   per-10s-window `running_skew` reaches **89** (ranks fall out of lockstep) even
   though final counts (5,120/rank) and per-rank average durations (0.058 %
   spread) stay even. This is the clearest macro confirmation of *temporal*
   underfeed — but at conc 2048 it costs no throughput (254 ms median TTFT, lots
   of slack), so it is a visibility/mechanism result, not a regression. The mixed
   shape (isl/osl=1024, job `2188157`) stays saturated (Waiting ~1000, never
   drains → no precondition); the isl=4096 prefill confirm (job `2188156`) **OOM'd
   at cudagraph capture** before serving. The first-round nsys captures all failed
   mechanically (see gotchas), so the drain shape was **resubmitted with retimed
   nsys** (`delay_secs=560`, job `2188338`, ptyche0162) and finalized offline. Its
   kernel-level EP barrier shows an **arrival-dominated** skew (START≈END, ~203 µs
   p50 / ~376 µs p99) that **rotates across ranks** (no fixed straggler) and is
   **not** compute-load skew (attn balanced 1.015×, expert GEMM 1.07-1.10×; the
   variance lives in the collective wait) — the kernel signature of temporal
   underfeed. **Node confound now CLOSED** by the saturated control (job `2191165`,
   ptyche0287, same identity perm + isl=2/osl=1024, only regime varies): the same
   barrier at *saturation* is ~79 µs p50 vs the drain's ~203 µs, so drainage
   causally widens the arrival skew **2.6× on the identical node + permutation** —
   it is regime-driven, not ptyche ring mapping. A residual ~2× offset (ptyche
   identity 79 µs vs lyris identity 37 µs; ptyche identity is arrival-dominated
   where lyris identity was ring-pinned) is a constant topology baseline that
   drainage stacks on top of. Compute stays balanced in both regimes.
7. `dep-bubble/FINDINGS-5-SWEEP.md` — **throughput-knee sweep + final synthesis
   (ptyche, job `2191166`).** Single-node concurrency sweep 2048→3072→4096→6144 at
   the canonical decode shape. **The knee is at conc ≈ 3072:** the queue still
   *fully drains* (Waiting=0) yet throughput has reached 97.4 % of its ceiling
   (64.1k of 65.8k tok/s) and macro `run_skew` peaks (p95 187, vs 1 at 2048).
   Throughput scales +15 % from 2048→3072 then flattens (+2.7 % to 6144); past the
   knee concurrency buys only TTFT (1.3 s → 38 s). **Final synthesis:** (cause) a
   drainable-queue + round-robin temporal underfeed, arrival-dominated and
   rotating, not count/expert/GPU/topology skew; (impact) **essentially nil on
   throughput** — see the fix test; mostly a throughput/latency-headroom effect;
   (fix, TESTED) the least-loaded control (job `2191349`, same shape, only
   `router-mode: least-loaded`) **collapses run_skew** (knee p95 187→117, −37 %;
   conc 4096 685→105, −85 %) but **does not move throughput** (±1.4 %) — so the
   skew is a *symptom*, not the throughput bottleneck; the ~66k ceiling is set by
   per-step EP-collective + GEMM cost, not arrival jitter. Least-loaded is a free
   balance improvement, not a throughput win at this shape. The throughput-gap
   regression itself stays fixed (PR 9915 + `stream-interval: 50`, #3).

If you only read one file about the **throughput gap**, read #3 ("did PR 9915 fix
it?" → yes, conditional on `stream-interval: 50`). For the **EP-barrier / temporal
skew** question, read #4 (mechanism, kernel-level + node-confound closed); for why
it doesn't appear under prefill load, read #5; for the drainable-queue test that
surfaces macro temporal skew, read #6; for the throughput-knee sweep that
quantifies its cost + the final cause/impact/fix synthesis, read #7.

## Layout

```
dep/
├── dp-imbalance-repro/        # REAL cluster repro (tracked in git)
│   ├── *.md                   # run logs + findings reports
│   ├── *.py                   # log/metrics post-processing scripts
│   ├── *-lyris-gb200-*.yaml   # srtctl recipes used for completed runs
│   ├── *-gb300-*.yaml         # staged GB300 recipes (not yet run)
│   └── post-9915/             # PR-9915 follow-up
│       └── instrumented/      # trace-enabled recipes (pin instrumentation hash)
└── router-gym/                # OFFLINE router-policy replay (untracked)
    ├── scenarios/             # gym scenario YAMLs
    ├── traces/                # mooncake + request traces converted from job 1880624
    └── dep-dp4-*/             # replay run outputs (manifest, results.jsonl, reports/)
```

`dp-imbalance-repro/` is committed; `router-gym/` is currently untracked working
output.

## Two Very Different Kinds of "Result" — Do Not Conflate

- **`dp-imbalance-repro/`** = real SLURM jobs on real GPUs. Numbers are
  wall-clock throughput/TTFT and are directly comparable.
- **`router-gym/`** = an **offline simulator** (Dynamo's `benchmarks/router/gym`,
  `replay_mode: offline`, `speedup_ratio: 1000`). A whole 81920-request run
  finishes in ~570ms of `duration_ms`. Its `output_throughput_tok_s` (~1.4e8)
  and sub-50ms TTFT are **simulator artifacts, not real performance.** Use gym
  results only for *relative* comparison of router policies under an analytic
  model, never as absolute throughput, and never against real runs.

## Running Things

### Submitting cluster recipes (dp-imbalance-repro)

Recipes are srt-slurm YAML. Submit via the srtctl MCP after MFA socket login:

```
srtctl_apply(config=<recipe.yaml>, cluster="lyris")
```

Outputs land on persistent Lyris Lustre:
`/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/<job_id>/`. The full srun
command and per-worker env/commands are in `logs/sweep_<job_id>.log`. Inspect
the recipe before submit with `srtctl dry-run -f <recipe.yaml>`.

Three router variants are the standard matrix:

| Variant | Frontend config |
|---|---|
| Round robin | `enable_multiple_frontends: true`, default routing |
| Least-loaded | `frontend.args.router-mode: least-loaded` |
| Dedicated KV | single frontend, `router-mode: kv`, queue threshold 64, KV events off |

### Analysis scripts

All three are stdlib-only Python 3.10+, run from the repo root.

```bash
# Join SA-Bench client trace with Dynamo router/request-plane/backend events
python dep/dp-imbalance-repro/trace_summary.py \
  --client-trace /logs/sa-bench_isl_2_osl_1024/request_trace_concurrency_8192_gpus_4.jsonl \
  --server-log '/logs/dynamo_request_trace_vllm_*.jsonl' \
  --server-log /path/to/frontend.log \
  --csv-dir /logs/dp_trace_join

# Per-DP queue depth + temporal skew from vLLM "Engine NNN:" backend log lines
python dep/dp-imbalance-repro/backend_log_summary.py /path/to/lyris0213_agg_w0.out

# Prometheus scrape snapshots -> gauges, histograms, underfeed estimate
python dep/dp-imbalance-repro/metrics_summary.py /logs/metrics \
  --glob 'frontend__*.prom' --backend-glob 'backend__*.prom' \
  --max-num-seqs 864 --interval-s 1.0
```

### Router gym replay

Runs in the `nvcr.io/nvidia/ai-dynamo/vllm-runtime` container against a checked
out Dynamo source (`dynamo replay`). A run writes `manifest.json`,
`results.jsonl` (one normalized record per variant), `reports/*.json` (full
replay report per variant), and `summary.md`. Scenarios live in
`router-gym/scenarios/`; the synthetic `dep_dp4.yaml` and the trace-derived
`dep_dp4_trace_1880624.yaml` are the two DEP scenarios.

## Instrumentation Branch

Server-side per-request tracing was added in Dynamo branch
`dev/connorc/dp-instrumentation-post9915`. Recipes under
`post-9915/instrumented/` pin `dynamo.install: true` + `dynamo.hash` to that
revision and set `DYN_REQUEST_TRACE_LOGGING=1` / `DYN_REQUEST_TRACE_DIR=/logs`
on frontend and backend. Events: `router_enqueued`, `router_assigned`,
`request_plane_*`, `backend_dp_{enter,first_token,done,...}`. It also adds
`dynamo_component_vllm_dp_requests_{running,waiting}{dp_rank=...}` gauges. See
`post-9915/instrumented/INSTRUMENTATION.md`.

## Gotchas (learned the hard way — don't relitigate these)

- **`stream-interval: 50` is load-bearing.** Omitting it from the vLLM backend
  config invalidates any PR-9915 conclusion. This caused one full wasted rerun.
- **`load-aware: true` is rejected** by the current frontend CLI. Use
  `router-mode: least-loaded` instead.
- **No `token-dp-balance` router mode exists.** Available modes: `round-robin`,
  `random`, `power-of-two`, `kv`, `direct`, `least-loaded`,
  `device-aware-weighted`. For token-aware balancing use the KV-router knobs
  (`--router-track-prefill-tokens`, etc.), not a new mode.
- **Backend trace files carry no `dp_rank`** (it logs as `None`). Use per-process
  file identity (`dynamo_request_trace_vllm_<pid>.jsonl`) as the DP-rank proxy.
- **Dynamo Prometheus scrapes don't expose per-DP running/waiting gauges**
  except on the instrumentation branch above. Direct vLLM exposes them natively.
- **Final per-rank counts being balanced proves nothing.** The failure mode is
  temporal: short windows where ranks hit `Waiting: 0` with `Running < 864`.
- **Discovery policy:** never `find` across `/home` or NFS fanout roots for
  models. Use the known shared root
  `/lustre/share/coreai_dlfw_dev/models/...` or `stat`/`ls` an explicit path.
- **Bumping `max-cudagraph-capture-size` to a large value OOMs at startup.** The
  isl=4096 confirm (job 2188156) set all three limits to 8192; vLLM then captured
  cudagraphs at sizes all the way to 8192 (4.95 GiB private pool) on top of the
  8192-ctx KV at `gpu-memory-utilization: 0.90`, and every DP worker died with
  `CUDA out of memory` in `capture_model()`. Prefill is token-budget bound
  (running ≈ 2) so big cudagraphs are useless — use `enforce-eager: true` (or a
  small `max-cudagraph-capture-size`) and/or drop util to ~0.80 when raising
  `max-model-len`.
- **nsys `delay_secs` is from worker launch and must beat teardown.** The
  de-saturated drain run (job 2188160) finished its traffic ~6 min after a ~460 s
  worker→traffic gap; `delay_secs=800` fired ~20 s before teardown and wrote
  nothing. Calibrate per run: worker-launch + traffic-gap + ~120 s. Also: the
  container's bind-mounted Nsight tree has only the **target** subtree, so
  time-based captures emit raw `.qdstrm` but never finalize to `.nsys-rep` (the
  host `QdstrmImporter` is absent) — finalize offline with a full Nsight install.

## Conventions

- Reports are Markdown with explicit dates, a results table, ASCII bar charts,
  and a `## Artifacts` section listing absolute Lyris output paths. Match this
  style when adding a report.
- Recipe filenames encode the axes:
  `qwen3-235b-a22b-vllm-agg-<cluster>-<gpu>-dp4-ep-<router-variant>[-suffix].yaml`.
- Keep GB200 (Lyris, `gpus_per_node: 4`) and GB300 recipes as parallel sets.
- Python here is stdlib-only and matches the parent repo's style (3.10+ syntax,
  type hints, 120-col). Verify scripts with `python -m py_compile`.
