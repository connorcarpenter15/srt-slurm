# GB300 DeepSeek-V4-Pro SGLang gRPC A/B campaign

This directory defines an internal, unofficial crossover A/B of Dynamo's integrated SGLang backend and the native-gRPC SGLang sidecar on the Ptyche GB300 NVL72 cluster. The public InferenceX curve is context only; the primary result is the same-image local sidecar-versus-legacy comparison.

## Locked inputs

- Dynamo: `beb91b0de5392af2bd36560b312c153e7dbed061`
- SGLang: `e2728ac504c00e37a284c7248693857b894e40e7`
- InferenceX recipe snapshot: `4dd213e53b2bb1dbaabe5a2634889185092a09d3`
- Base image: `lmsysorg/sglang:dev-cu13@sha256:4b140bc08eb4782b057109b084b6df94f74c3a66c6984ee383a1d6c3714994d5`
- Candidate image: `nvcr.io/nvidian/dynamo-dev/sglang-runtime:connorc-beb91b0-e2728ac-dsv4-gb300-ab-arm64`
- Model: `deepseek-ai/DeepSeek-V4-Pro` at revision `b5968e9190ef611bbf34a7229255be88a0e937c1`
- Workload: 8,192 input and 1,024 output tokens, Mooncake, no speculative decoding

The seven files in `canonical/` are byte-for-byte copies of the public DSV4 FP4 GB300 recipes. `render-recipes.py` derives both measured variants from each canonical file. Tests reject any A/B difference other than the name and the four native-sidecar fields.

## Render and validate locally

```bash
./campaigns/dsv4-pro-gb300-grpc-ab/render-recipes.py \
  --model-path /ptyche/path/to/pinned/DeepSeek-V4-Pro/snapshot

pytest -q \
  tests/test_dsv4_grpc_ab_campaign.py \
  tests/test_dsv4_grpc_ab_analysis.py \
  tests/test_sglang_native_grpc_sidecar.py
```

The default `deepseek-v4-pro` value is Ptyche's public-recipe model alias. Before running, verify that it resolves to the pinned revision above. If it does not, render every recipe with the immutable Ptyche snapshot path and commit the resulting manifest and recipe change before the first gate.

## Build the common ARM64 image

Run on a native ARM64 Docker builder with NGC authentication:

```bash
DYNAMO_REPO=/path/to/dynamo-dsv4-grpc-ab \
SGLANG_REPO=/path/to/sglang-dsv4-grpc-ab \
./campaigns/dsv4-pro-gb300-grpc-ab/build-image.sh
```

The build uses clean source archives at the locked commits. It compiles the sidecar, builds Dynamo and SGLang wheels, checks the SGLang server and sidecar proto descriptors byte-for-byte, installs the wheels without dependency resolution, runs `pip check`, and pushes the final ARM64 image. The build produces `artifacts/image-manifest.json`, the package lock, wheels, sidecar binary, source-proto hashes, and descriptor hash.

Validate the imported image on a GB300 node:

```bash
./campaigns/dsv4-pro-gb300-grpc-ab/verify-image.sh
```

Also load the DSV4 tokenizer/parser and one FP4 model shard during the preflight. `verify-image.sh` covers architecture, exact build metadata, the Dynamo imports, `sglang.srt.grpc._core`, `--grpc-port`, sidecar execution, and `pip check`.

## Gates

Dry-run and then execute each legacy/sidecar pair in this order:

1. `gates/smoke`: deterministic TP4 1P/1D, exact 8K prompt IDs and exact 1K returned output token IDs. Its setup runs the full image/model preflight once per participating node before starting telemetry or serving processes.
2. `gates/correctness-c01024`: the public five-node C1024 topology with one warmup and one measured wave.
3. `gates/stress-c02048`: the full-rack EP40 topology with exactly 4,096 measured requests at unlimited rate.

For the deterministic smoke, require identical token IDs with:

```bash
python3 campaigns/dsv4-pro-gb300-grpc-ab/compare-smoke.py \
  /path/to/legacy/deterministic-output.json \
  /path/to/sidecar/deterministic-output.json
```

For every gate require exact worker registrations, successful Mooncake initialization and KV handoff, zero request failures, complete tokens, and no fatal engine, Mooncake, NCCL, gRPC, or sidecar error.

## Measured campaign

`run-plan.json` is the authoritative 28-run order. Each point uses fresh processes and the crossover sequence:

1. pair 1: legacy, sidecar;
2. pair 2: sidecar, legacy.

Preserve both legs of a pair. When either leg is invalid, retain both and rerun the complete identical pair once into the corresponding `retry_artifact_dir` in `run-plan.json`. Stop that point after the same-spec failure repeats. Do not tune a leg during the campaign. Keep a pair on the same NVL72 placement where scheduler control permits. The analyzer retains every attempt but selects the first complete, fully valid attempt for each pair; it never combines legs from different attempts.

Copy every job's complete result directory under the run plan's `artifact_dir`. Each directory must contain the benchmark result JSON, frontend/client/worker/scheduler logs, the resolved recipe, scheduler metadata, GPU telemetry, and `validation.json`:

```json
{
  "expected_worker_registrations": 9,
  "observed_worker_registrations": 9,
  "mooncake_kv_transfer": true,
  "fatal_engine_errors": 0,
  "fatal_mooncake_errors": 0,
  "fatal_nccl_errors": 0,
  "fatal_grpc_errors": 0,
  "fatal_sidecar_errors": 0
}
```

The expected count is the number of logical prefill plus decode endpoints, not the number of distributed follower processes. Derive the observed count and fatal-error counts from retained logs; do not mark them clean by assumption.

## Results

After all 28 runs are present:

```bash
python3 campaigns/dsv4-pro-gb300-grpc-ab/analyze-results.py \
  --artifacts-root /path/to/campaign-artifacts \
  --campaign-manifest campaigns/dsv4-pro-gb300-grpc-ab/campaign-manifest.json \
  --run-plan campaigns/dsv4-pro-gb300-grpc-ab/run-plan.json \
  --public-curve campaigns/dsv4-pro-gb300-grpc-ab/public-curve-2026-07-13.json \
  --output-dir /path/to/campaign-artifacts/analysis
```

The analyzer emits `comparison.json`, `runs.csv`, `pareto-overlay.svg`, and `report.md`. It applies the InferenceX formulas, computes both per-run and paired means, crossover order effects, sidecar-versus-legacy deltas, telemetry summaries, and the geometric-mean throughput ratio over fully valid points. Missing hard validation evidence excludes a run; missing or abnormal telemetry is retained as an explicit warning under the campaign's locked validity rule.

Publish the completed report as `sidecar-info/sglang/benchmarks/benchmark_inferencex_dsv4_pro_gb300_grpc_ab_8k1k_<date>.md`, with the raw artifacts and image manifest linked from it. Apply no formal pass/fail threshold; call out paired deviations above 5%, sidecar-only failures, systematic curve shifts, and order or hardware effects.
