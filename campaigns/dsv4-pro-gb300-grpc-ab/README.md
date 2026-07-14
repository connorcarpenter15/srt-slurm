# GB300 DeepSeek-V4-Pro SGLang gRPC A/B campaign

This directory defines an internal, unofficial crossover A/B of Dynamo's integrated SGLang backend and the native-gRPC SGLang sidecar on the Lyris GB300 NVL72 cluster. The public InferenceX curve is context only; the primary result is the same-image local sidecar-versus-legacy comparison. Ptyche's `batch` partition is GB200 and exposes only about 184 GiB per GPU, so it cannot load the locked DSV4-Pro TP4 topology; the image preflight rejects GPUs below 260,000 MiB.

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
  --model-path /lustre/fsw/coreai_comparch_inferencex/models/dsv4-pro

pytest -q \
  tests/test_dsv4_grpc_ab_campaign.py \
  tests/test_dsv4_grpc_ab_analysis.py \
  tests/test_sglang_native_grpc_sidecar.py
```

The Lyris shared snapshot at `/lustre/fsw/coreai_comparch_inferencex/models/dsv4-pro` was originally cached from revision `89d501a`, but all 69 runtime files are blob-identical to the pinned `b5968e9` revision: 64 model shards, the safetensors index, config files, and tokenizer files. `configs/dsv4-model-runtime-blobs-b5968e9.json` records the exact pinned blob identities, and `configs/verify-dsv4-model.py` checks the mounted snapshot before setup and on every smoke node.

If the shared snapshot is unavailable, model staging remains reproducible and revision-locked:

```bash
MODEL_DIR=/lustre/fsw/coreai_comparch_inferencex/$USER/models/DeepSeek-V4-Pro-b5968e9 \
  ./campaigns/dsv4-pro-gb300-grpc-ab/stage-model.sh
```

The script downloads revision `b5968e9...`, verifies the safetensors index and all 64 non-empty shards, and writes `.campaign-model.json` only after the snapshot is complete. Render the recipes with that immutable directory rather than relying on the public runner's node-local alias.

After the model and candidate squash image are present, prepare the pinned Lyris runner environment from a compute-node job:

```bash
./campaigns/dsv4-pro-gb300-grpc-ab/prepare-lyris.sh
```

This installs the repository's pinned ARM64 NATS/etcd versions, writes the committed account/partition/model/image mapping, and performs a frozen runner dependency sync. It refuses to proceed unless the candidate image exists and the shared model matches every pinned runtime blob identity.

Then render all fourteen measured jobs through the actual Lyris configuration and retain the generated commands:

```bash
./campaigns/dsv4-pro-gb300-grpc-ab/dry-run-lyris.sh
```

The script checks every dry-run for its architecture name, immutable model path, and candidate squash image, and fails unless exactly fourteen artifacts are produced.

## Build the common ARM64 image

Run on a native ARM64 Docker builder with NGC authentication:

```bash
DYNAMO_REPO=/path/to/dynamo-dsv4-grpc-ab \
SGLANG_REPO=/path/to/sglang-dsv4-grpc-ab \
./campaigns/dsv4-pro-gb300-grpc-ab/build-image.sh
```

The build uses clean source archives at the locked commits. It compiles the sidecar, builds Dynamo and SGLang wheels, checks the SGLang server and sidecar proto descriptors byte-for-byte, installs the wheels without dependency resolution, runs `pip check`, and pushes the final ARM64 image. The pinned base has three unrelated ARM64 metadata defects: its NIXL metapackage requires both CUDA payloads, its cuSparseLt wheel uses pip's unrecognized SBSA alias, and unused MoviePy conflicts with Pillow 12. The build removes MoviePy and records deterministic NIXL/cuSparseLt metadata repairs in `base-python-metadata-repairs.json`; it does not fetch replacement runtime packages. The build produces `artifacts/image-manifest.json`, the package lock, wheels, sidecar binary, source-proto hashes, and descriptor hash.

If the native ARM64 Docker builder is unavailable, Lyris can produce the same pinned runtime as a local Enroot squash image:

```bash
sbatch --account=coreai_comparch_inferencex --partition=gb300-backfill \
  --nodes=1 --ntasks=1 --cpus-per-task=144 --mem=0 --time=04:00:00 \
  ./campaigns/dsv4-pro-gb300-grpc-ab/build-enroot-image.sh
```

The fallback imports the same base digest, checks out both exact commits, performs the same wheel, sidecar, descriptor, import, and `pip check` validation, and records the squash-image SHA-256 in its manifest. It is sufficient for the A/B because both legs consume that one immutable file. The NGC publication step must still be completed later from an authenticated ARM64 Docker builder.

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

The smoke retains the complete OpenAI response as `deterministic-output.response.json`. Dynamo's integrated SGLang backend at the pinned revision exposes only the first selected token through non-streaming completion logprobs when SGLang supplies per-chunk logprob metadata. When that occurs, the harness records `output_token_id_source: retokenized_output_text` and derives the comparison sequence with the pinned DSV4 tokenizer. It still requires exactly 1,024 comparison tokens, and the raw response makes this fallback auditable.

For every gate require exact worker registrations, successful Mooncake initialization and KV handoff, zero request failures, complete tokens, and no fatal engine, Mooncake, NCCL, gRPC, or sidecar error.

The resumable gate controller executes these pairs in order, compares the deterministic smoke outputs, retries a complete failed pair once, and stops before measurements if a gate still fails:

```bash
sbatch campaigns/dsv4-pro-gb300-grpc-ab/gate-controller.sbatch
```

On success it submits `campaign-controller.sbatch` automatically. Both controllers archive scheduler metadata and log-derived validation evidence rather than treating a successful Slurm exit as sufficient.

## Measured campaign

`run-plan.json` is the authoritative 28-run order. Each point uses fresh processes and the crossover sequence:

1. pair 1: legacy, sidecar;
2. pair 2: sidecar, legacy.

Preserve both legs of a pair. When either leg is invalid, retain both and rerun the complete identical pair once into the corresponding `retry_artifact_dir` in `run-plan.json`. Stop that point after the same-spec failure repeats. Do not tune a leg during the campaign. Keep a pair on the same NVL72 placement where scheduler control permits. The analyzer retains every attempt but selects the first complete, fully valid attempt for each pair; it never combines legs from different attempts.

To launch the measured controller directly after independently proving all gates:

```bash
sbatch campaigns/dsv4-pro-gb300-grpc-ab/campaign-controller.sbatch
```

The controller is stateful and submits only one measured leg at a time. It archives each job into the exact run-plan directory, derives worker-registration, Mooncake-transfer, fatal-error, scheduler, and benchmark evidence, and retries an entire pair once when either leg is invalid. Its lightweight GB200 allocation hands off to a successor before the eight-hour controller limit; an active GB300 benchmark continues across that handoff and is adopted by job ID from `controller-state.json`.

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
