# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locks for the DeepSeek-V4-Pro GB300 legacy/native-gRPC A/B campaign."""

from __future__ import annotations

import copy
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.benchmarks import get_runner
from srtctl.core.schema import SrtConfig

CAMPAIGN_DIR = Path("campaigns/dsv4-pro-gb300-grpc-ab")
CANONICAL_DIR = CAMPAIGN_DIR / "canonical"
RECIPE_DIR = Path("recipes/dsv4-pro-gb300-grpc-ab")
CANDIDATE_IMAGE = "nvcr.io/nvidian/dynamo-dev/sglang-runtime:connorc-beb91b0-e2728ac-dsv4-gb300-ab-arm64"
DYNAMO_COMMIT = "beb91b0de5392af2bd36560b312c153e7dbed061"
MODEL_REVISION = "b5968e9190ef611bbf34a7229255be88a0e937c1"
CUSTOM_TOKENIZER = "sa_bench_tokenizers.sglang_deepseek_v4.SGLangDeepseekV4Tokenizer"
SIDECAR_FIELDS = {
    "native_grpc_sidecar": True,
    "native_grpc_port": 50051,
    "sidecar_binary": "dynamo-sglang-sidecar",
    "sidecar_args": [
        "--sglang-connections",
        "8",
        "--health-deadline-secs",
        "1200",
    ],
}
PUBLIC_SHA256 = {
    "disagg-gb300-1p1d-tp4-tp4-2-c1.yaml": "7d31075153ca6d050e609a650435566e5a69c8f4759f7a40407443a5ded1b6d8",
    "disagg-gb300-1p1d-dep4-dep16-5-c1024.yaml": "00b75580ce853be5b6385347ccc8c72f1de2faa7bcfd65047c6f5a112d69c6ee",
    "disagg-gb300-15p1d-dep4-dep12-18-c12000.yaml": "ca4953b6f1939d09feb35da5b6ae8f1d3fc2002011f3494715b16f45af3189c2",
    "disagg-gb300-14p1d-dep4-dep16-18-c8192.yaml": "5daedbad1437f30d025087216e2088a53c2689a555818f180e6849a07b7f0a56",
    "disagg-gb300-12p1d-dep4-dep24-18-c3000.yaml": "ad23d994f2e1da0c84daee5c309cd0c681ced45b019a27ac6561e2d228532905",
    "disagg-gb300-10p1d-dep4-dep32-18-c2500.yaml": "ce140942cc974a724779fdaa75e9724838c6dce4e0a6a58d7ba188da0e457150",
    "disagg-gb300-8p1d-dep4-dep40-18-c2048.yaml": "5847b3783639055b5dc8e3b9dcee88668b980cd41ad25a12d78a866ed3fe765e",
}
MATRIX = {
    "c00001": (1, 1, 1, 2, 8, 4, 4),
    "c01024": (1024, 1, 4, 5, 20, 4, 16),
    "c12000": (12000, 15, 3, 18, 72, 4, 12),
    "c08192": (8192, 14, 4, 18, 72, 4, 16),
    "c03000": (3000, 12, 6, 18, 72, 4, 24),
    "c02500": (2500, 10, 8, 18, 72, 4, 32),
    "c02048": (2048, 8, 10, 18, 72, 4, 40),
}
PUBLIC_RESULT_IDS = {
    1: "420241",
    1024: "420239",
    12000: "420237",
    8192: "420240",
    3000: "420238",
    2500: "420235",
    2048: "420236",
}


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    assert isinstance(value, dict)
    return value


def _normalize(recipe: dict) -> dict:
    value = copy.deepcopy(recipe)
    value.pop("name")
    for key in SIDECAR_FIELDS:
        value["backend"].pop(key, None)
    return value


@pytest.mark.parametrize("filename,expected", PUBLIC_SHA256.items())
def test_public_recipe_snapshot_is_exact(filename: str, expected: str) -> None:
    assert hashlib.sha256((CANONICAL_DIR / filename).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("point_id", MATRIX)
def test_variants_only_differ_in_locked_architecture_fields(point_id: str) -> None:
    legacy = _load(RECIPE_DIR / "legacy" / f"{point_id}.yaml")
    sidecar = _load(RECIPE_DIR / "sidecar" / f"{point_id}.yaml")

    assert _normalize(legacy) == _normalize(sidecar)
    assert set(SIDECAR_FIELDS).isdisjoint(legacy["backend"])
    assert {key: sidecar["backend"][key] for key in SIDECAR_FIELDS} == SIDECAR_FIELDS
    for recipe in (legacy, sidecar):
        assert recipe["model"]["container"] == CANDIDATE_IMAGE
        assert recipe["dynamo"] == {"hash": DYNAMO_COMMIT, "install": False}
        assert recipe["srun_options"] == {"mem": "0"}
        assert recipe["setup_script"] == "dsv4-gpu-telemetry.sh"


@pytest.mark.parametrize("point_id,expected", MATRIX.items())
def test_public_matrix_and_dsv4_workload_are_locked(point_id: str, expected: tuple[int, ...]) -> None:
    concurrency, prefill_nodes, decode_nodes, total_nodes, total_gpus, prefill_size, decode_size = expected
    recipe = _load(RECIPE_DIR / "legacy" / f"{point_id}.yaml")
    resources = recipe["resources"]
    benchmark = recipe["benchmark"]
    server = recipe["backend"]["sglang_config"]

    assert resources["prefill_nodes"] == prefill_nodes
    assert resources["decode_nodes"] == decode_nodes
    assert prefill_nodes + decode_nodes == total_nodes
    assert total_nodes * resources["gpus_per_node"] == total_gpus
    assert resources["gpus_per_prefill"] == prefill_size
    assert resources["gpus_per_decode"] == decode_size
    assert benchmark == {
        "type": "sa-bench",
        "isl": 8192,
        "osl": 1024,
        "concurrencies": str(concurrency),
        "req_rate": "inf",
        "use_chat_template": False,
        "custom_tokenizer": CUSTOM_TOKENIZER,
    }
    for mode in ("prefill", "decode"):
        environment = recipe["backend"][f"{mode}_environment"]
        assert environment["SGLANG_DSV4_REASONING_EFFORT"] == "max"
        assert environment["SGLANG_DEFAULT_THINKING"] == "1"
        if point_id == "c00001":
            assert "SGLANG_OPT_USE_ONLINE_COMPRESS" not in environment
        else:
            assert environment["SGLANG_OPT_USE_ONLINE_COMPRESS"] == "1"
        assert environment["MC_FORCE_MNNVL"] == "1"
        assert server[mode]["disaggregation-transfer-backend"] == "mooncake"
        assert "speculative-algorithm" not in server[mode]


@pytest.mark.parametrize("point_id", MATRIX)
def test_all_fourteen_recipes_load(point_id: str) -> None:
    legacy = SrtConfig.from_yaml(RECIPE_DIR / "legacy" / f"{point_id}.yaml")
    sidecar = SrtConfig.from_yaml(RECIPE_DIR / "sidecar" / f"{point_id}.yaml")

    assert legacy.backend.native_grpc_sidecar is False
    assert sidecar.backend.native_grpc_sidecar is True
    assert sidecar.backend.native_grpc_port == 50051


@pytest.mark.parametrize("point_id", MATRIX)
def test_sidecar_leaders_and_followers_render_correctly(point_id: str) -> None:
    config = SrtConfig.from_yaml(RECIPE_DIR / "sidecar" / f"{point_id}.yaml")
    resources = config.resources
    nodes = [f"node{index}" for index in range(resources.total_nodes)]
    endpoints = config.backend.allocate_endpoints(
        num_prefill=resources.num_prefill,
        num_decode=resources.num_decode,
        num_agg=resources.num_agg,
        gpus_per_prefill=resources.gpus_per_prefill,
        gpus_per_decode=resources.gpus_per_decode,
        gpus_per_agg=resources.gpus_per_agg,
        gpus_per_node=resources.gpus_per_node,
        available_nodes=nodes,
    )
    processes = config.backend.endpoints_to_processes(endpoints)
    runtime = MagicMock()
    runtime.model_path = Path("/models/DeepSeek-V4-Pro")

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        for process in processes:
            peers = [
                candidate
                for candidate in processes
                if candidate.endpoint_mode == process.endpoint_mode
                and candidate.endpoint_index == process.endpoint_index
            ]
            command = config.backend.build_worker_command(process, peers, runtime)
            rendered = command[2] if command[:2] == ["bash", "-lc"] else " ".join(command)
            if process.is_leader:
                assert command[:2] == ["bash", "-lc"]
                assert "--grpc-port 50051" in rendered
                assert "dynamo-sglang-sidecar" in rendered
                assert "wait -n" in rendered
                if process.endpoint_mode == "prefill":
                    assert "--bootstrap-host 10.0.0.1" in rendered
            else:
                assert command[:3] == ["python3", "-m", "sglang.launch_server"]
                assert "--grpc-port" not in rendered
                assert "dynamo-sglang-sidecar" not in rendered


def test_gate_shapes_and_request_counts() -> None:
    for backend in ("legacy", "sidecar"):
        smoke_path = CAMPAIGN_DIR / "gates" / "smoke" / f"{backend}.yaml"
        smoke = _load(smoke_path)
        assert smoke["benchmark"] == {
            "type": "deterministic-smoke",
            "isl": 8192,
            "osl": 1024,
            "custom_tokenizer": CUSTOM_TOKENIZER,
            "use_chat_template": False,
        }
        assert smoke["setup_script"] == "dsv4-smoke-setup.sh"
        assert get_runner("deterministic-smoke").validate_config(SrtConfig.from_yaml(smoke_path)) == []

        correctness = _load(CAMPAIGN_DIR / "gates" / "correctness-c01024" / f"{backend}.yaml")
        assert correctness["benchmark"]["num_prompts_mult"] == 1
        assert correctness["benchmark"]["num_warmup_mult"] == 1

        stress = _load(CAMPAIGN_DIR / "gates" / "stress-c02048" / f"{backend}.yaml")
        assert int(stress["benchmark"]["concurrencies"]) * stress["benchmark"]["num_prompts_mult"] == 4096
        assert stress["benchmark"]["req_rate"] == "inf"


def test_run_plan_has_two_crossover_pairs_per_point() -> None:
    plan = json.loads((CAMPAIGN_DIR / "run-plan.json").read_text())

    assert len(plan["runs"]) == 28
    for point_id in MATRIX:
        point_runs = [run for run in plan["runs"] if run["point"] == point_id]
        assert [(run["pair"], run["backend"]) for run in point_runs] == [
            (1, "legacy"),
            (1, "sidecar"),
            (2, "sidecar"),
            (2, "legacy"),
        ]
        assert all("-retry-1/" in run["retry_artifact_dir"] for run in point_runs)
    assert plan["fresh_processes_per_run"] is True
    assert plan["pair_retry_limit"] == 1


def test_public_curve_snapshot_matches_locked_matrix() -> None:
    snapshot = json.loads((CAMPAIGN_DIR / "public-curve-2026-07-13.json").read_text())

    assert snapshot["source"]["inferencex_commit"] == "4dd213e53b2bb1dbaabe5a2634889185092a09d3"
    assert snapshot["filters"] == {
        "hardware": "gb300",
        "framework": "dynamo-sglang",
        "precision": "fp4",
        "disagg": True,
        "spec_method": "none",
        "isl": 8192,
        "osl": 1024,
    }
    assert {int(point["concurrency"]): point["id"] for point in snapshot["points"]} == PUBLIC_RESULT_IDS
    assert {int(point["concurrency"]): int(point["topology"]["total_gpus"]) for point in snapshot["points"]} == {
        values[0]: values[4] for values in MATRIX.values()
    }


def test_campaign_manifest_pins_model_revision() -> None:
    manifest = json.loads((CAMPAIGN_DIR / "campaign-manifest.json").read_text())

    assert manifest["cluster"] == {
        "name": "lyris",
        "partition": "gb300,gb300-backfill",
        "system": "GB300 NVL72",
        "minimum_gpu_memory_mib": 260000,
    }
    assert manifest["model"] == "deepseek-ai/DeepSeek-V4-Pro"
    assert manifest["model_revision"] == MODEL_REVISION


def test_lyris_config_selects_real_gb300_and_content_identical_model() -> None:
    config = _load(CAMPAIGN_DIR / "lyris-srtslurm.yaml")

    assert config["default_account"] == "coreai_comparch_inferencex"
    assert config["default_partition"] == "gb300,gb300-backfill"
    assert config["gpus_per_node"] == 4
    assert config["model_paths"]["deepseek-v4-pro"] == ("/lustre/fsw/coreai_comparch_inferencex/models/dsv4-pro")


def test_pinned_model_blob_manifest_covers_all_runtime_files() -> None:
    manifest = json.loads((Path("configs") / "dsv4-model-runtime-blobs-b5968e9.json").read_text())
    files = manifest["files"]

    assert manifest["revision"] == MODEL_REVISION
    assert len(files) == 69
    assert len([name for name in files if name.startswith("model-")]) == 64
    assert all(len(blob_id) in {40, 64} for blob_id in files.values())


def test_model_snapshot_verifier_accepts_matching_hf_metadata(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    metadata_dir = model_dir / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    shards = [f"model-{index:05d}-of-00064.safetensors" for index in range(1, 65)]
    files = {
        **{name: f"{index:064x}" for index, name in enumerate(shards, start=1)},
        "model.safetensors.index.json": "a" * 64,
        "config.json": "b" * 40,
        "generation_config.json": "c" * 40,
        "tokenizer.json": "d" * 40,
        "tokenizer_config.json": "e" * 40,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "repo": "deepseek-ai/DeepSeek-V4-Pro",
                "revision": MODEL_REVISION,
                "files": files,
            }
        )
    )
    index = {"weight_map": {f"weight-{i}": name for i, name in enumerate(shards)}}
    for name, blob_id in files.items():
        data = json.dumps(index) if name == "model.safetensors.index.json" else "data"
        (model_dir / name).write_text(data)
        (metadata_dir / f"{name}.metadata").write_text(f"old-cache-revision\n{blob_id}\n0.0\n")

    verify = runpy.run_path(Path("configs") / "verify-dsv4-model.py")["verify"]
    result = verify(model_dir, manifest_path)

    assert result["revision"] == MODEL_REVISION
    assert result["runtime_files"] == 69
    assert result["indexed_shards"] == 64


def test_deterministic_smoke_command_captures_output_artifact() -> None:
    config = SrtConfig.from_yaml(CAMPAIGN_DIR / "gates" / "smoke" / "legacy.yaml")
    runtime = MagicMock(frontend_port=8000)

    command = get_runner("deterministic-smoke").build_command(config, runtime)

    assert command[:2] == ["python3", "/srtctl-benchmarks/deterministic-smoke/run.py"]
    assert command[command.index("--isl") + 1] == "8192"
    assert command[command.index("--osl") + 1] == "1024"
    assert command[command.index("--tokenizer") + 1] == CUSTOM_TOKENIZER
    assert command[command.index("--tokenizer-root") + 1] == "/srtctl-benchmarks/sa-bench"
    assert command[command.index("--output") + 1] == "/logs/deterministic-smoke/deterministic-output.json"


def test_deterministic_smoke_falls_back_to_pinned_tokenizer_for_incomplete_logprobs() -> None:
    helper = runpy.run_path(Path("src/srtctl/benchmarks/scripts/deterministic-smoke/run.py"))["comparison_token_ids"]
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [11, 12, 13]
    result = {"choices": [{"logprobs": {"tokens": ["token_id:11"]}}]}

    token_ids, source, returned = helper(result, tokenizer, "output", 3)

    assert token_ids == [11, 12, 13]
    assert source == "retokenized_output_text"
    assert returned == [11]


def test_deterministic_smoke_prefers_complete_response_token_ids() -> None:
    helper = runpy.run_path(Path("src/srtctl/benchmarks/scripts/deterministic-smoke/run.py"))["comparison_token_ids"]
    tokenizer = MagicMock()
    result = {"choices": [{"logprobs": {"tokens": ["token_id:11", "token_id:12"]}}]}

    token_ids, source, returned = helper(result, tokenizer, "output", 2)

    assert token_ids == [11, 12]
    assert source == "response_logprobs"
    assert returned == [11, 12]
    tokenizer.encode.assert_not_called()


def test_smoke_comparison_requires_identical_token_ids(tmp_path: Path) -> None:
    baseline = {
        "isl": 8192,
        "osl": 1024,
        "prompt_token_count": 8192,
        "prompt_token_sha256": "prompt",
        "completion_tokens_reported": 1024,
        "output_token_ids": [1, 2, 3],
    }
    legacy = tmp_path / "legacy.json"
    sidecar = tmp_path / "sidecar.json"
    legacy.write_text(json.dumps(baseline))
    sidecar.write_text(json.dumps(baseline))
    script = CAMPAIGN_DIR / "compare-smoke.py"

    matched = subprocess.run([sys.executable, script, legacy, sidecar], check=True, capture_output=True, text=True)
    assert matched.stdout.strip() == "matched 3 output token IDs"

    sidecar.write_text(json.dumps({**baseline, "output_token_ids": [1, 9, 3]}))
    mismatched = subprocess.run([sys.executable, script, legacy, sidecar], capture_output=True, text=True)
    assert mismatched.returncode != 0
    assert "output token mismatch at index 1" in mismatched.stderr


def test_run_collection_derives_registration_transfer_and_fatal_evidence(tmp_path: Path) -> None:
    evaluate = runpy.run_path(CAMPAIGN_DIR / "collect-run.py")["evaluate"]
    run_dir = tmp_path / "run"
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    (logs / "sweep_1.log").write_text("Model is ready. Have 1 prefills and 1 decodes.\n")
    (logs / "node_prefill_w0.out").write_text("Topology discovery complete. Found 4 HCAs.\n")
    (logs / "node_decode_w0.out").write_text(
        "Topology discovery complete. Found 4 HCAs.\n"
        "WARN Ignore import error when loading an unrelated model.\n"
        "WARN dynamo_runtime::transports::etcd::lease grpc request error: connection refused.\n"
        "Decode batch, #running-req: 4, gen throughput (token/s): 100\n"
    )
    result = {
        "num_prompts": 2,
        "completed": 2,
        "errors": [None, None],
        "input_lens": [8192, 8192],
        "output_lens": [1024, 1024],
        "total_input_tokens": 16384,
        "total_output_tokens": 2048,
    }
    (run_dir / "results_concurrency_2.json").write_text(json.dumps(result))
    scheduler = {"root": {"state": "COMPLETED"}, "rows": []}

    collected = evaluate(run_dir, "sidecar", 2, scheduler)

    assert collected["valid"] is True
    assert collected["validation"]["observed_worker_registrations"] == 2
    assert collected["validation"]["mooncake_kv_transfer"] is True
    assert collected["validation"]["fatal_sidecar_errors"] == 0
    assert collected["validation"]["fatal_grpc_errors"] == 0

    (logs / "node_decode_w0.out").write_text(
        "Topology discovery complete. Found 4 HCAs.\n"
        "Decode batch, #running-req: 4\n"
        "RuntimeError: engine process failed\n"
    )
    failed = evaluate(run_dir, "legacy", 2, scheduler)
    assert failed["valid"] is False
    assert failed["validation"]["fatal_engine_errors"] == 1

    (logs / "node_decode_w0.out").write_text(
        "Topology discovery complete. Found 4 HCAs.\n"
        "Decode batch, #running-req: 4\n"
        "ERROR SGLang gRPC transport failed\n"
    )
    grpc_failed = evaluate(run_dir, "sidecar", 2, scheduler)
    assert grpc_failed["validation"]["fatal_grpc_errors"] == 1


def test_campaign_controller_preserves_crossover_pair_order_and_registration_count() -> None:
    controller = runpy.run_path(CAMPAIGN_DIR / "run-campaign.py")
    plan = json.loads((CAMPAIGN_DIR / "run-plan.json").read_text())

    grouped = controller["_group_plan"](plan)

    assert len(grouped) == 7
    assert [spec["backend"] for spec in grouped[0][1][0][1]] == ["legacy", "sidecar"]
    assert [spec["backend"] for spec in grouped[0][1][1][1]] == ["sidecar", "legacy"]
    assert controller["expected_registrations"](Path("recipes/dsv4-pro-gb300-grpc-ab/legacy/c12000.yaml")) == 16
    command = controller["build_apply_command"](Path("/runner/srtctl"), Path("/recipes/c00001.yaml"), ["a", "b"])
    assert command == [
        "/runner/srtctl",
        "apply",
        "-f",
        "/recipes/c00001.yaml",
        "-y",
        "--tags",
        "a,b",
    ]


def test_gate_plan_stops_on_failure_and_compares_smoke_tokens(tmp_path: Path) -> None:
    controller = runpy.run_path(CAMPAIGN_DIR / "run-campaign.py")
    plan = json.loads((CAMPAIGN_DIR / "gate-plan.json").read_text())
    assert plan["stop_campaign_on_point_failure"] is True
    assert [point for point, _pairs in controller["_group_plan"](plan)] == [
        "smoke",
        "correctness-c01024",
        "stress-c02048",
    ]

    legacy = tmp_path / "pair-1" / "1-legacy"
    sidecar = tmp_path / "pair-1" / "2-sidecar"
    legacy.mkdir(parents=True)
    sidecar.mkdir(parents=True)
    output = {
        "isl": 8192,
        "osl": 1024,
        "prompt_token_count": 8192,
        "prompt_token_sha256": "prompt",
        "completion_tokens_reported": 1024,
        "output_token_ids": [1, 2, 3],
    }
    (legacy / "deterministic-output.json").write_text(json.dumps(output))
    (sidecar / "deterministic-output.json").write_text(json.dumps(output))

    matched = controller["_compare_pair"]("smoke_tokens", [legacy, sidecar], CAMPAIGN_DIR)
    assert matched["valid"] is True
    assert (tmp_path / "pair-1" / "pair-comparison.json").is_file()

    (sidecar / "deterministic-output.json").write_text(json.dumps({**output, "output_token_ids": [1, 9, 3]}))
    mismatched = controller["_compare_pair"]("smoke_tokens", [legacy, sidecar], CAMPAIGN_DIR)
    assert mismatched["valid"] is False

    state = {"runs": {}, "points": {}}
    controller["_adopt_jobs"](state, plan, ["1:0:2368261"])
    assert state["runs"]["1:attempt-0"] == {
        "artifact_dir": "gates/smoke/pair-1/1-legacy",
        "job_id": "2368261",
        "status": "submitted",
        "adopted": True,
    }


def test_base_metadata_repair_is_idempotent_for_removed_requirement(tmp_path: Path) -> None:
    dist_info = tmp_path / "nixl-1.3.1.dist-info"
    dist_info.mkdir()
    metadata = dist_info / "METADATA"
    metadata.write_text("Name: nixl\nRequires-Dist: nixl-cu12==1.3.1\nRequires-Dist: nixl-cu13==1.3.1\n")
    (dist_info / "RECORD").write_text("nixl-1.3.1.dist-info/METADATA,,\n")
    distribution = MagicMock(_path=dist_info)
    distribution.locate_file.return_value = tmp_path
    repair = runpy.run_path(CAMPAIGN_DIR / "repair-base-python-metadata.py")["replace_once"]

    with patch("importlib.metadata.distribution", return_value=distribution):
        first = repair("nixl", "METADATA", "Requires-Dist: nixl-cu12==1.3.1\n", "")
        second = repair("nixl", "METADATA", "Requires-Dist: nixl-cu12==1.3.1\n", "")

    assert first["status"] == "applied"
    assert second["status"] == "already-applied"
    assert "nixl-cu12" not in metadata.read_text()
