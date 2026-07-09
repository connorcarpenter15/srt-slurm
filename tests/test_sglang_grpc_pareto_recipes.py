# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lock the GB300 native-gRPC campaign to the public InferenceX recipes."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.core.schema import SrtConfig

PUBLIC_DIR = Path("recipes/gb300-fp4/8k1k")
CAMPAIGN_DIR = Path("recipes/sglang-sidecar/gb300-fp4/8k1k")
CANDIDATE_IMAGE = "nvcr.io/nvidian/dynamo-dev/sglang-runtime:connorc-555695f436-cc7d6659fd-gb300-sidecar-arm64"
PUBLIC_SHA256 = {
    "low_latency.yaml": "e443d194e8a8062d06034856f029a7de6d0e3081cd512770179cae338150a1d0",
    "mid_curve.yaml": "7ebb31c4c078a4c716ce356da7f6e5e3754b3f781e5c46800117a1250764c594",
    "max_tpt.yaml": "d64bdd9d270393d5794c7b3224641df88f80b8a15987ae8753a7adc20ac7c255",
}
SIDECAR_FIELDS = {
    "native_grpc_sidecar",
    "native_grpc_port",
    "sidecar_binary",
    "sidecar_args",
}
COMPATIBILITY_ENV = "SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION"


def _load(path: Path) -> dict:
    with path.open() as stream:
        return yaml.safe_load(stream)


@pytest.mark.parametrize("filename,expected", PUBLIC_SHA256.items())
def test_public_recipe_snapshot_is_unchanged(filename: str, expected: str) -> None:
    content = (PUBLIC_DIR / filename).read_bytes()

    assert hashlib.sha256(content).hexdigest() == expected


@pytest.mark.parametrize("filename", PUBLIC_SHA256)
def test_campaign_only_changes_image_and_sidecar_fields(filename: str) -> None:
    public = _load(PUBLIC_DIR / filename)
    campaign = _load(CAMPAIGN_DIR / filename)

    assert campaign["name"] != public["name"]
    public.pop("name")
    campaign.pop("name")

    assert public["model"].pop("container") == "dynamo-sglang"
    assert campaign["model"].pop("container") == CANDIDATE_IMAGE

    sidecar = {key: campaign["backend"].pop(key) for key in SIDECAR_FIELDS}
    assert sidecar == {
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
    assert SIDECAR_FIELDS.isdisjoint(public["backend"])
    for environment in ("prefill_environment", "decode_environment"):
        assert campaign["backend"][environment].pop(COMPATIBILITY_ENV) == "0"
        assert COMPATIBILITY_ENV not in public["backend"][environment]
    assert campaign == public


@pytest.mark.parametrize("filename", PUBLIC_SHA256)
def test_campaign_recipe_loads_with_native_grpc_sidecar(filename: str) -> None:
    config = SrtConfig.from_yaml(CAMPAIGN_DIR / filename)

    assert config.model.container == CANDIDATE_IMAGE
    assert config.backend.native_grpc_sidecar is True
    assert config.backend.native_grpc_port == 50051
    assert config.backend.sidecar_args == [
        "--sglang-connections",
        "8",
        "--health-deadline-secs",
        "1200",
    ]


def test_campaign_matrix_has_eight_points_and_public_topologies() -> None:
    low = _load(CAMPAIGN_DIR / "low_latency.yaml")
    mid = _load(CAMPAIGN_DIR / "mid_curve.yaml")
    maximum = _load(CAMPAIGN_DIR / "max_tpt.yaml")

    assert low["resources"] == {
        "gpu_type": "gb300",
        "prefill_nodes": 1,
        "decode_nodes": 4,
        "prefill_workers": 1,
        "decode_workers": 4,
        "gpus_per_node": 4,
    }
    assert low["benchmark"]["concurrencies"] == "4x8x32x64"

    assert mid["resources"] == {
        "gpu_type": "gb300",
        "prefill_nodes": 6,
        "decode_nodes": 12,
        "prefill_workers": 6,
        "decode_workers": 1,
        "gpus_per_node": 4,
    }
    assert {key: mid["backend"]["sglang_config"]["decode"][key] for key in ("tp-size", "dp-size", "ep-size")} == {
        "tp-size": 48,
        "dp-size": 48,
        "ep-size": 48,
    }
    assert mid["benchmark"]["concurrencies"] == "512x2048x4096"

    assert maximum["resources"] == {
        "gpu_type": "gb300",
        "prefill_nodes": 10,
        "decode_nodes": 8,
        "prefill_workers": 10,
        "decode_workers": 1,
        "gpus_per_node": 4,
    }
    assert {key: maximum["backend"]["sglang_config"]["decode"][key] for key in ("tp-size", "dp-size", "ep-size")} == {
        "tp-size": 32,
        "dp-size": 32,
        "ep-size": 32,
    }
    assert maximum["benchmark"]["concurrencies"] == "2048"

    points = sum(len(recipe["benchmark"]["concurrencies"].split("x")) for recipe in (low, mid, maximum))
    assert points == 8


@pytest.mark.parametrize(
    "path",
    [
        *(CAMPAIGN_DIR / filename for filename in PUBLIC_SHA256),
        Path("campaigns/sglang-sidecar-gb300-fp4-8k1k/smoke.yaml"),
        Path("campaigns/sglang-sidecar-gb300-fp4-8k1k/correctness-gate.yaml"),
    ],
)
def test_pinned_sglang_fp4_swiglu_fusion_is_disabled(path: Path) -> None:
    backend = _load(path)["backend"]

    assert backend["prefill_environment"][COMPATIBILITY_ENV] == "0"
    assert backend["decode_environment"][COMPATIBILITY_ENV] == "0"


def test_smoke_recipe_is_one_prefill_one_decode_and_one_request() -> None:
    path = Path("campaigns/sglang-sidecar-gb300-fp4-8k1k/smoke.yaml")
    raw = _load(path)
    config = SrtConfig.from_yaml(path)

    assert config.resources.prefill_nodes == 1
    assert config.resources.decode_nodes == 1
    assert config.resources.gpus_per_prefill == 4
    assert config.resources.gpus_per_decode == 4
    assert {
        key: raw["benchmark"][key]
        for key in (
            "isl",
            "osl",
            "concurrencies",
            "num_prompts_mult",
            "num_warmup_mult",
        )
    } == {
        "isl": 8192,
        "osl": 1024,
        "concurrencies": "1",
        "num_prompts_mult": 1,
        "num_warmup_mult": 1,
    }


@pytest.mark.parametrize(
    "filename,expected_leaders,expected_followers,expected_gpus",
    [
        ("low_latency.yaml", 5, 0, 20),
        ("mid_curve.yaml", 7, 11, 72),
        ("max_tpt.yaml", 11, 7, 72),
    ],
)
def test_campaign_topologies_render_leader_and_follower_commands(
    filename: str,
    expected_leaders: int,
    expected_followers: int,
    expected_gpus: int,
) -> None:
    config = SrtConfig.from_yaml(CAMPAIGN_DIR / filename)
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
    runtime.model_path = Path("/models/DeepSeek-R1-0528-NVFP4-v2")
    leaders = followers = 0

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        for process in processes:
            endpoint_processes = [
                candidate
                for candidate in processes
                if candidate.endpoint_mode == process.endpoint_mode
                and candidate.endpoint_index == process.endpoint_index
            ]
            command = config.backend.build_worker_command(process, endpoint_processes, runtime)
            rendered = command[2] if command[:2] == ["bash", "-lc"] else " ".join(command)
            if process.is_leader:
                leaders += 1
                assert command[:2] == ["bash", "-lc"]
                assert "--grpc-port 50051" in rendered
                assert "dynamo-sglang-sidecar" in rendered
                if process.endpoint_mode == "prefill":
                    assert "--bootstrap-host 10.0.0.1" in rendered
            else:
                followers += 1
                assert command[:3] == ["python3", "-m", "sglang.launch_server"]
                assert "--grpc-port" not in rendered
                assert "dynamo-sglang-sidecar" not in rendered

    assert leaders == expected_leaders
    assert followers == expected_followers
    assert resources.prefill_gpus + resources.decode_gpus == expected_gpus
