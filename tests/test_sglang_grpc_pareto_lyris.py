# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "campaigns/sglang-sidecar-gb300-fp4-8k1k/render-lyris-recipes.py"
CAMPAIGN_DIR = SCRIPT.parent


def load_renderer():
    spec = importlib.util.spec_from_file_location("render_lyris_recipes", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def without_lyris_fields(data):
    normalized = copy.deepcopy(data)
    normalized.pop("name")
    normalized["model"].pop("path")
    normalized["model"].pop("container")
    normalized["frontend"].pop("nginx_container")
    return normalized


def test_renderer_changes_only_cluster_specific_fields():
    renderer = load_renderer()
    for source in renderer.SOURCES:
        original = yaml.safe_load(source.read_text())
        rendered = renderer.render_recipe(
            source,
            model_path="/lustre/model",
            candidate_image="/lustre/candidate.sqsh",
            nginx_container="nginx:1.27.4",
        )
        assert rendered["name"] == f"{original['name']}-lyris"
        assert rendered["model"]["path"] == "/lustre/model"
        assert rendered["model"]["container"] == "/lustre/candidate.sqsh"
        assert rendered["frontend"]["nginx_container"] == "nginx:1.27.4"
        assert without_lyris_fields(rendered) == without_lyris_fields(original)


def test_cli_writes_all_recipes_and_manifest(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model-path",
            "/lustre/model",
            "--candidate-image",
            "/lustre/candidate.sqsh",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
    )
    manifest = json.loads((tmp_path / "render-manifest.json").read_text())
    assert manifest["cluster"] == "lyris"
    assert manifest["partition"] == "gb300"
    assert len(manifest["files"]) == 5
    assert {entry["rendered"] for entry in manifest["files"]} == {
        "smoke.yaml",
        "correctness-gate.yaml",
        "low_latency.yaml",
        "mid_curve.yaml",
        "max_tpt.yaml",
    }


def test_candidate_image_uses_validated_runtime_dependency_lock():
    dockerfile = (CAMPAIGN_DIR / "Dockerfile").read_text()
    build_script = (CAMPAIGN_DIR / "build-image.sh").read_text()
    verification = (CAMPAIGN_DIR / "verify-image.sh").read_text()

    assert "lmsysorg/sglang:v0.5.14-cu130-runtime" in dockerfile
    assert "lmsysorg/sglang:v0.5.14-cu130-runtime" in build_script
    assert "pip install --no-cache-dir --no-deps --force-reinstall" in dockerfile
    for dependency in (
        "apache-tvm-ffi==0.1.11",
        "mistral-common==1.11.5",
        "sgl-deep-gemm==0.1.4",
        "tilelang==0.1.11",
        "transformers==5.12.1",
    ):
        assert dependency in dockerfile
        package, version = dependency.split("==", maxsplit=1)
        assert f'"{package}": "{version}"' in verification
