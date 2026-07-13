#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render locked DeepSeek-V4-Pro legacy/sidecar campaign recipes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DYNAMO_COMMIT = "beb91b0de5392af2bd36560b312c153e7dbed061"
SGLANG_COMMIT = "e2728ac504c00e37a284c7248693857b894e40e7"
INFERENCEX_COMMIT = "4dd213e53b2bb1dbaabe5a2634889185092a09d3"
CANDIDATE_IMAGE = "nvcr.io/nvidian/dynamo-dev/sglang-runtime:connorc-beb91b0-e2728ac-dsv4-gb300-ab-arm64"
MODEL_REPO = "deepseek-ai/DeepSeek-V4-Pro"
MODEL_REVISION = "b5968e9190ef611bbf34a7229255be88a0e937c1"

POINTS = (
    ("c00001", "disagg-gb300-1p1d-tp4-tp4-2-c1.yaml", 1),
    ("c01024", "disagg-gb300-1p1d-dep4-dep16-5-c1024.yaml", 1024),
    ("c12000", "disagg-gb300-15p1d-dep4-dep12-18-c12000.yaml", 12000),
    ("c08192", "disagg-gb300-14p1d-dep4-dep16-18-c8192.yaml", 8192),
    ("c03000", "disagg-gb300-12p1d-dep4-dep24-18-c3000.yaml", 3000),
    ("c02500", "disagg-gb300-10p1d-dep4-dep32-18-c2500.yaml", 2500),
    ("c02048", "disagg-gb300-8p1d-dep4-dep40-18-c2048.yaml", 2048),
)

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_recipe(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"recipe is not a mapping: {path}")
    return value


def base_variant(
    canonical: dict[str, Any],
    *,
    point_id: str,
    backend: str,
    model_path: str,
    image: str,
) -> dict[str, Any]:
    recipe = copy.deepcopy(canonical)
    recipe["name"] = f"dsv4-gb300-ab-{point_id}-{backend}"
    recipe["model"]["path"] = model_path
    recipe["model"]["container"] = image
    recipe["dynamo"] = {"hash": DYNAMO_COMMIT, "install": False}
    recipe["setup_script"] = "dsv4-gpu-telemetry.sh"
    if backend == "sidecar":
        recipe["backend"].update(copy.deepcopy(SIDECAR_FIELDS))
    elif backend != "legacy":
        raise ValueError(f"unknown backend: {backend}")
    return recipe


def normalized_for_ab(recipe: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(recipe)
    normalized.pop("name", None)
    backend = normalized["backend"]
    for key in SIDECAR_FIELDS:
        backend.pop(key, None)
    return normalized


def dump_recipe(path: Path, recipe: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(recipe, sort_keys=False, width=120))


def render(model_path: str, image: str) -> None:
    campaign_dir = Path(__file__).resolve().parent
    repo_root = campaign_dir.parents[1]
    canonical_dir = campaign_dir / "canonical"
    output_root = repo_root / "recipes" / "dsv4-pro-gb300-grpc-ab"
    gates_root = campaign_dir / "gates"
    manifest_points: list[dict[str, Any]] = []

    rendered: dict[str, dict[str, dict[str, Any]]] = {}
    for point_id, filename, concurrency in POINTS:
        canonical_path = canonical_dir / filename
        canonical = load_recipe(canonical_path)
        variants: dict[str, dict[str, Any]] = {}
        for backend in ("legacy", "sidecar"):
            recipe = base_variant(
                canonical,
                point_id=point_id,
                backend=backend,
                model_path=model_path,
                image=image,
            )
            dump_recipe(output_root / backend / f"{point_id}.yaml", recipe)
            variants[backend] = recipe

        if normalized_for_ab(variants["legacy"]) != normalized_for_ab(variants["sidecar"]):
            raise RuntimeError(f"unexpected A/B drift at {point_id}")

        rendered[point_id] = variants
        resources = canonical["resources"]
        manifest_points.append(
            {
                "id": point_id,
                "concurrency": concurrency,
                "canonical_recipe": filename,
                "canonical_sha256": sha256(canonical_path),
                "prefill_nodes": resources["prefill_nodes"],
                "decode_nodes": resources["decode_nodes"],
                "total_gpus": (
                    resources["prefill_nodes"] * resources["gpus_per_node"]
                    + resources["decode_nodes"] * resources["gpus_per_node"]
                ),
            }
        )

    # Matching 1P/1D deterministic output checks.
    for backend, recipe in rendered["c00001"].items():
        smoke = copy.deepcopy(recipe)
        smoke["name"] = f"dsv4-gb300-ab-smoke-{backend}"
        smoke["benchmark"] = {
            "type": "deterministic-smoke",
            "isl": 8192,
            "osl": 1024,
            "custom_tokenizer": ("sa_bench_tokenizers.sglang_deepseek_v4.SGLangDeepseekV4Tokenizer"),
            "use_chat_template": False,
        }
        smoke["setup_script"] = "dsv4-smoke-setup.sh"
        dump_recipe(gates_root / "smoke" / f"{backend}.yaml", smoke)

    # Multi-node gate: one warmup and one measured wave at concurrency 1024.
    for backend, recipe in rendered["c01024"].items():
        gate = copy.deepcopy(recipe)
        gate["name"] = f"dsv4-gb300-ab-correctness-c01024-{backend}"
        gate["benchmark"]["num_prompts_mult"] = 1
        gate["benchmark"]["num_warmup_mult"] = 1
        dump_recipe(gates_root / "correctness-c01024" / f"{backend}.yaml", gate)

    # Full-rack EP40 gate: exactly 4096 measured requests at unlimited rate.
    for backend, recipe in rendered["c02048"].items():
        gate = copy.deepcopy(recipe)
        gate["name"] = f"dsv4-gb300-ab-stress-c02048-{backend}"
        gate["benchmark"]["num_prompts_mult"] = 2
        gate["benchmark"]["num_warmup_mult"] = 1
        dump_recipe(gates_root / "stress-c02048" / f"{backend}.yaml", gate)

    manifest = {
        "schema_version": 1,
        "model": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_path": model_path,
        "image": image,
        "source_commits": {
            "dynamo": DYNAMO_COMMIT,
            "sglang": SGLANG_COMMIT,
            "inferencex": INFERENCEX_COMMIT,
        },
        "workload": {
            "isl": 8192,
            "osl": 1024,
            "request_rate": "inf",
            "num_prompts_multiplier": 10,
            "num_warmup_multiplier": 2,
            "random_range_ratio": 0.8,
            "speculative_decoding": False,
            "transfer_backend": "mooncake",
        },
        "sidecar": SIDECAR_FIELDS,
        "points": manifest_points,
        "crossover_order": [
            {"pair": 1, "order": ["legacy", "sidecar"]},
            {"pair": 2, "order": ["sidecar", "legacy"]},
        ],
    }
    (campaign_dir / "campaign-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    run_plan = []
    sequence = 0
    for point_id, _, _ in POINTS:
        for pair, order in (
            (1, ("legacy", "sidecar")),
            (2, ("sidecar", "legacy")),
        ):
            for order_index, backend in enumerate(order, start=1):
                sequence += 1
                run_plan.append(
                    {
                        "sequence": sequence,
                        "point": point_id,
                        "pair": pair,
                        "order_index": order_index,
                        "backend": backend,
                        "recipe": str(output_root.relative_to(repo_root) / backend / f"{point_id}.yaml"),
                        "artifact_dir": f"raw/{point_id}/pair-{pair}/{order_index}-{backend}",
                        "retry_artifact_dir": f"raw/{point_id}/pair-{pair}-retry-1/{order_index}-{backend}",
                    }
                )
    (campaign_dir / "run-plan.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fresh_processes_per_run": True,
                "same_placement_preferred_within_pair": True,
                "pair_retry_limit": 1,
                "runs": run_plan,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="deepseek-v4-pro",
        help="Ptyche-visible pinned model directory",
    )
    parser.add_argument("--image", default=CANDIDATE_IMAGE)
    args = parser.parse_args()
    render(args.model_path, args.image)


if __name__ == "__main__":
    main()
