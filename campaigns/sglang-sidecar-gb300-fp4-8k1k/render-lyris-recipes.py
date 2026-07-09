#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render Lyris-specific campaign copies without changing engine tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO_ROOT = CAMPAIGN_DIR.parents[1]
SOURCES = (
    CAMPAIGN_DIR / "smoke.yaml",
    CAMPAIGN_DIR / "correctness-gate.yaml",
    REPO_ROOT / "recipes/sglang-sidecar/gb300-fp4/8k1k/low_latency.yaml",
    REPO_ROOT / "recipes/sglang-sidecar/gb300-fp4/8k1k/mid_curve.yaml",
    REPO_ROOT / "recipes/sglang-sidecar/gb300-fp4/8k1k/max_tpt.yaml",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_recipe(
    source: Path,
    *,
    model_path: str,
    candidate_image: str,
    nginx_container: str,
) -> dict[str, Any]:
    data = yaml.safe_load(source.read_text())
    data["name"] = f"{data['name']}-lyris"
    data["model"]["path"] = model_path
    data["model"]["container"] = candidate_image
    data["frontend"]["nginx_container"] = nginx_container
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--candidate-image", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--nginx-container", default="nginx:1.27.4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered_files = []

    for source in SOURCES:
        output = args.output_dir / source.name
        data = render_recipe(
            source,
            model_path=args.model_path,
            candidate_image=args.candidate_image,
            nginx_container=args.nginx_container,
        )
        output.write_text(yaml.safe_dump(data, sort_keys=False))
        rendered_files.append(
            {
                "source": str(source.relative_to(REPO_ROOT)),
                "source_sha256": sha256(source),
                "rendered": output.name,
                "rendered_sha256": sha256(output),
            }
        )

    manifest = {
        "cluster": "lyris",
        "partition": "gb300",
        "model_path": args.model_path,
        "candidate_image": args.candidate_image,
        "nginx_container": args.nginx_container,
        "files": rendered_files,
    }
    (args.output_dir / "render-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
