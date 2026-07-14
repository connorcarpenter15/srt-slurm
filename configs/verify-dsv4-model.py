#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify a local DSV4 snapshot against pinned Hugging Face blob identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metadata_blob_id(path: Path) -> str:
    lines = path.read_text().splitlines()
    if len(lines) < 2 or not lines[1]:
        raise ValueError(f"invalid Hugging Face metadata: {path}")
    return lines[1]


def verify(model_dir: Path, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("model manifest has no files")

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shards = sorted(set(index.get("weight_map", {}).values()))
    if len(shards) != 64:
        raise ValueError(f"expected 64 indexed shards, found {len(shards)}")
    if set(shards) != {
        name for name in expected_files if name.startswith("model-")
    }:
        raise ValueError("indexed shards differ from the pinned runtime manifest")

    metadata_dir = model_dir / ".cache" / "huggingface" / "download"
    mismatches: list[str] = []
    missing: list[str] = []
    for name, expected_blob_id in sorted(expected_files.items()):
        data_path = model_dir / name
        metadata_path = metadata_dir / f"{name}.metadata"
        if not data_path.is_file() or data_path.stat().st_size == 0:
            missing.append(name)
            continue
        if not metadata_path.is_file():
            missing.append(f"{name}.metadata")
            continue
        observed_blob_id = _metadata_blob_id(metadata_path)
        if observed_blob_id != expected_blob_id:
            mismatches.append(
                f"{name}: {observed_blob_id} != {expected_blob_id}"
            )

    if missing or mismatches:
        raise ValueError(
            f"model snapshot verification failed: missing={missing}, mismatches={mismatches}"
        )

    return {
        "repo": manifest["repo"],
        "revision": manifest["revision"],
        "runtime_files": len(expected_files),
        "indexed_shards": len(shards),
        "model_dir": str(model_dir),
        "status": "content-identical",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = verify(args.model_dir, args.manifest)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
