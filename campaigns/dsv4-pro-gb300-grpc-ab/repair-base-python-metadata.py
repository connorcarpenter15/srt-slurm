#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repair known ARM64 metadata defects in the pinned SGLang base image."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def update_record(dist: importlib.metadata.Distribution, changed: Path) -> None:
    root = Path(dist.locate_file(""))
    record = Path(dist._path) / "RECORD"  # type: ignore[attr-defined]
    relative = changed.relative_to(root).as_posix()
    rows = list(csv.reader(record.read_text().splitlines()))
    digest = base64.urlsafe_b64encode(hashlib.sha256(changed.read_bytes()).digest()).decode().rstrip("=")
    replacements = 0
    for row in rows:
        if row[0] == relative:
            row[1] = f"sha256={digest}"
            row[2] = str(changed.stat().st_size)
            replacements += 1
    if replacements != 1:
        raise RuntimeError(f"expected one RECORD entry for {relative}, found {replacements}")
    with record.open("w", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def replace_once(distribution: str, filename: str, old: str, new: str) -> dict[str, str]:
    dist = importlib.metadata.distribution(distribution)
    path = Path(dist._path) / filename  # type: ignore[attr-defined]
    contents = path.read_text()
    old_count = contents.count(old)
    new_count = contents.count(new) if new else 0
    if old_count == 1 and new_count == 0:
        path.write_text(contents.replace(old, new))
        update_record(dist, path)
        status = "applied"
    elif old_count == 0 and (not new or new_count == 1):
        status = "already-applied"
    else:
        raise RuntimeError(f"expected exactly one {old!r} in {path}")
    return {
        "distribution": distribution,
        "file": filename,
        "old": old.rstrip("\n"),
        "new": new.rstrip("\n"),
        "status": status,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} OUTPUT_JSON")
    if platform.machine() not in {"aarch64", "arm64"}:
        raise SystemExit(f"metadata repairs are ARM64-only, got {platform.machine()}")

    repairs = [
        replace_once(
            "nixl",
            "METADATA",
            "Requires-Dist: nixl-cu12==1.3.1\n",
            "",
        ),
        replace_once(
            "nvidia-cusparselt-cu13",
            "WHEEL",
            "Tag: py3-none-manylinux2014_sbsa\n",
            "Tag: py3-none-manylinux2014_aarch64\n",
        ),
    ]
    try:
        importlib.metadata.distribution("moviepy")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("moviepy must be removed before applying base metadata repairs")

    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "base_image_defects": [
                    "nixl 1.3.1 incorrectly requires both CUDA 12 and CUDA 13 payloads",
                    "nvidia-cusparselt-cu13 uses the equivalent but pip-unsupported SBSA wheel tag",
                    "unused moviepy 2.2.1 conflicts with the base image's Pillow 12",
                ],
                "removed_distributions": ["moviepy==2.2.1"],
                "metadata_repairs": repairs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
