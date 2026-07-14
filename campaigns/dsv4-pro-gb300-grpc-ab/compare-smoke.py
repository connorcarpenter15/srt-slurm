#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require deterministic legacy and sidecar smoke outputs to match exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy", type=Path)
    parser.add_argument("sidecar", type=Path)
    args = parser.parse_args()

    legacy = json.loads(args.legacy.read_text())
    sidecar = json.loads(args.sidecar.read_text())
    for field in ("isl", "osl", "prompt_token_count", "prompt_token_sha256", "completion_tokens_reported"):
        if legacy[field] != sidecar[field]:
            raise SystemExit(f"smoke field differs: {field}: {legacy[field]!r} != {sidecar[field]!r}")
    if legacy["output_token_ids"] != sidecar["output_token_ids"]:
        for index, (legacy_id, sidecar_id) in enumerate(
            zip(legacy["output_token_ids"], sidecar["output_token_ids"], strict=False)
        ):
            if legacy_id != sidecar_id:
                raise SystemExit(f"output token mismatch at index {index}: {legacy_id} != {sidecar_id}")
        raise SystemExit(
            "output token arrays have different lengths: "
            f"{len(legacy['output_token_ids'])} != {len(sidecar['output_token_ids'])}"
        )
    print(f"matched {len(legacy['output_token_ids'])} output token IDs")


if __name__ == "__main__":
    main()
