#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resume gates after auditable recovery of a post-request smoke harness failure."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recover_state(state: dict[str, Any], comparison: str, collection: dict[str, Any]) -> dict[str, Any]:
    if not collection["valid"]:
        raise ValueError("recovered sidecar collection is not valid")
    recovery = collection["validation"]["scheduler_harness_recovery"]
    if not recovery["valid"]:
        raise ValueError("sidecar collection lacks a valid scheduler harness recovery")
    state["runs"]["2:attempt-1"].update(
        {
            "status": "collected",
            "valid": True,
            "harness_recovery": recovery,
        }
    )
    state.setdefault("pair_comparisons", {})["smoke:pair-1:attempt-1"] = {
        "comparator": "smoke_tokens",
        "output": comparison,
        "valid": True,
    }
    state.setdefault("pairs", {})["smoke:pair-1"] = "complete"
    state.setdefault("points", {})["smoke"] = "complete"
    state.pop("status", None)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()

    campaign = args.repo_root / "campaigns/dsv4-pro-gb300-grpc-ab"
    legacy = args.artifacts_root / "gates/smoke/pair-1-retry-1/1-legacy"
    sidecar = args.artifacts_root / "gates/smoke/pair-1-retry-1/2-sidecar"
    collector = _load_module(campaign / "collect-run.py")
    legacy_collection = collector.refresh(legacy, backend="legacy", expected_registrations=2)
    sidecar_collection = collector.refresh(sidecar, backend="sidecar", expected_registrations=2)
    if not legacy_collection["valid"]:
        raise SystemExit("legacy smoke collection is not valid")

    compared = subprocess.run(
        [
            sys.executable,
            str(campaign / "compare-smoke.py"),
            str(legacy / "logs/deterministic-smoke/deterministic-output.json"),
            str(sidecar / "logs/deterministic-smoke/deterministic-output.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    comparison = compared.stdout.strip()
    original = args.state.read_bytes()
    backup = args.state.with_name("gate-controller-state.before-cumulative-usage-recovery.json")
    if backup.exists() and backup.read_bytes() != original:
        raise SystemExit(f"refusing to overwrite differing recovery backup: {backup}")
    if not backup.exists():
        shutil.copy2(args.state, backup)

    state = recover_state(json.loads(original), comparison, sidecar_collection)
    temporary = args.state.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.state)
    audit = {
        "schema_version": 1,
        "recovered_at": datetime.now(UTC).isoformat(),
        "original_state_sha256": hashlib.sha256(original).hexdigest(),
        "legacy_job_id": state["runs"]["1:attempt-1"]["job_id"],
        "sidecar_job_id": state["runs"]["2:attempt-1"]["job_id"],
        "comparison": comparison,
        "scheduler_harness_recovery": sidecar_collection["validation"]["scheduler_harness_recovery"],
    }
    audit_path = args.artifacts_root / "gates/smoke/cumulative-usage-harness-recovery.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(comparison)


if __name__ == "__main__":
    main()
