#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the locked 28-leg DSV4 crossover plan with pair-level retries."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
SUBMITTED_RE = re.compile(r"Job (\d+) submitted")
STOP_REQUESTED = False


def _request_stop(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _load_collector(path: Path):
    spec = importlib.util.spec_from_file_location("dsv4_collect_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import collector: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def expected_registrations(recipe: Path) -> int:
    value = yaml.safe_load(recipe.read_text())
    resources = value["resources"]
    return int(resources["prefill_workers"]) + int(resources["decode_workers"])


def build_apply_command(srtctl: Path, recipe: Path, tags: list[str]) -> list[str]:
    return [str(srtctl), "apply", "-f", str(recipe), "-y", "--tags", ",".join(tags)]


def _submit(srtctl: Path, config: Path, recipe: Path, tags: list[str]) -> tuple[str, str]:
    environment = os.environ.copy()
    environment["SRTSLURM_CONFIG"] = str(config)
    command = build_apply_command(srtctl, recipe, tags)
    completed = subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
    output = f"{completed.stdout}\n{completed.stderr}"
    match = SUBMITTED_RE.search(output)
    if not match:
        raise RuntimeError(f"srtctl did not report a job id:\n{output}")
    return match.group(1), output


def _job_state(job_id: str) -> str | None:
    command = [
        "sacct",
        "-X",
        "-j",
        job_id,
        "--noheader",
        "--parsable2",
        "--format=JobIDRaw,State",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    for line in completed.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0] == job_id:
            return fields[1].split()[0].rstrip("+")
    return None


def _wait(job_id: str, deadline: float, poll_seconds: int) -> str | None:
    while not STOP_REQUESTED and time.monotonic() < deadline:
        state = _job_state(job_id)
        if state in TERMINAL_STATES:
            return state
        print(f"job {job_id}: {state or 'not-yet-visible'}", flush=True)
        time.sleep(poll_seconds)
    return None


def _artifact_spec(spec: dict[str, Any], attempt: int) -> dict[str, Any]:
    artifact_dir = spec["artifact_dir"] if attempt == 0 else spec["retry_artifact_dir"]
    return {**spec, "artifact_dir": artifact_dir, "attempt": attempt}


def _group_plan(plan: dict[str, Any]) -> list[tuple[str, list[tuple[int, list[dict[str, Any]]]]]]:
    points: list[tuple[str, list[tuple[int, list[dict[str, Any]]]]]] = []
    for run in plan["runs"]:
        if not points or points[-1][0] != run["point"]:
            points.append((run["point"], []))
        pairs = points[-1][1]
        if not pairs or pairs[-1][0] != int(run["pair"]):
            pairs.append((int(run["pair"]), []))
        pairs[-1][1].append(run)
    return points


def _collection(destination: Path) -> dict[str, Any] | None:
    path = destination / "collection.json"
    return _load_json(path) if path.is_file() else None


def _adopt_jobs(state: dict[str, Any], plan: dict[str, Any], adoptions: list[str]) -> None:
    known_sequences = {int(spec["sequence"]): spec for spec in plan["runs"]}
    for adoption in adoptions:
        try:
            sequence_text, attempt_text, job_id = adoption.split(":", 2)
            sequence = int(sequence_text)
            attempt = int(attempt_text)
        except ValueError as error:
            raise ValueError(f"invalid --adopt-job value: {adoption}") from error
        if sequence not in known_sequences:
            raise ValueError(f"unknown adopted sequence: {sequence}")
        if attempt < 0 or attempt > int(plan["pair_retry_limit"]):
            raise ValueError(f"invalid adopted attempt: {attempt}")
        run_key = f"{sequence}:attempt-{attempt}"
        spec = _artifact_spec(known_sequences[sequence], attempt)
        record = state["runs"].setdefault(run_key, {"artifact_dir": spec["artifact_dir"]})
        existing = record.get("job_id")
        if existing is not None and existing != job_id:
            raise ValueError(f"{run_key} already tracks job {existing}, cannot adopt {job_id}")
        record.update({"job_id": job_id, "status": "submitted", "adopted": True})


def _compare_pair(
    comparator: str | None,
    destinations: list[Path],
    campaign: Path,
) -> dict[str, Any]:
    if comparator is None:
        return {"valid": True, "comparator": None}
    if comparator != "smoke_tokens":
        raise ValueError(f"unknown pair comparator: {comparator}")
    if len(destinations) != 2:
        raise ValueError("smoke token comparison requires exactly two legs")
    outputs = []
    for destination in destinations:
        candidates = sorted(destination.rglob("deterministic-output.json"))
        if len(candidates) != 1:
            return {
                "valid": False,
                "comparator": comparator,
                "error": f"expected one deterministic output under {destination}, found {len(candidates)}",
            }
        outputs.append(candidates[0])
    completed = subprocess.run(
        [sys.executable, str(campaign / "compare-smoke.py"), *(str(path) for path in outputs)],
        capture_output=True,
        text=True,
    )
    result = {
        "valid": completed.returncode == 0,
        "comparator": comparator,
        "command": completed.args,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "outputs": [str(path) for path in outputs],
    }
    comparison_dir = Path(os.path.commonpath([str(path) for path in destinations]))
    (comparison_dir / "pair-comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run(args: argparse.Namespace) -> int:
    repo = args.repo_root.resolve()
    campaign = repo / "campaigns/dsv4-pro-gb300-grpc-ab"
    plan = _load_json(args.run_plan or campaign / "run-plan.json")
    state_path = args.state or args.artifacts_root / "controller-state.json"
    state = _load_json(state_path) if state_path.is_file() else {"schema_version": 1, "runs": {}, "points": {}}
    _adopt_jobs(state, plan, args.adopt_job)
    _save(state_path, state)
    collector = _load_collector(campaign / "collect-run.py")
    deadline = time.monotonic() + args.max_runtime_seconds

    for point, pairs in _group_plan(plan):
        point_state = state["points"].get(point)
        if point_state in {"complete", "failed"}:
            continue
        point_failed = False
        for pair, specs in pairs:
            pair_key = f"{point}:pair-{pair}"
            if state.get("pairs", {}).get(pair_key) == "complete":
                continue
            for attempt in range(int(plan["pair_retry_limit"]) + 1):
                results = []
                destinations = []
                for original in specs:
                    spec = _artifact_spec(original, attempt)
                    destination = args.artifacts_root / spec["artifact_dir"]
                    destinations.append(destination)
                    run_key = f"{spec['sequence']}:attempt-{attempt}"
                    record = state["runs"].setdefault(run_key, {"artifact_dir": spec["artifact_dir"]})
                    recipe = repo / spec["recipe"]
                    collected = _collection(destination)
                    if collected is not None:
                        collected = collector.refresh(
                            destination,
                            backend=spec["backend"],
                            expected_registrations=expected_registrations(recipe),
                        )
                        record.update({"status": "collected", "valid": collected["valid"]})
                        _save(state_path, state)
                    if collected is None:
                        if STOP_REQUESTED or time.monotonic() >= deadline:
                            _save(state_path, state)
                            return 75
                        job_id = record.get("job_id")
                        if job_id is None:
                            tags = ["dsv4-grpc-ab", point, f"pair-{pair}", spec["backend"], f"attempt-{attempt}"]
                            job_id, submission = _submit(args.srtctl, args.config, recipe, tags)
                            record.update({"job_id": job_id, "status": "submitted", "submission": submission})
                            _save(state_path, state)
                            print(f"submitted {run_key} as job {job_id}", flush=True)
                        terminal = _wait(job_id, deadline, args.poll_seconds)
                        if terminal is None:
                            _save(state_path, state)
                            return 75
                        record["slurm_state"] = terminal
                        record["status"] = "collecting"
                        _save(state_path, state)
                        collected = collector.collect(
                            args.outputs_root / job_id,
                            destination,
                            backend=spec["backend"],
                            expected_registrations=expected_registrations(recipe),
                            job_id=job_id,
                            recipe=recipe,
                            run_spec=spec,
                        )
                        record.update({"status": "collected", "valid": collected["valid"]})
                        _save(state_path, state)
                    results.append(collected)

                comparator = (plan.get("pair_comparators") or {}).get(point)
                pair_comparison = _compare_pair(comparator, destinations, campaign)
                state.setdefault("pair_comparisons", {})[f"{pair_key}:attempt-{attempt}"] = pair_comparison
                _save(state_path, state)
                if all(result["valid"] for result in results) and pair_comparison["valid"]:
                    state.setdefault("pairs", {})[pair_key] = "complete"
                    _save(state_path, state)
                    break
                if attempt == int(plan["pair_retry_limit"]):
                    state.setdefault("pairs", {})[pair_key] = "failed"
                    state["points"][point] = "failed"
                    _save(state_path, state)
                    point_failed = True
            if point_failed:
                break
        if point_failed and plan.get("stop_campaign_on_point_failure", False):
            state["status"] = "failed_gate"
            _save(state_path, state)
            return 2
        if not point_failed:
            state["points"][point] = "complete"
            _save(state_path, state)

    state["status"] = "complete"
    state["completed_at_epoch"] = time.time()
    _save(state_path, state)
    return 2 if any(value == "failed" for value in state["points"].values()) else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--srtctl", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-runtime-seconds", type=int, default=27000)
    parser.add_argument("--adopt-job", action="append", default=[])
    args = parser.parse_args()
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
