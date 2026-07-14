#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Archive one DSV4 A/B job and derive auditable validity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

REGISTRATION_RE = re.compile(r"Model is ready\. Have (\d+) prefills and (\d+) decodes\.")
ENGINE_FATAL_RE = re.compile(
    r"Traceback \(most recent call last\):|OutOfMemoryError|CUDA error:|"
    r"Segmentation fault|core dumped|\bFATAL\b|engine process.*(?:failed|exited)",
    re.IGNORECASE,
)
MOONCAKE_FATAL_RE = re.compile(
    r"(?:mooncake|transfer engine).*?(?:\bERROR\b|\bFATAL\b|failed|aborted)",
    re.IGNORECASE,
)
NCCL_FATAL_RE = re.compile(r"nccl.*?(?:\bERROR\b|\bFATAL\b|unhandled|failed|aborted)", re.IGNORECASE)
GRPC_FATAL_RE = re.compile(
    r"(?:grpc|http/2|http2).*?(?:\bERROR\b|\bFATAL\b|panic|failed|aborted)",
    re.IGNORECASE,
)
SIDECAR_FATAL_RE = re.compile(
    r"Traceback \(most recent call last\):|\b(?:ERROR|FATAL)\b|"
    r"(?i:panicked at|BackendEngineShutdown|process.*(?:failed|exited))",
)


def _text_files(log_dir: Path) -> list[Path]:
    return sorted(
        path for path in log_dir.glob("*") if path.is_file() and path.suffix.lower() in {".log", ".out", ".txt"}
    )


def _read(path: Path) -> str:
    return path.read_text(errors="replace")


def _matches(paths: list[Path], pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    matches = []
    for path in paths:
        for line_number, line in enumerate(_read(path).splitlines(), 1):
            if pattern.search(line):
                matches.append(
                    {
                        "file": path.name,
                        "line": line_number,
                        "text": line[-1000:],
                    }
                )
    return matches


def _grpc_matches(paths: list[Path], backend: str) -> list[dict[str, Any]]:
    if backend != "sidecar":
        return []
    return [
        match
        for match in _matches(paths, GRPC_FATAL_RE)
        if "dynamo_runtime::transports::etcd" not in match["text"]
    ]


def _registrations(logs: list[Path]) -> tuple[int, dict[str, int] | None]:
    last = None
    for path in logs:
        if not path.name.startswith("sweep_"):
            continue
        for match in REGISTRATION_RE.finditer(_read(path)):
            last = {"prefills": int(match.group(1)), "decodes": int(match.group(2))}
    return (sum(last.values()) if last else 0), last


def _benchmark_status(run_dir: Path) -> dict[str, Any]:
    results = sorted(run_dir.rglob("results_concurrency_*.json"))
    if len(results) == 1:
        raw = json.loads(results[0].read_text())
        input_lens = raw.get("input_lens") or []
        output_lens = raw.get("output_lens") or []
        expected = int(raw.get("num_prompts", len(output_lens)))
        completed = int(raw.get("completed", 0))
        errors = raw.get("errors") or []
        failures = max(expected - completed, sum(bool(error) for error in errors))
        token_counts_complete = (
            bool(output_lens)
            and int(raw.get("total_input_tokens", -1)) == sum(int(value) for value in input_lens)
            and int(raw.get("total_output_tokens", -1)) == sum(int(value) for value in output_lens)
        )
        return {
            "kind": "sa-bench",
            "result_file": str(results[0].relative_to(run_dir)),
            "expected_requests": expected,
            "completed_requests": completed,
            "failed_requests": failures,
            "token_counts_complete": token_counts_complete,
            "valid": failures == 0 and token_counts_complete and completed == expected,
        }

    deterministic = sorted(run_dir.rglob("deterministic-output.json"))
    if len(deterministic) == 1:
        raw = json.loads(deterministic[0].read_text())
        completion_tokens = raw.get("completion_tokens_normalized", raw.get("completion_tokens_reported", -1))
        valid = (
            int(raw.get("prompt_token_count", -1)) == int(raw.get("isl", -2))
            and int(completion_tokens) == int(raw.get("osl", -2))
            and len(raw.get("output_token_ids") or []) == int(raw.get("osl", -3))
        )
        return {
            "kind": "deterministic-smoke",
            "result_file": str(deterministic[0].relative_to(run_dir)),
            "expected_requests": 1,
            "completed_requests": int(valid),
            "failed_requests": int(not valid),
            "token_counts_complete": valid,
            "valid": valid,
        }

    return {
        "kind": "missing",
        "expected_requests": 0,
        "completed_requests": 0,
        "failed_requests": 1,
        "token_counts_complete": False,
        "valid": False,
    }


def _harness_scheduler_recovery(run_dir: Path, benchmark: dict[str, Any], root_state: str) -> dict[str, Any]:
    result = {"valid": False}
    if root_state != "FAILED" or benchmark.get("kind") != "deterministic-smoke" or not benchmark.get("valid"):
        return result
    recoveries = sorted(run_dir.rglob("harness-recovery.json"))
    deterministic = sorted(run_dir.rglob("deterministic-output.json"))
    responses = sorted(run_dir.rglob("deterministic-output.response.json"))
    if len(recoveries) != 1 or len(deterministic) != 1 or len(responses) != 1:
        return result
    recovery = json.loads(recoveries[0].read_text())
    artifact = json.loads(deterministic[0].read_text())
    response_sha256 = hashlib.sha256(responses[0].read_bytes()).hexdigest()
    expected_reported = int(artifact["osl"]) * (int(artifact["osl"]) + 1) // 2
    benchmark_logs = sorted((run_dir / "logs").glob("benchmark.out"))
    expected_failure = (
        f"expected {artifact['osl']} completion tokens, received {expected_reported}"
    )
    valid = (
        recovery.get("reason") == "sglang_native_grpc_cumulative_completion_usage"
        and recovery.get("response_sha256") == response_sha256 == artifact.get("raw_response_sha256")
        and artifact.get("completion_token_count_source") == "cumulative_chunk_sum"
        and int(artifact.get("completion_tokens_reported", -1)) == expected_reported
        and int(artifact.get("completion_tokens_normalized", -1)) == int(artifact["osl"])
        and len(benchmark_logs) == 1
        and expected_failure in _read(benchmark_logs[0])
    )
    return {
        "valid": valid,
        "reason": recovery.get("reason"),
        "response_sha256": response_sha256,
        "original_slurm_state": root_state,
    }


def _scheduler(job_id: str) -> dict[str, Any]:
    command = [
        "sacct",
        "-X",
        "-j",
        job_id,
        "--noheader",
        "--parsable2",
        "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) >= 5:
            rows.append(dict(zip(("job_id", "state", "exit_code", "elapsed", "nodes"), fields[:5], strict=True)))
    root = next((row for row in rows if row["job_id"] == job_id), rows[0] if rows else {})
    return {"root": root, "rows": rows, "command": command}


def evaluate(run_dir: Path, backend: str, expected_registrations: int, scheduler: dict[str, Any]) -> dict[str, Any]:
    log_dir = run_dir / "logs"
    logs = _text_files(log_dir)
    workers = [path for path in logs if "_prefill_w" in path.name or "_decode_w" in path.name]
    sidecars = workers + [path for path in logs if "sidecar" in path.name.lower()] if backend == "sidecar" else []
    observed, registration_detail = _registrations(logs)
    benchmark = _benchmark_status(run_dir)

    worker_text = "\n".join(_read(path) for path in workers)
    mooncake_initialized = "Topology discovery complete" in worker_text
    decode_activity = re.search(r"Decode batch, #running-req: [1-9]", worker_text) is not None
    mooncake_kv_transfer = mooncake_initialized and decode_activity and benchmark["valid"]

    fatal_matches = {
        "engine": _matches(workers, ENGINE_FATAL_RE),
        "mooncake": _matches(workers, MOONCAKE_FATAL_RE),
        "nccl": _matches(workers, NCCL_FATAL_RE),
        "grpc": _grpc_matches(sidecars, backend),
        "sidecar": _matches(sidecars, SIDECAR_FATAL_RE),
    }
    root_state = scheduler.get("root", {}).get("state", "UNKNOWN").split()[0].rstrip("+")
    scheduler_recovery = _harness_scheduler_recovery(run_dir, benchmark, root_state)
    evidence = {
        "expected_worker_registrations": expected_registrations,
        "observed_worker_registrations": observed,
        "mooncake_kv_transfer": mooncake_kv_transfer,
        "fatal_engine_errors": len(fatal_matches["engine"]),
        "fatal_mooncake_errors": len(fatal_matches["mooncake"]),
        "fatal_nccl_errors": len(fatal_matches["nccl"]),
        "fatal_grpc_errors": len(fatal_matches["grpc"]),
        "fatal_sidecar_errors": len(fatal_matches["sidecar"]),
        "backend": backend,
        "registration_detail": registration_detail,
        "scheduler_harness_recovery": scheduler_recovery,
        "mooncake_evidence": {
            "topology_initialized": mooncake_initialized,
            "decode_activity": decode_activity,
            "benchmark_completed": benchmark["valid"],
        },
        "fatal_matches": fatal_matches,
    }
    hard_errors = []
    if root_state != "COMPLETED" and not scheduler_recovery["valid"]:
        hard_errors.append(f"slurm_state:{root_state}")
    if observed != expected_registrations:
        hard_errors.append("worker_registration_mismatch")
    if not mooncake_kv_transfer:
        hard_errors.append("mooncake_kv_transfer_failed")
    if not benchmark["valid"]:
        hard_errors.append("benchmark_invalid")
    for field in (
        "fatal_engine_errors",
        "fatal_mooncake_errors",
        "fatal_nccl_errors",
        "fatal_grpc_errors",
        "fatal_sidecar_errors",
    ):
        if evidence[field]:
            hard_errors.append(field)
    return {
        "schema_version": 2,
        "valid": not hard_errors,
        "validity_errors": hard_errors,
        "benchmark": benchmark,
        "validation": evidence,
        "scheduler": scheduler,
    }


def refresh(run_dir: Path, *, backend: str, expected_registrations: int) -> dict[str, Any]:
    scheduler_path = run_dir / "scheduler.json"
    if not scheduler_path.is_file():
        raise FileNotFoundError(scheduler_path)
    scheduler = json.loads(scheduler_path.read_text())
    result = evaluate(run_dir, backend, expected_registrations, scheduler)
    (run_dir / "validation.json").write_text(json.dumps(result["validation"], indent=2, sort_keys=True) + "\n")
    (run_dir / "collection.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def collect(
    source: Path,
    destination: Path,
    *,
    backend: str,
    expected_registrations: int,
    job_id: str,
    recipe: Path | None = None,
    run_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.mkdir(exist_ok=True)
        (destination / "source-output-missing.txt").write_text(f"{source}\n")
    if recipe is not None:
        shutil.copy2(recipe, destination / "resolved-recipe.yaml")
    if run_spec is not None:
        (destination / "run-spec.json").write_text(json.dumps(run_spec, indent=2, sort_keys=True) + "\n")

    scheduler = _scheduler(job_id)
    result = evaluate(destination, backend, expected_registrations, scheduler)
    (destination / "scheduler.json").write_text(json.dumps(scheduler, indent=2, sort_keys=True) + "\n")
    (destination / "validation.json").write_text(json.dumps(result["validation"], indent=2, sort_keys=True) + "\n")
    (destination / "collection.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--backend", choices=("legacy", "sidecar"), required=True)
    parser.add_argument("--expected-registrations", type=int, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--recipe", type=Path)
    args = parser.parse_args()
    result = collect(
        args.source,
        args.destination,
        backend=args.backend,
        expected_registrations=args.expected_registrations,
        job_id=args.job_id,
        recipe=args.recipe,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
