#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze the locked DeepSeek-V4-Pro legacy/native-gRPC crossover campaign."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

LATENCIES = ("ttft", "tpot", "itl", "e2el")
METRIC_FIELDS = (
    "total_token_throughput",
    "output_token_throughput",
    "median_intvty",
    "tput_per_gpu",
    "output_tput_per_decode_gpu",
    *(f"{percentile}_{latency}" for latency in LATENCIES for percentile in ("median", "p99")),
    "achieved_request_rate",
)
VALIDATION_FIELDS = (
    "expected_worker_registrations",
    "observed_worker_registrations",
    "mooncake_kv_transfer",
    "fatal_engine_errors",
    "fatal_mooncake_errors",
    "fatal_nccl_errors",
    "fatal_grpc_errors",
    "fatal_sidecar_errors",
)


def percent_delta(candidate: float, baseline: float) -> float:
    return ((candidate - baseline) / baseline) * 100.0


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _find_result(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.rglob("results_concurrency_*.json"))
    if len(candidates) > 1:
        raise ValueError(f"multiple result files under {run_dir}: {candidates}")
    return candidates[0] if candidates else None


def _token_counts_complete(raw: dict[str, Any]) -> bool:
    input_lens = raw.get("input_lens")
    output_lens = raw.get("output_lens")
    if not isinstance(input_lens, list) or not isinstance(output_lens, list) or not output_lens:
        return False
    return int(raw.get("total_input_tokens", -1)) == sum(int(value) for value in input_lens) and int(
        raw.get("total_output_tokens", -1)
    ) == sum(int(value) for value in output_lens)


def _failed_requests(raw: dict[str, Any]) -> int:
    expected = int(raw.get("num_prompts", len(raw.get("output_lens", []))))
    completed = int(raw.get("completed", 0))
    failures = max(expected - completed, 0)
    errors = raw.get("errors", [])
    if isinstance(errors, list):
        failures = max(failures, sum(bool(error) for error in errors))
    return failures


def _metrics(raw: dict[str, Any], total_gpus: int, decode_gpus: int) -> dict[str, float]:
    median_tpot_ms = float(raw["median_tpot_ms"])
    metrics = {
        "total_token_throughput": float(raw["total_token_throughput"]),
        "output_token_throughput": float(raw["output_throughput"]),
        "median_intvty": 1000.0 / median_tpot_ms,
        "tput_per_gpu": float(raw["total_token_throughput"]) / total_gpus,
        "output_tput_per_decode_gpu": float(raw["output_throughput"]) / decode_gpus,
        "achieved_request_rate": float(raw["request_throughput"]),
    }
    for latency in LATENCIES:
        for percentile in ("median", "p99"):
            metrics[f"{percentile}_{latency}"] = float(raw[f"{percentile}_{latency}_ms"]) / 1000.0
    return metrics


def _float_cell(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    if not value or value.upper() in {"N/A", "[NOT SUPPORTED]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _telemetry(run_dir: Path) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    samples = 0
    for path in sorted(run_dir.rglob("gpu-telemetry-*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                samples += 1
                for output, source in (
                    ("gpu_utilization_percent", "utilization.gpu"),
                    ("memory_used_mib", "memory.used"),
                    ("power_draw_watts", "power.draw"),
                    ("sm_clock_mhz", "clocks.sm"),
                    ("memory_clock_mhz", "clocks.mem"),
                    ("temperature_c", "temperature.gpu"),
                ):
                    value = _float_cell(row, source)
                    if value is not None:
                        values[output].append(value)

    power_cap_active_samples = 0
    hardware_throttle_active_samples = 0
    throttle_samples = 0
    for path in sorted(run_dir.rglob("gpu-throttle-*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                throttle_samples += 1
                if row.get("sw_power_cap", "").strip().lower() == "active":
                    power_cap_active_samples += 1
                if any(
                    value.strip().lower() == "active"
                    for key, value in row.items()
                    if key not in {"timestamp", "index", "uuid", "sw_power_cap"}
                ):
                    hardware_throttle_active_samples += 1

    summary: dict[str, Any] = {
        "samples": samples,
        "throttle_samples": throttle_samples,
        "power_cap_active_samples": power_cap_active_samples,
        "hardware_throttle_active_samples": hardware_throttle_active_samples,
        "power_cap_observed": power_cap_active_samples > 0,
        "throttling_observed": hardware_throttle_active_samples > 0,
    }
    for key, field_values in values.items():
        summary[f"mean_{key}"] = fmean(field_values)
        summary[f"min_{key}"] = min(field_values)
        summary[f"max_{key}"] = max(field_values)
    return summary


def _validation(run_dir: Path, backend: str) -> tuple[dict[str, Any], list[str]]:
    path = run_dir / "validation.json"
    if not path.is_file():
        return {}, ["validation_evidence_missing"]
    evidence = _load_json(path)
    missing = [field for field in VALIDATION_FIELDS if field not in evidence]
    flags = [f"validation_field_missing:{field}" for field in missing]
    if not missing:
        if int(evidence["observed_worker_registrations"]) != int(evidence["expected_worker_registrations"]):
            flags.append("worker_registration_mismatch")
        if evidence["mooncake_kv_transfer"] is not True:
            flags.append("mooncake_kv_transfer_failed")
        for field in VALIDATION_FIELDS[3:]:
            if int(evidence[field]):
                flags.append(field)
        if backend == "legacy" and int(evidence["fatal_sidecar_errors"]):
            flags.append("legacy_reported_sidecar_errors")
    return evidence, flags


def process_run(
    artifacts_root: Path,
    run_spec: dict[str, Any],
    topology: dict[str, Any],
) -> dict[str, Any] | None:
    run_dir = artifacts_root / run_spec["artifact_dir"]
    result_path = _find_result(run_dir)
    if result_path is None:
        return None
    raw = _load_json(result_path)
    failed = _failed_requests(raw)
    tokens_complete = _token_counts_complete(raw)
    validation, validity_errors = _validation(run_dir, run_spec["backend"])
    telemetry = _telemetry(run_dir)
    if failed:
        validity_errors.append("request_failures")
    if not tokens_complete:
        validity_errors.append("incomplete_token_counts")
    warnings = []
    if telemetry["samples"] == 0:
        warnings.append("telemetry_missing")
    if telemetry["throttle_samples"] == 0:
        warnings.append("throttle_evidence_missing")
    if telemetry["throttling_observed"]:
        warnings.append("gpu_hardware_or_thermal_throttling")

    expected = int(raw.get("num_prompts", len(raw.get("output_lens", []))))
    completed = int(raw.get("completed", 0))
    return {
        **run_spec,
        "concurrency": int(topology["concurrency"]),
        "topology": topology,
        "metrics": _metrics(raw, int(topology["total_gpus"]), int(topology["decode_gpus"])),
        "failed_requests": failed,
        "completed_requests": completed,
        "expected_requests": expected,
        "total_input_tokens": int(raw.get("total_input_tokens", 0)),
        "total_output_tokens": int(raw.get("total_output_tokens", 0)),
        "token_counts_complete": tokens_complete,
        "validation": validation,
        "telemetry": telemetry,
        "validity_errors": sorted(set(validity_errors)),
        "warnings": sorted(set(warnings)),
        "flags": sorted(set(validity_errors + warnings)),
        "valid": not validity_errors,
        "result_file": str(result_path),
    }


def _paired_summary(runs: list[dict[str, Any]], topology: dict[str, Any]) -> dict[str, Any]:
    selected_runs: list[dict[str, Any]] = []
    pair_attempts: dict[str, Any] = {}
    for pair in (1, 2):
        pair_runs = [run for run in runs if int(run["pair"]) == pair]
        attempts = sorted({int(run["attempt"]) for run in pair_runs})
        selected_attempt = None
        attempt_status = []
        for attempt in attempts:
            attempt_runs = [run for run in pair_runs if int(run["attempt"]) == attempt]
            complete = {run["backend"] for run in attempt_runs} == {"legacy", "sidecar"}
            valid = complete and all(run["valid"] for run in attempt_runs)
            attempt_status.append({"attempt": attempt, "complete": complete, "valid": valid})
            if selected_attempt is None and valid:
                selected_attempt = attempt
                selected_runs.extend(attempt_runs)
        pair_attempts[str(pair)] = {
            "attempts": attempt_status,
            "selected_attempt": selected_attempt,
        }

    by_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in selected_runs:
        by_backend[run["backend"]].append(run)

    means: dict[str, dict[str, float] | None] = {}
    telemetry_means: dict[str, dict[str, float] | None] = {}
    for backend in ("legacy", "sidecar"):
        valid = [run for run in by_backend[backend] if run["valid"]]
        means[backend] = (
            {field: fmean(float(run["metrics"][field]) for run in valid) for field in METRIC_FIELDS}
            if len(valid) == 2
            else None
        )
        telemetry_means[backend] = (
            {
                field: fmean(float(run["telemetry"][field]) for run in valid)
                for field in (
                    "mean_gpu_utilization_percent",
                    "mean_power_draw_watts",
                    "mean_sm_clock_mhz",
                )
            }
            if len(valid) == 2
            and all(
                field in run["telemetry"]
                for run in valid
                for field in (
                    "mean_gpu_utilization_percent",
                    "mean_power_draw_watts",
                    "mean_sm_clock_mhz",
                )
            )
            else None
        )

    deltas = None
    if means["legacy"] is not None and means["sidecar"] is not None:
        deltas = {field: percent_delta(means["sidecar"][field], means["legacy"][field]) for field in METRIC_FIELDS}

    order_effects: dict[str, float | None] = {}
    for backend in ("legacy", "sidecar"):
        first = [run for run in by_backend[backend] if run["valid"] and int(run["order_index"]) == 1]
        second = [run for run in by_backend[backend] if run["valid"] and int(run["order_index"]) == 2]
        order_effects[backend] = (
            percent_delta(
                float(first[0]["metrics"]["tput_per_gpu"]),
                float(second[0]["metrics"]["tput_per_gpu"]),
            )
            if len(first) == 1 and len(second) == 1
            else None
        )

    return {
        "point": topology["id"],
        "concurrency": topology["concurrency"],
        "topology": topology,
        "pair_attempts": pair_attempts,
        "valid_runs": {backend: sum(run["valid"] for run in by_backend[backend]) for backend in ("legacy", "sidecar")},
        "paired_means": means,
        "paired_telemetry_means": telemetry_means,
        "sidecar_vs_legacy_percent": deltas,
        "throughput_order_effect_percent": order_effects,
        "highlight_over_5_percent": bool(
            deltas
            and (
                abs(deltas["tput_per_gpu"]) > 5.0
                or abs(deltas["output_tput_per_decode_gpu"]) > 5.0
                or abs(deltas["median_intvty"]) > 5.0
            )
        ),
    }


def build_comparison(
    artifacts_root: Path,
    manifest: dict[str, Any],
    run_plan: dict[str, Any],
    public_curve: dict[str, Any],
    *,
    allow_incomplete: bool,
) -> dict[str, Any]:
    public_by_concurrency = {int(point["concurrency"]): point for point in public_curve["points"]}
    topologies: dict[str, dict[str, Any]] = {}
    for point in manifest["points"]:
        public = public_by_concurrency[int(point["concurrency"])]
        topologies[point["id"]] = {
            **point,
            **public["topology"],
            "public_metrics": public["metrics"],
            "public_result_id": public["id"],
        }

    runs: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in run_plan["runs"]:
        primary_spec = {**spec, "attempt": 1}
        run = process_run(artifacts_root, primary_spec, topologies[spec["point"]])
        if run is None:
            missing.append(spec["artifact_dir"])
        else:
            runs.append(run)
        retry_spec = {
            **spec,
            "attempt": 2,
            "artifact_dir": spec["retry_artifact_dir"],
        }
        retry = process_run(artifacts_root, retry_spec, topologies[spec["point"]])
        if retry is not None:
            runs.append(retry)
    if missing and not allow_incomplete:
        raise ValueError(f"missing {len(missing)} planned runs; first missing: {missing[0]}")

    summaries = []
    for point in manifest["points"]:
        point_runs = [run for run in runs if run["point"] == point["id"]]
        summaries.append(_paired_summary(point_runs, topologies[point["id"]]))

    selected_attempts = {
        (summary["point"], int(pair)): details["selected_attempt"]
        for summary in summaries
        for pair, details in summary["pair_attempts"].items()
    }
    for run in runs:
        run["selected_for_pair_mean"] = int(run["attempt"]) == selected_attempts[(run["point"], int(run["pair"]))]

    ratios = [
        1.0 + summary["sidecar_vs_legacy_percent"]["tput_per_gpu"] / 100.0
        for summary in summaries
        if summary["sidecar_vs_legacy_percent"] is not None
    ]
    geometric_mean = math.exp(fmean(math.log(ratio) for ratio in ratios)) if ratios else None
    return {
        "campaign": manifest,
        "public_source": public_curve["source"],
        "planned_primary_runs": len(run_plan["runs"]),
        "collected_attempt_runs": len(runs),
        "missing_runs": missing,
        "runs": runs,
        "points": summaries,
        "geometric_mean_sidecar_legacy_tput_per_gpu_ratio": geometric_mean,
    }


def write_runs_csv(comparison: dict[str, Any], output: Path) -> None:
    fields = [
        "sequence",
        "point",
        "concurrency",
        "pair",
        "attempt",
        "order_index",
        "backend",
        "selected_for_pair_mean",
        "valid",
        "failed_requests",
        "completed_requests",
        "expected_requests",
        *METRIC_FIELDS,
        "mean_gpu_utilization_percent",
        "mean_power_draw_watts",
        "mean_sm_clock_mhz",
        "throttling_observed",
        "power_cap_observed",
        "flags",
        "result_file",
    ]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in comparison["runs"]:
            telemetry = run["telemetry"]
            writer.writerow(
                {
                    **{field: run.get(field) for field in fields},
                    **run["metrics"],
                    "mean_gpu_utilization_percent": telemetry.get("mean_gpu_utilization_percent"),
                    "mean_power_draw_watts": telemetry.get("mean_power_draw_watts"),
                    "mean_sm_clock_mhz": telemetry.get("mean_sm_clock_mhz"),
                    "throttling_observed": telemetry["throttling_observed"],
                    "power_cap_observed": telemetry["power_cap_observed"],
                    "flags": ";".join(run["flags"]),
                }
            )


def _scale(value: float, low: float, high: float, start: float, size: float) -> float:
    return start + size / 2 if high == low else start + ((value - low) / (high - low)) * size


def write_svg(comparison: dict[str, Any], public_curve: dict[str, Any], output: Path) -> None:
    series: dict[str, list[tuple[float, float, int]]] = {"public": [], "legacy": [], "sidecar": []}
    for point in public_curve["points"]:
        metrics = point["metrics"]
        series["public"].append((metrics["median_intvty"], metrics["tput_per_gpu"], point["concurrency"]))
    for point in comparison["points"]:
        for backend in ("legacy", "sidecar"):
            metrics = point["paired_means"][backend]
            if metrics is not None:
                series[backend].append((metrics["median_intvty"], metrics["tput_per_gpu"], point["concurrency"]))
    all_values = [value for values in series.values() for value in values]
    if not all_values:
        return
    left, top, width, height = 100, 60, 920, 540
    x_values, y_values = [v[0] for v in all_values], [v[1] for v in all_values]
    x_low, x_high = min(x_values) * 0.95, max(x_values) * 1.05
    y_low, y_high = 0.0, max(y_values) * 1.08
    colors = {"public": "#4b5563", "legacy": "#2563eb", "sidecar": "#dc2626"}
    labels = {"public": "Public context", "legacy": "Local legacy", "sidecar": "Local sidecar"}

    def xy(point: tuple[float, float, int]) -> tuple[float, float]:
        return (
            _scale(point[0], x_low, x_high, left, width),
            top + height - _scale(point[1], y_low, y_high, 0, height),
        )

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="700" viewBox="0 0 1100 700">',
        '<rect width="1100" height="700" fill="white"/>',
        '<text x="550" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">GB300 DeepSeek-V4-Pro 8K/1K SGLang A/B</text>',
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="black"/>',
        f'<text x="{left + width / 2}" y="655" text-anchor="middle" font-family="sans-serif">Median interactivity (tok/s/user)</text>',
        f'<text x="24" y="{top + height / 2}" text-anchor="middle" transform="rotate(-90 24 {top + height / 2})" font-family="sans-serif">Total throughput / GPU (tok/s/GPU)</text>',
    ]
    for index, name in enumerate(("public", "legacy", "sidecar")):
        ordered = sorted(series[name])
        if ordered:
            coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(xy, ordered))
            lines.append(f'<polyline points="{coordinates}" fill="none" stroke="{colors[name]}" stroke-width="2"/>')
        for point in ordered:
            x, y = xy(point)
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{colors[name]}"/>')
            lines.append(
                f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" font-family="sans-serif" font-size="10" fill="{colors[name]}">c{point[2]}</text>'
            )
        legend_x = 700 + index * 125
        lines.append(f'<rect x="{legend_x}" y="75" width="14" height="4" fill="{colors[name]}"/>')
        lines.append(
            f'<text x="{legend_x + 20}" y="82" font-family="sans-serif" font-size="11">{html.escape(labels[name])}</text>'
        )
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n")


def _fmt(value: float | None, precision: int = 2) -> str:
    return "—" if value is None else f"{value:.{precision}f}"


def write_markdown(comparison: dict[str, Any], output: Path) -> None:
    campaign = comparison["campaign"]
    ratio = comparison["geometric_mean_sidecar_legacy_tput_per_gpu_ratio"]
    lines = [
        "# GB300 NVL72 DeepSeek-V4-Pro SGLang gRPC A/B",
        "",
        "> Internal and unofficial. The public InferenceX curve is contextual; the primary comparison is the paired local sidecar-versus-legacy A/B.",
        "",
        "## Pinned inputs",
        "",
        f"- Dynamo: `{campaign['source_commits']['dynamo']}`",
        f"- SGLang: `{campaign['source_commits']['sglang']}`",
        f"- InferenceX recipes: `{campaign['source_commits']['inferencex']}`",
        f"- Image: `{campaign['image']}`",
        f"- Model: `{campaign['model']}` revision `{campaign['model_revision']}` at `{campaign['model_path']}`",
        "- Workload: 8,192 input / 1,024 output tokens; Mooncake; speculative decoding disabled",
        "",
        "## Summary",
        "",
        f"Collected {comparison['collected_attempt_runs']} attempt legs for {comparison['planned_primary_runs']} planned primary fresh-process runs.",
        f"Geometric-mean sidecar/legacy throughput-per-GPU ratio: **{_fmt(ratio, 4)}**.",
        "",
        "| C | P/D GPUs | Valid L/S | Legacy tput/GPU | Sidecar tput/GPU | Delta | Legacy intvty | Sidecar intvty | Delta | Output/decode GPU delta | L order effect | S order effect | Flags |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for point in comparison["points"]:
        legacy = point["paired_means"]["legacy"]
        sidecar = point["paired_means"]["sidecar"]
        delta = point["sidecar_vs_legacy_percent"]
        topology = point["topology"]
        flags = []
        if point["highlight_over_5_percent"]:
            flags.append(">5% paired deviation")
        point_runs = [run for run in comparison["runs"] if run["point"] == point["point"]]
        if any(run["backend"] == "sidecar" and not run["valid"] for run in point_runs):
            flags.append("sidecar invalid run")
        lines.append(
            "| {c} | {p}/{d} | {vl}/{vs} | {lt} | {st} | {dt} | {li} | {si} | {di} | {do} | {lo} | {so} | {flags} |".format(
                c=point["concurrency"],
                p=topology["prefill_gpus"],
                d=topology["decode_gpus"],
                vl=point["valid_runs"]["legacy"],
                vs=point["valid_runs"]["sidecar"],
                lt=_fmt(legacy["tput_per_gpu"] if legacy else None),
                st=_fmt(sidecar["tput_per_gpu"] if sidecar else None),
                dt=_fmt(delta["tput_per_gpu"] if delta else None),
                li=_fmt(legacy["median_intvty"] if legacy else None),
                si=_fmt(sidecar["median_intvty"] if sidecar else None),
                di=_fmt(delta["median_intvty"] if delta else None),
                do=_fmt(delta["output_tput_per_decode_gpu"] if delta else None),
                lo=_fmt(point["throughput_order_effect_percent"]["legacy"]),
                so=_fmt(point["throughput_order_effect_percent"]["sidecar"]),
                flags=", ".join(flags) or "—",
            )
        )
    lines.extend(
        [
            "",
            "## Paired means",
            "",
            "| C | Backend | Total tok/s | Output tok/s | TTFT med/p99 (s) | TPOT med/p99 (ms) | ITL med/p99 (ms) | E2E med/p99 (s) | Req/s | GPU util | Power W | SM MHz |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for point in comparison["points"]:
        for backend in ("legacy", "sidecar"):
            metrics = point["paired_means"][backend]
            telemetry = point["paired_telemetry_means"][backend]
            if metrics is None:
                lines.append(f"| {point['concurrency']} | {backend} | — | — | — | — | — | — | — | — | — | — |")
                continue
            lines.append(
                "| {c} | {backend} | {total:.2f} | {output:.2f} | {ttft:.3f}/{p99ttft:.3f} | "
                "{tpot:.3f}/{p99tpot:.3f} | {itl:.3f}/{p99itl:.3f} | {e2e:.3f}/{p99e2e:.3f} | "
                "{rate:.2f} | {util} | {power} | {clock} |".format(
                    c=point["concurrency"],
                    backend=backend,
                    total=metrics["total_token_throughput"],
                    output=metrics["output_token_throughput"],
                    ttft=metrics["median_ttft"],
                    p99ttft=metrics["p99_ttft"],
                    tpot=metrics["median_tpot"] * 1000,
                    p99tpot=metrics["p99_tpot"] * 1000,
                    itl=metrics["median_itl"] * 1000,
                    p99itl=metrics["p99_itl"] * 1000,
                    e2e=metrics["median_e2el"],
                    p99e2e=metrics["p99_e2el"],
                    rate=metrics["achieved_request_rate"],
                    util=_fmt(telemetry["mean_gpu_utilization_percent"] if telemetry else None),
                    power=_fmt(telemetry["mean_power_draw_watts"] if telemetry else None),
                    clock=_fmt(telemetry["mean_sm_clock_mhz"] if telemetry else None),
                )
            )
    lines.extend(
        [
            "",
            "## Per-run results",
            "",
            "| Seq | Point | Pair/attempt/order | Backend | Sel | Valid | Total/output tok/s | TTFT med/p99 (s) | TPOT med/p99 (ms) | ITL med/p99 (ms) | E2E med/p99 (s) | Req/s | Requests | Input/output tokens | GPU util / W / MHz | Throttle | Flags |",
            "|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for run in sorted(comparison["runs"], key=lambda item: (item["sequence"], item["attempt"])):
        metrics = run["metrics"]
        telemetry = run["telemetry"]
        lines.append(
            "| {sequence} | {point} | {pair}/{attempt}/{order} | {backend} | {selected} | {valid} | "
            "{total:.2f}/{output:.2f} | {ttft:.3f}/{p99ttft:.3f} | {tpot:.3f}/{p99tpot:.3f} | "
            "{itl:.3f}/{p99itl:.3f} | {e2e:.3f}/{p99e2e:.3f} | {rate:.2f} | {completed}/{expected} | "
            "{input_tokens}/{output_tokens} | {util}/{power}/{clock} | {throttle} | {flags} |".format(
                sequence=run["sequence"],
                point=run["point"],
                pair=run["pair"],
                attempt=run["attempt"],
                order=run["order_index"],
                backend=run["backend"],
                selected="yes" if run["selected_for_pair_mean"] else "no",
                valid="yes" if run["valid"] else "no",
                total=metrics["total_token_throughput"],
                output=metrics["output_token_throughput"],
                ttft=metrics["median_ttft"],
                p99ttft=metrics["p99_ttft"],
                tpot=metrics["median_tpot"] * 1000,
                p99tpot=metrics["p99_tpot"] * 1000,
                itl=metrics["median_itl"] * 1000,
                p99itl=metrics["p99_itl"] * 1000,
                e2e=metrics["median_e2el"],
                p99e2e=metrics["p99_e2el"],
                rate=metrics["achieved_request_rate"],
                completed=run["completed_requests"],
                expected=run["expected_requests"],
                input_tokens=run["total_input_tokens"],
                output_tokens=run["total_output_tokens"],
                util=_fmt(telemetry.get("mean_gpu_utilization_percent")),
                power=_fmt(telemetry.get("mean_power_draw_watts")),
                clock=_fmt(telemetry.get("mean_sm_clock_mhz")),
                throttle=(
                    "hardware/thermal"
                    if telemetry["throttling_observed"]
                    else "power-cap"
                    if telemetry["power_cap_observed"]
                    else "no"
                ),
                flags=", ".join(run["flags"]) or "—",
            )
        )
    lines.extend(
        [
            "",
            "All deltas are percentages with sidecar as candidate and legacy as baseline. Positive throughput/interactivity is better; positive latency is worse. Order effect compares the same backend when run first versus second in its pair.",
            "",
            "A run is excluded unless it has zero failed requests, internally complete token accounting, exact worker registrations, successful Mooncake KV transfer, and no fatal engine/Mooncake/NCCL/gRPC/sidecar errors. Power-cap activity is reported separately; missing telemetry and hardware or thermal throttling are retained as warnings rather than silently changing the campaign validity rule.",
            "",
            "No formal pass/fail threshold is applied. Paired deviations above 5%, sidecar-only failures, systematic shifts, and run-order or hardware variation are called out for review.",
        ]
    )
    if comparison["missing_runs"]:
        lines.extend(["", f"Incomplete campaign: {len(comparison['missing_runs'])} planned runs are missing."])
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--public-curve", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    manifest = _load_json(args.campaign_manifest)
    run_plan = _load_json(args.run_plan)
    public_curve = _load_json(args.public_curve)
    comparison = build_comparison(
        args.artifacts_root,
        manifest,
        run_plan,
        public_curve,
        allow_incomplete=args.allow_incomplete,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    write_runs_csv(comparison, args.output_dir / "runs.csv")
    write_svg(comparison, public_curve, args.output_dir / "pareto-overlay.svg")
    write_markdown(comparison, args.output_dir / "report.md")


if __name__ == "__main__":
    main()
