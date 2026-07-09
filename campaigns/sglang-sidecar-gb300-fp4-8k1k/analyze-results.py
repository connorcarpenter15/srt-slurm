#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native-gRPC sidecar SA-Bench results with the public curve."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

LATENCY_FIELDS = (
    "median_ttft",
    "p99_ttft",
    "median_tpot",
    "p99_tpot",
    "median_e2el",
    "p99_e2el",
)
REPORT_FIELDS = (
    "median_intvty",
    "tput_per_gpu",
    "output_tput_per_gpu",
    *LATENCY_FIELDS,
)
FILENAME_TOPOLOGY = re.compile(r"_ctx_(?P<prefill>\d+)_gen_(?P<decode>\d+)\.json$")


def percent_delta(candidate: float, public: float) -> float:
    return ((candidate - public) / public) * 100.0


def _topology_key(point: dict[str, Any]) -> tuple[int, int, int]:
    topology = point["topology"]
    return (
        int(point["concurrency"]),
        int(topology["prefill_gpus"]),
        int(topology["decode_gpus"]),
    )


def _candidate_topology(path: Path, raw: dict[str, Any]) -> tuple[int, int, int]:
    match = FILENAME_TOPOLOGY.search(path.name)
    if match is None:
        raise ValueError(f"result filename does not include _ctx_<gpus>_gen_<gpus>: {path}")
    return (
        int(raw["max_concurrency"]),
        int(match.group("prefill")),
        int(match.group("decode")),
    )


def _failed_requests(raw: dict[str, Any]) -> int:
    expected = int(raw.get("num_prompts", 0))
    completed = int(raw.get("completed", 0))
    if expected:
        return max(expected - completed, 0)
    return sum(bool(error) for error in raw.get("errors", []))


def _token_counts_complete(raw: dict[str, Any]) -> bool:
    output_lens = raw.get("output_lens", [])
    if not output_lens:
        return False
    return int(raw.get("total_output_tokens", -1)) == sum(int(length) for length in output_lens)


def process_result(
    path: Path,
    raw: dict[str, Any],
    public: dict[str, Any],
) -> dict[str, Any]:
    topology = public["topology"]
    total_gpus = int(topology["total_gpus"])
    decode_gpus = int(topology["decode_gpus"])
    median_tpot_ms = float(raw["median_tpot_ms"])
    metrics = {
        "median_intvty": 1000.0 / median_tpot_ms,
        "tput_per_gpu": float(raw["total_token_throughput"]) / total_gpus,
        "output_tput_per_gpu": float(raw["output_throughput"]) / decode_gpus,
    }
    for field in LATENCY_FIELDS:
        metrics[field] = float(raw[f"{field}_ms"]) / 1000.0

    public_metrics = public["metrics"]
    deltas = {field: percent_delta(metrics[field], float(public_metrics[field])) for field in REPORT_FIELDS}
    failed = _failed_requests(raw)
    token_counts_complete = _token_counts_complete(raw)
    flags = []
    if failed:
        flags.append("request_failures")
    if not token_counts_complete:
        flags.append("incomplete_token_counts")
    if abs(deltas["tput_per_gpu"]) > 5.0:
        flags.append("throughput_delta_over_5pct")

    return {
        "profile": public["profile"],
        "concurrency": int(raw["max_concurrency"]),
        "topology": topology,
        "metrics": metrics,
        "deltas_percent": deltas,
        "failed_requests": failed,
        "completed_requests": int(raw.get("completed", 0)),
        "expected_requests": int(raw.get("num_prompts", 0)),
        "achieved_request_rate": float(raw.get("request_throughput", 0.0)),
        "token_counts_complete": token_counts_complete,
        "flags": flags,
        "rerun_recommended": bool(flags),
        "source_file": str(path),
    }


def load_candidate_points(input_dir: Path, public_curve: dict[str, Any]) -> list[dict[str, Any]]:
    public_by_topology = {_topology_key(point): point for point in public_curve["points"]}
    candidates = []
    for path in sorted(input_dir.rglob("results_concurrency_*.json")):
        raw = json.loads(path.read_text())
        key = _candidate_topology(path, raw)
        if key not in public_by_topology:
            raise ValueError(f"no public point matches candidate topology {key}: {path}")
        candidates.append(process_result(path, raw, public_by_topology[key]))

    attempts: Counter[tuple[str, int]] = Counter()
    for point in candidates:
        key = (point["profile"], point["concurrency"])
        attempts[key] += 1
        point["attempt"] = attempts[key]
    return candidates


def write_csv(points: list[dict[str, Any]], output: Path) -> None:
    fields = [
        "profile",
        "concurrency",
        "attempt",
        "prefill_gpus",
        "decode_gpus",
        "total_gpus",
        *REPORT_FIELDS,
        "tput_per_gpu_delta_percent",
        "median_intvty_delta_percent",
        "failed_requests",
        "achieved_request_rate",
        "token_counts_complete",
        "flags",
        "source_file",
    ]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in points:
            row = {
                "profile": point["profile"],
                "concurrency": point["concurrency"],
                "attempt": point["attempt"],
                **{key: point["topology"][key] for key in ("prefill_gpus", "decode_gpus", "total_gpus")},
                **point["metrics"],
                "tput_per_gpu_delta_percent": point["deltas_percent"]["tput_per_gpu"],
                "median_intvty_delta_percent": point["deltas_percent"]["median_intvty"],
                "failed_requests": point["failed_requests"],
                "achieved_request_rate": point["achieved_request_rate"],
                "token_counts_complete": point["token_counts_complete"],
                "flags": ";".join(point["flags"]),
                "source_file": point["source_file"],
            }
            writer.writerow(row)


def _scale(value: float, low: float, high: float, start: float, size: float) -> float:
    if high == low:
        return start + size / 2
    return start + ((value - low) / (high - low)) * size


def write_svg(
    public_points: list[dict[str, Any]],
    candidate_points: list[dict[str, Any]],
    output: Path,
) -> None:
    left, top, plot_width, plot_height = 100, 60, 920, 540
    all_points = [
        (float(point["metrics"]["median_intvty"]), float(point["metrics"]["tput_per_gpu"]))
        for point in [*public_points, *candidate_points]
    ]
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_low, x_high = min(x_values) * 0.95, max(x_values) * 1.05
    y_low, y_high = 0.0, max(y_values) * 1.08

    def coordinates(point: dict[str, Any]) -> tuple[float, float]:
        metrics = point["metrics"]
        x = _scale(float(metrics["median_intvty"]), x_low, x_high, left, plot_width)
        y = top + plot_height - _scale(float(metrics["tput_per_gpu"]), y_low, y_high, 0, plot_height)
        return x, y

    def polyline(points: list[dict[str, Any]], color: str) -> str:
        ordered = sorted(points, key=lambda point: float(point["metrics"]["median_intvty"]))
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(coordinates, ordered))
        return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>'

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="700" viewBox="0 0 1100 700">',
        '<rect width="1100" height="700" fill="white"/>',
        '<text x="550" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">GB300 SGLang 8K/1K: native gRPC sidecar vs public</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        f'<text x="{left + plot_width / 2}" y="655" text-anchor="middle" font-family="sans-serif">Median interactivity (tok/s/user)</text>',
        f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_height / 2})" font-family="sans-serif">Total throughput / GPU (tok/s/GPU)</text>',
        polyline(public_points, "#1f2937"),
        polyline(candidate_points, "#dc2626"),
    ]
    for points, color, label_prefix in (
        (public_points, "#1f2937", "public"),
        (candidate_points, "#dc2626", "sidecar"),
    ):
        for point in points:
            x, y = coordinates(point)
            label = html.escape(f"{label_prefix} {point['profile']} c{point['concurrency']}")
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"/>')
            lines.append(
                f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" font-family="sans-serif" font-size="10" fill="{color}">{label}</text>'
            )
    lines.extend(
        [
            '<rect x="815" y="75" width="14" height="4" fill="#1f2937"/><text x="838" y="82" font-family="sans-serif" font-size="12">Public</text>',
            '<rect x="900" y="75" width="14" height="4" fill="#dc2626"/><text x="923" y="82" font-family="sans-serif" font-size="12">Native gRPC sidecar</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def write_markdown(
    public_curve: dict[str, Any],
    points: list[dict[str, Any]],
    output: Path,
) -> None:
    regressed = sum(point["deltas_percent"]["tput_per_gpu"] < -5.0 for point in points)
    systematic = bool(points) and regressed > len(points) / 2
    lines = [
        "# GB300 SGLang native-gRPC sidecar 8K/1K Pareto",
        "",
        f"Public snapshot: {public_curve['source']['result_date']} run, retrieved {public_curve['source']['retrieved_at']}.",
        "",
        "| Profile | C | Attempt | P/D GPUs | Interactivity | Δ intvty | Tput/GPU | Δ tput | Output/GPU | TTFT med/p99 (s) | TPOT med/p99 (ms) | E2E med/p99 (s) | Fail | Req/s | Flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for point in points:
        metrics = point["metrics"]
        delta = point["deltas_percent"]
        topology = point["topology"]
        lines.append(
            "| {profile} | {concurrency} | {attempt} | {prefill}/{decode} | {intvty:.2f} | {dint:+.2f}% | "
            "{tput:.2f} | {dtput:+.2f}% | {output:.2f} | {ttft:.3f}/{p99ttft:.3f} | "
            "{tpot:.3f}/{p99tpot:.3f} | {e2e:.3f}/{p99e2e:.3f} | {failed} | {rate:.2f} | {flags} |".format(
                profile=point["profile"],
                concurrency=point["concurrency"],
                attempt=point["attempt"],
                prefill=topology["prefill_gpus"],
                decode=topology["decode_gpus"],
                intvty=metrics["median_intvty"],
                dint=delta["median_intvty"],
                tput=metrics["tput_per_gpu"],
                dtput=delta["tput_per_gpu"],
                output=metrics["output_tput_per_gpu"],
                ttft=metrics["median_ttft"],
                p99ttft=metrics["p99_ttft"],
                tpot=metrics["median_tpot"] * 1000,
                p99tpot=metrics["p99_tpot"] * 1000,
                e2e=metrics["median_e2el"],
                p99e2e=metrics["p99_e2el"],
                failed=point["failed_requests"],
                rate=point["achieved_request_rate"],
                flags=", ".join(point["flags"]) or "—",
            )
        )
    lines.extend(
        [
            "",
            "No formal pass/fail threshold is applied. A point is marked for one rerun when it has failures, inconsistent output-token accounting, or an absolute throughput delta above 5%.",
            "",
            "The comparison combines gRPC transport overhead with the unavoidable SGLang build difference; it does not isolate transport overhead.",
        ]
    )
    if systematic:
        lines.extend(
            [
                "",
                "More than half of measured attempts regress throughput by over 5%. Run one same-build in-process control point before changing topology or tuning.",
            ]
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--public-curve", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    public_curve = json.loads(args.public_curve.read_text())
    points = load_candidate_points(args.input_dir, public_curve)
    if not points:
        raise SystemExit(f"no results_concurrency_*.json files found under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "public_source": public_curve["source"],
        "candidate_points": points,
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    write_csv(points, args.output_dir / "comparison.csv")
    write_svg(public_curve["points"], points, args.output_dir / "pareto-overlay.svg")
    write_markdown(public_curve, points, args.output_dir / "report.md")


if __name__ == "__main__":
    main()
