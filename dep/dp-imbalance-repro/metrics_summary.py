#!/usr/bin/env python3
"""Summarize selected Dynamo Prometheus scrape snapshots."""

from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

GAUGE_NAMES = {
    "dynamo_frontend_queued_requests",
    "dynamo_frontend_inflight_requests",
}

HISTOGRAM_NAMES = {
    "dynamo_request_plane_queue_seconds",
    "dynamo_request_plane_send_seconds",
    "dynamo_request_plane_roundtrip_ttft_seconds",
    "dynamo_router_overhead_scheduling_ms",
}

STATE_FRESHNESS_TERMS = ("forward", "fresh", "stale", "age", "slot", "state")
SAMPLE_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$")
INDEX_RE = re.compile(r"__(\d+)\.prom$")


def scrape_sort_key(path: Path) -> tuple[int, str]:
    match = INDEX_RE.search(path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**12, path.name)


def parse_prom(path: Path) -> dict[str, float]:
    values: dict[str, float] = defaultdict(float)
    with path.open("r", encoding="utf-8", errors="replace") as infile:
        for line in infile:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = SAMPLE_RE.match(line)
            if not match:
                continue
            name = match.group("name")
            value = float(match.group("value"))
            values[name] += value
    return values


def summarize(values: list[float]) -> str:
    if not values:
        return "count=0"
    ordered = sorted(values)
    return (
        f"count={len(values)} mean={statistics.fmean(values):.6g} "
        f"p50={ordered[len(ordered) // 2]:.6g} "
        f"p95={ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]:.6g} "
        f"max={ordered[-1]:.6g} last={values[-1]:.6g}"
    )


def histogram_mean_delta(samples: list[dict[str, float]], base_name: str) -> float | None:
    sum_name = f"{base_name}_sum"
    count_name = f"{base_name}_count"
    first = next((sample for sample in samples if sample.get(count_name, 0) > 0), None)
    last = next(
        (sample for sample in reversed(samples) if sample.get(count_name, 0) > 0),
        None,
    )
    if first is None or last is None:
        return None
    count_delta = last.get(count_name, 0) - first.get(count_name, 0)
    sum_delta = last.get(sum_name, 0) - first.get(sum_name, 0)
    if count_delta <= 0:
        count = last.get(count_name, 0)
        return None if count <= 0 else last.get(sum_name, 0) / count
    return sum_delta / count_delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scrape_dir", type=Path)
    parser.add_argument(
        "--glob",
        default="frontend__*.prom",
        help="Prometheus snapshot glob relative to scrape_dir.",
    )
    args = parser.parse_args()

    paths = sorted(args.scrape_dir.glob(args.glob), key=scrape_sort_key)
    samples = [parse_prom(path) for path in paths]

    print("# Dynamo Metrics Summary")
    print()
    print(f"scrape_dir: {args.scrape_dir}")
    print(f"scrapes: {len(samples)}")
    if not samples:
        return

    print()
    print("## Gauges")
    for name in sorted(GAUGE_NAMES):
        series = [sample.get(name, 0.0) for sample in samples]
        print(f"- {name}: {summarize(series)}")

    print()
    print("## Histograms")
    for name in sorted(HISTOGRAM_NAMES):
        counts = [sample.get(f"{name}_count", 0.0) for sample in samples]
        sums = [sample.get(f"{name}_sum", 0.0) for sample in samples]
        mean_delta = histogram_mean_delta(samples, name)
        mean_text = "n/a" if mean_delta is None else f"{mean_delta:.6g}"
        print(
            f"- {name}: last_count={counts[-1]:.0f} last_sum={sums[-1]:.6g} "
            f"window_mean={mean_text}"
        )

    names = set().union(*(sample.keys() for sample in samples))
    freshness_names = sorted(
        name
        for name in names
        if any(term in name.lower() for term in STATE_FRESHNESS_TERMS)
    )
    print()
    print("## State Freshness Candidate Metrics")
    if freshness_names:
        for name in freshness_names:
            print(f"- {name}")
    else:
        print("- none")


if __name__ == "__main__":
    main()
