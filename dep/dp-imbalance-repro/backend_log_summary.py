#!/usr/bin/env python3
"""Summarize per-DP vLLM engine queue metrics from Dynamo backend logs."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ENGINE_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)Z?.*"
    r"Engine (?P<rank>\d+): Avg prompt throughput: (?P<prompt>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<gen>[0-9.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[0-9.]+)%"
)


def parse_ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value).replace(tzinfo=dt.UTC)


def read_lines(path: Path | None) -> list[str]:
    if path is None:
        return sys.stdin.readlines()
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def quantiles(values: list[float]) -> tuple[float, float, float, float]:
    values = sorted(values)
    return (
        statistics.fmean(values),
        values[len(values) // 2],
        values[min(len(values) - 1, int(len(values) * 0.95))],
        values[-1],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", type=Path, help="Backend log path; stdin if omitted")
    parser.add_argument("--bin-seconds", type=int, default=10)
    args = parser.parse_args()

    rows = []
    for line in read_lines(args.log):
        line = ANSI_RE.sub("", line)
        match = ENGINE_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "ts": parse_ts(match.group("ts")),
                "rank": int(match.group("rank")),
                "prompt": float(match.group("prompt")),
                "gen": float(match.group("gen")),
                "running": int(match.group("running")),
                "waiting": int(match.group("waiting")),
                "kv": float(match.group("kv")),
            }
        )

    if not rows:
        print("No Engine metrics found.")
        return

    print("# Backend DP Queue Summary")
    print()
    print(f"samples: {len(rows)}")
    print(f"time_range_utc: {min(row['ts'] for row in rows).isoformat()} to {max(row['ts'] for row in rows).isoformat()}")
    print()

    by_rank: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_rank[int(row["rank"])].append(row)

    print("## Per Rank")
    for rank in sorted(by_rank):
        rank_rows = by_rank[rank]
        running = [float(row["running"]) for row in rank_rows]
        waiting = [float(row["waiting"]) for row in rank_rows]
        gen = [float(row["gen"]) for row in rank_rows]
        running_mean, running_p50, running_p95, running_max = quantiles(running)
        waiting_mean, waiting_p50, waiting_p95, waiting_max = quantiles(waiting)
        gen_mean, gen_p50, gen_p95, gen_max = quantiles(gen)
        print(
            f"- dp_rank={rank}: samples={len(rank_rows)} "
            f"running_mean={running_mean:.1f} running_p50={running_p50:.0f} "
            f"running_p95={running_p95:.0f} running_max={running_max:.0f} "
            f"waiting_mean={waiting_mean:.1f} waiting_p95={waiting_p95:.0f} "
            f"waiting_max={waiting_max:.0f} gen_mean={gen_mean:.1f} "
            f"gen_p50={gen_p50:.1f} gen_p95={gen_p95:.1f} gen_max={gen_max:.1f}"
        )
    print()

    bins: dict[int, list[dict[str, float]]] = defaultdict(list)
    first = min(row["ts"] for row in rows)
    for row in rows:
        offset = int((row["ts"] - first).total_seconds())
        bins[(offset // args.bin_seconds) * args.bin_seconds].append(row)

    skew_rows = []
    for offset, bin_rows in bins.items():
        latest_by_rank: dict[int, dict[str, float]] = {}
        for row in sorted(bin_rows, key=lambda item: item["ts"]):
            latest_by_rank[int(row["rank"])] = row
        if len(latest_by_rank) < 2:
            continue
        running = [int(row["running"]) for row in latest_by_rank.values()]
        waiting = [int(row["waiting"]) for row in latest_by_rank.values()]
        gen = [float(row["gen"]) for row in latest_by_rank.values()]
        skew_rows.append(
            {
                "offset": offset,
                "ranks": len(latest_by_rank),
                "running_skew": max(running) - min(running),
                "waiting_skew": max(waiting) - min(waiting),
                "gen_skew": max(gen) - min(gen),
                "running_max": max(running),
                "waiting_max": max(waiting),
            }
        )

    print("## Temporal Skew")
    for key in ("running_skew", "waiting_skew", "gen_skew"):
        values = [float(row[key]) for row in skew_rows]
        mean, p50, p95, max_value = quantiles(values)
        print(f"- {key}: mean={mean:.1f} p50={p50:.1f} p95={p95:.1f} max={max_value:.1f}")
    print()

    print("## Largest Waiting Skew Windows")
    for row in sorted(skew_rows, key=lambda item: (item["waiting_skew"], item["running_skew"]), reverse=True)[:10]:
        print(
            f"- t+{row['offset']}s: ranks={row['ranks']} "
            f"waiting_skew={row['waiting_skew']} running_skew={row['running_skew']} "
            f"waiting_max={row['waiting_max']} running_max={row['running_max']} "
            f"gen_skew={row['gen_skew']:.1f}"
        )


if __name__ == "__main__":
    main()
