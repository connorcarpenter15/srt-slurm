#!/usr/bin/env python3
"""Summarize SA-Bench client traces joined with Dynamo request-trace logs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TRACE_MARKER = "dynamo_request_trace"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return events


def extract_trace_json(line: str) -> dict[str, Any] | None:
    marker_pos = line.find(TRACE_MARKER)
    if marker_pos < 0:
        return None
    json_pos = line.find("{", marker_pos)
    if json_pos < 0:
        return None
    try:
        return json.loads(line[json_pos:])
    except json.JSONDecodeError:
        return None


def load_server_logs(paths: list[Path]) -> list[dict[str, Any]]:
    events = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as infile:
            for line in infile:
                event = extract_trace_json(line)
                if event is not None:
                    event["source_log"] = str(path)
                    events.append(event)
    return events


def first_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any] | None:
    candidates = [event for event in events if event.get("event") == event_name]
    if not candidates:
        return None
    return min(candidates, key=lambda event: event.get("wall_time_ns", 0))


def ns_delta_s(later: dict[str, Any] | None, earlier: dict[str, Any] | None) -> float | None:
    if later is None or earlier is None:
        return None
    later_ns = later.get("wall_time_ns")
    earlier_ns = earlier.get("wall_time_ns")
    if later_ns is None or earlier_ns is None:
        return None
    return (int(later_ns) - int(earlier_ns)) / 1_000_000_000


def summarize_values(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    values = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": values[len(values) // 2],
        "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
        "max": values[-1],
    }


def print_summary(client_events: list[dict[str, Any]], server_events: list[dict[str, Any]]) -> None:
    all_events = client_events + server_events
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in all_events:
        request_id = event.get("request_id")
        if request_id:
            by_request[str(request_id)].append(event)

    event_counts = Counter(str(event.get("event")) for event in all_events)
    router_assignments = [
        event for event in server_events if event.get("event") == "router_assigned"
    ]
    backend_enters = [
        event for event in server_events if event.get("event") == "backend_dp_enter"
    ]
    router_by_rank = Counter(str(event.get("dp_rank")) for event in router_assignments)
    backend_by_rank = Counter(str(event.get("dp_rank")) for event in backend_enters)

    deltas: dict[str, list[float]] = defaultdict(list)
    for request_events in by_request.values():
        submit = first_event(request_events, "client_submit")
        client_first = first_event(request_events, "client_first_token")
        router = first_event(request_events, "router_assigned")
        backend_enter = first_event(request_events, "backend_dp_enter")
        backend_first = first_event(request_events, "backend_dp_first_token")
        backend_done = first_event(request_events, "backend_dp_done")

        for name, later, earlier in (
            ("client_ttft_s", client_first, submit),
            ("router_to_backend_enter_s", backend_enter, router),
            ("backend_enter_to_first_token_s", backend_first, backend_enter),
            ("backend_duration_s", backend_done, backend_enter),
        ):
            delta = ns_delta_s(later, earlier)
            if delta is not None:
                deltas[name].append(delta)

    print("# Request Trace Summary")
    print()
    print(f"client_events: {len(client_events)}")
    print(f"server_events: {len(server_events)}")
    print(f"requests_seen: {len(by_request)}")
    print()
    print("## Event Counts")
    for event_name, count in sorted(event_counts.items()):
        print(f"- {event_name}: {count}")
    print()
    print("## Router Assignments By DP Rank")
    for rank, count in sorted(router_by_rank.items()):
        print(f"- dp_rank={rank}: {count}")
    print()
    print("## Backend Enters By DP Rank")
    for rank, count in sorted(backend_by_rank.items()):
        print(f"- dp_rank={rank}: {count}")
    print()
    print("## Timing Deltas")
    for name, values in sorted(deltas.items()):
        summary = summarize_values(values)
        if summary is None:
            continue
        print(
            f"- {name}: count={summary['count']:.0f} "
            f"mean={summary['mean']:.6f} p50={summary['p50']:.6f} "
            f"p95={summary['p95']:.6f} max={summary['max']:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client-trace",
        action="append",
        type=Path,
        default=[],
        help="SA-Bench request_trace_*.jsonl file. May be passed multiple times.",
    )
    parser.add_argument(
        "--server-log",
        action="append",
        type=Path,
        default=[],
        help="Dynamo/frontend/backend log file containing dynamo_request_trace lines.",
    )
    args = parser.parse_args()

    client_events: list[dict[str, Any]] = []
    for path in args.client_trace:
        client_events.extend(load_jsonl(path))
    server_events = load_server_logs(args.server_log)
    print_summary(client_events, server_events)


if __name__ == "__main__":
    main()
