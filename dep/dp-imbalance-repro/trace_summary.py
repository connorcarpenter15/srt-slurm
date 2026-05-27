#!/usr/bin/env python3
"""Summarize SA-Bench client traces joined with Dynamo request-trace logs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import Counter, defaultdict
from json import JSONDecoder
from pathlib import Path
from typing import Any

TRACE_MARKER = "dynamo_request_trace"
JSON_DECODER = JSONDecoder()


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
    stripped = line.strip()
    if stripped.startswith("{"):
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            event = None
        if isinstance(event, dict) and "event" in event:
            return event

    marker_pos = line.find(TRACE_MARKER)
    if marker_pos < 0:
        return None
    json_pos = line.find("{", marker_pos)
    if json_pos < 0:
        return None
    try:
        event, _ = JSON_DECODER.raw_decode(line[json_pos:])
        return event if isinstance(event, dict) else None
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


def expand_input_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        path_text = str(path)
        if any(char in path_text for char in "*?["):
            expanded.extend(Path(match) for match in sorted(glob.glob(path_text)))
        else:
            expanded.append(path)
    return expanded


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


def event_rank(event: dict[str, Any] | None) -> str:
    if event is None:
        return "unknown"
    rank = event.get("dp_rank")
    return "unknown" if rank is None else str(rank)


def print_value_summary(prefix: str, values: list[float]) -> None:
    summary = summarize_values(values)
    if summary is None:
        return
    print(
        f"- {prefix}: count={summary['count']:.0f} "
        f"mean={summary['mean']:.6f} p50={summary['p50']:.6f} "
        f"p95={summary['p95']:.6f} max={summary['max']:.6f}"
    )


def write_csv_outputs(
    csv_dir: Path,
    rows: list[dict[str, Any]],
    admission_rows: list[dict[str, Any]],
) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (csv_dir / "request_join.csv").open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if admission_rows:
        with (csv_dir / "admissions_per_second.csv").open(
            "w", encoding="utf-8", newline=""
        ) as outfile:
            writer = csv.DictWriter(outfile, fieldnames=sorted(admission_rows[0].keys()))
            writer.writeheader()
            writer.writerows(admission_rows)


def print_summary(
    client_events: list[dict[str, Any]],
    server_events: list[dict[str, Any]],
    *,
    csv_dir: Path | None = None,
) -> None:
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
    backend_by_source = Counter(
        Path(str(event.get("source_log", "unknown"))).name for event in backend_enters
    )
    joined_backend_by_source: Counter[str] = Counter()
    joined_backend_by_rank: Counter[str] = Counter()

    deltas: dict[str, list[float]] = defaultdict(list)
    deltas_by_rank: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    deltas_by_source: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    router_fields: dict[str, list[float]] = defaultdict(list)
    router_fields_by_rank: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    request_plane_fields: dict[str, list[float]] = defaultdict(list)
    joined_rows: list[dict[str, Any]] = []
    for request_id, request_events in by_request.items():
        submit = first_event(request_events, "client_submit")
        client_first = first_event(request_events, "client_first_token")
        router = first_event(request_events, "router_assigned")
        request_plane_enqueue = first_event(request_events, "request_plane_enqueue")
        request_plane_send_start = first_event(request_events, "request_plane_send_start")
        request_plane_send_done = first_event(request_events, "request_plane_send_done")
        request_plane_first = first_event(request_events, "request_plane_first_response")
        backend_enter = first_event(request_events, "backend_dp_enter")
        backend_first = first_event(request_events, "backend_dp_first_token")
        backend_done = first_event(request_events, "backend_dp_done")
        rank = event_rank(backend_enter if backend_enter is not None else router)
        if submit is not None and backend_enter is not None:
            joined_backend_by_rank[rank] += 1
            joined_backend_by_source[
                Path(str(backend_enter.get("source_log", "unknown"))).name
            ] += 1

        row: dict[str, Any] = {
            "request_id": request_id,
            "router_dp_rank": event_rank(router),
            "backend_dp_rank": event_rank(backend_enter),
        }

        for name, later, earlier in (
            ("client_ttft_s", client_first, submit),
            ("router_to_backend_enter_s", backend_enter, router),
            (
                "request_plane_enqueue_to_send_s",
                request_plane_send_start,
                request_plane_enqueue,
            ),
            (
                "request_plane_send_wall_s",
                request_plane_send_done,
                request_plane_send_start,
            ),
            (
                "request_plane_roundtrip_first_response_wall_s",
                request_plane_first,
                request_plane_send_start,
            ),
            (
                "request_plane_first_response_to_backend_first_token_s",
                backend_first,
                request_plane_first,
            ),
            ("backend_enter_to_first_token_s", backend_first, backend_enter),
            ("backend_duration_s", backend_done, backend_enter),
        ):
            delta = ns_delta_s(later, earlier)
            if delta is not None:
                deltas[name].append(delta)
                deltas_by_rank[rank][name].append(delta)
                row[name] = delta
                if backend_enter is not None and name.startswith("backend_"):
                    source_log = Path(
                        str(backend_enter.get("source_log", "unknown"))
                    ).name
                    deltas_by_source[source_log][name].append(delta)

        for request_plane_event in (
            request_plane_send_start,
            request_plane_send_done,
            request_plane_first,
        ):
            if request_plane_event is None:
                continue
            for source_key in (
                "queue_seconds",
                "send_seconds",
                "roundtrip_ttft_seconds",
            ):
                value = request_plane_event.get(source_key)
                if value is not None:
                    request_plane_fields[source_key].append(float(value))
                    row[f"request_plane_{source_key}"] = float(value)

        if router is not None:
            for source_key, output_key, scale in (
                ("scheduler_queue_delay_ms", "scheduler_queue_delay_s", 0.001),
                ("selected_decode_blocks", "selected_decode_blocks", 1.0),
                ("selected_prefill_tokens", "selected_prefill_tokens", 1.0),
                ("state_snapshot_age_ms", "state_snapshot_age_s", 0.001),
                ("snapshot_age_ms", "snapshot_age_s", 0.001),
                ("load_snapshot_age_ms", "load_snapshot_age_s", 0.001),
                ("pending_count_at_admit", "pending_count_at_admit", 1.0),
                ("pending_isl_tokens_at_admit", "pending_isl_tokens_at_admit", 1.0),
            ):
                value = router.get(source_key)
                if value is not None:
                    scaled = float(value) * scale
                    router_fields[output_key].append(scaled)
                    router_fields_by_rank[event_rank(router)][output_key].append(scaled)
                    row[output_key] = scaled

        if len(row) > 3:
            joined_rows.append(row)

    admission_events = backend_enters if backend_enters else router_assignments
    admission_rows: list[dict[str, Any]] = []
    if admission_events:
        base_ns = min(int(event.get("wall_time_ns", 0)) for event in admission_events)
        bucketed: dict[int, Counter[str]] = defaultdict(Counter)
        for event in admission_events:
            wall_time = event.get("wall_time_ns")
            if wall_time is None:
                continue
            bucket = max(0, (int(wall_time) - base_ns) // 1_000_000_000)
            bucketed[int(bucket)][event_rank(event)] += 1

        ranks = sorted({rank for counts in bucketed.values() for rank in counts})
        max_bucket = max(bucketed) if bucketed else 0
        for bucket in range(max_bucket + 1):
            for rank in ranks:
                admission_rows.append(
                    {
                        "second": bucket,
                        "dp_rank": rank,
                        "admissions": bucketed[bucket][rank],
                    }
                )

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
    print("## Backend Enters By Source Log")
    print("Use this as the DP-process proxy when trace events do not carry dp_rank.")
    for source_log, count in sorted(backend_by_source.items()):
        print(f"- {source_log}: {count}")
    print()
    print("## Joined Client Requests By Backend Source Log")
    print("Measured-run requests only; source log is the DP-process proxy.")
    for source_log, count in sorted(joined_backend_by_source.items()):
        print(f"- {source_log}: {count}")
    print()
    print("## Joined Client Requests By Backend DP Rank")
    for rank, count in sorted(joined_backend_by_rank.items()):
        print(f"- dp_rank={rank}: {count}")
    print()
    print("## Timing Deltas")
    for name, values in sorted(deltas.items()):
        print_value_summary(name, values)
    print()
    print("## Timing Deltas By DP Rank")
    for rank, rank_deltas in sorted(deltas_by_rank.items()):
        print(f"### dp_rank={rank}")
        for name, values in sorted(rank_deltas.items()):
            print_value_summary(name, values)
    print()
    print("## Backend Timing Deltas By Source Log")
    for source_log, source_deltas in sorted(deltas_by_source.items()):
        print(f"### {source_log}")
        for name, values in sorted(source_deltas.items()):
            print_value_summary(name, values)
    print()
    print("## Router Admission Fields")
    for name, values in sorted(router_fields.items()):
        print_value_summary(name, values)
    print()
    print("## Request Plane Fields")
    for name, values in sorted(request_plane_fields.items()):
        print_value_summary(name, values)
    print()
    print("## Router Admission Fields By DP Rank")
    for rank, rank_fields in sorted(router_fields_by_rank.items()):
        print(f"### dp_rank={rank}")
        for name, values in sorted(rank_fields.items()):
            print_value_summary(name, values)
    print()
    print("## Admissions Per Second By DP Rank")
    admissions_by_rank: dict[str, list[float]] = defaultdict(list)
    for row in admission_rows:
        admissions_by_rank[str(row["dp_rank"])].append(float(row["admissions"]))
    for rank, values in sorted(admissions_by_rank.items()):
        print_value_summary(f"dp_rank={rank}", values)

    if csv_dir is not None:
        write_csv_outputs(csv_dir, joined_rows, admission_rows)
        print()
        print(f"csv_dir: {csv_dir}")


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
        help=(
            "Dynamo frontend/backend log containing dynamo_request_trace lines, "
            "or direct dynamo_request_trace_*.jsonl file. May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Optional directory for joined request and admission-rate CSV outputs.",
    )
    args = parser.parse_args()

    client_events: list[dict[str, Any]] = []
    for path in expand_input_paths(args.client_trace):
        client_events.extend(load_jsonl(path))
    server_events = load_server_logs(expand_input_paths(args.server_log))
    print_summary(client_events, server_events, csv_dir=args.csv_dir)


if __name__ == "__main__":
    main()
