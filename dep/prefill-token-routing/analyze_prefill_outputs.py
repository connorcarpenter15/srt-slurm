#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

RESULT_FIELDS = (
    "duration",
    "completed",
    "total_input_tokens",
    "request_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
)

ACTIVE_PREFILL_RE = re.compile(
    r"^dynamo_frontend_worker_active_prefill_tokens\{(?P<labels>[^}]*)\}\s+(?P<value>[-+0-9.eE]+)$"
)
ASSIGNMENT_RE = re.compile(
    r"^dynamo_frontend_prefill_worker_assigned_(?P<kind>requests|input_tokens)_total"
    r"\{(?P<labels>[^}]*)\}\s+(?P<value>[-+0-9.eE]+)$"
)
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')


def parse_labels(label_text: str) -> dict[str, str]:
    return {key: value for key, value in LABEL_RE.findall(label_text)}


def maybe_float(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_match(text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return match.group(1)
    return None


def read_recipe_summary(output_dir: Path) -> dict[str, str | None]:
    summary: dict[str, str | None] = {
        "name": None,
        "router_mode": None,
        "worker_selection_policy": None,
    }
    for candidate in (output_dir / "recipe.lock.yaml", output_dir / "config.yaml"):
        if not candidate.exists():
            continue
        text = candidate.read_text(errors="replace")
        summary["name"] = summary["name"] or first_match(text, (r'^\s*name:\s*"?([^"\n]+)"?\s*$',))
        summary["router_mode"] = summary["router_mode"] or first_match(
            text,
            (
                r'^\s*router-mode:\s*"?([^"\n]+)"?\s*$',
                r'^\s*router_mode:\s*"?([^"\n]+)"?\s*$',
            ),
        )
        summary["worker_selection_policy"] = summary["worker_selection_policy"] or first_match(
            text,
            (
                r'^\s*router-worker-selection-policy:\s*"?([^"\n]+)"?\s*$',
                r'^\s*router_worker_selection_policy:\s*"?([^"\n]+)"?\s*$',
            ),
        )
    return summary


def result_files(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "logs").glob("sa-bench_*/results_*.json"))


def metrics_dir_for(result_file: Path) -> Path:
    metrics_name = result_file.stem.replace("results_", "metrics_trace_", 1)
    return result_file.parent / metrics_name


def parse_prefill_values(prom_file: Path) -> list[float]:
    values: list[float] = []
    for line in prom_file.read_text(errors="replace").splitlines():
        match = ACTIVE_PREFILL_RE.match(line)
        if not match:
            continue
        labels = parse_labels(match.group("labels"))
        if labels.get("worker_type") != "prefill":
            continue
        values.append(float(match.group("value")))
    return values


def metric_snapshot_sort_key(path: Path) -> tuple[int, int | str]:
    suffix = path.stem.rsplit("_", 1)[-1]
    if suffix == "baseline":
        return (0, 0)
    if suffix.isdigit():
        return (1, int(suffix))
    if suffix == "final":
        return (2, 0)
    return (3, suffix)


def metric_snapshot_files(metrics_dir: Path) -> list[Path]:
    return sorted(metrics_dir.glob("frontend__*.prom"), key=metric_snapshot_sort_key)


def parse_assignment_values(prom_file: Path) -> dict[str, dict[tuple[str, str], float]]:
    values: dict[str, dict[tuple[str, str], float]] = {"requests": {}, "input_tokens": {}}
    for line in prom_file.read_text(errors="replace").splitlines():
        match = ASSIGNMENT_RE.match(line)
        if not match:
            continue
        labels = parse_labels(match.group("labels"))
        worker_id = labels.get("worker_id")
        dp_rank = labels.get("dp_rank")
        if worker_id is None or dp_rank is None:
            continue
        values[match.group("kind")][(worker_id, dp_rank)] = float(match.group("value"))
    return values


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def imbalance(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"range": None, "normalized_range": None, "cv": None, "max_over_mean": None}
    mean = statistics.fmean(values)
    value_range = max(values) - min(values)
    return {
        "range": value_range,
        "normalized_range": value_range / mean if mean else None,
        "cv": statistics.pstdev(values) / mean if mean else None,
        "max_over_mean": max(values) / mean if mean else None,
    }


def summarize_assignment_counters(
    metrics_dir: Path,
    *,
    expected_workers: int,
    completed: float | int | None,
    total_input_tokens: float | int | None,
) -> dict[str, Any]:
    snapshots = [parse_assignment_values(path) for path in metric_snapshot_files(metrics_dir)]
    workers = sorted(
        {worker for snapshot in snapshots for kind in ("requests", "input_tokens") for worker in snapshot[kind]}
    )
    summary: dict[str, Any] = {
        "assignment_metric_snapshots": len(snapshots),
        "assignment_worker_count": len(workers),
        "assignment_missing_workers": max(0, expected_workers - len(workers)),
        "assignment_counter_resets": 0,
        "assigned_requests_total": None,
        "assigned_input_tokens_total": None,
        "assigned_request_range": None,
        "assigned_request_normalized_range": None,
        "assigned_request_cv": None,
        "assigned_request_max_over_mean": None,
        "assigned_token_range": None,
        "assigned_token_normalized_range": None,
        "assigned_token_cv": None,
        "assigned_token_max_over_mean": None,
        "assignment_window_count": 0,
        "assigned_token_window_median_normalized_range": None,
        "assigned_token_window_p95_normalized_range": None,
        "assigned_request_window_median_normalized_range": None,
        "assigned_request_window_p95_normalized_range": None,
        "assignment_request_reconciliation_error": None,
        "assignment_token_reconciliation_error": None,
        "assignment_reconciled": False,
        "assignment_workers": {},
    }
    if len(snapshots) < 2 or not workers:
        return summary

    resets = 0
    window_imbalance: dict[str, list[float]] = {"requests": [], "input_tokens": []}
    for previous, current in zip(snapshots, snapshots[1:], strict=False):
        for kind in ("requests", "input_tokens"):
            deltas: list[float] = []
            for worker in workers:
                before = previous[kind].get(worker, 0.0)
                after = current[kind].get(worker, before)
                if after < before:
                    resets += 1
                    continue
                deltas.append(after - before)
            if deltas and sum(deltas) > 0:
                normalized_range = imbalance(deltas)["normalized_range"]
                if normalized_range is not None:
                    window_imbalance[kind].append(normalized_range)

    first = snapshots[0]
    last = snapshots[-1]
    request_deltas = [last["requests"].get(worker, 0.0) - first["requests"].get(worker, 0.0) for worker in workers]
    token_deltas = [
        last["input_tokens"].get(worker, 0.0) - first["input_tokens"].get(worker, 0.0) for worker in workers
    ]
    negative_deltas = sum(value < 0 for value in request_deltas + token_deltas)
    resets += negative_deltas

    request_stats = imbalance(request_deltas)
    token_stats = imbalance(token_deltas)
    request_total = sum(request_deltas)
    token_total = sum(token_deltas)
    request_error = request_total - float(completed) if completed is not None else None
    token_error = token_total - float(total_input_tokens) if total_input_tokens is not None else None

    summary.update(
        {
            "assignment_counter_resets": resets,
            "assigned_requests_total": request_total,
            "assigned_input_tokens_total": token_total,
            "assigned_request_range": request_stats["range"],
            "assigned_request_normalized_range": request_stats["normalized_range"],
            "assigned_request_cv": request_stats["cv"],
            "assigned_request_max_over_mean": request_stats["max_over_mean"],
            "assigned_token_range": token_stats["range"],
            "assigned_token_normalized_range": token_stats["normalized_range"],
            "assigned_token_cv": token_stats["cv"],
            "assigned_token_max_over_mean": token_stats["max_over_mean"],
            "assignment_window_count": len(window_imbalance["input_tokens"]),
            "assigned_token_window_median_normalized_range": percentile(window_imbalance["input_tokens"], 0.5),
            "assigned_token_window_p95_normalized_range": percentile(window_imbalance["input_tokens"], 0.95),
            "assigned_request_window_median_normalized_range": percentile(window_imbalance["requests"], 0.5),
            "assigned_request_window_p95_normalized_range": percentile(window_imbalance["requests"], 0.95),
            "assignment_request_reconciliation_error": request_error,
            "assignment_token_reconciliation_error": token_error,
            "assignment_reconciled": (
                len(workers) == expected_workers and resets == 0 and request_error == 0 and token_error == 0
            ),
            "assignment_workers": {
                f"{worker_id}:{dp_rank}": {"requests": request_delta, "input_tokens": token_delta}
                for (worker_id, dp_rank), request_delta, token_delta in zip(
                    workers, request_deltas, token_deltas, strict=True
                )
            },
        }
    )
    return summary


def summarize_prefill_skew(metrics_dir: Path) -> dict[str, float | int | None]:
    snapshots = 0
    nonzero_snapshots = 0
    worker_count = 0
    skews_all: list[float] = []
    skews_nonzero: list[float] = []
    totals_all: list[float] = []

    snapshot_values: list[list[float]] = []
    for prom_file in metric_snapshot_files(metrics_dir):
        values = parse_prefill_values(prom_file)
        if not values:
            continue
        snapshot_values.append(values)
        snapshots += 1
        worker_count = max(worker_count, len(values))
        total = sum(values)
        skew = max(values) - min(values)
        totals_all.append(total)
        skews_all.append(skew)
        if total > 0:
            nonzero_snapshots += 1
            skews_nonzero.append(skew)

    def mean_or_none(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    peak_total = max(totals_all, default=0.0)
    plateau_skews: list[float] = []
    plateau_normalized: list[float] = []
    if peak_total > 0:
        for values in snapshot_values:
            if sum(values) < 0.9 * peak_total:
                continue
            skew = max(values) - min(values)
            plateau_skews.append(skew)
            mean = statistics.fmean(values)
            if mean > 0:
                plateau_normalized.append(skew / mean)

    return {
        "prefill_metric_snapshots": snapshots,
        "prefill_nonzero_snapshots": nonzero_snapshots,
        "prefill_worker_count": worker_count or None,
        "prefill_max_skew": max(skews_nonzero, default=None),
        "prefill_avg_skew_nonzero": mean_or_none(skews_nonzero),
        "prefill_avg_skew_all": mean_or_none(skews_all),
        "prefill_max_total_tokens": max(totals_all, default=None),
        "prefill_avg_total_tokens": mean_or_none(totals_all),
        "prefill_plateau_snapshots": len(plateau_skews),
        "prefill_plateau_mean_skew": mean_or_none(plateau_skews),
        "prefill_plateau_p95_skew": percentile(plateau_skews, 0.95),
        "prefill_plateau_mean_normalized_skew": mean_or_none(plateau_normalized),
        "prefill_plateau_p95_normalized_skew": percentile(plateau_normalized, 0.95),
    }


def summarize_result(output_dir: Path, result_file: Path, *, expected_prefill_workers: int) -> dict[str, Any]:
    data = json.loads(result_file.read_text())
    row: dict[str, Any] = {
        "job_id": output_dir.name,
        "result": result_file.name,
    }
    row.update(read_recipe_summary(output_dir))
    for field in RESULT_FIELDS:
        row[field] = maybe_float(data.get(field))

    metrics_dir = metrics_dir_for(result_file)
    if metrics_dir.exists():
        row.update(summarize_prefill_skew(metrics_dir))
        row.update(
            summarize_assignment_counters(
                metrics_dir,
                expected_workers=expected_prefill_workers,
                completed=row.get("completed"),
                total_input_tokens=row.get("total_input_tokens"),
            )
        )
    else:
        row.update(
            {
                "prefill_metric_snapshots": 0,
                "prefill_nonzero_snapshots": 0,
                "prefill_worker_count": None,
                "prefill_max_skew": None,
                "prefill_avg_skew_nonzero": None,
                "prefill_avg_skew_all": None,
                "prefill_max_total_tokens": None,
                "prefill_avg_total_tokens": None,
                "prefill_plateau_snapshots": 0,
                "prefill_plateau_mean_skew": None,
                "prefill_plateau_p95_skew": None,
                "prefill_plateau_mean_normalized_skew": None,
                "prefill_plateau_p95_normalized_skew": None,
            }
        )
        row.update(
            summarize_assignment_counters(
                metrics_dir,
                expected_workers=expected_prefill_workers,
                completed=row.get("completed"),
                total_input_tokens=row.get("total_input_tokens"),
            )
        )
    return row


def summarize_output(output_dir: Path, *, expected_prefill_workers: int) -> list[dict[str, Any]]:
    files = result_files(output_dir)
    if not files:
        return [{"job_id": output_dir.name, "error": "no results_*.json found"}]
    return [
        summarize_result(output_dir, result_file, expected_prefill_workers=expected_prefill_workers)
        for result_file in files
    ]


def resolve_output(path_or_job: str, output_root: Path) -> Path:
    path = Path(path_or_job)
    if path.exists():
        return path
    if path_or_job.isdigit():
        return output_root / path_or_job
    return path


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_markdown(rows: list[dict[str, Any]]) -> None:
    columns = (
        "job_id",
        "name",
        "worker_selection_policy",
        "completed",
        "total_input_tokens",
        "request_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "prefill_metric_snapshots",
        "prefill_nonzero_snapshots",
        "prefill_max_skew",
        "prefill_avg_skew_nonzero",
        "prefill_plateau_p95_normalized_skew",
        "assigned_token_normalized_range",
        "assigned_token_window_p95_normalized_range",
        "assigned_request_normalized_range",
        "assignment_worker_count",
        "assignment_reconciled",
    )
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        if "error" in row:
            print(f"| {row.get('job_id')} | {row['error']} |" + " |" * (len(columns) - 2))
            continue
        print("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize sa-bench result JSON and active-prefill-token skew for prefill routing jobs."
    )
    parser.add_argument("outputs", nargs="+", help="Output directories or numeric job IDs.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/connorc/work/srt-slurm/outputs"),
        help="Root used when an output argument is a numeric job ID.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a Markdown table.")
    parser.add_argument(
        "--expected-prefill-workers",
        type=int,
        default=6,
        help="Expected number of prefill worker/rank series (default: 6).",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for arg in args.outputs:
        rows.extend(
            summarize_output(
                resolve_output(arg, args.output_root),
                expected_prefill_workers=args.expected_prefill_workers,
            )
        )

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print_markdown(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
