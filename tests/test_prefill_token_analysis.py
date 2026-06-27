from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "dep" / "prefill-token-routing" / "analyze_prefill_outputs.py"
SPEC = importlib.util.spec_from_file_location("analyze_prefill_outputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)

TRACE_SCRIPT = SCRIPT.with_name("generate_prefill_gate_traces.py")
TRACE_SPEC = importlib.util.spec_from_file_location("generate_prefill_gate_traces", TRACE_SCRIPT)
assert TRACE_SPEC is not None and TRACE_SPEC.loader is not None
trace_generator = importlib.util.module_from_spec(TRACE_SPEC)
TRACE_SPEC.loader.exec_module(trace_generator)

SHADOW_SCRIPT = SCRIPT.with_name("analyze_shadow_divergence.py")
SHADOW_SPEC = importlib.util.spec_from_file_location("analyze_shadow_divergence", SHADOW_SCRIPT)
assert SHADOW_SPEC is not None and SHADOW_SPEC.loader is not None
shadow_analysis = importlib.util.module_from_spec(SHADOW_SPEC)
SHADOW_SPEC.loader.exec_module(shadow_analysis)


def assignment_lines(worker_id: int, rank: str, requests: int, tokens: int, active_tokens: int) -> str:
    return (
        "dynamo_frontend_prefill_worker_assigned_requests_total"
        f'{{worker_id="{worker_id}",dp_rank="{rank}"}} {requests}\n'
        "dynamo_frontend_prefill_worker_assigned_input_tokens_total"
        f'{{worker_id="{worker_id}",dp_rank="{rank}"}} {tokens}\n'
        "dynamo_frontend_worker_active_prefill_tokens"
        f'{{worker_id="{worker_id}",dp_rank="{rank}",worker_type="prefill"}} {active_tokens}\n'
    )


def write_snapshot(path: Path, workers: list[tuple[int, str, int, int, int]]) -> None:
    path.write_text("".join(assignment_lines(*worker) for worker in workers))


def test_assignment_summary_reconciles_and_reports_imbalance(tmp_path: Path) -> None:
    write_snapshot(
        tmp_path / "frontend__baseline.prom",
        [(10, "none", 2, 128, 0), (20, "0", 1, 64, 0)],
    )
    write_snapshot(
        tmp_path / "frontend__0.prom",
        [(10, "none", 3, 256, 128), (20, "0", 3, 320, 256)],
    )
    write_snapshot(
        tmp_path / "frontend__final.prom",
        [(10, "none", 5, 640, 512), (20, "0", 4, 896, 448)],
    )

    summary = analysis.summarize_assignment_counters(
        tmp_path,
        expected_workers=2,
        completed=6,
        total_input_tokens=1344,
    )

    assert summary["assignment_worker_count"] == 2
    assert summary["assignment_missing_workers"] == 0
    assert summary["assigned_requests_total"] == 6
    assert summary["assigned_input_tokens_total"] == 1344
    assert summary["assignment_window_count"] == 2
    assert summary["assigned_token_normalized_range"] == 320 / 672
    assert summary["assignment_reconciled"] is True
    assert summary["assignment_workers"] == {
        "10:none": {"requests": 3.0, "input_tokens": 512.0},
        "20:0": {"requests": 3.0, "input_tokens": 832.0},
    }


def test_assignment_summary_flags_reset_missing_worker_and_reconciliation(tmp_path: Path) -> None:
    write_snapshot(tmp_path / "frontend__baseline.prom", [(10, "0", 5, 500, 0)])
    write_snapshot(tmp_path / "frontend__final.prom", [(10, "0", 4, 400, 0)])

    summary = analysis.summarize_assignment_counters(
        tmp_path,
        expected_workers=2,
        completed=4,
        total_input_tokens=400,
    )

    assert summary["assignment_missing_workers"] == 1
    assert summary["assignment_counter_resets"] > 0
    assert summary["assignment_reconciled"] is False


def test_prefill_plateau_uses_ninety_percent_of_peak(tmp_path: Path) -> None:
    write_snapshot(
        tmp_path / "frontend__baseline.prom",
        [(10, "0", 0, 0, 10), (20, "0", 0, 0, 0)],
    )
    write_snapshot(
        tmp_path / "frontend__0.prom",
        [(10, "0", 0, 0, 100), (20, "0", 0, 0, 100)],
    )
    write_snapshot(
        tmp_path / "frontend__final.prom",
        [(10, "0", 0, 0, 120), (20, "0", 0, 0, 80)],
    )

    summary = analysis.summarize_prefill_skew(tmp_path)

    assert summary["prefill_plateau_snapshots"] == 2
    assert summary["prefill_plateau_mean_skew"] == 20
    assert summary["prefill_plateau_p95_normalized_skew"] == 0.38


def test_gate_trace_generator_is_balanced_unique_and_prefix_bearing() -> None:
    no_cache = trace_generator.no_cache_rows()
    prefix_reuse, shared_tokens, total_tokens = trace_generator.prefix_reuse_rows()

    assert len(no_cache) == trace_generator.REQUEST_COUNT
    assert len(prefix_reuse) == trace_generator.REQUEST_COUNT
    assert {row["input_length"] for row in no_cache} == set(trace_generator.BUCKETS)
    flattened_hashes = [hash_id for row in no_cache for hash_id in row["hash_ids"]]
    assert len(flattened_hashes) == len(set(flattened_hashes))
    assert shared_tokens / total_tokens == 1408 / 1920
    assert len({row["group_id"] for row in prefix_reuse}) == 32


def test_shadow_divergence_excludes_ties(tmp_path: Path) -> None:
    report = tmp_path / "shadow.jsonl"
    rows = [
        {
            "prefill_primary_policy": "default",
            "prefill_shadow_policy": "prefill-token-balance",
            "prefill_worker_idx": 0,
            "prefill_shadow_worker_idx": 1,
            "prefill_primary_tie_count": 1,
            "prefill_shadow_tie_count": 1,
        },
        {
            "prefill_primary_policy": "default",
            "prefill_shadow_policy": "prefill-token-balance",
            "prefill_worker_idx": 0,
            "prefill_shadow_worker_idx": 1,
            "prefill_primary_tie_count": 2,
            "prefill_shadow_tie_count": 1,
        },
        {
            "prefill_primary_policy": "default",
            "prefill_shadow_policy": "prefill-token-balance",
            "prefill_worker_idx": 2,
            "prefill_shadow_worker_idx": 2,
            "prefill_primary_tie_count": 1,
            "prefill_shadow_tie_count": 1,
        },
    ]
    report.write_text("".join(f"{json.dumps(row)}\n" for row in rows))

    summary = shadow_analysis.summarize(report)

    assert summary["total_requests"] == 3
    assert summary["comparable_unique_requests"] == 2
    assert summary["divergent_unique_requests"] == 1
    assert summary["unique_choice_divergence"] == 0.5
