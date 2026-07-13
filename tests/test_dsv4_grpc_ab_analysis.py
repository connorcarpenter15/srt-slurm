# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DeepSeek-V4-Pro crossover result analyzer."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path("campaigns/dsv4-pro-gb300-grpc-ab/analyze-results.py")


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("dsv4_grpc_ab_analysis", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(root: Path, spec: dict, throughput: float, *, valid: bool = True) -> None:
    run_dir = root / spec["artifact_dir"]
    run_dir.mkdir(parents=True)
    raw = {
        "completed": 2 if valid else 1,
        "request_throughput": throughput / 1000,
        "total_token_throughput": throughput,
        "output_throughput": throughput / 4,
        "total_input_tokens": 16_384 if valid else 8_192,
        "total_output_tokens": 2_048 if valid else 1_024,
        "input_lens": [8_192, 8_192],
        "output_lens": [1_024, 1_024 if valid else 0],
        "errors": ["", "" if valid else "request failed"],
        "median_ttft_ms": 1_000,
        "p99_ttft_ms": 2_000,
        "median_tpot_ms": 10,
        "p99_tpot_ms": 20,
        "median_itl_ms": 11,
        "p99_itl_ms": 21,
        "median_e2el_ms": 3_000,
        "p99_e2el_ms": 4_000,
    }
    (run_dir / "results_concurrency_1.json").write_text(json.dumps(raw))
    validation = {
        "expected_worker_registrations": 2,
        "observed_worker_registrations": 2,
        "mooncake_kv_transfer": True,
        "fatal_engine_errors": 0,
        "fatal_mooncake_errors": 0,
        "fatal_nccl_errors": 0,
        "fatal_grpc_errors": 0,
        "fatal_sidecar_errors": 0,
    }
    (run_dir / "validation.json").write_text(json.dumps(validation))
    (run_dir / "gpu-telemetry-node.csv").write_text(
        "timestamp,index,uuid,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu\n"
        "now,0,GPU-0,90,1000,500,1500,2000,60\n"
    )
    (run_dir / "gpu-throttle-node.csv").write_text(
        "timestamp,index,uuid,sw_power_cap,hw_slowdown,hw_thermal_slowdown,sw_thermal_slowdown\n"
        "now,0,GPU-0,Not Active,Not Active,Not Active,Not Active\n"
    )


def inputs() -> tuple[dict, dict, dict]:
    manifest = {
        "source_commits": {"dynamo": "d", "sglang": "s", "inferencex": "i"},
        "image": "image",
        "model": "model",
        "model_revision": "model-revision",
        "model_path": "model-path",
        "points": [
            {
                "id": "c00001",
                "concurrency": 1,
                "prefill_nodes": 1,
                "decode_nodes": 1,
                "total_gpus": 8,
            }
        ],
    }
    run_plan = {
        "runs": [
            {
                "sequence": index,
                "point": "c00001",
                "pair": pair,
                "order_index": order,
                "backend": backend,
                "recipe": f"{backend}.yaml",
                "artifact_dir": f"raw/pair-{pair}/{order}-{backend}",
                "retry_artifact_dir": f"raw/pair-{pair}-retry-1/{order}-{backend}",
            }
            for index, (pair, order, backend) in enumerate(
                ((1, 1, "legacy"), (1, 2, "sidecar"), (2, 1, "sidecar"), (2, 2, "legacy")),
                start=1,
            )
        ]
    }
    public = {
        "source": {"retrieved_at": "2026-07-13"},
        "points": [
            {
                "id": "public-1",
                "concurrency": 1,
                "topology": {"prefill_gpus": 4, "decode_gpus": 4, "total_gpus": 8},
                "metrics": {"median_intvty": 90, "tput_per_gpu": 100},
            }
        ],
    }
    return manifest, run_plan, public


def test_analysis_calculates_paired_mean_delta_order_effect_and_telemetry(tmp_path: Path) -> None:
    module = load_analysis_module()
    manifest, run_plan, public = inputs()
    throughputs = (800, 880, 968, 840)
    for spec, throughput in zip(run_plan["runs"], throughputs, strict=True):
        write_run(tmp_path, spec, throughput)

    comparison = module.build_comparison(
        tmp_path,
        manifest,
        run_plan,
        public,
        allow_incomplete=False,
    )

    point = comparison["points"][0]
    assert point["paired_means"]["legacy"]["tput_per_gpu"] == 102.5
    assert point["paired_means"]["sidecar"]["tput_per_gpu"] == 115.5
    assert point["sidecar_vs_legacy_percent"]["tput_per_gpu"] == 100 * 13 / 102.5
    assert point["throughput_order_effect_percent"]["legacy"] == 100 * (100 - 105) / 105
    assert point["throughput_order_effect_percent"]["sidecar"] == 100 * (121 - 110) / 110
    assert comparison["geometric_mean_sidecar_legacy_tput_per_gpu_ratio"] == pytest.approx(115.5 / 102.5)
    assert comparison["runs"][0]["telemetry"]["mean_power_draw_watts"] == 500
    assert comparison["runs"][0]["valid"] is True
    assert comparison["runs"][0]["selected_for_pair_mean"] is True
    assert point["paired_telemetry_means"]["legacy"]["mean_sm_clock_mhz"] == 1500

    module.write_runs_csv(comparison, tmp_path / "runs.csv")
    module.write_svg(comparison, public, tmp_path / "pareto.svg")
    module.write_markdown(comparison, tmp_path / "report.md")
    assert (tmp_path / "runs.csv").is_file()
    assert (tmp_path / "pareto.svg").is_file()
    assert "## Per-run results" in (tmp_path / "report.md").read_text()


def test_invalid_run_is_excluded_and_missing_campaign_is_rejected(tmp_path: Path) -> None:
    module = load_analysis_module()
    manifest, run_plan, public = inputs()
    write_run(tmp_path, run_plan["runs"][0], 800, valid=False)

    comparison = module.build_comparison(
        tmp_path,
        manifest,
        run_plan,
        public,
        allow_incomplete=True,
    )
    run = comparison["runs"][0]
    assert run["valid"] is False
    assert "request_failures" in run["flags"]
    assert "incomplete_token_counts" in run["flags"]
    assert comparison["points"][0]["paired_means"]["legacy"] is None

    try:
        module.build_comparison(
            tmp_path,
            manifest,
            run_plan,
            public,
            allow_incomplete=False,
        )
    except ValueError as error:
        assert "missing 3 planned runs" in str(error)
    else:
        raise AssertionError("incomplete campaign was accepted")


def test_complete_retry_pair_replaces_invalid_pair_without_mixing_legs(tmp_path: Path) -> None:
    module = load_analysis_module()
    manifest, run_plan, public = inputs()
    primary_throughputs = (800, 880, 968, 840)
    for spec, throughput in zip(run_plan["runs"], primary_throughputs, strict=True):
        write_run(tmp_path, spec, throughput, valid=spec["sequence"] != 2)

    for spec, throughput in zip(run_plan["runs"][:2], (900, 990), strict=True):
        retry_spec = {**spec, "artifact_dir": spec["retry_artifact_dir"]}
        write_run(tmp_path, retry_spec, throughput)

    comparison = module.build_comparison(
        tmp_path,
        manifest,
        run_plan,
        public,
        allow_incomplete=False,
    )

    point = comparison["points"][0]
    assert point["pair_attempts"]["1"]["selected_attempt"] == 2
    assert point["pair_attempts"]["2"]["selected_attempt"] == 1
    assert point["paired_means"]["legacy"]["tput_per_gpu"] == (112.5 + 105) / 2
    assert point["paired_means"]["sidecar"]["tput_per_gpu"] == (123.75 + 121) / 2
    assert comparison["collected_attempt_runs"] == 6
    invalid_primary = [run for run in comparison["runs"] if run["sequence"] == 2 and run["attempt"] == 1]
    assert len(invalid_primary) == 1
    assert invalid_primary[0]["valid"] is False
    assert invalid_primary[0]["selected_for_pair_mean"] is False
