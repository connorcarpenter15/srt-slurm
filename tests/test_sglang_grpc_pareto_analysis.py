# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GB300 native-gRPC Pareto result processor."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("campaigns/sglang-sidecar-gb300-fp4-8k1k/analyze-results.py")


def test_analysis_uses_inferencex_formulas_and_retains_failures(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "report"
    input_dir.mkdir()
    public_path = tmp_path / "public.json"
    public = {
        "source": {"result_date": "2026-02-12", "retrieved_at": "2026-07-09"},
        "points": [
            {
                "profile": "low_latency",
                "concurrency": 4,
                "topology": {
                    "prefill_gpus": 4,
                    "decode_gpus": 16,
                    "total_gpus": 20,
                },
                "metrics": {
                    "median_intvty": 100.0,
                    "tput_per_gpu": 80.0,
                    "output_tput_per_gpu": 40.0,
                    "median_ttft": 1.0,
                    "p99_ttft": 2.0,
                    "median_tpot": 0.01,
                    "p99_tpot": 0.02,
                    "median_e2el": 3.0,
                    "p99_e2el": 4.0,
                },
            }
        ],
    }
    public_path.write_text(json.dumps(public))
    raw = {
        "max_concurrency": 4,
        "num_prompts": 2,
        "completed": 1,
        "request_throughput": 0.5,
        "total_token_throughput": 2000.0,
        "output_throughput": 800.0,
        "total_output_tokens": 1024,
        "output_lens": [1024],
        "median_ttft_ms": 1000.0,
        "p99_ttft_ms": 2000.0,
        "median_tpot_ms": 10.0,
        "p99_tpot_ms": 20.0,
        "median_e2el_ms": 3000.0,
        "p99_e2el_ms": 4000.0,
    }
    result_path = input_dir / "results_concurrency_4_gpus_20_ctx_4_gen_16.json"
    result_path.write_text(json.dumps(raw))

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-dir",
            str(input_dir),
            "--public-curve",
            str(public_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    comparison = json.loads((output_dir / "comparison.json").read_text())
    point = comparison["candidate_points"][0]
    assert point["metrics"]["median_intvty"] == 100.0
    assert point["metrics"]["tput_per_gpu"] == 100.0
    assert point["metrics"]["output_tput_per_gpu"] == 50.0
    assert point["metrics"]["median_tpot"] == 0.01
    assert point["deltas_percent"]["tput_per_gpu"] == 25.0
    assert point["failed_requests"] == 1
    assert point["token_counts_complete"] is True
    assert point["rerun_recommended"] is True
    assert point["flags"] == ["request_failures", "throughput_delta_over_5pct"]
    assert (output_dir / "comparison.csv").is_file()
    assert (output_dir / "pareto-overlay.svg").is_file()
    assert (output_dir / "report.md").is_file()
