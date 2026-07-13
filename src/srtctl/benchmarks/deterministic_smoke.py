# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic completion capture used by engine-architecture smoke tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("deterministic-smoke")
class DeterministicSmokeRunner(BenchmarkRunner):
    @property
    def name(self) -> str:
        return "Deterministic output smoke"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/deterministic-smoke/run.py"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "deterministic-smoke")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        if config.benchmark.isl is None:
            errors.append("benchmark.isl is required for deterministic-smoke")
        if config.benchmark.osl is None:
            errors.append("benchmark.osl is required for deterministic-smoke")
        if not config.benchmark.custom_tokenizer:
            errors.append("benchmark.custom_tokenizer is required for deterministic-smoke")
        return errors

    def build_command(self, config: SrtConfig, runtime: RuntimeContext) -> list[str]:
        return [
            "python3",
            self.script_path,
            "--url",
            f"http://localhost:{runtime.frontend_port}/v1/completions",
            "--model",
            config.served_model_name,
            "--model-path",
            "/model",
            "--tokenizer",
            config.benchmark.custom_tokenizer or "",
            "--tokenizer-root",
            "/srtctl-benchmarks/sa-bench",
            "--isl",
            str(config.benchmark.isl),
            "--osl",
            str(config.benchmark.osl),
            "--output",
            "/logs/deterministic-smoke/deterministic-output.json",
        ]
