# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for lifecycle-coupled native engine and Dynamo sidecars."""

import shlex
from typing import TYPE_CHECKING

from srtctl.ports import DYN_SYSTEM_PORT_BASE

if TYPE_CHECKING:
    from srtctl.core.topology import Process


def sidecar_grpc_port(base_port: int, process: "Process") -> int:
    """Return a deterministic gRPC port that stays unique for co-located workers."""
    port = base_port + max(process.sys_port - DYN_SYSTEM_PORT_BASE, 0)
    if not 1 <= port <= 65535:
        raise ValueError(f"sidecar_port must resolve between 1 and 65535, got {port}")
    return port


def build_sidecar_launch_command(
    *,
    engine: list[str],
    sidecar: list[str],
    grpc_port: int,
    engine_name: str,
    startup_timeout: int,
    rank_zero_only: bool = False,
) -> list[str]:
    """Run an engine and its sidecar together, stopping both when either exits."""
    if startup_timeout < 1:
        raise ValueError(f"sidecar_startup_timeout must be at least 1, got {startup_timeout}")

    rank_guard = ""
    if rank_zero_only:
        rank_guard = """if [[ "${SLURM_PROCID:-0}" != "0" ]]; then
    set +e
    wait "${ENGINE_PID}"
    status=$?
    set -e
    exit "${status}"
fi
"""

    compound = f"""set -euo pipefail
ENGINE_PID=
SIDECAR_PID=
cleanup() {{
    status=$?
    trap - EXIT INT TERM
    if [[ -n "${{SIDECAR_PID}}" ]] && kill -0 "${{SIDECAR_PID}}" 2>/dev/null; then kill "${{SIDECAR_PID}}" 2>/dev/null || true; fi
    if [[ -n "${{ENGINE_PID}}" ]] && kill -0 "${{ENGINE_PID}}" 2>/dev/null; then kill "${{ENGINE_PID}}" 2>/dev/null || true; fi
    if [[ -n "${{SIDECAR_PID}}" ]]; then wait "${{SIDECAR_PID}}" 2>/dev/null || true; fi
    if [[ -n "${{ENGINE_PID}}" ]]; then wait "${{ENGINE_PID}}" 2>/dev/null || true; fi
    exit "${{status}}"
}}
trap cleanup EXIT INT TERM
{shlex.join(engine)} &
ENGINE_PID=$!
{rank_guard}port_ready=0
for _ in $(seq 1 {startup_timeout}); do
    if ! kill -0 "${{ENGINE_PID}}" 2>/dev/null; then
        echo "{engine_name} exited before native gRPC became ready" >&2
        exit 1
    fi
    if (exec 3<>/dev/tcp/127.0.0.1/{grpc_port}) 2>/dev/null; then
        exec 3>&-
        port_ready=1
        break
    fi
    sleep 1
done
if [[ "${{port_ready}}" != 1 ]]; then
    echo "Timed out waiting for {engine_name} native gRPC on port {grpc_port}" >&2
    exit 1
fi
{shlex.join(sidecar)} &
SIDECAR_PID=$!
set +e
wait -n "${{ENGINE_PID}}" "${{SIDECAR_PID}}"
status=$?
set -e
if [[ "${{status}}" == 0 ]]; then status=1; fi
exit "${{status}}"
"""
    return ["bash", "-lc", compound]
