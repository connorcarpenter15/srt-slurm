# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SGLang native-gRPC sidecar launch mode."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.backends import SGLangProtocol, SGLangServerConfig
from srtctl.core.topology import Process


def _process(node: str, rank: int, mode: str = "prefill") -> Process:
    return Process(
        node=node,
        gpu_indices=frozenset(range(4)),
        sys_port=8081 + rank,
        http_port=30000 if rank == 0 else 0,
        endpoint_mode=mode,
        endpoint_index=0,
        node_rank=rank,
        bootstrap_port=30001 if mode == "prefill" else None,
    )


def _runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.model_path = Path("/models/DeepSeek-R1-0528-NVFP4-v2")
    runtime.is_hf_model = False
    return runtime


def test_prefill_leader_launches_native_grpc_and_sidecar() -> None:
    process = _process("node0", 0)
    backend = SGLangProtocol(
        native_grpc_sidecar=True,
        sglang_config=SGLangServerConfig(
            prefill={
                "disaggregation-mode": "prefill",
                "disaggregation-bootstrap-port": 30001,
                "tensor-parallel-size": 4,
            }
        ),
    )

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(process, [process], _runtime())

    assert command[:2] == ["bash", "-lc"]
    script = command[2]
    assert "python3 -m sglang.launch_server" in script
    assert "--model-path /model" in script
    assert "--grpc-port 50051" in script
    assert "dynamo-sglang-sidecar --sglang-endpoint 127.0.0.1:50051" in script
    assert "--bootstrap-host 10.0.0.1" in script
    assert "--sglang-connections 8" in script
    assert "--health-deadline-secs 1200" in script
    assert script.count("--disaggregation-mode prefill") == 1
    assert script.count("--disaggregation-bootstrap-port 30001") == 1
    assert "/dev/tcp/127.0.0.1/50051" in script
    assert "trap cleanup EXIT INT TERM" in script
    assert 'wait -n "${ENGINE_PID}" "${SIDECAR_PID}"' in script


def test_decode_leader_does_not_pass_bootstrap_host() -> None:
    process = _process("node0", 0, mode="decode")
    backend = SGLangProtocol(
        native_grpc_sidecar=True,
        sglang_config=SGLangServerConfig(decode={"tensor-parallel-size": 4}),
    )

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(process, [process], _runtime())

    script = command[2]
    assert "--disaggregation-mode decode" in script
    assert "--bootstrap-host" not in script


def test_distributed_follower_launches_engine_only() -> None:
    leader = _process("node0", 0, mode="decode")
    follower = _process("node1", 1, mode="decode")
    backend = SGLangProtocol(
        native_grpc_sidecar=True,
        sglang_config=SGLangServerConfig(decode={"tp-size": 8}),
    )

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(follower, [leader, follower], _runtime())

    assert command[:3] == ["python3", "-m", "sglang.launch_server"]
    assert command[command.index("--dist-init-addr") + 1] == "10.0.0.1:8300"
    assert command[command.index("--node-rank") + 1] == "1"
    assert "--grpc-port" not in command
    assert "dynamo-sglang-sidecar" not in command


def test_invalid_native_grpc_port_is_rejected() -> None:
    process = _process("node0", 0)
    backend = SGLangProtocol(native_grpc_sidecar=True, native_grpc_port=0)

    with (
        patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        pytest.raises(ValueError, match="native_grpc_port must be between 1 and 65535"),
    ):
        backend.build_worker_command(process, [process], _runtime())
