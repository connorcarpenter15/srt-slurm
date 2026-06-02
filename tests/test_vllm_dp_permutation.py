# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SRT_DP_GPU_PERMUTATION control knob in the vLLM DP+EP backend.

The knob decouples ring position (= --data-parallel-rank = node_rank = nsys
rank{N} file) from the physical GPU (CUDA_VISIBLE_DEVICES), for the ring-vs-
hardware experiment in dep/dep-bubble/FINDINGS-2-NSYS.md. ring position must
stay sequential 0..N-1; only the backing GPU permutes.
"""

import pytest

from srtctl.backends import VLLMProtocol, VLLMServerConfig
from srtctl.core.topology import Endpoint


def _build_processes():
    backend = VLLMProtocol(
        vllm_config=VLLMServerConfig(
            aggregated={"data-parallel-size": 4, "enable-expert-parallel": True},
        )
    )
    endpoint = Endpoint(
        mode="agg",
        index=0,
        nodes=("node0",),
        gpu_indices=frozenset(range(4)),
        gpus_per_node=4,
    )
    return backend.endpoints_to_processes([endpoint])


def _rank_to_gpu(processes):
    # node_rank holds dp_rank (ring position); gpu_indices is the single backing GPU.
    return {p.node_rank: sorted(p.gpu_indices)[0] for p in processes}


def test_default_is_identity(monkeypatch):
    monkeypatch.delenv("SRT_DP_GPU_PERMUTATION", raising=False)
    procs = _build_processes()
    assert len(procs) == 4
    assert _rank_to_gpu(procs) == {0: 0, 1: 1, 2: 2, 3: 3}


def test_reverse_permutes_gpu_not_ring_position(monkeypatch):
    monkeypatch.setenv("SRT_DP_GPU_PERMUTATION", "reverse")
    procs = _build_processes()
    # Ring position (node_rank -> --data-parallel-rank -> nsys rank{N}) stays 0..3.
    assert sorted(p.node_rank for p in procs) == [0, 1, 2, 3]
    # Physical GPU is reversed: dp_rank r backed by GPU 3-r.
    assert _rank_to_gpu(procs) == {0: 3, 1: 2, 2: 1, 3: 0}
    # CUDA_VISIBLE_DEVICES (single GPU) is the permuted physical device, decoupled
    # from the ring position.
    cvd = {p.node_rank: p.cuda_visible_devices for p in procs}
    assert cvd == {0: "3", 1: "2", 2: "1", 3: "0"}


def test_explicit_permutation(monkeypatch):
    monkeypatch.setenv("SRT_DP_GPU_PERMUTATION", "2,0,3,1")
    procs = _build_processes()
    assert sorted(p.node_rank for p in procs) == [0, 1, 2, 3]
    assert _rank_to_gpu(procs) == {0: 2, 1: 0, 2: 3, 3: 1}


def test_invalid_permutation_rejected(monkeypatch):
    monkeypatch.setenv("SRT_DP_GPU_PERMUTATION", "0,1,2")  # missing GPU 3
    with pytest.raises(ValueError, match="not a permutation"):
        _build_processes()


def test_blank_permutation_is_identity(monkeypatch):
    monkeypatch.setenv("SRT_DP_GPU_PERMUTATION", "   ")
    procs = _build_processes()
    assert _rank_to_gpu(procs) == {0: 0, 1: 1, 2: 2, 3: 3}
