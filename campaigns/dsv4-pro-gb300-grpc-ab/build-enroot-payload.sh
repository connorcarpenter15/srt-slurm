#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly DYNAMO_BASE_COMMIT="beb91b0de5392af2bd36560b312c153e7dbed061"
readonly DYNAMO_COMMIT="ba4c325301b23e3c5b1c76d61a3185edeea2d039"
readonly SGLANG_COMMIT="e2728ac504c00e37a284c7248693857b894e40e7"
readonly BASE_IMAGE="lmsysorg/sglang:dev-cu13@sha256:4b140bc08eb4782b057109b084b6df94f74c3a66c6984ee383a1d6c3714994d5"
readonly BUILD_TIMESTAMP="${BUILD_TIMESTAMP:?BUILD_TIMESTAMP is required}"
readonly REUSE_BUILD_ARTIFACTS="${REUSE_BUILD_ARTIFACTS:-false}"
export CARGO_TARGET_DIR="/cargo-target"
readonly CARGO_TARGET_DIR

test "$(git -C /src/dynamo rev-parse HEAD)" = "${DYNAMO_COMMIT}"
test "$(git -C /src/sglang rev-parse HEAD)" = "${SGLANG_COMMIT}"

mkdir -p "/campaign-artifacts/build" "${CARGO_TARGET_DIR}"
if [[ "${REUSE_BUILD_ARTIFACTS}" != "true" ]]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        cmake \
        libclang-dev \
        libssl-dev \
        patchelf \
        pkg-config \
        protobuf-compiler
    rm -rf /var/lib/apt/lists/*

    python3 -m pip install --no-cache-dir \
        build \
        hatchling \
        'maturin[patchelf]>=1.9,<2' \
        'setuptools>=61.0' \
        'setuptools-rust>=1.10' \
        'setuptools-scm>=8.0' \
        tomlkit \
        wheel

    rm -f \
        /campaign-artifacts/build/ai_dynamo-*.whl \
        /campaign-artifacts/build/ai_dynamo_runtime-*.whl \
        /campaign-artifacts/build/sglang-*.whl \
        /campaign-artifacts/build/dynamo-sglang-sidecar \
        /campaign-artifacts/build/server-descriptor.bin \
        /campaign-artifacts/build/sidecar-descriptor.bin

    mkdir -p /tmp/sglang-sidecar /tmp/sglang-server
    cp /src/dynamo/lib/sglang-sidecar/proto/sglang.proto /tmp/sglang-sidecar/sglang.proto
    cp /src/sglang/proto/sglang/runtime/v1/sglang.proto /tmp/sglang-server/sglang.proto
    protoc \
        --experimental_allow_proto3_optional \
        -I /tmp/sglang-sidecar \
        --descriptor_set_out=/campaign-artifacts/build/sidecar-descriptor.bin \
        /tmp/sglang-sidecar/sglang.proto
    protoc \
        --experimental_allow_proto3_optional \
        -I /tmp/sglang-server \
        --descriptor_set_out=/campaign-artifacts/build/server-descriptor.bin \
        /tmp/sglang-server/sglang.proto
    cmp \
        /campaign-artifacts/build/sidecar-descriptor.bin \
        /campaign-artifacts/build/server-descriptor.bin
    sha256sum \
        /src/dynamo/lib/sglang-sidecar/proto/sglang.proto \
        /src/sglang/proto/sglang/runtime/v1/sglang.proto \
        /campaign-artifacts/build/server-descriptor.bin \
        > /campaign-artifacts/build/proto-sha256.txt

    (
        cd /src/dynamo
        cargo test \
            --locked \
            --package dynamo-sglang-sidecar
        cargo build \
            --locked \
            --release \
            --package dynamo-sglang-sidecar
    )
    cp "${CARGO_TARGET_DIR}/release/dynamo-sglang-sidecar" /campaign-artifacts/build/

    python3 -m build \
        --wheel \
        --no-isolation \
        --outdir /campaign-artifacts/build \
        /src/dynamo
    (
        cd /src/dynamo/lib/bindings/python
        maturin build \
            --locked \
            --release \
            --features 'kv-indexer,slot-tracker,select-service,mm-routing' \
            --out /campaign-artifacts/build
    )

    (
        cd /src/sglang/python
        env CARGO_TARGET_DIR=/campaign-artifacts/sglang-cargo-target \
            SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.0.0+e2728ac504c \
            SGLANG_BUILD_RUST_EXTS=grpc \
            python3 -m build --wheel --no-isolation
        cp dist/sglang-*.whl /campaign-artifacts/build/
    )
fi

sidecar_path="/campaign-artifacts/build/dynamo-sglang-sidecar"
dynamo_wheel_path="$(printf '%s\n' /campaign-artifacts/build/ai_dynamo-*-none-any.whl | head -n1)"
dynamo_runtime_wheel_path="$(printf '%s\n' /campaign-artifacts/build/ai_dynamo_runtime-*.whl | head -n1)"
sglang_wheel_path="$(printf '%s\n' /campaign-artifacts/build/sglang-*.whl | head -n1)"
descriptor_path="/campaign-artifacts/build/server-descriptor.bin"

for artifact in \
    "${sidecar_path}" \
    "${dynamo_wheel_path}" \
    "${dynamo_runtime_wheel_path}" \
    "${sglang_wheel_path}" \
    "${descriptor_path}"; do
    test -s "${artifact}"
done

sidecar_sha256="$(sha256sum "${sidecar_path}" | awk '{print $1}')"
dynamo_wheel_sha256="$(sha256sum "${dynamo_wheel_path}" | awk '{print $1}')"
dynamo_runtime_wheel_sha256="$(sha256sum "${dynamo_runtime_wheel_path}" | awk '{print $1}')"
sglang_wheel_sha256="$(sha256sum "${sglang_wheel_path}" | awk '{print $1}')"
descriptor_sha256="$(sha256sum "${descriptor_path}" | awk '{print $1}')"

python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
    "${dynamo_runtime_wheel_path}" \
    "${dynamo_wheel_path}" \
    "${sglang_wheel_path}"

install -d /usr/local/libexec /usr/local/share/dynamo-sglang-sidecar
python3 -m pip uninstall --yes moviepy
python3 /campaign/repair-base-python-metadata.py \
    /usr/local/share/dynamo-sglang-sidecar/base-python-metadata-repairs.json
install -m 0755 "${sidecar_path}" /usr/local/libexec/dynamo-sglang-sidecar.real
install -m 0755 /campaign/dynamo-sglang-sidecar-wrapper.sh /usr/local/bin/dynamo-sglang-sidecar
printf '%s\n' \
    "dynamo_commit=${DYNAMO_COMMIT}" \
    "dynamo_base_commit=${DYNAMO_BASE_COMMIT}" \
    "sglang_commit=${SGLANG_COMMIT}" \
    "sidecar_sha256=${sidecar_sha256}" \
    "dynamo_wheel_sha256=${dynamo_wheel_sha256}" \
    "dynamo_runtime_wheel_sha256=${dynamo_runtime_wheel_sha256}" \
    "sglang_wheel_sha256=${sglang_wheel_sha256}" \
    "proto_descriptor_sha256=${descriptor_sha256}" \
    "architecture=linux/arm64" \
    "cuda_version=13.0" \
    "base_image=${BASE_IMAGE}" \
    "build_timestamp=${BUILD_TIMESTAMP}" \
    > /usr/local/share/dynamo-sglang-sidecar/build-info
python3 -m pip freeze --all > /usr/local/share/dynamo-sglang-sidecar/package-lock.txt
cp /usr/local/share/dynamo-sglang-sidecar/build-info /campaign-artifacts/build/
cp /usr/local/share/dynamo-sglang-sidecar/package-lock.txt /campaign-artifacts/build/
cp /usr/local/share/dynamo-sglang-sidecar/base-python-metadata-repairs.json \
    /campaign-artifacts/build/

python3 -m pip check
python3 -c 'import dynamo._core; import dynamo.frontend; import dynamo.sglang'
python3 -c 'from sglang.srt.grpc import _core; assert hasattr(_core, "start_server")'
python3 -c 'import deep_gemm, flashinfer, mooncake, sgl_kernel; from sglang.srt.layers.attention.dsv4 import metadata; from sglang.srt.layers.quantization import mxfp4; from sglang.srt.mem_cache import swa_memory_pool; from sglang.srt.models import deepseek_v4; from sglang.srt.parser import reasoning_parser'
python3 -m sglang.launch_server --help | grep -F -- '--grpc-port'
dynamo-sglang-sidecar --build-info
