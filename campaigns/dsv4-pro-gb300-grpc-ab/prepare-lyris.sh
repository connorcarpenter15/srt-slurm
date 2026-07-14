#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly MODEL_DIR="/lustre/fsw/coreai_comparch_inferencex/models/dsv4-pro"
readonly IMAGE_PATH="/lustre/fsw/coreai_comparch_inferencex/connorc/artifacts/dsv4-grpc-ab-beb91b0-e2728ac-arm64.sqsh"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly REPO_ROOT

test -s "${IMAGE_PATH}"
python3 "${REPO_ROOT}/configs/verify-dsv4-model.py" \
    --model-dir "${MODEL_DIR}" \
    --manifest "${REPO_ROOT}/configs/dsv4-model-runtime-blobs-b5968e9.json"

cp "${SCRIPT_DIR}/lyris-srtslurm.yaml" "${REPO_ROOT}/srtslurm.yaml"
mkdir -p "/lustre/fsw/coreai_comparch_inferencex/connorc/dsv4-grpc-ab/outputs"

# srtslurm.yaml is present before setup, so make downloads only the pinned
# ARM64 NATS/etcd binaries and never enters its interactive cluster prompt.
make -C "${REPO_ROOT}" setup ARCH=aarch64

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
UV_CACHE_DIR="/lustre/fsw/coreai_comparch_inferencex/connorc/.cache/uv" \
    uv sync --directory "${REPO_ROOT}" --frozen --no-dev

printf 'Lyris campaign ready: model=%s image=%s\n' "${MODEL_DIR}" "${IMAGE_PATH}"
