#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly EXPECTED_REVISION="b5968e9190ef611bbf34a7229255be88a0e937c1"
readonly MODEL_DIR="/lustre/fsw/coreai_comparch_inferencex/connorc/models/DeepSeek-V4-Pro-b5968e9"
readonly IMAGE_PATH="/lustre/fsw/coreai_comparch_inferencex/connorc/artifacts/dsv4-grpc-ab-beb91b0-e2728ac-arm64.sqsh"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly REPO_ROOT

test -s "${IMAGE_PATH}"
python3 - "${MODEL_DIR}/.campaign-model.json" "${EXPECTED_REVISION}" <<'PY'
import json
import pathlib
import sys

marker_path = pathlib.Path(sys.argv[1])
expected_revision = sys.argv[2]
marker = json.loads(marker_path.read_text())
if marker.get("revision") != expected_revision:
    raise SystemExit(
        f"model revision mismatch: {marker.get('revision')!r} != {expected_revision!r}"
    )
if marker.get("indexed_shards") != 64:
    raise SystemExit(f"expected 64 model shards, got {marker.get('indexed_shards')!r}")
PY

cp "${SCRIPT_DIR}/ptyche-srtslurm.yaml" "${REPO_ROOT}/srtslurm.yaml"
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

printf 'Ptyche campaign ready: model=%s image=%s\n' "${MODEL_DIR}" "${IMAGE_PATH}"

