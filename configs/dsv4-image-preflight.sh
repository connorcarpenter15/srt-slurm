#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

expected_dynamo="${EXPECTED_DYNAMO_COMMIT:-beb91b0de5392af2bd36560b312c153e7dbed061}"
expected_sglang="${EXPECTED_SGLANG_COMMIT:-e2728ac504c00e37a284c7248693857b894e40e7}"
expected_model_revision="${EXPECTED_MODEL_REVISION:-b5968e9190ef611bbf34a7229255be88a0e937c1}"

test "$(uname -m)" = "aarch64"
python3 -m pip check
python3 -c 'import dynamo._core; import dynamo.frontend; import dynamo.sglang'
python3 -c 'from sglang.srt.grpc import _core; assert hasattr(_core, "start_server")'
python3 -c 'import deep_gemm, flashinfer, mooncake, sgl_kernel; from sglang.srt.layers.attention.dsv4 import metadata; from sglang.srt.layers.quantization import mxfp4; from sglang.srt.mem_cache import swa_memory_pool; from sglang.srt.models import deepseek_v4; from sglang.srt.parser import reasoning_parser'
python3 -m sglang.launch_server --help | grep -F -- '--grpc-port'
dynamo-sglang-sidecar --help >/dev/null
build_info="$(dynamo-sglang-sidecar --build-info)"
grep -F "dynamo_commit=${expected_dynamo}" <<<"${build_info}"
grep -F "sglang_commit=${expected_sglang}" <<<"${build_info}"
grep -F 'architecture=linux/arm64' <<<"${build_info}"
grep -F 'cuda_version=13.0' <<<"${build_info}"
test -s /usr/local/share/dynamo-sglang-sidecar/package-lock.txt

if [[ -n "${MODEL_PATH:-}" ]]; then
    python3 -c 'from transformers import AutoConfig, AutoTokenizer; import sys; config = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True); tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True); assert config is not None and tokenizer.vocab_size > 0' "${MODEL_PATH}"
    python3 - "${MODEL_PATH}/.campaign-model.json" "${expected_model_revision}" <<'PY'
import json
import pathlib
import sys

marker = json.loads(pathlib.Path(sys.argv[1]).read_text())
if marker.get("revision") != sys.argv[2] or marker.get("indexed_shards") != 64:
    raise SystemExit(f"invalid campaign model marker: {marker!r}")
PY
fi

printf '%s\n' "${build_info}"
