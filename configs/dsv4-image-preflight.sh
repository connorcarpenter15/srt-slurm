#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

expected_dynamo="${EXPECTED_DYNAMO_COMMIT:-beb91b0de5392af2bd36560b312c153e7dbed061}"
expected_sglang="${EXPECTED_SGLANG_COMMIT:-e2728ac504c00e37a284c7248693857b894e40e7}"
expected_model_revision="${EXPECTED_MODEL_REVISION:-b5968e9190ef611bbf34a7229255be88a0e937c1}"
minimum_gpu_memory_mib="${MINIMUM_GPU_MEMORY_MIB:-260000}"

test "$(uname -m)" = "aarch64"
python3 - "${minimum_gpu_memory_mib}" <<'PY'
import subprocess
import sys

minimum = int(sys.argv[1])
output = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
values = [int(line.strip()) for line in output.splitlines() if line.strip()]
if not values or min(values) < minimum:
    raise SystemExit(
        f"GB300 preflight requires >= {minimum} MiB per GPU; observed {values}"
    )
PY
python3 -m pip check
python3 -c 'import dynamo._core; import dynamo.frontend; import dynamo.sglang'
python3 -c 'from sglang.srt.grpc import _core; assert hasattr(_core, "start_server")'
python3 -c 'import deep_gemm, flashinfer, mooncake, sgl_kernel; from sglang.srt.layers.attention.dsv4 import metadata; from sglang.srt.layers.quantization import mxfp4; from sglang.srt.mem_cache import swa_memory_pool; from sglang.srt.models import deepseek_v4; from sglang.srt.parser import reasoning_parser'
python3 -m sglang.launch_server --help | grep -F -- '--grpc-port'
build_info="$(dynamo-sglang-sidecar --build-info)"
sidecar_path="/usr/local/libexec/dynamo-sglang-sidecar.real"
test -x "${sidecar_path}"
expected_sidecar_sha256="$(awk -F= '$1 == "sidecar_sha256" {print $2}' <<<"${build_info}")"
test -n "${expected_sidecar_sha256}"
test "$(sha256sum "${sidecar_path}" | awk '{print $1}')" = "${expected_sidecar_sha256}"
grep -F "dynamo_commit=${expected_dynamo}" <<<"${build_info}"
grep -F "sglang_commit=${expected_sglang}" <<<"${build_info}"
grep -F 'architecture=linux/arm64' <<<"${build_info}"
grep -F 'cuda_version=13.0' <<<"${build_info}"
test -s /usr/local/share/dynamo-sglang-sidecar/package-lock.txt

if [[ -n "${MODEL_PATH:-}" ]]; then
    python3 -c 'from transformers import AutoConfig, AutoTokenizer; import sys; config = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True); tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True); assert config is not None and tokenizer.vocab_size > 0' "${MODEL_PATH}"
    model_manifest="${MODEL_RUNTIME_MANIFEST:-/configs/dsv4-model-runtime-blobs-b5968e9.json}"
    python3 /configs/verify-dsv4-model.py \
        --model-dir "${MODEL_PATH}" \
        --manifest "${model_manifest}" | grep -F "${expected_model_revision}"
fi

printf '%s\n' "${build_info}"
