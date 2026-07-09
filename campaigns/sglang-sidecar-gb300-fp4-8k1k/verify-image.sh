#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

expected_dynamo="${EXPECTED_DYNAMO_COMMIT:-555695f4367986db3fb7d86184be7c84eabdad73}"
expected_sglang="${EXPECTED_SGLANG_COMMIT:-cc7d6659fd68694797892d0d863b2549a5b61b69}"

test "$(uname -m)" = "aarch64"
python3 -c 'from sglang.srt.grpc import _core; assert hasattr(_core, "start_server")'
python3 -m sglang.launch_server --help | grep -F -- '--grpc-port'
dynamo-sglang-sidecar --help >/dev/null
build_info="$(dynamo-sglang-sidecar --build-info)"
grep -F "dynamo_commit=${expected_dynamo}" <<<"${build_info}"
grep -F "sglang_commit=${expected_sglang}" <<<"${build_info}"
grep -F 'architecture=linux/arm64' <<<"${build_info}"
grep -F 'cuda_version=13.0' <<<"${build_info}"

printf '%s\n' "${build_info}"
