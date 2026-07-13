#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -eq 1 && ( "$1" == "--version" || "$1" == "--build-info" ) ]]; then
    cat /usr/local/share/dynamo-sglang-sidecar/build-info
    exit 0
fi

exec /usr/local/libexec/dynamo-sglang-sidecar.real "$@"
