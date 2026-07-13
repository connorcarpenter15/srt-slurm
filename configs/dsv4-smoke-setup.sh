#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

hostname_short="$(hostname -s)"
lock_dir="/logs/.image-preflight-${hostname_short}.lock"
complete_file="/logs/image-preflight-${hostname_short}.complete"
failed_file="/logs/image-preflight-${hostname_short}.failed"
log_file="/logs/image-preflight-${hostname_short}.log"

if mkdir "${lock_dir}" 2>/dev/null; then
    if MODEL_PATH=/model bash /configs/dsv4-image-preflight.sh >"${log_file}" 2>&1; then
        touch "${complete_file}"
    else
        status=$?
        touch "${failed_file}"
        cat "${log_file}" >&2
        exit "${status}"
    fi
else
    ready=0
    for _ in $(seq 1 900); do
        if [[ -f "${complete_file}" ]]; then
            ready=1
            break
        fi
        if [[ -f "${failed_file}" ]]; then
            cat "${log_file}" >&2
            exit 1
        fi
        sleep 1
    done
    if [[ "${ready}" != 1 ]]; then
        echo "Timed out waiting for image preflight on ${hostname_short}" >&2
        exit 1
    fi
fi

bash /configs/dsv4-gpu-telemetry.sh
