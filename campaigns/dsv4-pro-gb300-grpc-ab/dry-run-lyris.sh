#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly REPO_ROOT
readonly OUTPUT_DIR="/lustre/fsw/coreai_comparch_inferencex/connorc/dsv4-grpc-ab-ba4c325301/dry-runs"
readonly SRTCTL="${REPO_ROOT}/.venv/bin/srtctl"

export SRTSLURM_CONFIG="${REPO_ROOT}/srtslurm.yaml"
mkdir -p "${OUTPUT_DIR}"
test -x "${SRTCTL}"
grep -F -- "gb300,gb300-backfill" "${SRTSLURM_CONFIG}" >/dev/null
grep -F -- "dsv4-grpc-ab-ba4c325301-e2728ac-arm64.sqsh" "${SRTSLURM_CONFIG}" >/dev/null

for architecture in legacy sidecar; do
    while IFS= read -r recipe; do
        point="$(basename -- "${recipe}" .yaml)"
        output="${OUTPUT_DIR}/${point}-${architecture}.txt"
        COLUMNS=300 "${SRTCTL}" dry-run -f "${recipe}" > "${output}"
        test -s "${output}"
        grep -F -- "${architecture}" "${output}" >/dev/null
        grep -F -- "/models/dsv4-pro" "${output}" >/dev/null
        grep -F -- "dsv4-grpc-ab-ba4c325301-e2728ac-arm64.sqsh" "${output}" >/dev/null
    done < <(printf '%s\n' "${REPO_ROOT}"/recipes/dsv4-pro-gb300-grpc-ab/"${architecture}"/*.yaml | sort)
done

test "$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.txt' | wc -l)" -eq 14
printf 'Validated 14 Lyris GB300 A/B dry-runs in %s\n' "${OUTPUT_DIR}"
