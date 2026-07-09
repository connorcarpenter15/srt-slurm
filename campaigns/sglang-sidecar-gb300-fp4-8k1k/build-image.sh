#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly DYNAMO_COMMIT="555695f4367986db3fb7d86184be7c84eabdad73"
readonly SGLANG_COMMIT="cc7d6659fd68694797892d0d863b2549a5b61b69"
readonly BASE_IMAGE="lmsysorg/sglang:v0.5.8.post1-cu130-runtime"
readonly DEFAULT_IMAGE="nvcr.io/nvidian/dynamo-dev/sglang-runtime:connorc-555695f436-cc7d6659fd-gb300-sidecar-arm64"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly DYNAMO_REPO="${DYNAMO_REPO:?set DYNAMO_REPO to a Dynamo checkout containing ${DYNAMO_COMMIT}}"
readonly SGLANG_REPO="${SGLANG_REPO:?set SGLANG_REPO to an SGLang checkout containing ${SGLANG_COMMIT}}"
readonly IMAGE="${IMAGE:-${DEFAULT_IMAGE}}"
readonly PUSH="${PUSH:-true}"
readonly ARTIFACT_DIR="${ARTIFACT_DIR:-${SCRIPT_DIR}/artifacts}"
readonly BUILD_TIMESTAMP="${BUILD_TIMESTAMP:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
readonly TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sglang-sidecar-image.XXXXXX")"

cleanup() {
    rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

git -C "${DYNAMO_REPO}" cat-file -e "${DYNAMO_COMMIT}^{commit}"
git -C "${SGLANG_REPO}" cat-file -e "${SGLANG_COMMIT}^{commit}"

mkdir -p "${TEMP_DIR}/dynamo" "${TEMP_DIR}/sglang" "${ARTIFACT_DIR}/build"
git -C "${DYNAMO_REPO}" archive "${DYNAMO_COMMIT}" | tar -x -C "${TEMP_DIR}/dynamo"
git -C "${SGLANG_REPO}" archive "${SGLANG_COMMIT}" | tar -x -C "${TEMP_DIR}/sglang"

common_args=(
    --platform linux/arm64
    --build-context "dynamo-source=${TEMP_DIR}/dynamo"
    --build-context "sglang-source=${TEMP_DIR}/sglang"
    --build-arg "BASE_IMAGE=${BASE_IMAGE}"
    --build-arg "DYNAMO_COMMIT=${DYNAMO_COMMIT}"
    --build-arg "SGLANG_COMMIT=${SGLANG_COMMIT}"
    --build-arg "BUILD_TIMESTAMP=${BUILD_TIMESTAMP}"
    --file "${SCRIPT_DIR}/Dockerfile"
)

docker buildx build \
    "${common_args[@]}" \
    --target campaign-artifacts \
    --output "type=local,dest=${ARTIFACT_DIR}/build" \
    "${SCRIPT_DIR}"

sidecar_sha256="$(sha256sum "${ARTIFACT_DIR}/build/dynamo-sglang-sidecar" | awk '{print $1}')"
wheel_path="$(printf '%s\n' "${ARTIFACT_DIR}"/build/sglang-*.whl | head -n1)"
wheel_sha256="$(sha256sum "${wheel_path}" | awk '{print $1}')"
metadata_file="${ARTIFACT_DIR}/build-metadata.json"

publish_args=(--load)
if [[ "${PUSH}" == "true" ]]; then
    publish_args=(--push)
fi

docker buildx build \
    "${common_args[@]}" \
    --build-arg "SIDECAR_SHA256=${sidecar_sha256}" \
    --build-arg "SGLANG_WHEEL_SHA256=${wheel_sha256}" \
    --metadata-file "${metadata_file}" \
    --provenance=true \
    --sbom=true \
    --tag "${IMAGE}" \
    "${publish_args[@]}" \
    "${SCRIPT_DIR}"

image_digest=""
if [[ -f "${metadata_file}" ]]; then
    image_digest="$(python3 - "${metadata_file}" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    metadata = json.load(stream)
print(metadata.get("containerimage.digest", ""))
PY
)"
fi

python3 - \
    "${ARTIFACT_DIR}/image-manifest.json" \
    "${IMAGE}" \
    "${image_digest}" \
    "${DYNAMO_COMMIT}" \
    "${SGLANG_COMMIT}" \
    "${sidecar_sha256}" \
    "${wheel_path}" \
    "${wheel_sha256}" \
    "${BASE_IMAGE}" \
    "${BUILD_TIMESTAMP}" <<'PY'
import json
import pathlib
import sys

(
    output,
    image,
    digest,
    dynamo_commit,
    sglang_commit,
    sidecar_sha256,
    wheel_path,
    wheel_sha256,
    base_image,
    build_timestamp,
) = sys.argv[1:]
manifest = {
    "image": image,
    "image_digest": digest or None,
    "source_commits": {
        "dynamo": dynamo_commit,
        "sglang": sglang_commit,
    },
    "artifacts": {
        "dynamo-sglang-sidecar": {"sha256": sidecar_sha256},
        pathlib.Path(wheel_path).name: {"sha256": wheel_sha256},
    },
    "architecture": "linux/arm64",
    "cuda_version": "13.0",
    "base_image": base_image,
    "build_timestamp": build_timestamp,
}
path = pathlib.Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

printf 'Candidate image: %s\nManifest: %s\n' "${IMAGE}" "${ARTIFACT_DIR}/image-manifest.json"
