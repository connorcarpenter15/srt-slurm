#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly DYNAMO_COMMIT="beb91b0de5392af2bd36560b312c153e7dbed061"
readonly SGLANG_COMMIT="e2728ac504c00e37a284c7248693857b894e40e7"
readonly BASE_IMAGE="lmsysorg/sglang:dev-cu13@sha256:4b140bc08eb4782b057109b084b6df94f74c3a66c6984ee383a1d6c3714994d5"
readonly DEFAULT_IMAGE="nvcr.io/nvidian/dynamo-dev/sglang-runtime:connorc-beb91b0-e2728ac-dsv4-gb300-ab-arm64"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly DYNAMO_REPO="${DYNAMO_REPO:?set DYNAMO_REPO to a Dynamo checkout containing ${DYNAMO_COMMIT}}"
readonly SGLANG_REPO="${SGLANG_REPO:?set SGLANG_REPO to an SGLang checkout containing ${SGLANG_COMMIT}}"
readonly IMAGE="${IMAGE:-${DEFAULT_IMAGE}}"
readonly PUSH="${PUSH:-true}"
readonly ARTIFACT_DIR="${ARTIFACT_DIR:-${SCRIPT_DIR}/artifacts}"
readonly BUILD_TIMESTAMP="${BUILD_TIMESTAMP:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dsv4-grpc-ab-image.XXXXXX")"
readonly TEMP_DIR

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

sidecar_path="${ARTIFACT_DIR}/build/dynamo-sglang-sidecar"
dynamo_wheel_path="$(printf '%s\n' "${ARTIFACT_DIR}"/build/ai_dynamo-*-none-any.whl | head -n1)"
dynamo_runtime_wheel_path="$(printf '%s\n' "${ARTIFACT_DIR}"/build/ai_dynamo_runtime-*.whl | head -n1)"
sglang_wheel_path="$(printf '%s\n' "${ARTIFACT_DIR}"/build/sglang-*.whl | head -n1)"
descriptor_path="${ARTIFACT_DIR}/build/server-descriptor.bin"
proto_hashes_path="${ARTIFACT_DIR}/build/proto-sha256.txt"
build_info_path="${ARTIFACT_DIR}/build/build-info"

for artifact in \
    "${sidecar_path}" \
    "${dynamo_wheel_path}" \
    "${dynamo_runtime_wheel_path}" \
    "${sglang_wheel_path}" \
    "${descriptor_path}" \
    "${proto_hashes_path}" \
    "${build_info_path}" \
    "${ARTIFACT_DIR}/build/base-python-metadata-repairs.json" \
    "${ARTIFACT_DIR}/build/package-lock.txt"; do
    test -s "${artifact}"
done

sidecar_sha256="$(sha256sum "${sidecar_path}" | awk '{print $1}')"
dynamo_wheel_sha256="$(sha256sum "${dynamo_wheel_path}" | awk '{print $1}')"
dynamo_runtime_wheel_sha256="$(sha256sum "${dynamo_runtime_wheel_path}" | awk '{print $1}')"
sglang_wheel_sha256="$(sha256sum "${sglang_wheel_path}" | awk '{print $1}')"
descriptor_sha256="$(sha256sum "${descriptor_path}" | awk '{print $1}')"
metadata_file="${ARTIFACT_DIR}/build-metadata.json"

# The campaign-artifacts target is built before the artifact hashes are known,
# so its build-info file contains empty hash fields. Replace that preliminary
# copy with the exact metadata that the final runtime image will contain.
printf '%s\n' \
    "dynamo_commit=${DYNAMO_COMMIT}" \
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
    > "${build_info_path}"

publish_args=(--load)
if [[ "${PUSH}" == "true" ]]; then
    publish_args=(--push)
fi

docker buildx build \
    "${common_args[@]}" \
    --build-arg "SIDECAR_SHA256=${sidecar_sha256}" \
    --build-arg "DYNAMO_WHEEL_SHA256=${dynamo_wheel_sha256}" \
    --build-arg "DYNAMO_RUNTIME_WHEEL_SHA256=${dynamo_runtime_wheel_sha256}" \
    --build-arg "SGLANG_WHEEL_SHA256=${sglang_wheel_sha256}" \
    --build-arg "PROTO_DESCRIPTOR_SHA256=${descriptor_sha256}" \
    --metadata-file "${metadata_file}" \
    --provenance=true \
    --sbom=true \
    --target final \
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
    "${BASE_IMAGE}" \
    "${BUILD_TIMESTAMP}" \
    "${sidecar_path}" \
    "${dynamo_wheel_path}" \
    "${dynamo_runtime_wheel_path}" \
    "${sglang_wheel_path}" \
    "${descriptor_path}" \
    "${proto_hashes_path}" \
    "${build_info_path}" \
    "${ARTIFACT_DIR}/build/base-python-metadata-repairs.json" \
    "${ARTIFACT_DIR}/build/package-lock.txt" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    output,
    image,
    digest,
    dynamo_commit,
    sglang_commit,
    base_image,
    build_timestamp,
    *artifact_names,
) = sys.argv[1:]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


artifacts = {}
for name in artifact_names:
    path = pathlib.Path(name)
    artifacts[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}

manifest = {
    "schema_version": 1,
    "image": image,
    "image_digest": digest or None,
    "source_commits": {"dynamo": dynamo_commit, "sglang": sglang_commit},
    "base_image": base_image,
    "architecture": "linux/arm64",
    "cuda_version": "13.0",
    "build_timestamp": build_timestamp,
    "artifacts": artifacts,
    "proto_compatibility": "descriptor-byte-identical",
}
path = pathlib.Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

printf 'Candidate image: %s\nManifest: %s\n' "${IMAGE}" "${ARTIFACT_DIR}/image-manifest.json"
