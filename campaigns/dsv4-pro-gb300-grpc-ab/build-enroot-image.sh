#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly DYNAMO_COMMIT="beb91b0de5392af2bd36560b312c153e7dbed061"
readonly SGLANG_COMMIT="e2728ac504c00e37a284c7248693857b894e40e7"
readonly BASE_DIGEST="sha256:4b140bc08eb4782b057109b084b6df94f74c3a66c6984ee383a1d6c3714994d5"
readonly BASE_URI="docker://lmsysorg/sglang@${BASE_DIGEST}"
readonly SCRATCH_ROOT="${SCRATCH_ROOT:-/lustre/fsw/coreai_comparch_inferencex/connorc}"
readonly ARTIFACT_DIR="${ARTIFACT_DIR:-${SCRATCH_ROOT}/artifacts/dsv4-grpc-ab-image}"
readonly FINAL_IMAGE="${FINAL_IMAGE:-${SCRATCH_ROOT}/artifacts/dsv4-grpc-ab-beb91b0-e2728ac-arm64.sqsh}"
readonly SOURCE_ROOT="${SOURCE_ROOT:-${SCRATCH_ROOT}/src}"
readonly BASE_SQSH="${ARTIFACT_DIR}/base-sglang-dev-cu13-4b140bc.sqsh"
readonly BUILD_TIMESTAMP="${BUILD_TIMESTAMP:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
readonly ENROOT_NAME="dsv4-grpc-ab-beb91b0-e2728ac"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly DYNAMO_REPO="${SOURCE_ROOT}/dynamo-${DYNAMO_COMMIT:0:9}"
readonly SGLANG_REPO="${SOURCE_ROOT}/sglang-${SGLANG_COMMIT:0:9}"

export ENROOT_CACHE_PATH="${SCRATCH_ROOT}/.cache/enroot"
export ENROOT_DATA_PATH="${SCRATCH_ROOT}/.local/share/enroot"
export ENROOT_RUNTIME_PATH="${SCRATCH_ROOT}/.run/enroot"
export ENROOT_TEMP_PATH="${SCRATCH_ROOT}/.tmp/enroot"

mkdir -p \
    "${ARTIFACT_DIR}/build" \
    "${SOURCE_ROOT}" \
    "${ENROOT_CACHE_PATH}" \
    "${ENROOT_DATA_PATH}" \
    "${ENROOT_RUNTIME_PATH}" \
    "${ENROOT_TEMP_PATH}" \
    "$(dirname -- "${FINAL_IMAGE}")"

stage_source() {
    local url="$1"
    local commit="$2"
    local destination="$3"

    if [[ ! -d "${destination}/.git" ]]; then
        git clone --filter=blob:none --no-checkout "${url}" "${destination}"
    fi
    if ! git -C "${destination}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        git -C "${destination}" fetch --depth=1 origin "${commit}"
    fi
    git -C "${destination}" checkout --detach "${commit}"
    test "$(git -C "${destination}" rev-parse HEAD)" = "${commit}"
}

stage_source https://github.com/ai-dynamo/dynamo.git "${DYNAMO_COMMIT}" "${DYNAMO_REPO}"
stage_source https://github.com/sgl-project/sglang.git "${SGLANG_COMMIT}" "${SGLANG_REPO}"

if [[ ! -s "${BASE_SQSH}" ]]; then
    enroot import -o "${BASE_SQSH}" "${BASE_URI}"
fi
if [[ ! -d "${ENROOT_DATA_PATH}/${ENROOT_NAME}" ]]; then
    enroot create --name "${ENROOT_NAME}" "${BASE_SQSH}"
fi

enroot start \
    --root \
    --rw \
    --env "BUILD_TIMESTAMP=${BUILD_TIMESTAMP}" \
    --mount "${SCRIPT_DIR}:/campaign" \
    --mount "${DYNAMO_REPO}:/src/dynamo" \
    --mount "${SGLANG_REPO}:/src/sglang" \
    --mount "${ARTIFACT_DIR}:/campaign-artifacts" \
    "${ENROOT_NAME}" \
    /bin/bash /campaign/build-enroot-payload.sh

rm -f "${FINAL_IMAGE}"
mksquashfs \
    "${ENROOT_DATA_PATH}/${ENROOT_NAME}" \
    "${FINAL_IMAGE}" \
    -comp zstd \
    -noappend

python3 - \
    "${ARTIFACT_DIR}/image-manifest.json" \
    "${FINAL_IMAGE}" \
    "${DYNAMO_COMMIT}" \
    "${SGLANG_COMMIT}" \
    "${BASE_URI}" \
    "${BUILD_TIMESTAMP}" \
    "${ARTIFACT_DIR}/build" <<'PY'
import hashlib
import json
import pathlib
import platform
import sys

output, image_name, dynamo_commit, sglang_commit, base_uri, build_timestamp, build_dir = sys.argv[1:]
image = pathlib.Path(image_name)
build = pathlib.Path(build_dir)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


artifact_names = [
    "build-info",
    "package-lock.txt",
    "dynamo-sglang-sidecar",
    "server-descriptor.bin",
    "sidecar-descriptor.bin",
    "proto-sha256.txt",
]
artifact_names += sorted(path.name for path in build.glob("*.whl"))
artifacts = {}
for name in artifact_names:
    path = build / name
    if not path.is_file():
        raise SystemExit(f"missing build artifact: {path}")
    artifacts[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}

manifest = {
    "schema_version": 1,
    "format": "enroot-squashfs",
    "image": str(image),
    "image_sha256": sha256(image),
    "image_bytes": image.stat().st_size,
    "source_commits": {"dynamo": dynamo_commit, "sglang": sglang_commit},
    "base_image": base_uri,
    "architecture": platform.machine(),
    "cuda_version": "13.0",
    "build_timestamp": build_timestamp,
    "artifacts": artifacts,
    "proto_compatibility": "descriptor-byte-identical",
}
path = pathlib.Path(output)
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

printf 'Candidate squash image: %s\nManifest: %s\n' \
    "${FINAL_IMAGE}" \
    "${ARTIFACT_DIR}/image-manifest.json"
