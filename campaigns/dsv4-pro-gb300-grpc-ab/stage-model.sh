#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly MODEL_REPO="deepseek-ai/DeepSeek-V4-Pro"
readonly MODEL_REVISION="b5968e9190ef611bbf34a7229255be88a0e937c1"
readonly MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the immutable campaign model directory}"
readonly HF_HOME="${HF_HOME:-$(dirname -- "${MODEL_DIR}")/.hf-cache}"
readonly TOOL_VENV="${TOOL_VENV:-$(dirname -- "${MODEL_DIR}")/.venv-huggingface-hub}"

mkdir -p "${MODEL_DIR}" "${HF_HOME}"

if [[ -x "${TOOL_VENV}/bin/hf" ]]; then
    hf_bin="${TOOL_VENV}/bin/hf"
elif command -v hf >/dev/null 2>&1; then
    hf_bin="$(command -v hf)"
else
    python3 -m venv "${TOOL_VENV}"
    "${TOOL_VENV}/bin/python" -m pip install \
        --disable-pip-version-check \
        'huggingface_hub[hf_xet]==0.35.3'
    hf_bin="${TOOL_VENV}/bin/hf"
fi

export HF_HOME
export HF_XET_HIGH_PERFORMANCE=1
"${hf_bin}" download \
    "${MODEL_REPO}" \
    --repo-type model \
    --revision "${MODEL_REVISION}" \
    --local-dir "${MODEL_DIR}"

python3 - "${MODEL_DIR}" "${MODEL_REPO}" "${MODEL_REVISION}" <<'PY'
import json
import pathlib
import sys

model_dir = pathlib.Path(sys.argv[1])
repo = sys.argv[2]
revision = sys.argv[3]
index_path = model_dir / "model.safetensors.index.json"

if not (model_dir / "config.json").is_file():
    raise SystemExit("missing config.json")
if not index_path.is_file():
    raise SystemExit("missing model.safetensors.index.json")

index = json.loads(index_path.read_text())
weight_map = index.get("weight_map")
if not isinstance(weight_map, dict) or not weight_map:
    raise SystemExit("model index has no weight_map")

shards = sorted(set(weight_map.values()))
if len(shards) != 64:
    raise SystemExit(f"expected 64 indexed model shards, found {len(shards)}")

missing = [name for name in shards if not (model_dir / name).is_file()]
empty = [name for name in shards if (model_dir / name).is_file() and not (model_dir / name).stat().st_size]
if missing or empty:
    raise SystemExit(f"incomplete model: missing={missing}, empty={empty}")

marker = {
    "repo": repo,
    "revision": revision,
    "indexed_shards": len(shards),
    "total_indexed_shard_bytes": sum((model_dir / name).stat().st_size for name in shards),
}
(model_dir / ".campaign-model.json").write_text(
    json.dumps(marker, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(marker, sort_keys=True))
PY
