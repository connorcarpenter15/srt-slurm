#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

BLOCK_SIZE = 64
REQUEST_COUNT = 10_240
BUCKETS = (128, 256, 512, 1024)
GROUPS_PER_BUCKET = 8
OUTPUT_LENGTH = 1
SEED = 20_260_626


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_cache_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    next_hash = 1_000_000
    per_bucket = REQUEST_COUNT // len(BUCKETS)
    for input_length in BUCKETS:
        blocks = math.ceil(input_length / BLOCK_SIZE)
        for index in range(per_bucket):
            hashes = list(range(next_hash, next_hash + blocks))
            next_hash += blocks
            rows.append(
                {
                    "session_id": f"nocache-{input_length}-{index}",
                    "input_length": input_length,
                    "output_length": OUTPUT_LENGTH,
                    "hash_ids": hashes,
                    "timestamp": 0.0,
                }
            )
    random.Random(SEED).shuffle(rows)
    return rows


def prefix_reuse_rows() -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    next_hash = 10_000_000
    shared_tokens = 0
    total_tokens = 0
    per_group = REQUEST_COUNT // (len(BUCKETS) * GROUPS_PER_BUCKET)
    group_id = 0
    for input_length in BUCKETS:
        blocks = math.ceil(input_length / BLOCK_SIZE)
        shared_blocks = max(1, math.floor(blocks * 0.75))
        for bucket_group in range(GROUPS_PER_BUCKET):
            prefix_hashes = list(range(next_hash, next_hash + shared_blocks))
            next_hash += shared_blocks
            for request_index in range(per_group):
                unique_blocks = blocks - shared_blocks
                suffix_hashes = list(range(next_hash, next_hash + unique_blocks))
                next_hash += unique_blocks
                rows.append(
                    {
                        "session_id": f"prefix-{input_length}-{bucket_group}-{request_index}",
                        "group_id": group_id,
                        "input_length": input_length,
                        "output_length": OUTPUT_LENGTH,
                        "hash_ids": prefix_hashes + suffix_hashes,
                        "timestamp": 0.0,
                    }
                )
                shared_tokens += min(shared_blocks * BLOCK_SIZE, input_length)
                total_tokens += input_length
            group_id += 1
    random.Random(SEED).shuffle(rows)
    return rows, shared_tokens, total_tokens


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic prefill routing gate traces.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("traces"),
        help="Output directory (default: traces/ next to this script).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    no_cache = no_cache_rows()
    prefix_reuse, shared_tokens, total_tokens = prefix_reuse_rows()
    no_cache_path = args.output_dir / "prefill-gate-no-cache.jsonl"
    prefix_path = args.output_dir / "prefill-gate-prefix-reuse.jsonl"
    checksums = {
        no_cache_path.name: write_jsonl(no_cache_path, no_cache),
        prefix_path.name: write_jsonl(prefix_path, prefix_reuse),
    }
    manifest = {
        "block_size": BLOCK_SIZE,
        "buckets": list(BUCKETS),
        "bucket_counts": dict(sorted(Counter(row["input_length"] for row in no_cache).items())),
        "groups_per_bucket": GROUPS_PER_BUCKET,
        "output_length": OUTPUT_LENGTH,
        "prefix_shared_token_ratio": shared_tokens / total_tokens,
        "request_count": REQUEST_COUNT,
        "seed": SEED,
        "sha256": checksums,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
