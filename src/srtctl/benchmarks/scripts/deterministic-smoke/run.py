#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pathlib
import re
import sys
import urllib.request


def load_tokenizer(path: str, model_path: str):
    module_name, class_name = path.rsplit(".", 1)
    tokenizer_class = getattr(importlib.import_module(module_name), class_name)
    return tokenizer_class.from_pretrained(model_path, trust_remote_code=True)


def repeated_prompt(tokenizer, length: int) -> list[int]:
    seed = (
        "Dynamo and SGLang deterministic transport validation. "
        "Preserve every token while comparing serving architectures. "
    )
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("tokenizer produced an empty deterministic seed")
    return (seed_ids * ((length + len(seed_ids) - 1) // len(seed_ids)))[:length]


def output_token_ids(logprobs) -> list[int]:
    """Extract Dynamo's token_id:<id> representation in response order."""
    result: list[int] = []

    def visit(value) -> None:
        if isinstance(value, str):
            match = re.fullmatch(r"token_id:(\d+)", value)
            if match:
                result.append(int(match.group(1)))
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(logprobs)
    return result


def comparison_token_ids(result: dict, tokenizer, text: str, expected: int) -> tuple[list[int], str, list[int]]:
    """Return a complete token sequence while recording its provenance.

    Dynamo's legacy SGLang response can expose only the first generated token
    through completion logprobs when SGLang emits per-chunk (rather than
    cumulative) logprob metadata. In that case, re-tokenize the completed text
    with the campaign's pinned tokenizer. The raw response remains alongside
    the normalized artifact so this fallback is explicit and auditable.
    """
    returned = output_token_ids(result["choices"][0].get("logprobs"))
    if len(returned) == expected:
        return returned, "response_logprobs", returned

    retokenized = tokenizer.encode(text, add_special_tokens=False)
    if len(retokenized) != expected:
        raise RuntimeError(
            "Dynamo returned "
            f"{len(returned)} token ids and re-tokenization produced "
            f"{len(retokenized)} tokens; expected {expected}"
        )
    return retokenized, "retokenized_output_text", returned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-root", type=pathlib.Path, required=True)
    parser.add_argument("--isl", type=int, required=True)
    parser.add_argument("--osl", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.tokenizer_root))
    tokenizer = load_tokenizer(args.tokenizer, args.model_path)
    prompt_ids = repeated_prompt(tokenizer, args.isl)
    payload = {
        "model": args.model,
        "prompt": prompt_ids,
        "max_tokens": args.osl,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": False,
        "logprobs": 0,
        "return_tokens_as_token_ids": True,
    }
    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=7200) as response:
        result = json.load(response)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.output.with_name(f"{args.output.stem}.response.json")
    raw_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    text = result["choices"][0]["text"]
    usage = result.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens != args.osl:
        raise RuntimeError(f"expected {args.osl} completion tokens, received {completion_tokens}")

    retokenized = tokenizer.encode(text, add_special_tokens=False)
    actual_output_ids, token_id_source, returned_output_ids = comparison_token_ids(result, tokenizer, text, args.osl)
    artifact = {
        "model": args.model,
        "isl": args.isl,
        "osl": args.osl,
        "prompt_token_count": len(prompt_ids),
        "prompt_token_sha256": hashlib.sha256(json.dumps(prompt_ids, separators=(",", ":")).encode()).hexdigest(),
        "completion_tokens_reported": completion_tokens,
        "output_token_ids": actual_output_ids,
        "output_token_id_source": token_id_source,
        "response_output_token_ids": returned_output_ids,
        "retokenized_output_ids": retokenized,
        "output_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "finish_reason": result["choices"][0].get("finish_reason"),
        "output_text": text,
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
