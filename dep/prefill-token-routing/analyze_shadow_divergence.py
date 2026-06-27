#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(path: Path) -> dict[str, Any]:
    total = 0
    comparable = 0
    divergent = 0
    primary_ties = 0
    shadow_ties = 0
    primary_policy: str | None = None
    shadow_policy: str | None = None
    with path.open(encoding="utf-8") as rows:
        for line in rows:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            primary_policy = row.get("prefill_primary_policy") or primary_policy
            shadow_policy = row.get("prefill_shadow_policy") or shadow_policy
            primary_tie_count = row.get("prefill_primary_tie_count")
            shadow_tie_count = row.get("prefill_shadow_tie_count")
            primary_worker = row.get("prefill_worker_idx")
            shadow_worker = row.get("prefill_shadow_worker_idx")
            if primary_tie_count != 1:
                primary_ties += 1
            if shadow_tie_count != 1:
                shadow_ties += 1
            if (
                primary_tie_count == 1
                and shadow_tie_count == 1
                and primary_worker is not None
                and shadow_worker is not None
            ):
                comparable += 1
                divergent += primary_worker != shadow_worker

    return {
        "path": str(path),
        "primary_policy": primary_policy,
        "shadow_policy": shadow_policy,
        "total_requests": total,
        "comparable_unique_requests": comparable,
        "divergent_unique_requests": divergent,
        "unique_choice_divergence": divergent / comparable if comparable else None,
        "primary_tied_requests": primary_ties,
        "shadow_tied_requests": shadow_ties,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze offline shadow worker-selection divergence.")
    parser.add_argument("reports", nargs="+", type=Path, help="Per-request replay JSONL reports.")
    parser.add_argument(
        "--require",
        action="append",
        type=Path,
        default=[],
        help="Report that must meet --threshold; may be repeated.",
    )
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    paths = list(dict.fromkeys([*args.reports, *args.require]))
    summaries = [summarize(path) for path in paths]
    by_path = {Path(summary["path"]).resolve(): summary for summary in summaries}
    required = [by_path[path.resolve()] for path in args.require]
    gate_passed = bool(required) and all(
        summary["unique_choice_divergence"] is not None and summary["unique_choice_divergence"] >= args.threshold
        for summary in required
    )
    report = {"threshold": args.threshold, "gate_passed": gate_passed, "reports": summaries}
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
