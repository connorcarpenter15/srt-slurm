#!/usr/bin/env python3
"""Per-concurrency-point DP queue-skew + throughput for a single-node sweep job.

The concurrency sweep (recipe ...-sweep-conc2048to6144) runs several concurrency
points back-to-back in ONE job on ONE node, so the backend vLLM log holds all
points' `Engine NNN:` interval lines concatenated. To get per-point running/
waiting skew we must slice that stream by each point's measured window.

Segmentation is tz-free: each measured point writes its own
`request_trace_concurrency_<C>_gpus_<G>.jsonl` (warmup runs do NOT get a trace),
whose `wall_time_ns` epoch timestamps bound the point exactly. We convert the
ISO-UTC Engine timestamps to epoch seconds and keep only the rows inside each
point's [min, max] wall-clock window.

For each point we report:
  - per-rank running/waiting means (is the queue draining? Waiting ~ 0?)
  - temporal running_skew/waiting_skew quantiles (do ranks fall out of lockstep?)
  - throughput + TTFT pulled from benchmark-rollup.json

The knee we are hunting: the largest concurrency where Waiting still ~ 0 (queue
drains) but output throughput has stopped scaling with concurrency — i.e. where
the temporal underfeed starts costing tok/s instead of being absorbed by slack.

Usage:
  python3 sweep_skew.py <job_log_dir> <node_prefix> <conc1,conc2,...> [--gpus 4] [--bin-seconds 10]
e.g.
  python3 sweep_skew.py outputs/2191166/logs ptyche0267 2048,3072,4096,6144
expects:
  <job_log_dir>/<node_prefix>_agg_w0.out
  <job_log_dir>/sa-bench_isl_2_osl_1024/request_trace_concurrency_<C>_gpus_<G>.jsonl
  <job_log_dir>/benchmark-rollup.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ENGINE_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)Z?.*"
    r"Engine (?P<rank>\d+): Avg prompt throughput: (?P<prompt>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<gen>[0-9.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[0-9.]+)%"
)


def engine_epoch(ts: str) -> float:
    return dt.datetime.fromisoformat(ts).replace(tzinfo=dt.UTC).timestamp()


def parse_engine_rows(log_path: Path) -> list[dict]:
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_RE.sub("", line)
        m = ENGINE_RE.search(line)
        if not m:
            continue
        rows.append(
            {
                "epoch": engine_epoch(m.group("ts")),
                "rank": int(m.group("rank")),
                "gen": float(m.group("gen")),
                "running": int(m.group("running")),
                "waiting": int(m.group("waiting")),
                "kv": float(m.group("kv")),
            }
        )
    return rows


def trace_window(trace_path: Path) -> tuple[float, float] | None:
    lo = hi = None
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            ns = json.loads(line).get("wall_time_ns")
        except json.JSONDecodeError:
            continue
        if ns is None:
            continue
        sec = ns / 1e9
        lo = sec if lo is None else min(lo, sec)
        hi = sec if hi is None else max(hi, sec)
    if lo is None:
        return None
    return lo, hi


def quantiles(values: list[float]) -> tuple[float, float, float, float]:
    values = sorted(values)
    return (
        statistics.fmean(values),
        values[len(values) // 2],
        values[min(len(values) - 1, int(len(values) * 0.95))],
        values[-1],
    )


def rollup_by_conc(rollup_path: Path) -> dict[int, dict]:
    if not rollup_path.exists():
        return {}
    data = json.loads(rollup_path.read_text())
    out = {}
    for run in data.get("runs", []):
        out[int(run["concurrency"])] = run
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", type=Path)
    ap.add_argument("node_prefix")
    ap.add_argument("concurrencies", help="comma-separated, e.g. 2048,3072,4096,6144")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--isl", type=int, default=2)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--bin-seconds", type=int, default=10)
    args = ap.parse_args()

    concs = [int(c) for c in args.concurrencies.split(",")]
    backend_log = args.log_dir / f"{args.node_prefix}_agg_w0.out"
    trace_dir = args.log_dir / f"sa-bench_isl_{args.isl}_osl_{args.osl}"
    rollup = rollup_by_conc(args.log_dir / "benchmark-rollup.json")

    rows = parse_engine_rows(backend_log)
    if not rows:
        print(f"No Engine metrics found in {backend_log}")
        return
    print(f"# Concurrency-sweep DP skew  ({backend_log.name}, {len(rows)} Engine samples)\n")

    header = (
        f"{'conc':>6} {'agg_gen_tok/s':>13} {'roll_tok/s':>11} {'req/s':>8} {'ttft_mean_ms':>13} "
        f"{'run/rank':>9} {'wait/rank':>10} {'run_skew p50/p95/max':>22} {'wait_skew max':>13} {'samp':>5}"
    )
    print(header)
    print("-" * len(header))

    for conc in concs:
        trace = trace_dir / f"request_trace_concurrency_{conc}_gpus_{args.gpus}.jsonl"
        win = trace_window(trace) if trace.exists() else None
        if win is None:
            print(f"{conc:>6}  (no request trace at {trace.name} — skipped)")
            continue
        lo, hi = win
        pt = [r for r in rows if lo <= r["epoch"] <= hi]
        if not pt:
            print(f"{conc:>6}  (no Engine rows in window — skipped)")
            continue

        by_rank: dict[int, list[dict]] = defaultdict(list)
        for r in pt:
            by_rank[r["rank"]].append(r)
        run_means = [statistics.fmean([x["running"] for x in by_rank[k]]) for k in by_rank]
        wait_means = [statistics.fmean([x["waiting"] for x in by_rank[k]]) for k in by_rank]
        gen_means = {k: statistics.fmean([x["gen"] for x in by_rank[k]]) for k in by_rank}
        run_per_rank = statistics.fmean(run_means)
        wait_per_rank = statistics.fmean(wait_means)
        # Backend aggregate decode throughput = sum of per-rank mean gen tok/s.
        # Robust to client-side rollup truncation (the rollup throughput_toks can
        # under-report when a point ends mid-wave); this is the true serving rate.
        agg_gen = sum(gen_means.values())

        bins: dict[int, dict[int, dict]] = defaultdict(dict)
        base = min(r["epoch"] for r in pt)
        for r in sorted(pt, key=lambda x: x["epoch"]):
            b = int((r["epoch"] - base) // args.bin_seconds)
            bins[b][r["rank"]] = r
        run_skew, wait_skew = [], []
        for b, latest in bins.items():
            if len(latest) < 2:
                continue
            rr = [v["running"] for v in latest.values()]
            ww = [v["waiting"] for v in latest.values()]
            run_skew.append(max(rr) - min(rr))
            wait_skew.append(max(ww) - min(ww))
        if run_skew:
            _, rs50, rs95, rsmax = quantiles([float(x) for x in run_skew])
            wsmax = max(wait_skew)
        else:
            rs50 = rs95 = rsmax = wsmax = float("nan")

        ru = rollup.get(conc, {})
        thru = ru.get("throughput_toks")
        reqps = ru.get("request_throughput")
        ttft = ru.get("ttft_mean_ms")

        def f(v, nd=0):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"

        print(
            f"{conc:>6} {agg_gen:>13.0f} {f(thru):>11} {f(reqps,1):>8} {f(ttft,0):>13} "
            f"{run_per_rank:>9.0f} {wait_per_rank:>10.0f} "
            f"{rs50:>6.0f}/{rs95:>4.0f}/{rsmax:>4.0f}{'':>6} {wsmax:>13.0f} {len(pt):>5}"
        )

    print(
        "\nUse agg_gen_tok/s (backend, robust) as the throughput signal; roll_tok/s is "
        "client-side and under-reports on truncated points.\nKnee = largest conc with "
        "wait/rank ~ 0 (queue drains) but agg_gen_tok/s no longer rising with conc.\n"
        "run_skew>0 with wait/rank~0 => temporal underfeed; if it coincides with the "
        "agg_gen plateau, the skew is costing tok/s."
    )


if __name__ == "__main__":
    main()
