#!/usr/bin/env python3
"""Per-rank kernel-shape decomposition for the DEP EP-bubble straggler test.

Discriminator: attention (fmha decode) is DP-balanced -> control for raw GPU speed.
Expert GEMM (bmm_E2m1*) is EP-load-skewed -> treatment.
  rank3 slower ONLY in expert GEMM (attn equal) => expert-load skew.
  rank3 slower in BOTH attn and expert GEMM     => hardware/execution asymmetry.
Also reports gridX min/max per bucket: varying grid => dynamic problem size (token count);
constant grid => padded (CUDA-graph capture), so lean on the duration-ratio argument.
"""
import sqlite3
import sys

BASE = "/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1977752/logs/profiles/agg"
RANKS = [0, 1, 2, 3]

BUCKETS = [
    ("attn_decode", "fmhaSm100fKernel_QkvE4m3OBfloat16H128PagedKvCausalP16VarSeqQ16Kv128%"),
    ("expert_up", "bmm_E2m1_E2m1E2m1%"),
    ("expert_down", "bmm_Bfloat16_E2m1E2m1%"),
    ("fi_fp4_gemm", "void cutlass::device_kernel<flashinfer::gemm::DeviceGemmFp4GemmSm100%"),
    ("moe_finalize", "void moe::dev::finalize::finalizeKernelVecLoad%"),
    ("topk_gating", "void vllm::moe::topkGating%"),
    ("combine_rs", "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL%"),
    ("dispatch_ag", "ncclDevKernel_AllGather_RING_LL%"),
]

case_sql = "\n".join(
    f"    WHEN s.value LIKE '{pat}' THEN '{name}'" for name, pat in BUCKETS
)
QUERY = f"""
SELECT bucket, COUNT(*) n, SUM(dur)/1e6 ms, AVG(dur)/1e3 avg_us,
       MIN(gx) gx_min, MAX(gx) gx_max, MIN(gy) gy_min, MAX(gy) gy_max
FROM (
  SELECT (k.end-k.start) dur, k.gridX gx, k.gridY gy,
    CASE
{case_sql}
    ELSE NULL END bucket
  FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds s ON k.demangledName = s.id
)
WHERE bucket IS NOT NULL
GROUP BY bucket;
"""

# results[bucket][rank] = dict
results = {name: {} for name, _ in BUCKETS}
for r in RANKS:
    con = sqlite3.connect(f"file:{BASE}/lyris0243_agg_w0_rank{r}_profile.sqlite?mode=ro", uri=True)
    for row in con.execute(QUERY):
        bucket, n, ms, avg_us, gxn, gxx, gyn, gyx = row
        results[bucket][r] = dict(n=n, ms=ms, avg_us=avg_us, gx=(gxn, gxx), gy=(gyn, gyx))
    con.close()

def fmt_avg_table():
    print("=== per-kernel AVG duration (us), by rank ===")
    print(f"{'bucket':<14} " + "  ".join(f"r{r:>8}" for r in RANKS) + "   slow/fast")
    for name, _ in BUCKETS:
        row = results[name]
        if not row:
            continue
        avgs = {r: row.get(r, {}).get("avg_us") for r in RANKS}
        vals = [avgs[r] for r in RANKS if avgs[r] is not None]
        ratio = (max(vals) / min(vals)) if vals and min(vals) > 0 else float("nan")
        cells = "  ".join(f"{(avgs[r] if avgs[r] is not None else 0):>9.2f}" for r in RANKS)
        print(f"{name:<14} {cells}   {ratio:>5.3f}x")

def fmt_total_table():
    print("\n=== per-kernel TOTAL time (ms), by rank (compute work proxy) ===")
    print(f"{'bucket':<14} " + "  ".join(f"r{r:>8}" for r in RANKS))
    for name, _ in BUCKETS:
        row = results[name]
        if not row:
            continue
        cells = "  ".join(f"{(row.get(r, {}).get('ms') or 0):>9.1f}" for r in RANKS)
        print(f"{name:<14} {cells}")

def fmt_grid_table():
    print("\n=== gridX [min,max] and gridY [min,max], by rank (constant => padded) ===")
    for name, _ in BUCKETS:
        row = results[name]
        if not row:
            continue
        print(f"  {name}:")
        for r in RANKS:
            d = row.get(r)
            if d:
                print(f"    r{r}: gridX[{d['gx'][0]},{d['gx'][1]}]  gridY[{d['gy'][0]},{d['gy'][1]}]  n={d['n']}")

fmt_avg_table()
fmt_total_table()
fmt_grid_table()
