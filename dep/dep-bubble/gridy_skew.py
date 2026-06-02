#!/usr/bin/env python3
"""Sharp expert-skew test: mean gridY (token-tile dim) per rank for the dynamic
grouped-GEMM kernels. gridY scales with tokens routed to a rank's local experts,
so rank3 mean gridY > others => expert-load skew. Balanced => no skew.
attn gridY is fixed (control)."""
import sqlite3

BASE = "/lustre/fsw/coreai_dlfw_dev/connorc/srt-slurm/outputs/1977752/logs/profiles/agg"
RANKS = [0, 1, 2, 3]
BUCKETS = [
    ("expert_up", "bmm_E2m1_E2m1E2m1%"),
    ("expert_down", "bmm_Bfloat16_E2m1E2m1%"),
    ("fi_fp4_gemm", "void cutlass::device_kernel<flashinfer::gemm::DeviceGemmFp4GemmSm100%"),
    ("moe_finalize", "void moe::dev::finalize::finalizeKernelVecLoad%"),
    ("topk_gating", "void vllm::moe::topkGating%"),
]

# AVG, and p50/p90 via window-free approximation: AVG + STDDEV-ish not in sqlite,
# so also pull a coarse histogram (count by gridY bucket of width 8) for expert_up.
def avg_gridy(con, pat):
    row = con.execute(
        "SELECT AVG(k.gridY), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName=s.id WHERE s.value LIKE ?", (pat,)
    ).fetchone()
    return row[0], row[1]

res = {name: {} for name, _ in BUCKETS}
for r in RANKS:
    con = sqlite3.connect(f"file:{BASE}/lyris0243_agg_w0_rank{r}_profile.sqlite?mode=ro", uri=True)
    for name, pat in BUCKETS:
        avg, n = avg_gridy(con, pat)
        res[name][r] = (avg, n)
    con.close()

print("=== mean gridY (token-tile dim) per rank ===")
print(f"{'bucket':<14} " + "  ".join(f"r{r:>9}" for r in RANKS) + "    max/min")
for name, _ in BUCKETS:
    avgs = [res[name][r][0] for r in RANKS]
    ratio = max(avgs) / min(avgs) if min(avgs) else float("nan")
    cells = "  ".join(f"{a:>10.3f}" for a in avgs)
    print(f"{name:<14} {cells}    {ratio:.4f}x")
