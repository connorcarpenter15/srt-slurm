#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRACE_DIR=${TRACE_DIR:-"${SCRIPT_DIR}/traces"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/shadow-gate-results"}
PYTHON=${PYTHON:-python3}

mkdir -p "$OUTPUT_DIR"
"$PYTHON" "$SCRIPT_DIR/generate_prefill_gate_traces.py" --output-dir "$TRACE_DIR"

ENGINE_ARGS='{"block_size":64,"speedup_ratio":1000.0,"worker_type":"prefill","enable_prefix_caching":true,"max_num_batched_tokens":2048,"max_num_seqs":512}'
DECODE_ARGS='{"block_size":64,"speedup_ratio":1000.0,"worker_type":"decode","enable_prefix_caching":true,"max_num_batched_tokens":2048,"max_num_seqs":512}'

run_direction() {
    local trace_name=$1
    local primary=$2
    local shadow=$3
    local output_prefix="${OUTPUT_DIR}/${trace_name}-${primary}-primary"
    "$PYTHON" -m dynamo.replay \
        "${TRACE_DIR}/prefill-gate-${trace_name}.jsonl" \
        --replay-mode offline \
        --router-mode kv_router \
        --num-prefill-workers 6 \
        --num-decode-workers 2 \
        --replay-concurrency 1024 \
        --trace-block-size 64 \
        --router-config "{\"router_worker_selection_policy\":\"${primary}\",\"router_temperature\":0.0}" \
        --shadow-router-worker-selection-policy "$shadow" \
        --prefill-engine-args "$ENGINE_ARGS" \
        --decode-engine-args "$DECODE_ARGS" \
        --report-json "${output_prefix}.json" \
        --report-jsonl "${output_prefix}.jsonl"
}

for trace_name in no-cache prefix-reuse; do
    run_direction "$trace_name" default prefill-token-balance
    run_direction "$trace_name" prefill-token-balance default
done

"$PYTHON" "$SCRIPT_DIR/analyze_shadow_divergence.py" \
    "${OUTPUT_DIR}/no-cache-default-primary.jsonl" \
    "${OUTPUT_DIR}/no-cache-prefill-token-balance-primary.jsonl" \
    --require "${OUTPUT_DIR}/prefix-reuse-default-primary.jsonl" \
    --require "${OUTPUT_DIR}/prefix-reuse-prefill-token-balance-primary.jsonl" \
    --threshold 0.05 \
    --output "${OUTPUT_DIR}/gate-summary.json"
