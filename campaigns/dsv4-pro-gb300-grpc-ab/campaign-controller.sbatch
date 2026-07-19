#!/bin/bash
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=coreai_comparch_inferencex-dsv4.grpc-ab-controller
#SBATCH --output=/lustre/fsw/coreai_comparch_inferencex/connorc/logs/dsv4-grpc-ab-controller-%j.out

set -uo pipefail

REPO=/lustre/fsw/coreai_comparch_inferencex/connorc/srt-slurm-dsv4-grpc-ab
ROOT=/lustre/fsw/coreai_comparch_inferencex/connorc/dsv4-grpc-ab-ba4c325301
SCRIPT=${REPO}/campaigns/dsv4-pro-gb300-grpc-ab/campaign-controller.sbatch
TIME_LIMIT=${CONTROLLER_TIME_LIMIT:-08:00:00}
controller_pid=

# shellcheck disable=SC2329 # Invoked by the USR1/TERM trap.
forward_usr1() {
  if [[ -n ${controller_pid} ]]; then
    kill -USR1 "${controller_pid}" 2>/dev/null || true
  fi
}
# gb200-backfill uses PreemptMode=CANCEL with a grace period. Catch TERM so
# run-campaign.py can checkpoint and this wrapper can submit its successor.
trap forward_usr1 USR1 TERM

"${REPO}/.venv/bin/python" "${REPO}/campaigns/dsv4-pro-gb300-grpc-ab/run-campaign.py" \
  --repo-root "${REPO}" \
  --srtctl "${REPO}/.venv/bin/srtctl" \
  --config "${REPO}/srtslurm.yaml" \
  --outputs-root "${ROOT}/outputs" \
  --artifacts-root "${ROOT}/artifacts" \
  --max-runtime-seconds 27000 &
controller_pid=$!
while true; do
  wait "${controller_pid}"
  status=$?
  if ! kill -0 "${controller_pid}" 2>/dev/null; then
    break
  fi
done

if [[ ${status} -eq 75 ]]; then
  echo "Controller handoff requested; resubmitting from persistent state."
  sbatch --time="${TIME_LIMIT}" --export="ALL,CONTROLLER_TIME_LIMIT=${TIME_LIMIT}" "${SCRIPT}"
  exit $?
fi

analysis_args=(
  --artifacts-root "${ROOT}/artifacts"
  --campaign-manifest "${REPO}/campaigns/dsv4-pro-gb300-grpc-ab/campaign-manifest.json"
  --run-plan "${REPO}/campaigns/dsv4-pro-gb300-grpc-ab/run-plan.json"
  --public-curve "${REPO}/campaigns/dsv4-pro-gb300-grpc-ab/public-curve-2026-07-13.json"
  --output-dir "${ROOT}/artifacts/analysis"
)
if [[ ${status} -ne 0 ]]; then
  analysis_args+=(--allow-incomplete)
fi

analysis_status=0
"${REPO}/.venv/bin/python" \
  "${REPO}/campaigns/dsv4-pro-gb300-grpc-ab/analyze-results.py" \
  "${analysis_args[@]}" || analysis_status=$?
if [[ -f ${ROOT}/artifacts/analysis/report.md ]]; then
  report_date=$(date +%F)
  cp "${ROOT}/artifacts/analysis/report.md" \
    "${ROOT}/artifacts/analysis/benchmark_inferencex_dsv4_pro_gb300_grpc_ab_8k1k_${report_date}.md"
fi

if [[ ${status} -eq 0 && ${analysis_status} -ne 0 ]]; then
  echo "The campaign completed, but result analysis failed." >&2
  exit "${analysis_status}"
fi

exit "${status}"
