#!/bin/bash
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=coreai_comparch_inferencex-dsv4.grpc-ab-gates
#SBATCH --output=/lustre/fsw/coreai_comparch_inferencex/connorc/logs/dsv4-grpc-ab-gates-%j.out

set -uo pipefail

REPO=/lustre/fsw/coreai_comparch_inferencex/connorc/srt-slurm-dsv4-grpc-ab
ROOT=/lustre/fsw/coreai_comparch_inferencex/connorc/dsv4-grpc-ab
CAMPAIGN=${REPO}/campaigns/dsv4-pro-gb300-grpc-ab
SCRIPT=${CAMPAIGN}/gate-controller.sbatch

"${REPO}/.venv/bin/python" "${CAMPAIGN}/run-campaign.py" \
  --repo-root "${REPO}" \
  --srtctl "${REPO}/.venv/bin/srtctl" \
  --config "${REPO}/srtslurm.yaml" \
  --outputs-root "${ROOT}/outputs" \
  --artifacts-root "${ROOT}/artifacts" \
  --run-plan "${CAMPAIGN}/gate-plan.json" \
  --state "${ROOT}/artifacts/gate-controller-state.json" \
  --max-runtime-seconds 27000
status=$?

if [[ ${status} -eq 75 ]]; then
  echo "Gate controller handoff requested; resubmitting from persistent state."
  sbatch "${SCRIPT}"
  exit $?
fi

if [[ ${status} -eq 0 ]]; then
  echo "All gates passed; submitting the locked 28-leg crossover campaign."
  sbatch "${CAMPAIGN}/campaign-controller.sbatch"
  exit $?
fi

echo "A correctness gate failed; the measured campaign was not submitted." >&2
exit "${status}"
