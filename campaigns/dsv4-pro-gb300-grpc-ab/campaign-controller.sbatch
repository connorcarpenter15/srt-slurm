#!/bin/bash
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=coreai_comparch_inferencex-dsv4.grpc-ab-controller
#SBATCH --output=/lustre/fsw/coreai_comparch_inferencex/connorc/logs/dsv4-grpc-ab-controller-%j.out

set -uo pipefail

REPO=/lustre/fsw/coreai_comparch_inferencex/connorc/srt-slurm-dsv4-grpc-ab
ROOT=/lustre/fsw/coreai_comparch_inferencex/connorc/dsv4-grpc-ab
SCRIPT=${REPO}/campaigns/dsv4-pro-gb300-grpc-ab/campaign-controller.sbatch
TIME_LIMIT=${CONTROLLER_TIME_LIMIT:-08:00:00}

"${REPO}/.venv/bin/python" "${REPO}/campaigns/dsv4-pro-gb300-grpc-ab/run-campaign.py" \
  --repo-root "${REPO}" \
  --srtctl "${REPO}/.venv/bin/srtctl" \
  --config "${REPO}/srtslurm.yaml" \
  --outputs-root "${ROOT}/outputs" \
  --artifacts-root "${ROOT}/artifacts" \
  --max-runtime-seconds 27000
status=$?

if [[ ${status} -eq 75 ]]; then
  echo "Controller handoff requested; resubmitting from persistent state."
  sbatch --time="${TIME_LIMIT}" --export="ALL,CONTROLLER_TIME_LIMIT=${TIME_LIMIT}" "${SCRIPT}"
  exit $?
fi

exit "${status}"
