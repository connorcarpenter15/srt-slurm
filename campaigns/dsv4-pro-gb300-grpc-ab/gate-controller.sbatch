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
#SBATCH --job-name=coreai_comparch_inferencex-dsv4.grpc-ab-gates
#SBATCH --output=/lustre/fsw/coreai_comparch_inferencex/connorc/logs/dsv4-grpc-ab-gates-%j.out

set -uo pipefail

REPO=/lustre/fsw/coreai_comparch_inferencex/connorc/srt-slurm-dsv4-grpc-ab
ROOT=/lustre/fsw/coreai_comparch_inferencex/connorc/dsv4-grpc-ab-ba4c325301
CAMPAIGN=${REPO}/campaigns/dsv4-pro-gb300-grpc-ab
SCRIPT=${CAMPAIGN}/gate-controller.sbatch
TIME_LIMIT=${CONTROLLER_TIME_LIMIT:-08:00:00}
adopt_args=()
controller_pid=
if [[ -n ${ADOPT_SMOKE_LEGACY_JOB:-} ]]; then
  adopt_args+=(--adopt-job "1:0:${ADOPT_SMOKE_LEGACY_JOB}")
fi

# shellcheck disable=SC2329 # Invoked by the USR1/TERM trap.
forward_usr1() {
  if [[ -n ${controller_pid} ]]; then
    kill -USR1 "${controller_pid}" 2>/dev/null || true
  fi
}
# gb200-backfill uses PreemptMode=CANCEL with a grace period. Catch TERM so
# run-campaign.py can checkpoint and this wrapper can submit its successor.
trap forward_usr1 USR1 TERM

"${REPO}/.venv/bin/python" "${CAMPAIGN}/run-campaign.py" \
  --repo-root "${REPO}" \
  --srtctl "${REPO}/.venv/bin/srtctl" \
  --config "${REPO}/srtslurm.yaml" \
  --outputs-root "${ROOT}/outputs" \
  --artifacts-root "${ROOT}/artifacts" \
  --run-plan "${CAMPAIGN}/gate-plan.json" \
  --state "${ROOT}/artifacts/gate-controller-state.json" \
  --max-runtime-seconds 27000 \
  "${adopt_args[@]}" &
controller_pid=$!
while true; do
  wait "${controller_pid}"
  status=$?
  if ! kill -0 "${controller_pid}" 2>/dev/null; then
    break
  fi
done

if [[ ${status} -eq 75 ]]; then
  echo "Gate controller handoff requested; resubmitting from persistent state."
  sbatch --time="${TIME_LIMIT}" --export="ALL,CONTROLLER_TIME_LIMIT=${TIME_LIMIT}" "${SCRIPT}"
  exit $?
fi

if [[ ${status} -eq 0 ]]; then
  echo "All gates passed; submitting the locked 28-leg crossover campaign."
  sbatch --time="${TIME_LIMIT}" --export="ALL,CONTROLLER_TIME_LIMIT=${TIME_LIMIT}" \
    "${CAMPAIGN}/campaign-controller.sbatch"
  exit $?
fi

echo "A correctness gate failed; the measured campaign was not submitted." >&2
exit "${status}"
