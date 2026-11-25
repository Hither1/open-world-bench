#!/usr/bin/env bash
#SBATCH --job-name=owb-ckpt-bank
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=assets/evaluation/checkpoint-bank/collect-%x-%j.log

# Collect one Skylands campaign checkpoint bank on the current code.
#   sbatch --export=ALL,TASK=a owb/scripts/checkpoint_bank.slurm
# Not requeued: a requeue would restart into an existing actions.jsonl and refuse.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit through sbatch from the repository root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
"${OWB_PYTHON:-/n/home13/chloe00/miniforge3/bin/python}" -u -B -m owb.scripts.checkpoint_bank collect \
  --world "${WORLD:-skylands_campaign}" --task "${TASK:?set TASK=a or b}" --seed "${SEED:-7}"
