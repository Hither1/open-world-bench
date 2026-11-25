#!/usr/bin/env bash
#SBATCH --job-name=owb-model-sweep
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm-%x-%j.log

# Run one model sweep (several tasks x checkpoint starts concurrently) and its report.
# The first argument picks the world; the rest go to the runner unchanged:
#
#   sbatch owb/scripts/model_sweep.slurm wildlands --task a b --start-milestone 0 3 6 9 \
#       --stop-after-milestones 3 --condition perception_oracle --output runs/NAME \
#       --transport codex --account-home ~/.codex-roboharness04 \
#       --codex-executable .cache/codex-sol/node_modules/.bin/codex
#
# Skylands campaign worlds use about 10 GB each: raise --mem to ~12G per concurrent episode
# (sbatch --mem=100G ... skylands ... for 8 episodes).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit through sbatch from the repository root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
world="${1:?first argument: wildlands or skylands}"; shift
case "$world" in
  wildlands) module=owb.scripts.wildlands_model ;;
  skylands)  module=owb.scripts.skylands_model ;;
  *) echo "unknown world $world" >&2; exit 2 ;;
esac
output=""; prev=""
for arg in "$@"; do [[ "$prev" == "--output" ]] && output="$arg"; prev="$arg"; done
PY="${OWB_PYTHON:-/n/home13/chloe00/miniforge3/bin/python}"
"$PY" -u -B -m "$module" "$@"
[[ -n "$output" ]] && "$PY" -B -m owb.scripts.checkpoint_report "$output"
