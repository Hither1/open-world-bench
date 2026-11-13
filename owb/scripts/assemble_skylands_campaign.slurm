#!/usr/bin/env bash
#SBATCH --job-name=owb-skylands-video
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=assets/demos/skylands-campaign/assemble-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
task="${1:?Pass task a or b}"
case "$task" in a|b) ;; *) exit 2;; esac
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/.cache/skylands-campaign/matplotlib-assembly-${task}"
mkdir -p "$MPLCONFIGDIR"
/n/home13/chloe00/miniforge3/bin/python -u -B -m owb.scripts.skylands_campaign_merge --task "$task"
/n/home13/chloe00/miniforge3/bin/python -u -B -m owb.scripts.skylands_campaign_frame_records --task "$task"
