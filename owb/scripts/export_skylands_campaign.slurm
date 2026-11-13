#!/usr/bin/env bash
#SBATCH --job-name=owb-skylands-export
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-1%2
#SBATCH --output=assets/demos/skylands-campaign/export-%A-%a.log
# Replay each verified campaign trace (positions, events, rewards, milestone
# digests) and export static/dynamic presentation snapshots plus the camera
# visibility review. Safe to rerun after preemption: incomplete outputs are redone.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/.cache/skylands-campaign/matplotlib-export-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"
py="${OWB_PYTHON:-/n/home13/chloe00/miniforge3/bin/python}"
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then task=a; else task=b; fi
cache=.cache/skylands-campaign
if [ ! -f "$cache/video-${task}/frames.json" ]; then
  rm -rf "$cache/video-${task}"
  "$py" -u -B -m owb.scripts.skylands_campaign_video export \
    --directory "assets/demos/skylands-campaign/task-${task}-seed-7" --output "$cache/video-${task}"
fi
if [ ! -f "$cache/video-camera-${task}/frames.json" ] || [ ! -f "$cache/camera-only-${task}/frames.json" ]; then
  rm -rf "$cache/video-camera-${task}" "$cache/camera-only-${task}"
  "$py" -u -B -m owb.scripts.skylands_campaign_camera --task "$task"
fi
