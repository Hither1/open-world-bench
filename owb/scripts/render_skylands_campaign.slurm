#!/usr/bin/env bash
#SBATCH --job-name=owb-skylands-full-cycles
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=assets/demos/skylands-campaign/render-%A-%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
export OWB_CAMPAIGN_RENDER_HELPERS="$PWD/assets/demos/skylands-campaign/render-source"
export MPLCONFIGDIR="$PWD/.cache/skylands-campaign/matplotlib-${SLURM_ARRAY_TASK_ID}"
export CUDA_CACHE_PATH="$PWD/.cache/skylands-cycles/cuda-0"
mkdir -p "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then task=a; else task=b; fi
trace="assets/demos/skylands-campaign/task-${task}-seed-7"
scene=".cache/skylands-campaign/video-${task}"
frames=".cache/skylands-campaign/full-frames-${task}"
py=/n/home13/chloe00/miniforge3/bin/python
stage_dir="${SLURM_TMPDIR:-/tmp}/owb-skylands-render-${SLURM_JOB_ID}"
blender=$("$py" owb/scripts/stage_skylands_blender.py "$stage_dir")
export CUDA_CACHE_PATH="$stage_dir/cuda"
if [ ! -f "$scene/frames.json" ]; then
  "$py" -u -B -m owb.scripts.skylands_campaign_video export --directory "$trace" --output "$scene"
fi
"$blender" --background --factory-startup --python-exit-code 2 --python "$OWB_CAMPAIGN_RENDER_HELPERS/blender_campaign_baked.py" -- --directory "$scene" --output "$frames" --samples 64 --threads 8 --device OPTIX
"$py" -u -B -m owb.scripts.skylands_campaign_video assemble --directory "$scene" --render-dir "$frames" --output "$trace"
