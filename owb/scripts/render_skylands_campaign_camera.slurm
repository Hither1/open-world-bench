#!/usr/bin/env bash
#SBATCH --job-name=owb-skylands-camera
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtxa6000|rtx6000pro|a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G  # Blender holds ~52 GB with the presentation landscape; 64 GB was OOM-killed
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --output=assets/demos/skylands-campaign/render-camera-%A-%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export OWB_CAMPAIGN_RENDER_HELPERS="$PWD/assets/demos/skylands-campaign/render-source"
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then task=a; else task=b; fi
count=$(/n/home13/chloe00/miniforge3/bin/python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["entries"]))' ".cache/skylands-campaign/camera-only-${task}/frames.json")
if [ "$count" = 0 ]; then exit 0; fi
stage_dir="${SLURM_TMPDIR:-/tmp}/owb-skylands-render-${SLURM_JOB_ID}"
blender_path=$(/n/home13/chloe00/miniforge3/bin/python owb/scripts/stage_skylands_blender.py "$stage_dir")
export CUDA_CACHE_PATH="$stage_dir/cuda"
mkdir -p "$CUDA_CACHE_PATH"
"$blender_path" --background --factory-startup --python-exit-code 2 --python "$OWB_CAMPAIGN_RENDER_HELPERS/blender_campaign_baked.py" -- --directory ".cache/skylands-campaign/camera-only-${task}" --output ".cache/skylands-campaign/camera-frames-${task}" --samples 64 --threads 8 --device OPTIX
