#!/usr/bin/env bash
#SBATCH --job-name=owb-skylands-cycles
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtxa6000|rtx6000pro|a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G  # Blender holds ~52 GB with the presentation landscape; 64 GB was OOM-killed
#SBATCH --time=3:00:00  # landscape geometry adds render time; frames resume on requeue
#SBATCH --array=0-5%6
#SBATCH --output=assets/demos/skylands-campaign/render-chunk-%A-%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export OWB_CAMPAIGN_RENDER_HELPERS="$PWD/assets/demos/skylands-campaign/render-source"
stage_dir="${SLURM_TMPDIR:-/tmp}/owb-skylands-render-${SLURM_JOB_ID}"
blender_path=$(/n/home13/chloe00/miniforge3/bin/python owb/scripts/stage_skylands_blender.py "$stage_dir")
export CUDA_CACHE_PATH="$stage_dir/cuda"
mkdir -p "$CUDA_CACHE_PATH"
case "$SLURM_ARRAY_TASK_ID" in
  0) task=a; first=0; last=279;;
  1) task=a; first=279; last=557;;
  2) task=b; first=0; last=293;;
  3) task=b; first=585; last=878;;
  4) task=b; first=878; last=1170;;
  5) task=b; first=293; last=585;;
  *) exit 2;;
esac
"$blender_path" --background --factory-startup --python-exit-code 2 --python "$OWB_CAMPAIGN_RENDER_HELPERS/blender_campaign_baked.py" -- --directory ".cache/skylands-campaign/video-${task}" --output ".cache/skylands-campaign/final-frames-${task}-${SLURM_ARRAY_TASK_ID}" --first "$first" --last "$last" --samples 64 --threads 8 --device OPTIX
