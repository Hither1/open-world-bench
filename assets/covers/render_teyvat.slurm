#!/usr/bin/env bash
#SBATCH --job-name=owb-teyvat-cover
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtxa6000|rtx6000pro|a40|a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=.cache/teyvat-cover-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
destination="${1:?Pass a new output directory}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
cover_local="${SLURM_TMPDIR:-/tmp}/owb-teyvat-cover-${SLURM_JOB_ID}"
mkdir -p "$cover_local"
export TMPDIR="$cover_local" CUDA_CACHE_PATH="$cover_local/cuda"
py="${OWB_PYTHON:-python}"
blender="$("$py" owb/scripts/stage_blender.py "$cover_local/blender")"
nvidia-smi --query-gpu=name,driver_version --format=csv
"$py" -B assets/covers/export_teyvat.py "$destination"
"$blender" --background --factory-startup --python-exit-code 1 \
  --python assets/covers/render.py -- \
  --scene "$destination/teyvat.npz" --output "$destination/teyvat.png" \
  --blend "$destination/teyvat.blend" --samples 128 --threads 8 --device OPTIX
