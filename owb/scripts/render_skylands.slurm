#!/usr/bin/env bash
#SBATCH --job-name=owb-skylands-cycles
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=assets/demos/skylands-reference/render-%A-%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/.cache/skylands-cycles/matplotlib-${SLURM_ARRAY_TASK_ID}"
export CUDA_CACHE_PATH="$PWD/.cache/skylands-cycles/cuda-0"
mkdir -p "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
case "$SLURM_ARRAY_TASK_ID" in
  0) task=a; first=1; last=15 ;;
  1) task=b; first=1; last=24 ;;
esac
python -u -B -m owb.scripts.skylands_reference verify --task "$task"
python -u -B -m owb.scripts.skylands_reference render --task "$task" --device CUDA --first "$first" --last "$last"
python -u -B -m owb.scripts.skylands_reference assemble --task "$task"
