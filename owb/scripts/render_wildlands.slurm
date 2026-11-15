#!/usr/bin/env bash
#SBATCH --job-name=owb-wildlands-cycles
#SBATCH --partition=gpu_requeue
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtxa6000|rtx6000pro|a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00  # ~20 frames x ~13 s measured on an RTX A6000; frames resume on requeue
#SBATCH --array=0-5%6
#SBATCH --output=assets/demos/wildlands-reference/render-chunk-%A-%a.log

# One GPU per milestone range; owb/scripts/assemble_wildlands.slurm joins them.
# Submit both through owb/scripts/regenerate_wildlands.sh from the repository root.
# Each chunk replays its trace from the start with full checks, then renders its
# checkpoint states in one persistent Blender process (Cycles OptiX, 48 samples,
# native 960x540, OpenImageDenoise). Python exports the next state while Blender
# renders the previous one.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script through sbatch from the repository root}"
test -f owb/scripts/wildlands_reference.py
# Ranges are balanced by checkpoint frame count (A: 20+18, B: 20+18+19+15).
case "$SLURM_ARRAY_TASK_ID" in
  0) task=a; first=1;  last=5;;
  1) task=a; first=6;  last=12;;
  2) task=b; first=1;  last=5;;
  3) task=b; first=6;  last=13;;
  4) task=b; first=14; last=18;;
  5) task=b; first=19; last=24;;
  *) exit 2;;
esac
py="${OWB_PYTHON:-/n/home13/chloe00/miniforge3/bin/python}"
# Node-local disk for Blender and the large per-frame scene snapshots.
local_dir="${SLURM_TMPDIR:-/tmp}/owb-wildlands-${SLURM_JOB_ID}"
mkdir -p "$local_dir/tmp" "$local_dir/cuda"
trap 'rm -rf "$local_dir"' EXIT
export TMPDIR="$local_dir/tmp" CUDA_CACHE_PATH="$local_dir/cuda"
export OWB_BLENDER="$("$py" owb/scripts/stage_blender.py "$local_dir/blender")"
export OWB_CYCLES_DEVICE=OPTIX
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export OWB_FRAME_CACHE="${OWB_FRAME_CACHE:-$PWD/.cache/wildlands-cycles}"
export MPLCONFIGDIR="$OWB_FRAME_CACHE/matplotlib-$SLURM_ARRAY_TASK_ID"
mkdir -p "$MPLCONFIGDIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "Task $task milestones $first-$last; Blender $OWB_BLENDER"
"$py" -u -B -m owb.scripts.wildlands_reference \
  --render-chunk "assets/demos/wildlands-reference/task-${task}-seed-7" \
  --cache "$OWB_FRAME_CACHE" --milestones "$first" "$last" --checkpoints --samples 48 --threads 4
