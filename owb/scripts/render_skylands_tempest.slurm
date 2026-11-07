#!/usr/bin/env bash
#SBATCH --job-name=owb-tempest-cycles
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_rtx
#SBATCH --constraint=rtx6000pro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=assets/demos/skylands-tempest/render-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export OWB_CAMPAIGN_RENDER_HELPERS="$PWD/owb/render"
export CUDA_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/owb-tempest-cuda-${SLURM_JOB_ID}"
mkdir -p "$CUDA_CACHE_PATH"
py=/n/home13/chloe00/miniforge3/bin/python
trace="${TEMPEST_TRACE:-assets/demos/skylands-tempest/extended-seed-7}"
scene="${TEMPEST_SCENE:-.cache/skylands-tempest/extended-scenes}"
frames="${TEMPEST_FRAMES:-assets/demos/skylands-tempest/previews}"
blender_path="$PWD/.cache/render-tools/blender-4.5.4-linux-x64/blender"
if [ ! -x "$blender_path" ]; then
  blender_path=$("$py" owb/scripts/stage_skylands_blender.py "${SLURM_TMPDIR:-/tmp}/owb-tempest-render-${SLURM_JOB_ID}")
fi
if [ ! -f "$scene/frames.json" ]; then
  if [ "${TEMPEST_SHOWCASE:-0}" = 1 ]; then
    "$py" -u -B -m owb.scripts.skylands_tempest_showcase --directory "$trace" --output "$scene"
  else
    "$py" -u -B -m owb.scripts.skylands_tempest export --directory "$trace" --output "$scene"
  fi
fi
"$blender_path" --background --factory-startup --python-exit-code 2 --python "$OWB_CAMPAIGN_RENDER_HELPERS/blender_campaign_baked.py" -- \
  --directory "$scene" --output "$frames" --samples 64 --threads 8 --device OPTIX
