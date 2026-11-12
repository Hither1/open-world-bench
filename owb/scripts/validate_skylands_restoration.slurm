#!/usr/bin/env bash
#SBATCH --job-name=owb-restoration
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
#SBATCH --output=.cache/skylands-restoration/validation-%A-%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
py=/n/home13/chloe00/miniforge3/bin/python
case "$SLURM_ARRAY_TASK_ID" in
  0)
    "$py" -u -B -m owb.scripts.skylands_restoration run --objective extended --seed 7 \
      --route synchronization --order monitoring anchors cooling \
      --directory assets/demos/skylands-restoration/extended-seed-7
    ;;
  1)
    "$py" -u -B -m owb.scripts.skylands_restoration run --objective extended --seed 23 \
      --route grounding --order anchors cooling monitoring \
      --directory assets/demos/skylands-restoration/extended-grounding-seed-23
    ;;
  2)
    "$py" -u -B -m owb.scripts.skylands_restoration run --objective open --seed 7 \
      --route capacitors --directory assets/demos/skylands-restoration/open-capacitors-seed-7
    ;;
  3)
    "$py" -u -B -m owb.scripts.skylands_restoration run --objective open --seed 23 \
      --route grounding --directory assets/demos/skylands-restoration/open-grounding-seed-23
    ;;
esac
