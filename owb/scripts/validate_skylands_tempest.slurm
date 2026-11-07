#!/usr/bin/env bash
#SBATCH --job-name=owb-tempest-validate
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
#SBATCH --output=assets/demos/skylands-tempest/validation-%A-%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
py=/n/home13/chloe00/miniforge3/bin/python
case "$SLURM_ARRAY_TASK_ID" in
  0)
    "$py" -u -B -m owb.scripts.skylands_tempest_recollect --candidate .cache/tempest-development/candidate-sync \
      --directory assets/demos/skylands-tempest/release-extended-seed-7 --objective extended --seed 7 \
      --route synchronization --order monitoring anchors cooling
    ;;
  1)
    "$py" -u -B -m owb.scripts.skylands_tempest_recollect --candidate .cache/tempest-development/candidate-grounding \
      --directory assets/demos/skylands-tempest/release-extended-grounding-seed-23 --objective extended --seed 23 \
      --route grounding --order anchors cooling monitoring
    ;;
  2)
    "$py" -u -B -m owb.scripts.skylands_tempest_recollect --candidate .cache/tempest-development/candidate-open-capacitors \
      --directory assets/demos/skylands-tempest/release-open-capacitors-seed-7 --objective open --seed 7 --route capacitors
    "$py" -u -B -m owb.scripts.skylands_tempest_recollect --candidate .cache/tempest-development/candidate-open-grounding \
      --directory assets/demos/skylands-tempest/release-open-grounding-seed-23 --objective open --seed 23 --route grounding
    ;;
  3)
    "$py" -u -B -m pytest -q -p no:cacheprovider \
      --basetemp="${SLURM_TMPDIR:-/tmp}/owb-tempest-tests-${SLURM_JOB_ID}" \
      --junitxml=assets/demos/skylands-tempest/tests.xml \
      tests/test_skylands_tempest.py tests/test_skylands_reference.py tests/test_skylands_campaign_render.py \
      tests/test_skylands.py tests/test_framework.py
    ;;
esac
