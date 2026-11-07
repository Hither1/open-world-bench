#!/usr/bin/env bash
#SBATCH --job-name=owb-tempest-final
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_rtx
#SBATCH --constraint=rtx6000pro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=assets/demos/skylands-tempest/final-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
children=()
for task_index in 0 1 2 3; do
  (
    export SLURM_ARRAY_TASK_ID="$task_index"
    bash owb/scripts/validate_skylands_tempest.slurm
  ) >"assets/demos/skylands-tempest/validation-${SLURM_JOB_ID}-${task_index}.log" 2>&1 &
  children+=("$!")
done
result=0
for child in "${children[@]}"; do
  wait "$child" || result=1
done
if [ "$result" != 0 ]; then
  echo 'Validation failed; inspect the per-task validation logs.'
  exit 1
fi
export TEMPEST_TRACE=assets/demos/skylands-tempest/release-extended-seed-7
export TEMPEST_SCENE=.cache/skylands-tempest/verified-trace-scenes
export TEMPEST_FRAMES=assets/demos/skylands-tempest/trace-previews
export TEMPEST_SHOWCASE=0
bash owb/scripts/render_skylands_tempest.slurm
