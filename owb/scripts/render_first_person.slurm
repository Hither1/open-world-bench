#!/usr/bin/env bash
# First-person Cycles recordings (character's eyes) from verified reference traces.
#   GPU stages:  sbatch --array=0-7 owb/scripts/render_first_person.slurm skylands-render a 8
#                sbatch --array=0-5 owb/scripts/render_first_person.slurm teyvat-render a 6
#   CPU stages (--wrap skips the GPU directives below):
#     sbatch -p shared -c 4 --mem 48G -t 4:00:00 -o "$OWB_RENDER_SCRATCH/logs/%x-%j.log" --wrap "bash owb/scripts/render_first_person.slurm skylands-export a"
#     sbatch -p shared -c 4 --mem 16G -t 1:00:00 -o ... --wrap "bash owb/scripts/render_first_person.slurm skylands-assemble a 8"
#     sbatch -p shared -c 2 --mem 8G -t 0:30:00 -o ... --wrap "bash owb/scripts/render_first_person.slurm teyvat-assemble a"
# Thousands of intermediate files go to OWB_RENDER_SCRATCH (netscratch), not the repo.
#SBATCH --job-name=owb-first-person
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtxa6000|rtx6000pro|a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/n/netscratch/sham_lab/Lab/chloe00/owb-render-scratch/logs/%x-%A-%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
operation="${1:?operation}"; task="${2:?task a or b}"; chunks="${3:-8}"
export OWB_RENDER_SCRATCH="${OWB_RENDER_SCRATCH:-/n/netscratch/sham_lab/Lab/chloe00/owb-render-scratch}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MPLCONFIGDIR="$OWB_RENDER_SCRATCH/matplotlib-${SLURM_JOB_ID}"
python=/n/home13/chloe00/miniforge3/bin/python
module=owb.scripts.skylands_campaign_first_person
chunk="${SLURM_ARRAY_TASK_ID:-0}"
case "$operation" in
  skylands-export)   "$python" -u -B -m "$module" export --task "$task" ${STOP_STEP:+--stop-step "$STOP_STEP"};;
  skylands-assemble) "$python" -u -B -m "$module" assemble --task "$task" --chunks "$chunks";;
  skylands-render)
    directory="$OWB_RENDER_SCRATCH/skylands-campaign/first-person-${PILOT:+pilot-}$task"
    if [ -n "${PILOT:-}" ]; then read -r first last <<<"${PILOT_RANGE:?}"; else
      read -r first last < <("$python" -B -m "$module" range --task "$task" --chunk "$chunk" --chunks "$chunks"); fi
    output="$OWB_RENDER_SCRATCH/skylands-campaign/first-person-${PILOT:+pilot-}frames-$task-$chunk"
    export OWB_CAMPAIGN_RENDER_HELPERS="$PWD/owb/render"
    stage_dir="${SLURM_TMPDIR:-/tmp}/owb-skylands-render-${SLURM_JOB_ID}-${chunk}"
    blender_path=$("$python" owb/scripts/stage_skylands_blender.py "$stage_dir")
    export CUDA_CACHE_PATH="$stage_dir/cuda"; mkdir -p "$CUDA_CACHE_PATH"
    "$blender_path" --background --factory-startup --python-exit-code 2 --python "$OWB_CAMPAIGN_RENDER_HELPERS/blender_campaign_baked.py" -- \
      --directory "$directory" --output "$output" --first "$first" --last "$last" --samples 64 --threads 8 --device OPTIX;;
  teyvat-render)
    # Twelve milestones split into contiguous intervals; each replays from reset and checks every action.
    first=$(( chunk * 12 / chunks + 1 )); last=$(( (chunk + 1) * 12 / chunks ))
    export OWB_CYCLES_DEVICE=CUDA
    "$python" -u -B -m owb.scripts.teyvat_reference --render-chunk "assets/demos/teyvat-reference-cycles-first-person/task-$task-seed-7" \
      --milestones "$first" "$last" --cache "$OWB_RENDER_SCRATCH/teyvat-first-person" --render-stride 2 --fps 10 \
      --view first_person --samples 48 --threads 8 ${MAX_FRAMES:+--max-render-frames "$MAX_FRAMES"};;
  teyvat-assemble)
    "$python" -u -B -m owb.scripts.teyvat_reference --assemble "assets/demos/teyvat-reference-cycles-first-person/task-$task-seed-7" \
      --cache "$OWB_RENDER_SCRATCH/teyvat-first-person";;
  *) echo "unknown operation $operation" >&2; exit 2;;
esac
