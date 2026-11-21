#!/usr/bin/env bash
# Regenerate both Skylands campaign walkthroughs with the current presentation
# code (landscape, lighting, materials). The verified action traces are reused
# unchanged; every replay is checked again during export.
#
# Run from the repository root on a Slurm login node:
#   bash owb/scripts/regenerate_skylands_campaign.sh
# Then, after the final job succeeds:
#   cp assets/demos/skylands-campaign/task-a-seed-7/walkthrough.mp4 videos/skylands/skylands-15-milestones.mp4
#   cp assets/demos/skylands-campaign/task-b-seed-7/walkthrough.mp4 videos/skylands/skylands-24-milestones.mp4
set -euo pipefail
[ -f owb/render/blender_campaign.py ] || { echo "Run from the repository root" >&2; exit 1; }
if [ -n "$(git status --porcelain -- owb open_world_bench assets/demos/skylands-campaign/render-source)" ]; then
  echo "Commit render/simulator changes first; the render contract records source hashes." >&2; exit 1
fi
for f in blender_scene.py blender_campaign.py blender_campaign_baked.py; do
  cmp -s "owb/render/$f" "assets/demos/skylands-campaign/render-source/$f" || {
    echo "assets/demos/skylands-campaign/render-source/$f differs from owb/render/$f; copy it first." >&2; exit 1; }
done
stamp=$(date +%Y%m%d-%H%M%S)
# Scene caches from earlier renders would otherwise be reused as-is.
if [ -e .cache/skylands-campaign ]; then mv .cache/skylands-campaign ".cache/skylands-campaign.before-${stamp}"; fi
mkdir -p .cache/skylands-campaign
# The merge step only assembles when no walkthrough exists; archive the old ones.
archive=".cache/skylands-published-before-${stamp}"
for task in a b; do
  dir="assets/demos/skylands-campaign/task-${task}-seed-7"
  mkdir -p "$archive/task-${task}"
  for f in preview.png chapters.json video-manifest.json rendered-frame-verification.json \
           camera-verification.json frame-records.json playback-verification.json frames.json geometry.json; do
    if [ -e "$dir/$f" ]; then cp -p "$dir/$f" "$archive/task-${task}/"; fi
  done
  if [ -e "$dir/walkthrough.mp4" ]; then mv "$dir/walkthrough.mp4" "$archive/task-${task}/"; fi
done
echo "Archived previous outputs in $archive"
export_job=$(sbatch --parsable owb/scripts/export_skylands_campaign.slurm)
render_job=$(sbatch --parsable --dependency="afterok:${export_job}" owb/scripts/render_skylands_campaign_chunks.slurm)
camera_job=$(sbatch --parsable --dependency="afterok:${export_job}" owb/scripts/render_skylands_campaign_camera.slurm)
# Sequential assembly: the second job rebuilds the index from both new videos.
assemble_a=$(sbatch --parsable --dependency="afterok:${render_job}:${camera_job}" owb/scripts/assemble_skylands_campaign.slurm a)
assemble_b=$(sbatch --parsable --dependency="afterok:${assemble_a}" owb/scripts/assemble_skylands_campaign.slurm b)
echo "Submitted: export ${export_job} -> render ${render_job}, camera ${camera_job} -> assemble ${assemble_a}, ${assemble_b}"
