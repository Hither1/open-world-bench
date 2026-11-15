#!/usr/bin/env bash
# Render both Wildlands reference walkthroughs in Cycles with the current
# presentation code. The verified action traces are reused unchanged and are
# replay-checked again by every render chunk and by the assembly job.
#
# Run from the repository root on a Slurm login node:
#   bash owb/scripts/regenerate_wildlands.sh
set -euo pipefail
[ -f owb/scripts/wildlands_reference.py ] || { echo "Run from the repository root" >&2; exit 1; }
py="${OWB_PYTHON:-/n/home13/chloe00/miniforge3/bin/python}"
cache="${OWB_FRAME_CACHE:-$PWD/.cache/wildlands-cycles}"
stamp=$(date +%Y%m%d-%H%M%S)
# Resumable frames are keyed to source hashes, but start clean so no stale
# chunk layout can mix with the new one.
if [ -e "$cache" ]; then mv "$cache" "${cache}.before-${stamp}"; echo "Moved previous frames to ${cache}.before-${stamp}"; fi
mkdir -p "$cache"
# Assembly never overwrites a published video; archive existing ones.
archive=".cache/wildlands-published-before-${stamp}"
for task in a b; do
  dir="assets/demos/wildlands-reference/task-${task}-seed-7"
  for f in walkthrough.mp4 summary.json chapters.json preview.png; do
    if [ -e "$dir/$f" ]; then mkdir -p "$archive/task-${task}"; mv "$dir/$f" "$archive/task-${task}/"; fi
  done
done
[ -d "$archive" ] && echo "Archived previous outputs in $archive"
# Fill the checksummed Blender mirror once, before parallel chunks need it.
"$py" owb/scripts/stage_blender.py --mirror-only >/dev/null
render_job=$(sbatch --parsable owb/scripts/render_wildlands.slurm)
assemble_job=$(sbatch --parsable --dependency="afterok:${render_job}" owb/scripts/assemble_wildlands.slurm)
echo "Submitted: render ${render_job} (6 chunks) -> assemble ${assemble_job}"
