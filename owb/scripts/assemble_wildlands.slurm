#!/usr/bin/env bash
#SBATCH --job-name=owb-wildlands-video
#SBATCH --partition=serial_requeue
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=assets/demos/wildlands-reference/assemble-%j.log

# Runs after every render chunk succeeds: replay-verifies both traces, encodes and
# decodes each walkthrough from the verified chunk frames, and builds the player.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script through sbatch from the repository root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export OWB_FRAME_CACHE="${OWB_FRAME_CACHE:-$PWD/.cache/wildlands-cycles}"
export MPLCONFIGDIR="$OWB_FRAME_CACHE/matplotlib-assemble"
mkdir -p "$MPLCONFIGDIR"
# A preempted encode leaves only the partial file; the published file is never touched.
rm -f assets/demos/wildlands-reference/task-{a,b}-seed-7/walkthrough.partial.mp4
"${OWB_PYTHON:-/n/home13/chloe00/miniforge3/bin/python}" -u -B - <<'PY'
import json,os
from pathlib import Path
from owb.scripts.wildlands_reference import verify,assemble,sha
from owb.scripts.wildlands_demo_index import main as build_index
from owb.render.video import resolve_ffmpeg
root=Path('assets/demos/wildlands-reference'); cache=Path(os.environ['OWB_FRAME_CACHE'])
print('FFmpeg:',resolve_ffmpeg(),flush=True)
for task in 'ab':
    folder=root/f'task-{task}-seed-7'
    print('Verifying',task,verify(folder),flush=True)
    if not (folder/'walkthrough.mp4').exists():
        assemble(folder,cache)
    else:
        summary=json.loads((folder/'summary.json').read_text())
        assert sha(folder/'walkthrough.mp4')==summary['video_sha256']
build_index()
print('Both walkthroughs and chapter player are complete.',flush=True)
PY
