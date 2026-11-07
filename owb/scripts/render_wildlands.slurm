#!/usr/bin/env bash
#SBATCH --job-name=owb-wildlands-cycles
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=assets/demos/wildlands-reference/render-%j.log

# Submit from the repository root. This script requests exactly one GPU and
# renders sequentially, preserving all geometry, samples, and native resolution.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script through sbatch from the repository root}"
test -f owb/scripts/wildlands_reference.py
export OWB_CYCLES_DEVICE=CUDA
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export OWB_FRAME_CACHE="${OWB_FRAME_CACHE:-$PWD/.cache/wildlands-cycles}"
export MPLCONFIGDIR="$OWB_FRAME_CACHE/matplotlib"
export CUDA_CACHE_PATH="$OWB_FRAME_CACHE/cuda"
mkdir -p "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"

python -u -B - <<'PY'
import json,os,subprocess,sys
from pathlib import Path
from owb.scripts.wildlands_reference import verify,render_chunk,assemble
from owb.scripts.wildlands_demo_index import main as build_index
from owb.render.presentation import resolve_blender
from owb.render.video import resolve_ffmpeg

root=Path('assets/demos/wildlands-reference')
cache=Path(os.environ['OWB_FRAME_CACHE'])
print('Blender:',resolve_blender(),'FFmpeg:',resolve_ffmpeg(),flush=True)
subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],check=True)
for task,count in [('a',12),('b',24)]:
    folder=root/f'task-{task}-seed-7'
    print('Verifying',task,verify(folder),flush=True)
    for milestone in range(1,count+1):
        render_chunk(folder,cache,milestone,milestone,checkpoints=True,samples=48,threads=4)
    if not (folder/'walkthrough.mp4').exists():
        assemble(folder,cache)
    else:
        from owb.scripts.wildlands_reference import sha
        summary=json.loads((folder/'summary.json').read_text())
        assert sha(folder/'walkthrough.mp4')==summary['video_sha256']
build_index()
print('Both walkthroughs and chapter player are complete.',flush=True)
PY
