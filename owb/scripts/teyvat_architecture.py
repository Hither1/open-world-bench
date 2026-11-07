"""Export actual task architecture and optionally render it with GPU Cycles.

Only the presentation camera changes. No task entities or terrain are moved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from open_world_bench.reference_tasks import ReferenceTaskEnv
from owb.render.presentation import resolve_blender
from owb.render.scene_export import export_scene
from owb.scripts.teyvat_reference import render_source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=('a', 'b'), default='b')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--device', choices=('CUDA', 'OPTIX'), default='CUDA')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--size', type=int, nargs=2, default=(1280, 720))
    args = parser.parse_args()
    if min(*args.size, args.samples) <= 0:
        parser.error('size and samples must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    views = []
    with ReferenceTaskEnv(args.task, seed=args.seed, render_enabled=False) as env:
        env.reset(seed=args.seed)
        digest = env.state_digest()
        for kind in ('timber_house', 'pavilion', 'ruin_gate', 'windmill'):
            building = next(b for b in env.world.architecture.buildings if b.kind == kind)
            env.cam.eye = (np.array((14., 11., -21.)) @ building.rotation.T+building.pos).astype(np.float32)
            env.cam.pivot = np.asarray(building.pos, np.float32)+(0, 2.6, 0)
            snapshot = export_scene(env, args.output/f'{kind}.npz', width=args.size[0], height=args.size[1])
            assert env.state_digest() == digest
            views.append(dict(kind=kind, building=building.metadata(), snapshot=snapshot.name,
                              image=f'{kind}.png', camera=dict(eye=env.cam.eye.tolist(), target=env.cam.pivot.tolist())))
        manifest = dict(task=args.task, seed=args.seed, state_digest=digest,
                        scope='Actual compact reference course; presentation cameras only',
                        renderer=dict(engine='Blender Cycles', device=args.device,
                                      width=args.size[0], height=args.size[1], samples=args.samples,
                                      denoising=True, resolution_percentage=100),
                        buildings=[b.metadata() for b in env.world.architecture.buildings],
                        views=views, sources=render_source_hashes())
    if not args.export_only:
        blender = resolve_blender()
        manifest['renderer']['blender_version'] = subprocess.check_output([blender, '--version'], text=True).splitlines()[0]
        script = Path(__file__).resolve().parents[1]/'render/blender_scene.py'
        for view in views:
            with (args.output/f"{view['kind']}.log").open('w') as log:
                subprocess.run([blender, '--background', '--factory-startup', '--python-exit-code', '1',
                                '--python', str(script), '--', '--scene', str(args.output/view['snapshot']),
                                '--output', str(args.output/view['image']), '--device', args.device,
                                '--samples', str(args.samples), '--threads', '4'], stdout=log, stderr=subprocess.STDOUT, check=True)
            view['image_sha256'] = hashlib.sha256((args.output/view['image']).read_bytes()).hexdigest()
            print(f"Rendered {view['image']}", flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
