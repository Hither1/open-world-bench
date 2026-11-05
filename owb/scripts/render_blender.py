"""Export and render actual environment snapshots with an external Blender 4.5+.

python -m owb.scripts.render_blender --env teyvat wildlands --output /tmp/owb-blender
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import numpy as np

from ..bench import make_env
from ..render.scene_export import export_scene
from ..render.presentation import resolve_blender


def showcase(env):
    """Use the fixed README viewpoints. This is explicitly a presentation mode."""
    native = env.native
    if env.name == "teyvat":
        native.char.yaw = -0.5
        native.cam = native.camera_rig.reset(native.char, yaw=native.char.yaw)
    else:
        towers = [lm for lm in native.landmarks if lm.kind == "tower"]
        if towers:
            tower = max(towers, key=lambda lm: lm.pos[2])
            native.agent.pos = np.array(tower.pos, np.float32)
            native.agent.yaw = float(np.arctan2(native.world.size_z / 2 - native.agent.pos[2],
                                               native.world.size_x / 2 - native.agent.pos[0]))
        native.agent.pitch = -0.12
        native.elements.climate.time = 0.38
        native.cfg.render_cfg.third_person = True
        native.cfg.render_cfg.cam_distance = 5
        native.cfg.render_cfg.cam_pitch_offset = -0.16
        native._frame = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", nargs="+", choices=("teyvat", "wildlands"), default=["teyvat", "wildlands"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--variant", choices=("small", "full"), default="full")
    parser.add_argument("--size", nargs=2, type=int, default=(960, 540), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--blender", help="path to the Blender executable (or OWB_BLENDER)")
    parser.add_argument("--export-only", action="store_true", help="write snapshots without requiring Blender")
    parser.add_argument("--view", choices=("agent", "showcase"), default="agent")
    parser.add_argument("--output", type=Path, required=True, help="a new output directory")
    args = parser.parse_args()
    if min(*args.size, args.samples, args.threads) <= 0:
        parser.error("size, samples and threads must be positive")
    if len(set(args.env)) != len(args.env):
        parser.error("each environment must appear only once")
    try:
        blender = resolve_blender(args.blender) if not args.export_only else None
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    script = Path(__file__).resolve().parents[1] / "render" / "blender_scene.py"
    for name in args.env:
        destination = args.output / name
        destination.mkdir()
        # Keep reset inexpensive: snapshot resolution is independent of RGB observations.
        with make_env(name, seed=args.seed, variant=args.variant, image_size=(32, 24)) as env:
            env.reset(seed=args.seed)
            if args.view == "showcase":
                showcase(env)
            export_scene(env, destination / "scene.npz", width=args.size[0], height=args.size[1])
        print(f"Exported {destination / 'scene.npz'}", flush=True)
        if not args.export_only:
            subprocess.run([blender, "--background", "--factory-startup", "--python-exit-code", "1",
                            "--python", str(script), "--", "--scene", str(destination / "scene.npz"),
                            "--output", str(destination / "render.png"), "--blend", str(destination / "scene.blend"),
                            "--glb", str(destination / "scene.glb"), "--samples", str(args.samples),
                            "--threads", str(args.threads)], check=True)


if __name__ == "__main__":
    main()
