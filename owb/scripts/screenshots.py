"""Recreate the README screenshots: python -m owb.scripts.screenshots.

Requires the video extra for PNG output. Both scenes use seed 7 and fixed
showcase viewpoints. Teyvat and Wildlands use Blender Cycles by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from open_world_bench import make_env

from ..core.config import EnvConfig, RenderConfig
from ..env.wildlands import WildlandsEnv
from ..env.skylands import SkylandsEnv, default_config as skylands_config
from ..render.presentation import PresentationRenderer, add_render_arguments


SEED = 7
WIDTH, HEIGHT = 960, 540


def capture_wildlands(style="landscape", quality="high", view="overlook", **rendering):
    env = WildlandsEnv(EnvConfig(
        obs_rgb=False,
        render_cfg=RenderConfig(width=WIDTH, height=HEIGHT, max_dist=200 if view == "overlook" else 112,
                                style=style, quality=quality, supersample=2 if quality == "high" else 1,
                                third_person=True, cam_distance=5, cam_pitch_offset=-0.16, hud=True),
    ))
    try:
        env.reset(seed=SEED)
        if view == "overlook":
            # Look inland from the southernmost generated survey tower.
            tower = max((landmark for landmark in env.landmarks if landmark.kind == "tower"),
                        key=lambda landmark: landmark.pos[2])
            env.agent.pos = np.array(tower.pos, dtype=np.float32)
            env.agent.yaw = float(np.arctan2(
                env.world.size_z / 2 - env.agent.pos[2],
                env.world.size_x / 2 - env.agent.pos[0],
            ))
            env.agent.pitch = -0.12
        elif view == "woodland":
            # A ground-level viewpoint in the generated forest; no scene edits.
            env.agent.pos = np.array((40.5, env.world.spawn_y(40, 157) + 0.01, 157.5), np.float32)
            env.agent.yaw, env.agent.pitch = 0.66, -0.02
            env.cfg.render_cfg.cam_distance = 4.5
            env.cfg.render_cfg.cam_pitch_offset = -0.07
        else:
            env.cfg.render_cfg.cam_pitch_offset = -0.10
        env.elements.climate.time = 0.38
        with PresentationRenderer(width=WIDTH, height=HEIGHT, **rendering) as renderer:
            return renderer.render(env)
    finally:
        env.close()


def capture_teyvat(**rendering):
    env = make_env(seed=SEED, backend="soft")
    try:
        env.reset(seed=SEED)
        # Keep the Windrest spawn, framing the coast and nearby waypoint.
        env.char.yaw = -0.5
        env.cam = env.camera_rig.reset(env.char, yaw=env.char.yaw)
        with PresentationRenderer(width=WIDTH, height=HEIGHT, **rendering) as renderer:
            return renderer.render(env)
    finally:
        env.close()


def capture_skylands(layer="sky", quality="balanced"):
    cfg = skylands_config()
    cfg.obs_rgb = False
    cfg.render_cfg = RenderConfig(width=WIDTH, height=HEIGHT, max_dist=150,
                                  third_person=True, cam_distance=5, cam_pitch_offset=-0.10, hud=True,
                                  quality=quality)
    with SkylandsEnv(cfg) as env:
        env.reset(seed=SEED)
        edge, sky, z, _ = env.world.meta["bridge"]
        surface = env.world.meta["layer_heights"][1]
        if layer == "sky":
            env.agent.pos = np.array((edge - 5.5, sky + 1.01, z + 0.5), np.float32)
            env.agent.yaw, env.agent.pitch = 0.0, -0.10
        elif layer == "surface":
            env.agent.pos = np.array((8.5, surface + 1.01, env.world.size_z - 9.5), np.float32)
            env.agent.yaw = float(np.arctan2(z + 0.5 - env.agent.pos[2], edge + 8 - env.agent.pos[0]))
            env.agent.pitch = 0.48
        else:
            env.agent.pos = np.array((6.5, 4.01, z + 8.5), np.float32)
            env.agent.yaw, env.agent.pitch = -0.5, 0.05
        env.elements.climate.time = 0.38
        env._frame = None
        return env.render()


def main():
    global WIDTH, HEIGHT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/screenshots"))
    parser.add_argument("--environment", choices=("all", "wildlands", "teyvat", "skylands"), default="all")
    parser.add_argument("--skylands-layer", choices=("sky", "surface", "depths"), default="sky")
    parser.add_argument("--wildlands-view", choices=("overlook", "camp", "woodland"), default="overlook")
    parser.add_argument("--quality", choices=("fast", "balanced", "high"), default="high",
                        help="Wildlands presentation tier; high uses 2x supersampling")
    parser.add_argument("--size", type=int, nargs=2, default=(WIDTH, HEIGHT), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--skylands-quality", choices=("fast", "balanced", "high"), default="balanced")
    parser.add_argument("--compare-wildlands", action="store_true", help="also capture the voxel backend from the same camera")
    add_render_arguments(parser)
    args = parser.parse_args()
    if min(*args.size, args.samples, args.threads) <= 0:
        parser.error("image dimensions must be positive")
    WIDTH, HEIGHT = args.size
    rendering=dict(backend=args.renderer,blender=args.blender,samples=args.samples,threads=args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, capture in (("wildlands", lambda: capture_wildlands(quality=args.quality, view=args.wildlands_view, **rendering)), ("teyvat", lambda: capture_teyvat(**rendering)),
                          ("skylands", lambda: capture_skylands(args.skylands_layer, args.skylands_quality))):
        if args.environment not in ("all", name):
            continue
        suffix = f"-{args.skylands_layer}" if name == "skylands" and args.skylands_layer != "sky" else ""
        if name == "wildlands" and args.wildlands_view != "overlook":
            suffix = f"-{args.wildlands_view}"
        path = args.output_dir / f"{name}{suffix}.png"
        iio.imwrite(path, capture())
        print(f"Saved {path} ({WIDTH} x {HEIGHT}, seed {SEED})", flush=True)
    if args.compare_wildlands:
        suffix = f"-{args.wildlands_view}" if args.wildlands_view != "overlook" else ""
        path = args.output_dir / f"wildlands{suffix}-voxel.png"
        iio.imwrite(path, capture_wildlands(style="voxel", quality=args.quality, view=args.wildlands_view, backend="cpu"))
        print(f"Saved {path} (same seed, camera, time, and HUD)", flush=True)


if __name__ == "__main__":
    main()
