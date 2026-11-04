"""Recreate the README screenshots: python -m owb.scripts.screenshots.

Requires the video extra for PNG output. Both scenes use seed 7 and fixed
showcase viewpoints; images come directly from the CPU renderers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from open_world_bench import make_env

from ..core.config import EnvConfig, RenderConfig
from ..env.wildlands import WildlandsEnv


SEED = 7
WIDTH, HEIGHT = 960, 540


def capture_wildlands(style="landscape"):
    env = WildlandsEnv(EnvConfig(
        obs_rgb=False,
        render_cfg=RenderConfig(width=WIDTH, height=HEIGHT, max_dist=200, style=style,
                                third_person=True, cam_distance=5, cam_pitch_offset=-0.16, hud=True),
    ))
    try:
        env.reset(seed=SEED)
        # Look inland from the southernmost generated survey tower.
        tower = max((landmark for landmark in env.landmarks if landmark.kind == "tower"),
                    key=lambda landmark: landmark.pos[2])
        env.agent.pos = np.array(tower.pos, dtype=np.float32)
        env.agent.yaw = float(np.arctan2(
            env.world.size_z / 2 - env.agent.pos[2],
            env.world.size_x / 2 - env.agent.pos[0],
        ))
        env.agent.pitch = -0.12
        env.elements.climate.time = 0.38
        return env.render()
    finally:
        env.close()


def capture_teyvat():
    env = make_env(seed=SEED, backend="soft")
    try:
        env.reset(seed=SEED)
        # Keep the Windrest spawn, framing the coast and nearby waypoint.
        env.char.yaw = -0.5
        env.cam = env.camera_rig.reset(env.char, yaw=env.char.yaw)
        return env.render_hd(width=WIDTH, height=HEIGHT, hud=False)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/screenshots"))
    parser.add_argument("--environment", choices=("all", "wildlands", "teyvat"), default="all")
    parser.add_argument("--compare-wildlands", action="store_true", help="also capture the voxel backend from the same camera")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, capture in (("wildlands", capture_wildlands), ("teyvat", capture_teyvat)):
        if args.environment not in ("all", name):
            continue
        path = args.output_dir / f"{name}.png"
        iio.imwrite(path, capture())
        print(f"Saved {path} ({WIDTH} x {HEIGHT}, seed {SEED})", flush=True)
    if args.compare_wildlands:
        path = args.output_dir / "wildlands-voxel.png"
        iio.imwrite(path, capture_wildlands(style="voxel"))
        print(f"Saved {path} (same seed, camera, time, and HUD)", flush=True)


if __name__ == "__main__":
    main()
