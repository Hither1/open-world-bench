"""Export faithful, camera-only cover views of the seed-7 environments.

Run from the repository root: python -B assets/covers/export.py OUTPUT_DIRECTORY
The output must be new. No simulator actions, lighting edits, or terrain edits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from owb.core.config import EnvConfig, RenderConfig
from owb.env.wildlands import WildlandsEnv
from owb.env.skylands import SkylandsEnv, default_config
from owb.render.scene_export import capture_scene


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_fingerprint(env):
    # Hash the whole environment except its display/cache objects and camera
    # callback. This covers terrain, entities, clocks, inventory and all RNGs.
    state = {key: value for key, value in vars(env).items()
             if key not in {"renderer", "_renderer", "_frame", "presentation_camera"}}
    return hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest()


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    sources = ["assets/covers/export.py", "owb/render/scene_export.py",
               "owb/render/blender_scene.py", "owb/render/landscape.py",
               "owb/render/models.py", "owb/render/dressing.py",
               "owb/world/skylands.py", "owb/world/worldgen.py"]
    for name in ("wildlands", "skylands"):
        cfg = EnvConfig() if name == "wildlands" else default_config()
        cfg.obs_rgb = False
        cfg.render_cfg = RenderConfig(width=1920, height=1080, quality="high",
                                      max_dist=400, fov=68, third_person=True)
        env = WildlandsEnv(cfg) if name == "wildlands" else SkylandsEnv(cfg)
        with env:
            env.reset(seed=7)
            if name == "wildlands":
                tower = max((lm for lm in env.landmarks if lm.kind == "tower"),
                            key=lambda lm: lm.pos[2])
                eye = np.array(tower.pos) + (0, 6, -4)
                target = np.array((125., 47., 120.))
                scope = "Full-size procedural Wildlands sandbox, reset state"
            else:
                edge, sky, z, _ = env.world.meta["bridge"]
                eye = np.array((edge - 20., sky + 10., z + 0.5))
                target = np.array((edge + 10., sky + 1., z + 0.5))
                scope = "Aster Reach introductory construction course, full-size variant, reset state"
            env.presentation_camera = lambda: (eye.copy(), target.copy())
            before = state_fingerprint(env)
            rng_before = pickle.dumps(env.np_random.bit_generator.state)
            print("EXPORT", name, "camera", eye.tolist(), target.tolist(), flush=True)
            scene = capture_scene(env, width=1920, height=1080)
            after = state_fingerprint(env)
            assert before == after, "Scene export changed simulator state"
            assert rng_before == pickle.dumps(env.np_random.bit_generator.state)
            scene.metadata["cover"] = dict(scope=scope, camera_only=True,
                simulation_state_unchanged=True, simulation_rng_unchanged=True,
                agent_repositioned=False, actions_executed=0, state_sha256=before)
            scene.save(output / f"{name}.npz")
            records[name] = dict(scope=scope, seed=7, world_shape=list(env.world.shape),
                camera=scene.metadata["camera"], time_of_day=scene.metadata["time_of_day"],
                state_sha256_before=before, state_sha256_after=after,
                simulation_rng_unchanged=True, agent_position=env.agent.pos.tolist(),
                actions_executed=0, scene_sha256=sha(output / f"{name}.npz"),
                presentation_landscape=scene.metadata.get("presentation_landscape"),
                source_sha256={p: sha(ROOT / p) for p in sources})
            print("EXPORTED", name, flush=True)
    (output / "scenes.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
