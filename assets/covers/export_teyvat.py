"""Export the full seed-7 Teyvat island for its camera-only cover."""
from dataclasses import replace
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from export import sha, state_fingerprint
from open_world_bench import make_env, RenderSettings
from owb.render.scene_export import capture_scene


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    # This small buffer is only for reset observations. Cycles exports at 1080p.
    with make_env(seed=7, backend="soft", render=RenderSettings(width=32, height=24)) as env:
        env.reset(seed=7)
        before = state_fingerprint(env)
        rng_before = pickle.dumps((env.rng.bit_generator.state, env.np_random.bit_generator.state))
        original_camera = env.cam
        eye = env.char.pos.astype(float) + (27., 24., -30.)
        target = env.char.pos.astype(float) + (-34., 17., 50.)
        try:
            env.cam = replace(original_camera, eye=eye.astype(np.float32), pivot=target.astype(np.float32))
            print("EXPORT teyvat", eye.tolist(), target.tolist(), flush=True)
            scene = capture_scene(env, width=1920, height=1080)
        finally:
            env.cam = original_camera
        after = state_fingerprint(env)
        assert before == after, "Cover export changed simulator state"
        assert rng_before == pickle.dumps((env.rng.bit_generator.state, env.np_random.bit_generator.state))
        scope = "Full 1024-metre procedural Teyvat island, seed-7 reset state"
        scene.metadata["cover"] = dict(scope=scope, camera_only=True,
            simulation_state_unchanged=True, simulation_rng_unchanged=True,
            agent_repositioned=False, actions_executed=0, state_sha256=before)
        scene.save(output / "teyvat.npz")
        sources = ["assets/covers/export_teyvat.py", "assets/covers/export.py",
                   "assets/covers/render.py", "owb/render/scene_export.py",
                   "owb/render/blender_scene.py", "open_world_bench/render/assets.py",
                   "open_world_bench/render/soft_renderer.py", "open_world_bench/world.py"]
        record = dict(scope=scope, seed=7, island_extent_metres=env.world.terrain.cfg.size,
            camera=scene.metadata["camera"], time_of_day=scene.metadata["time_of_day"],
            state_sha256_before=before, state_sha256_after=after,
            simulation_rng_unchanged=True, agent_position=env.char.pos.tolist(),
            actions_executed=0, scene_sha256=sha(output / "teyvat.npz"),
            source_sha256={p: sha(ROOT / p) for p in sources})
        (output / "scenes.json").write_text(json.dumps({"teyvat": record}, indent=2) + "\n")
        print("EXPORTED teyvat; simulator state and RNG preserved", flush=True)


if __name__ == "__main__":
    main()
