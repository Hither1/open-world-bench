"""Full-world architecture stills from the recorded campaign's exact reset state.

The camera visits the architecture; the character and simulation never move.
This is a reset-state showcase, not an executed agent trajectory or video.
"""
import argparse
from pathlib import Path

import numpy as np

from owb.scripts.skylands_tempest import replay_start, sources
from owb.scripts.skylands_reference import save, sha
from owb.render.skylands_campaign import static_scene, dynamic_scene, grass_prototypes


def export(directory, output):
    output.mkdir(parents=True, exist_ok=False)
    (output/"states").mkdir()
    env, manifest = replay_start(directory)
    try:
        digest = env.state_digest()
        scene, points, variants = static_scene(env)
        assert env.state_digest() == digest
        scene.save(output/"static.npz")
        np.savez_compressed(output/"ground-cover.npz", points=points, variants=variants)
        grass_prototypes().save(output/"grass-prototypes.npz")
        save(output/"geometry.json", scene.metadata["geometry_contract"])
        entries = []
        for name, kind, region in (("Shipwrights and cargo works", "boathouse", "B02"),
                                   ("Deep cooling works", "pump_house", "B15"),
                                   ("Anchor foundry", "foundry", "B17"),
                                   ("Sky monitoring observatory", "observatory", "B19"),
                                   ("Tempest Engine", None, None)):
            state = dynamic_scene(env)
            if kind:
                building = next(b for b in env.architecture if b["kind"] == kind and b["id"].startswith(region+"-"))
                focus = np.asarray(building["pos"], float)+(0, 5 if kind != "observatory" else 8, 0)
                offset = np.array((-24., 14., -31.))
            else:
                focus = env.engine_pos+(0, 9, 0)
                offset = np.array((-50., 28., -60.))
            state.metadata["camera"] = dict(eye=(focus+offset).tolist(), target=focus.tolist(), fov_y=48.)
            state.metadata["presentation"] = "Reset-state architecture showcase; camera only"
            assert env.state_digest() == digest
            file = f"{len(entries):06}.npz"
            state.save(output/"states"/file)
            entries.append(dict(index=len(entries), scene="states/"+file, scene_sha256=sha(output/"states"/file),
                                step=0, seconds=0., milestone=region or "engine", title=name, digest=digest,
                                position=env.agent.pos.tolist()))
            print("SHOWCASE EXPORTED", name, flush=True)
        save(output/"frames.json", dict(task="tempest", seed=manifest["seed"], trace_sha256=manifest["actions_sha256"],
             world_shape=manifest["world_shape"], source_seconds=0., entries=entries,
             artifact_hashes={name: sha(output/name) for name in ("static.npz", "ground-cover.npz", "grass-prototypes.npz")},
             renderer=dict(engine="Blender Cycles", device="OPTIX GPU", width=1280, height=720, samples=64,
                           denoising=True, max_bounces=6, upscaling=False, full_world_geometry=True),
             sources={**sources(), "owb/scripts/skylands_tempest_showcase.py": sha(Path(__file__))},
             playback="Reset-state architecture stills. Camera moves only; no agent actions or simulation edits."))
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    export(a.directory, a.output)
