"""Settlement stills from a Skylands campaign's exact reset state.

The camera visits hamlets, farmsteads and working buildings; the character and
simulation never move. This is a reset-state showcase, not an executed agent
trajectory or video. Render the export with ``blender_campaign_baked.py``.
"""
import argparse
from pathlib import Path

import numpy as np

from owb.core import blocks as B
from owb.env.skylands_campaign import SkylandsCampaignEnv
from owb.render.skylands_campaign import static_scene, dynamic_scene, grass_prototypes
from owb.scripts.skylands_campaign import sources
from owb.scripts.skylands_reference import save, sha
from owb.world.settlements import door_normal

EXTRA_SOURCES = ("owb/world/settlements.py", "owb/render/settlements.py", "owb/render/tempest.py",
                 "owb/world/tempest.py", "owb/scripts/skylands_campaign_showcase.py")


def clear_view(env, target, offset, turns=8):
    """First offset, turned about the vertical, whose eye is in open air and
    whose sight line to ``target`` crosses no solid voxel."""
    vox = env.world.vox
    for k in range(turns):
        a = k*2*np.pi/turns
        c, s = np.cos(a), np.sin(a)
        eye = target+(offset[0]*c-offset[2]*s, offset[1], offset[0]*s+offset[2]*c)
        ray = target+(eye-target)*np.linspace(.12, 1, 160)[:, None]
        cells = np.floor(ray).astype(int)
        inside = ((cells >= 0) & (cells < vox.shape)).all(axis=1)
        if not inside.all():
            continue
        if not B.SOLID[vox[cells[:, 0], cells[:, 1], cells[:, 2]]].any():
            return eye
    return target+offset


def views(env):
    arch = env.architecture
    first = env.milestones[0]
    hamlet = [r for r in arch if r["id"].startswith(first.id+"-")]
    centre = np.mean([r["pos"] for r in hamlet], axis=0)
    out = [("Aster Reach landing hamlet", centre+(0, 3, 0), np.array((-46., 30., -52.)), 50.)]
    stead = next(r for r in arch if r["id"].endswith("cottage-0") and r["id"].startswith("farmstead"))
    n = door_normal(stead["rotation"]); side = np.array((-n[1], n[0]))
    focus = np.asarray(stead["pos"], float)+(.5, 3., .5)
    off = np.array((n[0]*14+side[0]*8, 6., n[1]*14+side[1]*8))
    out.append(("Trailside farmstead", focus, off, 46.))
    second = [r for r in arch if r["kind"] not in ("cottage",) and r["layer"] == 1][1]
    out.append((f"Working building: {second['kind'].replace('_', ' ')}",
                np.asarray(second["pos"], float)+(0, 5, 0), np.array((-22., 12., -29.)), 48.))
    depth = next(r for r in arch if r["layer"] == 0)
    out.append((f"Depths outpost: {depth['kind'].replace('_', ' ')}",
                np.asarray(depth["pos"], float)+(0, 4, 0), np.array((-17., 8., -22.)), 52.))
    sky = next(r for r in arch if r["layer"] == 2)
    out.append((f"Sky island {sky['kind'].replace('_', ' ')}",
                np.asarray(sky["pos"], float)+(0, 7, 0), np.array((-28., 13., -34.)), 48.))
    return out


def export(task, seed, output):
    output.mkdir(parents=True, exist_ok=False)
    (output/"states").mkdir()
    env = SkylandsCampaignEnv(task, seed)
    env.reset(seed=seed)
    try:
        digest = env.state_digest()
        scene, points, variants = static_scene(env)
        assert env.state_digest() == digest
        scene.save(output/"static.npz")
        np.savez_compressed(output/"ground-cover.npz", points=points, variants=variants)
        grass_prototypes().save(output/"grass-prototypes.npz")
        save(output/"geometry.json", scene.metadata["geometry_contract"])
        entries = []
        for title, focus, offset, fov in views(env):
            state = dynamic_scene(env)
            eye = clear_view(env, focus, offset)
            state.metadata["camera"] = dict(eye=eye.tolist(), target=focus.tolist(), fov_y=fov)
            state.metadata["presentation"] = "Reset-state settlement showcase; camera only"
            assert env.state_digest() == digest
            file = f"{len(entries):06}.npz"
            state.save(output/"states"/file)
            entries.append(dict(index=len(entries), scene="states/"+file, scene_sha256=sha(output/"states"/file),
                                step=0, seconds=0., milestone="settlements", title=title, digest=digest,
                                position=env.agent.pos.tolist()))
            print("SHOWCASE EXPORTED", title, eye.round(1).tolist(), flush=True)
        here = Path(__file__).resolve().parents[2]
        save(output/"frames.json", dict(task=task, seed=seed, trace_sha256=None, world_shape=list(env.world.shape),
             source_seconds=0., entries=entries, initial_digest=digest,
             buildings=len(env.architecture), building_kinds=sorted({r["kind"] for r in env.architecture}),
             artifact_hashes={name: sha(output/name) for name in ("static.npz", "ground-cover.npz", "grass-prototypes.npz")},
             renderer=dict(engine="Blender Cycles", device="OPTIX GPU", width=1280, height=720, samples=64,
                           denoising=True, max_bounces=6, upscaling=False, full_world_geometry=True),
             sources={**sources(), **{p: sha(here/p) for p in EXTRA_SOURCES}},
             playback="Reset-state settlement stills. Camera moves only; no agent actions or simulation edits."))
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("a", "b"), default="a")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    export(a.task, a.seed, a.output)
