"""Collect, independently replay, and export original Tempest campaign evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from owb.sims.voxel.env.skylands_tempest import SkylandsTempestEnv
from owb.bench.baselines.skylands_tempest import SkylandsTempestPolicy
from owb.scripts.skylands_campaign import SOURCES as BASE_SOURCES
from owb.scripts.skylands_reference import save, sha, plain, ROOT, action_from_json


SOURCES = BASE_SOURCES+(
    "owb/sims/voxel/env/skylands_tempest.py", "owb/sims/voxel/world/tempest.py", "owb/render/tempest.py",
    "owb/bench/baselines/skylands_tempest.py", "owb/scripts/skylands_tempest.py",
    "owb/render/skylands_campaign.py",
)


def sources():
    return {p: sha(ROOT/p) for p in SOURCES}


def collect(directory, objective, seed, route, order):
    directory.mkdir(parents=True, exist_ok=False)
    source_hashes = sources()
    env = SkylandsTempestEnv(objective, seed)
    start = time.monotonic()
    print("GENERATE", objective, route, seed, flush=True)
    obs, _ = env.reset(seed=seed)
    initial = env.state_digest()
    policy = SkylandsTempestPolicy(env, route=route, order=order)
    counts = Counter()
    last = None
    print("WORLD", env.world.shape, "buildings", len(env.architecture), flush=True)
    try:
        with (directory/"actions.jsonl").open("x") as stream:
            while not env._ended:
                action = policy(obs)
                if not env.action_space.contains(action):
                    raise RuntimeError("Reference emitted invalid action")
                before = env.agent.pos.copy()
                obs, reward, terminated, truncated, info = env.step(action)
                counts.update(e["type"] for e in info["events"])
                row = dict(step=env.steps, stage=policy.stage, action=plain(action),
                           position_before=before.tolist(), position_after=env.agent.pos.tolist(),
                           seconds=env.elapsed, reward=reward, terminated=terminated, truncated=truncated,
                           events=info["events"], explanation=policy.explanation)
                if last != policy.stage or env.steps % 1000 == 0 or env._ended:
                    row["digest"] = env.state_digest()
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                if last != policy.stage:
                    print("STAGE", policy.stage, env.steps, round(env.elapsed, 1), "health", env.agent.health, flush=True)
                    last = policy.stage
                if env.steps % 1000 == 0:
                    stream.flush()
                    progress = dict(stage=policy.stage, step=env.steps, position=env.agent.pos.tolist(),
                                    health=env.agent.health, wall_seconds=time.monotonic()-start)
                    save(directory/"progress.json", progress)
                    print("PROGRESS", progress, flush=True)
        if not env.success or env.agent.deaths:
            raise RuntimeError("Reference did not complete the final live objective")
        if sources() != source_hashes:
            raise RuntimeError("Source files changed during collection; recollect from the final version")
        manifest = dict(format="owb-tempest-trace-v1", objective=objective, seed=seed, route=route,
                        order=order, initial_digest=initial, final_digest=env.state_digest(),
                        evaluation=env.evaluation(), events=dict(counts), world_shape=list(env.world.shape),
                        actions_sha256=sha(directory/"actions.jsonl"), sources=source_hashes,
                        controller="Privileged reference; changes simulator only through legal actions",
                        limitations=["Authored geometry; held-out layout generalization is not established.",
                                     "Rated cargo craft and grid collision mechanisms, not general rigid-body assemblies.",
                                     "Mechanical hazard encounter, not humanoid combat AI.",
                                     "No learned-agent or memory-benefit results are claimed."])
        save(directory/"reference-manifest.json", manifest)
        print("COLLECTED", manifest["evaluation"], flush=True)
    except BaseException:
        save(directory/"failure.json", dict(stage=policy.stage, step=env.steps, position=env.agent.pos.tolist(),
                                            explanation=policy.explanation, evaluation=env.evaluation()))
        raise
    finally:
        env.close()


def replay_start(directory):
    manifest = json.loads((directory/"reference-manifest.json").read_text())
    if sha(directory/"actions.jsonl") != manifest["actions_sha256"]:
        raise ValueError("Trace checksum mismatch")
    env = SkylandsTempestEnv(manifest["objective"], manifest["seed"])
    env.reset(seed=manifest["seed"])
    if env.state_digest() != manifest["initial_digest"]:
        env.close()
        raise ValueError("Initial world differs from the recorded campaign")
    return env, manifest


def checked_step(env, row):
    np.testing.assert_array_equal(env.agent.pos, row["position_before"])
    action = action_from_json(row["action"])
    if not env.action_space.contains(action):
        raise ValueError("Invalid recorded action")
    _, reward, terminated, truncated, info = env.step(action)
    np.testing.assert_array_equal(env.agent.pos, row["position_after"])
    if (reward != row["reward"] or terminated != row["terminated"] or truncated != row["truncated"]
            or plain(info["events"]) != row["events"]):
        raise ValueError(f"Replay differs at action {row['step']}")
    if "digest" in row and env.state_digest() != row["digest"]:
        raise ValueError(f"State digest differs at action {row['step']}")


def verify(directory):
    env, manifest = replay_start(directory)
    try:
        with (directory/"actions.jsonl").open() as stream:
            for line in stream:
                checked_step(env, json.loads(line))
        if not env.success or env.state_digest() != manifest["final_digest"]:
            raise ValueError("Final objective or world digest differs")
        result = dict(success=True, checked_actions=env.steps, checked_positions=True, checked_events=True,
                      checked_rewards=True, final_live_predicate=env.system_operational(),
                      final_digest=env.state_digest(), evaluation=env.evaluation(), sources=sources())
        save(directory/"verification.json", result)
        print("VERIFIED", directory, env.steps, flush=True)
    finally:
        env.close()


def export(directory, output):
    """A small set of full-world Cycles scenes from an independently checked trace."""
    from owb.render.skylands_campaign import static_scene, dynamic_scene, grass_prototypes
    output.mkdir(parents=True, exist_ok=False)
    (output/"states").mkdir()
    env, manifest = replay_start(directory)
    entries = []
    try:
        before = env.state_digest()
        scene, points, variants = static_scene(env)
        assert env.state_digest() == before, "Presentation mutated simulation"
        scene.save(output/"static.npz")
        np.savez_compressed(output/"ground-cover.npz", points=points, variants=variants)
        grass_prototypes().save(output/"grass-prototypes.npz")
        save(output/"geometry.json", scene.metadata["geometry_contract"])
        print("STATIC EXPORTED", len(points), "grass clumps", flush=True)
        selected = {"B02", "B12", "B15", "B17", "B19", "B24", "grounding_canal", "lock_0", "grounding_socket", "stabilize"}
        seen = set()
        with (directory/"actions.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                checked_step(env, row)
                stage = row["stage"]
                # Wait until physically near the feature, never move the agent for a view.
                close = np.linalg.norm(env.agent.pos-env.engine_pos) < 45 if not stage.startswith("B") else np.linalg.norm(env.agent.pos-env.by_id[stage].pos) < 24
                if stage not in selected or stage in seen or not close:
                    continue
                seen.add(stage)
                before = env.state_digest()
                snapshot = dynamic_scene(env)
                if stage.startswith("B"):
                    buildings = [b for b in env.architecture if b["id"].startswith(stage+"-")]
                    feature = np.asarray(buildings[0]["pos"], float) if buildings else env.agent.pos.astype(float)
                    focus = feature+(0, 5, 0)
                    offset = np.array((-24., 19., -31.))
                else:
                    focus = env.engine_pos+(0, 9, 0)
                    offset = np.array((-50., 28., -60.))
                snapshot.metadata["camera"] = dict(eye=(focus+offset).tolist(), target=focus.tolist(), fov_y=48.)
                assert env.state_digest() == before, "Presentation mutated simulation"
                name = f"{len(entries):06}.npz"
                snapshot.save(output/"states"/name)
                entries.append(dict(index=len(entries), scene="states/"+name, scene_sha256=sha(output/"states"/name),
                                    step=env.steps, seconds=env.elapsed, milestone=stage, title=stage,
                                    digest=before, stage=stage, position=env.agent.pos.tolist(),
                                    source_start_step=env.steps, source_end_step=env.steps, duration=1., speed=1))
                print("EXPORTED", stage, env.steps, flush=True)
        assert env.success and env.state_digest() == manifest["final_digest"]
        save(output/"frames.json", dict(task="tempest", seed=manifest["seed"], trace_sha256=manifest["actions_sha256"],
             world_shape=manifest["world_shape"], source_seconds=env.elapsed, entries=entries,
             artifact_hashes={name: sha(output/name) for name in ("static.npz", "ground-cover.npz", "grass-prototypes.npz")},
             renderer=dict(engine="Blender Cycles", device="OPTIX GPU", width=1280, height=720, samples=64,
                           denoising=True, max_bounces=6, upscaling=False, full_world_geometry=True), sources=sources(),
             playback="Still previews of verified campaign states; not a continuous recording."))
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("collect", "verify", "export"))
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--objective", choices=("open", "extended"), default="extended")
    p.add_argument("--route", choices=("synchronization", "grounding", "capacitors"), default="synchronization")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--order", nargs=3, default=("cooling", "anchors", "monitoring"))
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.operation == "collect":
        collect(a.directory, a.objective, a.seed, a.route, a.order)
    elif a.operation == "verify":
        verify(a.directory)
    else:
        if not a.output:
            p.error("export requires --output")
        export(a.directory, a.output)


if __name__ == "__main__":
    main()
