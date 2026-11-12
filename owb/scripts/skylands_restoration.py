"""Collect and independently replay the 32-checkpoint restoration campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from owb.env.skylands_restoration import SUITE, SkylandsRestorationEnv
from owb.bench.baselines.skylands_restoration import SkylandsRestorationPolicy
from owb.scripts.skylands_reference import ROOT, action_from_json, plain, save, sha
from owb.scripts.skylands_tempest import SOURCES as BASE_SOURCES, checked_step


# Headless evidence tracks simulation and controller sources. Presentation-only
# edits have no bearing on these traces; a render must record its own sources.
SOURCES = tuple(p for p in BASE_SOURCES if not p.startswith("owb/render/"))+(
    "owb/env/skylands_restoration.py", "owb/bench/baselines/skylands_restoration.py",
    "owb/scripts/skylands_restoration.py",
)


def sources():
    return {p: sha(ROOT/p) for p in SOURCES}


def collect(directory, objective, seed, route, order):
    directory.mkdir(parents=True, exist_ok=False)
    hashes = sources()
    env = SkylandsRestorationEnv(objective, seed)
    start = time.monotonic()
    print("GENERATE", objective, seed, route, flush=True)
    obs, _ = env.reset(seed=seed)
    initial = env.state_digest()
    policy = SkylandsRestorationPolicy(env, route=route, order=order)
    last = None
    try:
        with (directory/"actions.jsonl").open("x") as stream:
            while not env._ended:
                action = policy(obs)
                if not env.action_space.contains(action):
                    raise ValueError("Reference emitted invalid action")
                before = env.agent.pos.copy()
                obs, reward, terminated, truncated, info = env.step(action)
                row = dict(step=env.steps, stage=policy.stage, action=plain(action),
                           position_before=before.tolist(), position_after=env.agent.pos.tolist(),
                           seconds=env.elapsed, reward=reward, terminated=terminated, truncated=truncated,
                           events=plain(info["events"]), explanation=policy.explanation)
                if last != policy.stage or env.steps % 1000 == 0 or env._ended:
                    row["digest"] = env.state_digest()
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                if last != policy.stage:
                    print("STAGE", policy.stage, env.steps, round(env.elapsed, 1), "health", env.agent.health, flush=True)
                    last = policy.stage
                if env.steps % 1000 == 0:
                    stream.flush()
                    progress = dict(step=env.steps, stage=policy.stage, position=env.agent.pos.tolist(),
                                    explanation=policy.explanation, wall_seconds=time.monotonic()-start)
                    save(directory/"progress.json", progress)
                    print("PROGRESS", progress, flush=True)
        if not env.success or env.agent.deaths:
            raise RuntimeError("Reference failed the live objective")
        if hashes != sources():
            raise RuntimeError("Sources changed during collection; recollect against final sources")
        save(directory/"reference-manifest.json", dict(
            format=SUITE, objective=objective, seed=seed, route=route, order=order,
            initial_digest=initial, final_digest=env.state_digest(), evaluation=env.evaluation(),
            actions_sha256=sha(directory/"actions.jsonl"), sources=hashes,
            policy_profiles=policy.profiles, wall_seconds=time.monotonic()-start,
            controller="Privileged geometry routing; service profiles read from observations; legal actions only",
            limitations=["Authored geometry, not held-out layout generalization.",
                         "Reference itinerary, not a shortest-path lower bound or learned-agent result.",
                         "Low-level actions and simulated time are not model calls or memory-causality evidence."]))
        print("COLLECTED", env.evaluation(), flush=True)
    except BaseException:
        save(directory/"failure.json", dict(stage=policy.stage, step=env.steps,
            position=env.agent.pos.tolist(), explanation=policy.explanation,
            profiles=policy.profiles, evaluation=env.evaluation()))
        raise
    finally:
        env.close()


def replay_start(directory):
    manifest = json.loads((directory/"reference-manifest.json").read_text())
    if manifest["format"] != SUITE or manifest["sources"] != sources():
        raise ValueError("Suite or simulator source mismatch")
    if sha(directory/"actions.jsonl") != manifest["actions_sha256"]:
        raise ValueError("Action trace checksum mismatch")
    env = SkylandsRestorationEnv(manifest["objective"], manifest["seed"])
    env.reset(seed=manifest["seed"])
    if env.state_digest() != manifest["initial_digest"]:
        env.close()
        raise ValueError("Initial world mismatch")
    return env, manifest


def verify(directory):
    env, manifest = replay_start(directory)
    try:
        with (directory/"actions.jsonl").open() as stream:
            for line in stream:
                checked_step(env, json.loads(line))
        if not env.success or not env.system_operational() or env.agent.deaths:
            raise ValueError("Replay failed live objective")
        if env.state_digest() != manifest["final_digest"]:
            raise ValueError("Replay final state differs")
        result = dict(success=True, checked_actions=env.steps, checked_positions=True,
                      checked_events=True, checked_rewards=True, checked_termination=True,
                      final_live_predicate=True, final_digest=env.state_digest(),
                      actions_sha256=manifest["actions_sha256"], sources=sources(), evaluation=env.evaluation())
        save(directory/"verification.json", result)
        print("VERIFIED", env.steps, "actions;", len(env.evaluation()["campaign_milestones"]), "checkpoints", flush=True)
        return result
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("collect", "verify", "run"))
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--objective", choices=("open", "extended"), default="extended")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--route", choices=("synchronization", "grounding", "capacitors"), default="synchronization")
    p.add_argument("--order", nargs=3, default=("monitoring", "anchors", "cooling"))
    a = p.parse_args()
    if a.command in ("collect", "run"):
        collect(a.directory, a.objective, a.seed, a.route, a.order)
    if a.command in ("verify", "run"):
        verify(a.directory)


if __name__ == "__main__":
    main()
