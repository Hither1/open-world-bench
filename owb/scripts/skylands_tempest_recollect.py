"""Execute a candidate action sequence in a fresh current-version Tempest world.

Use after a scoring/interface refinement: every action is executed again, all
positions must match, and new events/digests come from the current simulator.
The candidate is input, never a source of claimed state or milestone credit.
"""
import argparse
from collections import Counter
from pathlib import Path
import json

import numpy as np

from owb.env.skylands_tempest import SkylandsTempestEnv
from owb.scripts.skylands_tempest import sources, verify
from owb.scripts.skylands_reference import action_from_json, plain, save, sha


def collect(candidate, directory, objective, seed, route, order):
    directory.mkdir(parents=True, exist_ok=False)
    candidate_path = candidate/"actions.jsonl"
    candidate_hash = sha(candidate_path)
    source_hashes = sources()
    source_hashes["owb/scripts/skylands_tempest_recollect.py"] = sha(Path(__file__))
    with SkylandsTempestEnv(objective, seed) as env:
        env.reset(seed=seed)
        initial = env.state_digest()
        counts = Counter()
        print("REEXECUTE", candidate, flush=True)
        with candidate_path.open() as incoming, (directory/"actions.jsonl").open("x") as outgoing:
            for line in incoming:
                old = json.loads(line)
                action = action_from_json(old["action"])
                if not env.action_space.contains(action):
                    raise ValueError("Candidate action is invalid")
                np.testing.assert_array_equal(env.agent.pos, old["position_before"])
                before = env.agent.pos.copy()
                _, reward, terminated, truncated, info = env.step(action)
                np.testing.assert_array_equal(env.agent.pos, old["position_after"])
                row = dict(step=env.steps, stage=old["stage"], action=plain(action),
                           position_before=before.tolist(), position_after=env.agent.pos.tolist(),
                           seconds=env.elapsed, reward=reward, terminated=terminated, truncated=truncated,
                           events=info["events"], explanation=old["explanation"])
                if "digest" in old or env._ended:
                    row["digest"] = env.state_digest()
                counts.update(e["type"] for e in info["events"])
                outgoing.write(json.dumps(row, allow_nan=False)+"\n")
                if env.steps % 5000 == 0:
                    outgoing.flush()
                    print("REEXECUTED", env.steps, flush=True)
        if not env.success or env.agent.deaths:
            raise RuntimeError("Candidate does not complete the current live objective")
        current = sources()
        current["owb/scripts/skylands_tempest_recollect.py"] = sha(Path(__file__))
        if current != source_hashes or sha(candidate_path) != candidate_hash:
            raise RuntimeError("Inputs changed during re-execution")
        save(directory/"reference-manifest.json", dict(
            format="owb-tempest-trace-v1", objective=objective, seed=seed, route=route, order=order,
            initial_digest=initial, final_digest=env.state_digest(), evaluation=env.evaluation(),
            events=dict(counts), world_shape=list(env.world.shape), actions_sha256=sha(directory/"actions.jsonl"),
            sources=source_hashes, candidate_actions_sha256=candidate_hash,
            controller="Privileged reference action sequence, re-executed through step() on this source version",
            positions_match_candidate=True,
            limitations=["Scripted action-only reference, not learned-agent or memory-benefit results.",
                         "Authored layout and rated grid-stepped cargo craft; no general rigid-body assemblies."]))
        print("COLLECTED CURRENT VERSION", env.evaluation(), flush=True)
    verify(directory)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--objective", choices=("open", "extended"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--route", choices=("synchronization", "grounding", "capacitors"), required=True)
    p.add_argument("--order", nargs=3, default=("cooling", "anchors", "monitoring"))
    a = p.parse_args()
    collect(a.candidate, a.directory, a.objective, a.seed, a.route, a.order)
