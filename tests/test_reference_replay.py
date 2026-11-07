"""Full replay of the four recorded reference traces (behaviour freeze).

Each test rebuilds the environment exactly as the recording script did
(``owb/scripts/teyvat_reference.py`` / ``owb/scripts/wildlands_reference.py``:
same task, seed 7, observation rendering disabled), replays every action in
``actions.jsonl`` and compares against the recorded files:

* per step: action validity, positions, native events, reward, simulated time,
  and every milestone ``state_digest``;
* at the end: the final digest, the course verification dict, completed
  milestones, event counts, and the counts/hashes in ``verification.json``,
  ``reference-manifest.json`` and (Teyvat) ``summary.json``.

Not asserted, because they are not reproducible from simulation alone or
change by design: the per-file source hashes (``validated_source_sha256`` /
``sources``; several are already stale relative to the current tree), video
hashes, render timings, and the ``runtime`` versions block.

Nothing here renders or writes into ``assets/``.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TEYVAT = ROOT / "assets/demos/teyvat-reference"
WILDLANDS = ROOT / "assets/demos/wildlands-reference"


def _load(directory):
    rows = [json.loads(line) for line in (directory / "actions.jsonl").read_text().splitlines()]
    manifest = json.loads((directory / "reference-manifest.json").read_text())
    verification = json.loads((directory / "verification.json").read_text())
    trace_sha = hashlib.sha256((directory / "actions.jsonl").read_bytes()).hexdigest()
    return rows, manifest, verification, trace_sha


def _roundtrip(value):
    """Compare in the representation the recorder wrote (JSON)."""
    return json.loads(json.dumps(value, allow_nan=False))


# --------------------------------------------------------------------- Teyvat

def _teyvat_action(raw):
    # Same restoration as teyvat_reference.verify_recording.
    return {k: np.asarray(v, dtype=np.int8 if k == "buttons" else np.float32)
            if k in ("move", "look", "buttons") else v for k, v in raw.items()}


@pytest.mark.parametrize("task", ["a", "b"])
def test_teyvat_reference_trace_replays_exactly(task):
    from open_world_bench.reference_tasks import ReferenceTaskEnv
    from owb.scripts.teyvat_reference import jsonable

    directory = TEYVAT / f"task-{task}-seed-7"
    rows, manifest, verification, trace_sha = _load(directory)
    summary = json.loads((directory / "summary.json").read_text())

    # Trace identity and recorded bookkeeping agree with each other.
    assert manifest["task"] == task and manifest["seed"] == 7
    assert trace_sha == verification["actions_sha256"] == summary["actions_sha256"]
    assert len(rows) == manifest["decisions"] == verification["checked_actions"]
    assert [r["step"] for r in rows] == list(range(1, len(rows) + 1))
    assert {k: summary[k] for k in manifest} == manifest
    milestone_rows = [r for r in rows if "state_digest" in r]
    assert len(milestone_rows) == len(manifest["milestones"]) == verification["checked_milestone_hashes"]

    env = ReferenceTaskEnv(task, manifest["seed"], render_enabled=False)
    try:
        env.reset(seed=manifest["seed"])
        assert env.state_digest() == manifest["initial_digest"]
        counts = Counter()
        checked_digests = 0
        for i, row in enumerate(rows):
            step = row["step"]
            assert env.char.pos.tolist() == row["position_before"], f"position_before at step {step}"
            assert env.course.current.id == row["milestone"], f"milestone at step {step}"
            assert env.course.op.id == row["operation"], f"operation at step {step}"
            action = _teyvat_action(row["action"])
            assert env.action_space.contains(action), f"action at step {step}"
            _, reward, terminated, truncated, info = env.step(action)
            counts.update(info["events"])
            assert env.char.pos.tolist() == row["position_after"], f"position_after at step {step}"
            assert info["event_details"] == row["events"], f"events at step {step}"
            assert reward == row["reward"], f"reward at step {step}"
            assert env.world.elapsed == row["simulated_seconds"], f"simulated_seconds at step {step}"
            assert ("milestone_complete" in info["events"]) == ("state_digest" in row), f"milestone at step {step}"
            if "state_digest" in row:
                assert env.state_digest() == row["state_digest"], f"state_digest at {row['milestone']} (step {step})"
                checked_digests += 1
            # The recorder stops at the first terminated/truncated step.
            ended = bool(terminated or truncated)
            assert ended == (i == len(rows) - 1), f"episode end flag at step {step}"
        assert checked_digests == len(manifest["milestones"])

        assert env.state_digest() == manifest["final_digest"]
        result = env.course.verify()
        assert result["success"] and env.quests.all_complete
        assert result == manifest["verification"] == verification["verification"]
        assert _roundtrip(jsonable(env.course.completed)) == manifest["milestones"]
        assert dict(counts) == manifest["event_counts"]
        assert counts["player_defeated"] == 0
        assert env.world.elapsed == manifest["simulated_seconds"]
    finally:
        env.close()


# ------------------------------------------------------------------ Wildlands

def _wildlands_action(raw):
    # Same restoration as wildlands_reference.restore_action.
    out = dict(raw)
    for key in ("move", "look", "vector"):
        out[key] = np.array(out[key], np.float32)
    out["buttons"] = np.array(out["buttons"], np.int8)
    out["ingredients"] = np.array(out["ingredients"], np.int64)
    return out


@pytest.mark.parametrize("task", ["a", "b"])
def test_wildlands_reference_trace_replays_exactly(task):
    from owb.env.wildlands_reference import WildlandsReferenceEnv, plain

    directory = WILDLANDS / f"task-{task}-seed-7"
    rows, manifest, verification, trace_sha = _load(directory)

    assert manifest["task"] == task and manifest["seed"] == 7
    assert trace_sha == manifest["actions_sha256"] == verification["actions_sha256"]
    assert len(rows) == manifest["decisions"] == verification["checked_actions"]
    assert [r["step"] for r in rows] == list(range(1, len(rows) + 1))
    assert verification["final_digest"] == manifest["final_digest"]
    assert len(manifest["milestones"]) == manifest["required"] == verification["checked_milestones"]
    assert [r["step"] for r in rows if "digest" in r] == [m["step"] for m in manifest["milestones"]]

    with WildlandsReferenceEnv(task, manifest["seed"]) as env:
        env.reset(seed=manifest["seed"])
        assert env.state_digest() == manifest["initial_digest"]
        counts = Counter()
        for row in rows:
            step = row["step"]
            assert not env._ended, f"episode ended before step {step}"
            assert env.agent.pos.tolist() == row["position_before"], f"position_before at step {step}"
            assert env.current.id == row["milestone"], f"milestone at step {step}"
            action = _wildlands_action(row["action"])
            assert env.action_space.contains(action), f"action at step {step}"
            _, reward, _, _, info = env.step(action)
            counts.update(e["type"] for e in info["events"])
            assert env.steps == step
            assert env.agent.pos.tolist() == row["position_after"], f"position_after at step {step}"
            assert plain(info["events"]) == row["events"], f"events at step {step}"
            assert reward == row["reward"], f"reward at step {step}"
            assert env.elapsed == row["seconds"], f"seconds at step {step}"
            if "digest" in row:
                assert env.state_digest() == row["digest"], f"digest at {row['milestone']} (step {step})"
        # The recorder loops `while not env._ended`.
        assert env._ended

        assert env.success and not env.agent.deaths
        assert len(env.completed) == manifest["required"]
        assert env.state_digest() == manifest["final_digest"]
        assert _roundtrip(plain(env.completed)) == manifest["milestones"]
        assert dict(counts) == manifest["events"]
        assert env.agent.health == manifest["health"]
        assert env.elapsed == manifest["simulated_seconds"]
        assert env.steps == manifest["decisions"]
