"""The reference-course TaskSpecs agree with the courses and their recorded traces.

``owb/tasks/specs/{teyvat,wildlands}/reference_course_{a,b}.yaml`` declare each course's
milestones (operation_complete / milestone_complete events, requires chains/DAG, memory
roles). Fed the recorded event stream, ``MilestoneEvaluator`` must fire the same milestone
ids, in the same order, at the same steps as the course ledger (``course.completed`` /
``env.completed``) did when the trace was recorded, and live on a replay.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from owb.tasks import MilestoneEvaluator, load_task
from owb.tasks.setups import get_setup
from owb.tasks.teyvat.course import evaluator_events as teyvat_events
from owb.tasks.wildlands.course import evaluator_events as wildlands_events

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "assets/demos"
CASES = [("teyvat", "a"), ("teyvat", "b"), ("wildlands", "a"), ("wildlands", "b")]


def _rows(sim, task):
    directory = DEMOS / f"{sim}-reference/task-{task}-seed-7"
    rows = [json.loads(line) for line in (directory / "actions.jsonl").read_text().splitlines()]
    manifest = json.loads((directory / "reference-manifest.json").read_text())
    return rows, manifest


def _recorded_milestones(sim, rows, manifest):
    """(id, step) of every milestone completion, from the recorded trace."""
    if sim == "teyvat":
        out = [(e["milestone"], r["step"]) for r in rows for e in r["events"]
               if e["type"] == "milestone_complete"]
        assert [m for m, _ in out] == [m["id"] for m in manifest["milestones"]]
        assert [s for _, s in out] == [r["step"] for r in rows if "state_digest" in r]
        return out
    out = [(m["id"], m["step"]) for m in manifest["milestones"]]
    assert out == [(e["milestone"], r["step"]) for r in rows for e in r["events"]
                   if e["type"] == "milestone_complete"]
    return out


def _fired(evaluator, events, step):
    return [(mid, step) for mid in evaluator.update(events, None, step)]


@pytest.mark.parametrize("sim,task", CASES)
def test_spec_matches_the_course_structure(sim, task):
    spec = load_task(f"{sim}.reference_course_{task}")
    assert spec.sim == sim and spec.setup == f"{sim}.reference_course"
    assert spec.setup_params == {"course": task}
    assert callable(get_setup(spec.setup))
    roles = {m.role for m in spec.milestones}
    assert roles <= {"learning", "interference", "transfer", "composition", "adaptation", "recall"}
    if sim == "teyvat":
        from owb.sims.teyvat.reference_tasks import ReferenceTaskEnv
        env = ReferenceTaskEnv(task, 7, render_enabled=False)
        try:
            course = env.course.milestones
            assert [m.id for m in spec.milestones] == [m.id for m in course]
            assert [m.title for m in spec.milestones] == [m.title for m in course]
            for i, (m, cm) in enumerate(zip(spec.milestones, course)):
                assert m.requires == ((course[i - 1].id,) if i else ())
                preds = m.done["preds"] if m.done.kind == "all" else (m.done,)
                assert [(p["type"], p["targets"]) for p in preds] == \
                    [("operation_complete", (o.id,)) for o in cm.operations]
            assert env.course.setup.record["course"] == task
        finally:
            env.close()
    else:
        from owb.sims.voxel.env.wildlands_reference import WildlandsReferenceEnv
        with WildlandsReferenceEnv(task, 7) as env:
            env.reset(seed=7)
            assert [m.id for m in spec.milestones] == [m.id for m in env.milestones]
            assert [m.title for m in spec.milestones] == [m.title for m in env.milestones]
            for m, em in zip(spec.milestones, env.milestones):
                assert m.requires == em.prereqs
                assert (m.done.kind, m.done["type"], m.done["targets"]) == \
                    ("event", "milestone_complete", (em.id,))
            assert env.setup.record["calibration"] == env.calibration


@pytest.mark.parametrize("sim,task", CASES)
def test_evaluator_fires_the_recorded_milestones(sim, task):
    rows, manifest = _rows(sim, task)
    adapt = teyvat_events if sim == "teyvat" else wildlands_events
    evaluator = MilestoneEvaluator(load_task(f"{sim}.reference_course_{task}"))
    evaluator.reset(None)
    fired = []
    for row in rows:
        fired += _fired(evaluator, adapt(row["events"]), row["step"])
    assert fired == _recorded_milestones(sim, rows, manifest)
    assert evaluator.success and evaluator.progress == 1.0


def _teyvat_action(raw):
    return {k: np.asarray(v, dtype=np.int8 if k == "buttons" else np.float32)
            if k in ("move", "look", "buttons") else v for k, v in raw.items()}


def _wildlands_action(raw):
    out = dict(raw)
    for key in ("move", "look", "vector"):
        out[key] = np.array(out[key], np.float32)
    out["buttons"] = np.array(out["buttons"], np.int8)
    out["ingredients"] = np.array(out["ingredients"], np.int64)
    return out


def test_evaluator_matches_the_live_teyvat_ledger():
    from owb.sims.teyvat.reference_tasks import ReferenceTaskEnv
    rows, _ = _rows("teyvat", "a")
    evaluator = MilestoneEvaluator(load_task("teyvat.reference_course_a"))
    env = ReferenceTaskEnv("a", 7, render_enabled=False)
    try:
        env.reset(seed=7)
        evaluator.reset(None)
        fired, ledger = [], []
        for step, row in enumerate(rows, 1):
            before = len(env.course.completed)
            _, _, _, _, info = env.step(_teyvat_action(row["action"]))
            ledger += [(m["id"], step) for m in env.course.completed[before:]]
            fired += _fired(evaluator, teyvat_events(info["event_details"]), step)
        assert fired == ledger and len(ledger) == len(env.course.milestones)
        assert evaluator.success == env.course.verify()["success"] is True
    finally:
        env.close()


def test_evaluator_matches_the_live_wildlands_ledger():
    from owb.sims.voxel.env.wildlands_reference import WildlandsReferenceEnv
    rows, _ = _rows("wildlands", "a")
    evaluator = MilestoneEvaluator(load_task("wildlands.reference_course_a"))
    with WildlandsReferenceEnv("a", 7) as env:
        env.reset(seed=7)
        evaluator.reset(None)
        fired = []
        for row in rows:
            _, _, _, _, info = env.step(_wildlands_action(row["action"]))
            fired += _fired(evaluator, wildlands_events(info["events"]), env.steps)
        assert fired == [(r["id"], r["step"]) for r in env.completed]
        assert evaluator.success and env.success
