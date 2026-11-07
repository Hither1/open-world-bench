"""TaskSpec schema, YAML/JSON round trips, the registry, evaluator semantics, and the
Minecraft port onto MilestoneEvaluator."""
import hashlib
import json

import numpy as np
import pytest

from owb.tasks import (
    MINECRAFT_TASKS, Budget, MilestoneEvaluator, MilestoneTracker, MinecraftTask, Predicate,
    State, TaskMilestone, TaskSpec, available_tasks, dump, load, load_task,
)

yaml = pytest.importorskip("yaml")


def spec(*milestones, **kw):
    return TaskSpec(id=kw.pop("id", "t"), sim=kw.pop("sim", "test"), title="T", instruction="",
                    milestones=milestones, **kw)


def ms(mid, done, **kw):
    return TaskMilestone(mid, mid, "s", done if isinstance(done, Predicate) else Predicate.from_dict(done), **kw)


EV = {"kind": "event", "type": "ping"}


# ---------------------------------------------------------------- validation
def test_predicate_validation():
    with pytest.raises(ValueError, match="unknown predicate kind"):
        Predicate("teleport", {})
    with pytest.raises(ValueError, match="requires argument 'objects'"):
        Predicate("stat_increase", {"stat": "pickup"})
    with pytest.raises(ValueError, match="unknown arguments"):
        Predicate("reach", {"position": (0, 0, 0), "radiuss": 3})
    with pytest.raises(ValueError, match="three coordinates"):
        Predicate("reach", {"position": (0, 0)})
    p = Predicate.from_dict({"kind": "all", "preds": [EV, {"kind": "reach", "position": [1, 2, 3]}]})
    assert p["preds"][1]["radius"] == 6.0 and p["preds"][1]["position"] == (1, 2, 3)
    assert Predicate.from_dict({"kind": "event", "args": {"type": "ping"}}) == Predicate.from_dict(EV)


def test_taskspec_validation():
    with pytest.raises(ValueError, match="duplicate"):
        spec(ms("a", EV), ms("a", EV))
    with pytest.raises(ValueError, match="unknown milestone 'z'"):
        spec(ms("a", EV, requires=("z",)))
    with pytest.raises(ValueError, match="cycle"):
        spec(ms("a", EV, requires=("c",)), ms("b", EV, requires=("a",)), ms("c", EV, requires=("b",)))
    with pytest.raises(ValueError, match="no milestones"):
        spec()


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_round_trip(tmp_path, suffix):
    s = spec(ms("a", {"kind": "altitude", "min_height": 3, "time_window": [0.9, 0.1]}, role="learning"),
             ms("b", {"kind": "any", "preds": [EV, {"kind": "inventory_has", "item": "x", "count": 2}]},
                requires=("a",), reward=2.5, hidden_from_agent=False),
             ms("c", {"kind": "sequence", "targets": ["p", "q"]}, role="optional", expires_after=("b",)),
             budget=Budget(steps=100, tokens=10_000), setup="test.course", setup_params={"perm": (2, 0, 1)})
    for original in (s, *(load_task(t) for t in available_tasks())):
        path = dump(original, tmp_path / f"task{suffix}")
        assert load(path) == original


def test_registry_lists_shipped_specs():
    assert {"mine_diamond_from_scratch", "ender_dragon"} <= set(available_tasks())
    assert available_tasks("minecraft") == ("ender_dragon", "mine_diamond_from_scratch")
    with pytest.raises(KeyError, match="ender_dragon"):
        load_task("nether_portal")


# ----------------------------------------------------------------- semantics
def test_requires_unlock_on_the_next_update_and_counts_start_when_eligible():
    s = spec(ms("a", EV), ms("b", dict(EV, count=2), requires=("a",)))
    ev = MilestoneEvaluator(s)
    ev.reset(None)
    ping = {"type": "ping"}
    assert ev.update([ping, ping, ping], None, 1) == ["a"]     # b is not eligible yet
    assert ev.update([ping], None, 2) == []                      # 1 of 2 since eligible
    assert ev.update([ping], None, 3) == ["b"]
    assert ev.achieved == {"a": 1, "b": 3} and ev.success and ev.progress == 1.0
    json.dumps(ev.snapshot())


def test_distinct_sequence_expiry_and_compound():
    s = spec(
        ms("chests", {"kind": "event", "type": "open", "targets": ["c0", "c1"], "count": 2, "distinct": True}),
        ms("order", {"kind": "sequence", "type": "press", "targets": ["x", "y"]}),
        ms("bonus", EV, role="optional", expires_after=("chests",)),
        ms("both", {"kind": "all", "preds": [{"kind": "event", "type": "hit", "count": 2},
                                             {"kind": "reach", "position": [0, 0, 0], "radius": 1}]}),
    )
    ev = MilestoneEvaluator(s)
    ev.reset(State())
    far, near = State(position=(5, 0, 0)), State(position=(0.5, 9, 0))
    o = lambda e: {"type": "open", "entity": e}
    p = lambda e: {"type": "press", "entity": e}
    assert ev.update([o("c0"), o("c0"), p("x"), p("x"), {"type": "hit"}], far, 1) == []
    assert ev.update([o("c1"), p("x"), {"type": "hit"}], far, 2) == ["chests"]
    assert ev.update([p("y"), {"type": "ping"}], near, 3) == ["order", "both"]   # bonus expired
    assert ev.progress == 0.75 and ev.success          # the optional bonus is not required


def test_deliver_uses_the_inventory_before_the_hand_over():
    s = spec(ms("give", {"kind": "deliver", "item": "gem", "count": 2, "targets": ["npc"]}))
    ev = MilestoneEvaluator(s)
    ev.reset(State(inventory={"gem": 1}))
    talk = {"type": "talk", "entity": "npc"}
    # The sim deducts on the same step, so the post-step inventory shows 0.
    assert ev.update([talk], State(inventory={"gem": 1}), 1) == []
    assert ev.update([{"type": "collect", "item": "gem", "amount": 1}, talk],
                     State(inventory={"gem": 0}), 2) == ["give"]


def test_missing_state_never_satisfies_state_predicates():
    s = spec(ms("r", {"kind": "reach", "position": [0, 0, 0]}),
             ms("i", {"kind": "inventory_has", "item": "x"}),
             ms("t", {"kind": "altitude", "min_height": 0, "time_window": [0, 1]}))
    ev = MilestoneEvaluator(s)
    ev.reset(None)
    assert ev.update([], State(), 1) == []
    assert ev.update([], State(position=(0, 1, 0)), 2) == ["r"]


# ------------------------------------------------------------- Minecraft port
# sha256 of the task data as defined in Python before the port to YAML (owb/tasks/minecraft.py
# at 156b720): name, title, text, budget, seed and every (identity, event, objects, stage).
ORIGINAL_DIGESTS = {"mine_diamond_from_scratch": "37b89032a0d5df8b", "ender_dragon": "d0a75b8945c08409"}


@pytest.mark.parametrize("name", sorted(ORIGINAL_DIGESTS))
def test_minecraft_specs_match_the_original_definitions(name):
    t = MINECRAFT_TASKS[name]
    data = [t.name, t.title, t.text, t.step_budget, t.world_seed,
            [[m.identity, m.event, list(m.objects), m.stage] for m in t.milestones]]
    assert hashlib.sha256(json.dumps(data).encode()).hexdigest()[:16] == ORIGINAL_DIGESTS[name]
    assert MinecraftTask.from_spec(load_task(name)) == t
    assert MinecraftTask.from_spec(t.to_spec()) == t


def reference_update(task, prev, now, achieved, step):
    """The pre-port MilestoneTracker.update rule, verbatim."""
    fired = []
    for m in task.milestones:
        if m.identity in achieved:
            continue
        delta = 0.0
        for obj in m.objects:
            delta += now[m.event].get(obj, 0.0) - prev[m.event].get(obj, 0.0)
            if delta > 0:
                achieved[m.identity] = step
                fired.append(m.identity)
                break
    return fired


@pytest.mark.parametrize("name", sorted(MINECRAFT_TASKS))
def test_tracker_matches_the_pre_port_rule_on_random_statistics(name):
    task = MINECRAFT_TASKS[name]
    rng = np.random.default_rng(len(name))
    objects = sorted({(m.event, o) for m in task.milestones for o in m.objects})

    def info():
        # Sparse, noisy and sometimes decreasing statistics (e.g. a counter reset).
        out = {e: {} for e, _ in objects}
        for e, o in objects:
            if rng.random() < 0.5:
                out[e][o] = np.array(int(rng.integers(0, 3)))
        return out

    tracker = MilestoneTracker(task)
    first = info()
    tracker.reset(first)
    snap = lambda i: {e: {o: float(np.asarray(i.get(e, {}).get(o, 0)).item())
                          for (ee, o) in objects if ee == e} for e, _ in objects}
    prev, achieved = snap(first), {}
    for step in range(1, 60):
        now_info = info()
        expected = reference_update(task, prev, snap(now_info), achieved, step)
        assert tracker.update(now_info, step) == expected
        prev = snap(now_info)
    assert tracker.achieved == achieved
