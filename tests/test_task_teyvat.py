"""QuestGraph -> TaskSpec mapping: MilestoneEvaluator must agree with QuestEngine."""
import json

import numpy as np
import pytest

from owb.sims.teyvat.events import Event
from owb.sims.teyvat.quests import ObjKind, Objective, Quest, QuestEngine, QuestGraph, Stage
from owb.tasks import MilestoneEvaluator, State, TaskSpec
from owb.tasks.schema import GATE_ROLE, OPTIONAL_ROLE
from owb.tasks.teyvat import builtin_questline_spec, questgraph_to_spec

P1, P2, P3, P4 = (10.0, 2.0, 10.0), (-20.0, 30.0, 5.0), (40.0, -6.0, -40.0), (0.0, 1.0, -30.0)


def tricky_graph() -> QuestGraph:
    """Every ObjKind, an optional objective in a mixed stage, an all-optional final stage,
    a prerequisite join, a wrap-around time window and an item grant."""
    O = Objective
    return QuestGraph([
        Quest("intro", "Intro", "", stages=[
            Stage([O("i_talk", ObjKind.TALK, "talk", targets=("npc_a",))]),
            Stage([O("i_collect", ObjKind.COLLECT, "collect", item="gem", count=2),
                   O("i_scout", ObjKind.REACH, "scout", position=P1, radius=4.0, optional=True)]),
            Stage([O("i_deliver", ObjKind.DELIVER, "deliver", targets=("npc_a",), item="gem", count=2)]),
        ]),
        Quest("trials", "Trials", "", prerequisites=("intro",), grants=("token",), stages=[
            Stage([O("t_levers", ObjKind.ACTIVATE, "levers", targets=("lever_0", "lever_1", "lever_2"), count=2),
                   O("t_foes", ObjKind.DEFEAT, "foes", targets=("foe_0", "foe_1"), count=2)]),
            Stage([O("t_seq", ObjKind.SEQUENCE, "seq", targets=("p0", "p1", "p2"))]),
            Stage([O("t_bonus", ObjKind.OPEN_CHEST, "bonus", targets=("chest_0", "chest_1"), optional=True)]),
        ]),
        Quest("heights", "Heights", "", prerequisites=("intro",), stages=[
            Stage([O("h_climb", ObjKind.ALTITUDE, "climb", min_height=10.0, position=P2, radius=8.0,
                     time_window=(0.8, 0.1))]),
            Stage([O("h_dive", ObjKind.DIVE, "dive", position=P3, radius=5.0)]),
        ]),
        Quest("finale", "Finale", "", prerequisites=("trials", "heights"), stages=[
            Stage([O("f_reach", ObjKind.REACH, "reach", position=P4, radius=4.0)]),
            Stage([O("f_chest", ObjKind.OPEN_CHEST, "chest", targets=("chest_2",))]),
            Stage([O("f_high", ObjKind.ALTITUDE, "high", min_height=5.0)]),
        ]),
    ])


def objectives_done(engine: QuestEngine) -> set[str]:
    return set().union(*(p.done for p in engine.progress.values()))


def compare(graph: QuestGraph, steps):
    """Feed (events, position, time) per step to both; assert identical completions."""
    spec = questgraph_to_spec(graph, id="test")
    engine, ev = QuestEngine(graph), MilestoneEvaluator(spec)
    gates = {m.id for m in spec.milestones if m.role == GATE_ROLE}
    ev.reset(State(inventory=dict(engine.inventory)))
    native_steps: dict[str, int] = {}
    for step, (events, pos, tod) in enumerate(steps, start=1):
        before = objectives_done(engine)
        engine.consume(events, {"player_pos": np.asarray(pos, np.float32), "time_of_day": tod})
        new = objectives_done(engine) - before
        native_steps.update({oid: step for oid in new})
        fired = ev.update(events, State(np.asarray(pos, np.float32), tod, dict(engine.inventory)), step)
        assert set(fired) - gates == new, step
        assert ev.progress == pytest.approx(engine.completion())
        assert ev.success == engine.all_complete, step
    assert {k: v for k, v in ev.achieved.items() if k not in gates} == native_steps
    json.dumps(ev.snapshot())
    return engine, ev


def test_mapping_structure():
    spec = questgraph_to_spec(tricky_graph(), id="test", instruction="Finish the trials.")
    by = {m.id: m for m in spec.milestones}
    assert [m.id for m in spec.milestones if m.role != GATE_ROLE] == [
        o.id for q in tricky_graph().quests for s in q.stages for o in s.objectives]
    assert by["i_collect"].requires == ("i_talk",)
    assert by["i_scout"].role == OPTIONAL_ROLE and by["i_scout"].expires_after == ("i_collect",)
    assert by["i_deliver"].requires == ("i_collect",)           # optional does not gate
    assert by["t_levers"].requires == by["h_climb"].requires == ("i_deliver",)
    assert by["trials:stage2"].role == GATE_ROLE and by["t_bonus"].expires_after == ("trials:stage2",)
    assert by["f_reach"].requires == ("trials:stage2", "h_dive")  # prerequisite join
    assert by["t_levers"].done.to_dict() == {
        "kind": "event", "type": "activate", "targets": ["lever_0", "lever_1", "lever_2"],
        "count": 2, "distinct": True}
    assert TaskSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def random_steps(seed: int, n: int):
    rng = np.random.default_rng(seed)
    pool = ([Event("talk", {"entity": e}) for e in ("npc_a", "npc_b")]
            + [Event("collect", {"entity": "g", "item": "gem", "amount": 1}),
               Event("collect", {"entity": "j", "item": "junk", "amount": 2})]
            + [Event("activate", {"entity": e}) for e in ("lever_0", "lever_1", "lever_2",
                                                          "p0", "p1", "p2", "p1", "p2", "other")]
            + [Event("defeat", {"entity": e}) for e in ("foe_0", "foe_1", "foe_9")]
            + [Event("open_chest", {"entity": e}) for e in ("chest_0", "chest_1", "chest_2")]
            + [Event("move_jump")])
    sites = np.array([P1, P2, P3, P4, (0.0, 0.0, 0.0)])
    out = []
    for _ in range(n):
        k = rng.integers(0, 4)
        events = [pool[i] for i in rng.integers(0, len(pool), k)]
        pos = sites[rng.integers(len(sites))] + rng.normal(0, 3.5, 3) * (1, 6, 1)
        out.append((events, pos, float(rng.random())))
    return out


@pytest.mark.parametrize("seed", range(6))
def test_random_streams_match_quest_engine(seed):
    engine, ev = compare(tricky_graph(), random_steps(seed, 1500))
    # The streams are rich enough to finish the questline, with and without the bonus.
    assert engine.all_complete and ev.success


def scripted_steps(graph: QuestGraph, rng: np.random.Generator, limit: int = 400):
    """An oracle that works on the engine's active objectives, plus noise. It yields
    lazily so that each step can look at the engine state left by the previous one."""
    engine = QuestEngine(graph)
    seq_pos: dict[str, int] = {}
    for _ in range(limit):
        if engine.all_complete:
            return
        events, pos, tod = [], np.zeros(3), 0.5
        active = engine.active_objectives()
        if active:
            _, o = active[int(rng.integers(len(active)))]
            k = o.kind
            if k == ObjKind.DELIVER and engine.inventory.get(o.item, 0) < o.count:
                # The real world hands the item over with a `collect` event (chests, pickups).
                events.append(Event("collect", {"entity": "x", "item": o.item, "amount": 1}))
            elif k in (ObjKind.TALK, ObjKind.DELIVER):
                events.append(Event("talk", {"entity": o.targets[0]}))
            elif k == ObjKind.COLLECT:
                events.append(Event("collect", {"entity": "x", "item": o.item, "amount": 1}))
            elif k in (ObjKind.ACTIVATE, ObjKind.OPEN_CHEST, ObjKind.DEFEAT):
                events.append(Event(k.value, {"entity": o.targets[int(rng.integers(len(o.targets)))]}))
            elif k == ObjKind.SEQUENCE:
                i = seq_pos.get(o.id, 0)
                wrong = rng.random() < 0.2
                events.append(Event("activate", {"entity": o.targets[(i + 1) % len(o.targets)] if wrong
                                                 else o.targets[i % len(o.targets)]}))
                seq_pos[o.id] = 0 if wrong else i + 1
            if o.position is not None:
                pos = np.array(o.position, dtype=np.float64)
                if k == ObjKind.ALTITUDE:
                    pos[1] = max(pos[1], o.min_height + 1.0)
                if k == ObjKind.DIVE:
                    pos[1] -= 0.5
            elif k == ObjKind.ALTITUDE:
                pos[1] = o.min_height + 1.0
            if o.time_window is not None and rng.random() < 0.7:
                tod = float(np.mean(o.time_window))
        if rng.random() < 0.3:
            events.append(Event("collect", {"entity": "n", "item": "noise", "amount": 1}))
        world = {"player_pos": pos.astype(np.float32), "time_of_day": tod}
        engine.consume(events, world)
        yield events, pos, tod


def test_builtin_questline_matches_quest_engine_to_completion():
    spec = builtin_questline_spec(0)
    # Same world as the common env builds.
    import owb
    with owb.make_env("teyvat", seed=0, image_size=(8, 6)) as env:
        graph = env.unwrapped.native.graph
        assert questgraph_to_spec(graph, id=spec.id, setup_params=spec.setup_params) == spec
    assert spec.sim == "teyvat" and len([m for m in spec.milestones if m.role != GATE_ROLE]) == \
        graph.total_objectives
    for seed in range(3):
        steps = list(scripted_steps(graph, np.random.default_rng(seed)))
        engine, ev = compare(graph, steps)
        assert engine.all_complete and ev.success, seed


def test_real_env_progress_never_exceeds_native_completion():
    import owb
    from owb.bench.rollout import RandomPolicy
    from owb.tasks.env import TeyvatStateView

    env = owb.make_env("teyvat", max_steps=40, image_size=(32, 24))
    try:
        obs, info = env.reset(seed=3)
        spec = questgraph_to_spec(env.native.graph, id="teyvat.windborne_ledger")
        view = TeyvatStateView(env.native)
        ev = MilestoneEvaluator(spec)
        ev.reset(view)
        policy = RandomPolicy()
        policy.reset(3)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            ev.update(info["events"], view, info["steps"])
            assert ev.progress <= info["task_progress"] + 1e-9
            assert set(ev.achieved) - {m.id for m in spec.milestones if m.role == GATE_ROLE} \
                <= objectives_done(env.native.quests)
            done = terminated or truncated
        assert info["steps"] == 40
    finally:
        env.close()
