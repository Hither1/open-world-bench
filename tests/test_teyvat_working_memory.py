"""Teyvat working-memory probes WM-TA06, WM-TB09 and WM-TB10 (docs/working-memory-probes.md).

Placement in these tests (setting the character's position or state) is test setup only;
the reference runs use native actions.
"""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from open_world_bench import working_memory_probes as wmp
from open_world_bench.character import MoveState
from open_world_bench.reference_policy import ReferencePolicy
from open_world_bench.reference_tasks import ReferenceTaskEnv

ROOT = Path(__file__).resolve().parents[1]
VIEW_RANGE = 160.0          # compact camera far plane and render range


def compact(task, **kw):
    env = ReferenceTaskEnv(task, 7, render_enabled=False, **kw)
    env.reset(seed=7)
    return env


def select(env, mid, kind):
    """Test setup: make an operation current."""
    c = env.course
    c.index = next(i for i, m in enumerate(c.milestones) if m.id == mid)
    c.operation_index = next(i for i, o in enumerate(c.current.operations) if o.kind == kind)
    return c.op


def stand_at(env, eid, offset=(0.0, 0.0, 0.5)):
    env.char.pos = env.world.get(eid).pos.copy() + np.array(offset, np.float32)


def entry_events(env):
    return [e for e in env.course.events if e.type.startswith("wm_")]


def panel_entity(op, symbol):
    return next(e for e, v in op.state["symbols"].items() if v == symbol)


# ---------------------------------------------------------------- content
def test_content_is_deterministic_and_drawn_apart_from_the_simulation_rng():
    a, b = compact("b"), compact("b")
    legacy = compact("b", working_memory=False)
    try:
        for pid in (wmp.TB09, wmp.TB10):
            assert a.course.wm[pid].state() == b.course.wm[pid].state()
        assert a.course.wm_marks == b.course.wm_marks
        assert len(set(a.course.wm_marks.values())) == wmp.CHEST_COUNT
        # Neither the env, world nor course generator is consumed by the probes or their fixtures.
        assert a.rng.bit_generator.state == legacy.rng.bit_generator.state
        assert a.world.rng.bit_generator.state == legacy.world.rng.bit_generator.state
        assert a.course.rng.bit_generator.state == legacy.course.rng.bit_generator.state
        assert a.course.relay_elements == legacy.course.relay_elements
        # Existing sites and operation ids are unchanged; WM operations are additions.
        old = {o.id: o.state["site"] for m in legacy.course.milestones for o in m.operations}
        new = {o.id: o.state["site"] for m in a.course.milestones for o in m.operations}
        assert {k: new[k] for k in old} == old
        assert sorted(set(new) - set(old)) == ["wm09_cue", "wm10_panel"]
    finally:
        for x in (a, b, legacy):
            x.close()
    other = wmp.make_probes("b", 8)
    assert [p.items for p in other.values()] != [p.items for p in a.course.wm.values()]


def test_distractor_marks_share_some_but_not_all_glyphs():
    for seed in range(20):
        probe = wmp.make_probes("b", seed)[wmp.TB09]
        marks, target = wmp.chest_marks(probe, seed)
        assert marks[target] == tuple(probe.items) and len(set(marks)) == wmp.CHEST_COUNT
        for k, mark in enumerate(marks):
            if k != target:
                shared = sum(x == y for x, y in zip(mark, probe.items))
                assert 1 <= shared <= 2 and set(mark) != set(probe.items)
            assert wmp.mark_from_code(wmp.mark_code(mark)) == mark


def test_legacy_course_reproduces_the_archived_traces_start():
    for task in "ab":
        manifest = json.loads((ROOT / f"assets/demos/teyvat-reference-cycles/task-{task}-seed-7/"
                                      "reference-manifest.json").read_text())
        with ReferenceTaskEnv(task, 7, render_enabled=False, working_memory=False) as env:
            env.reset(seed=7)
            assert env.state_digest() == manifest["initial_digest"]
            assert env.course.wm == {}


def test_use_sites_are_out_of_view_of_their_sources():
    for task in "ab":
        with ReferenceTaskEnv(task, 7, render_enabled=False) as env:
            env.reset(seed=7)
            assert env.cfg.camera.far <= VIEW_RANGE and env.cfg.render.max_view_distance <= VIEW_RANGE
            for pid, s in env.course.wm_summary().items():
                assert s["distance"] > VIEW_RANGE, pid


# ------------------------------------------------------- cue visibility
def test_cue_is_shown_only_in_reach_and_for_five_seconds():
    env = compact("a")
    try:
        op = select(env, "A06", "exchange")
        env.course.inventory["casing"] = 1
        stand_at(env, op.entities[0])
        env.course.interact(env.char)
        probe = env.course.wm[wmp.TA06]
        cue = wmp.cue_text(probe, compact=True)
        assert probe.cue_reads == 1 and cue in env._observe()["local_text"]
        assert ", ".join(probe.items[:-1]) not in env.course.last_feedback
        neutral = ReferencePolicy.neutral()
        t0 = env.world.elapsed
        while env.world.elapsed - t0 < 4.8:
            obs, *_ = env.step(neutral)
            assert cue in obs["local_text"]
        for _ in range(3):
            obs, *_ = env.step(neutral)
        assert env.world.elapsed - t0 > 5.0 and cue not in obs["local_text"]
        # Read again, then leave reach: it clears at once and stays cleared on return.
        env.course.interact(env.char)
        assert probe.cue_reads == 2 and env.course.wm_cue.cue is not None
        env.char.pos += np.array([6.0, 0, 0], np.float32)
        obs, *_ = env.step(neutral)
        assert cue not in obs["local_text"]
        env.char.pos -= np.array([6.0, 0, 0], np.float32)
        obs, *_ = env.step(neutral)
        assert cue not in obs["local_text"]
    finally:
        env.close()


@pytest.fixture(scope="module")
def reference_b():
    """A compact Task B reference run: observations and WM events per decision."""
    env = compact("b")
    obs = env._observe()
    policy = ReferencePolicy(env)
    rows = []
    for _ in range(9000):
        obs, _, done, truncated, info = env.step(policy(obs))
        rows.append(dict(obs={k: obs[k] for k in ("local_text", "instruction")}, time=env.world.elapsed,
                         events=info["event_details"], pos=env.char.pos.copy(), info=info))
        if done or truncated:
            break
    yield env, rows
    env.close()


def test_reference_completes_with_correct_first_attempts(reference_b):
    env, rows = reference_b
    assert env.course.verify()["success"] and env.quests.all_complete
    for pid, s in env.course.wm_summary().items():
        assert s["solved"] and s["first_attempt_correct"] and s["attempts"] == 1 and s["cue_reads"] == 1
        assert 10.0 <= s["first_attempt_delay"]["seconds"] <= 60.0, pid


def test_cue_never_reaches_observations_outside_its_window(reference_b):
    env, rows = reference_b
    cues = {pid: wmp.cue_text(p, compact=True) for pid, p in env.course.wm.items()}
    reads = {}
    for row in rows:
        for e in row["events"]:
            if e["type"] == "wm_cue_read":
                reads[e["probe"]] = row["time"]
        text = row["obs"]["local_text"] + " " + row["obs"]["instruction"]
        for pid, cue in cues.items():
            visible = cue in text
            assert not visible or row["time"] - reads[pid] <= wmp.CUE_SECONDS + 1e-6
            items = ", ".join(env.course.wm[pid].items)
            assert items not in row["obs"]["instruction"]
            if any(e["type"] in ("wm_entry", "wm_attempt") for e in row["events"]):
                assert not visible, "cue visible at the use site"
                assert items not in text
        # Evaluator-only fields stay out of the info dict that model runners see.
        assert "working_memory" not in row["info"]


def test_events_carry_no_per_item_correctness(reference_b):
    _, rows = reference_b
    entries = [e for r in rows for e in r["events"] if e["type"] == "wm_entry"]
    assert entries and all(set(e) == {"type", "probe", "count"} for e in entries)
    reads = [e for r in rows for e in r["events"] if e["type"] == "wm_cue_read"]
    assert reads and all(set(e) == {"type", "probe", "read_index"} for e in reads)


# ------------------------------------------------- entries and recovery
def test_wrong_lever_sequence_is_recoverable_without_item_feedback():
    env = compact("a")
    try:
        c = env.course
        op = select(env, "A06", "exchange")
        c.inventory["casing"] = 1
        stand_at(env, op.entities[0])
        c.interact(env.char)
        op = c.op
        assert op.kind == "panel"
        probe = c.wm[wmp.TA06]
        wrong = [s for s in wmp.LEVER_COLOURS if s not in probe.items][:1] + list(probe.items[1:])
        feedback = []
        for symbol in wrong:
            stand_at(env, panel_entity(op, symbol), (0, 0, .8))
            c.interact(env.char)
            feedback.append(c.last_feedback)
        assert probe.attempts == 1 and probe.first_attempt_correct is False and not op.complete
        assert "ferry_open" not in c.inventory and probe.buffer == []
        # Entry feedback names the lever and a count, never whether it was right.
        right_first = ReferenceTaskEnv("a", 7, render_enabled=False)
        right_first.reset(seed=7)
        try:
            r = right_first.course
            o = select(right_first, "A06", "exchange")
            r.inventory["casing"] = 1
            stand_at(right_first, o.entities[0])
            r.interact(right_first.char)
            stand_at(right_first, panel_entity(r.op, probe.items[0]), (0, 0, .8))
            r.interact(right_first.char)
            assert r.last_feedback.split(":")[1] == feedback[0].split(":")[1] == " click (1 set)."
        finally:
            right_first.close()
        # The mechanic can be revisited, then the correct entry solves the panel.
        stand_at(env, c.wm_sources[wmp.TA06])
        c.interact(env.char)
        assert probe.cue_reads == 2 and c.op is op
        for symbol in probe.expected:
            stand_at(env, panel_entity(op, symbol), (0, 0, .8))
            c.interact(env.char)
        assert probe.solved and op.complete and c.inventory["ferry_open"] == 1
        s = probe.summary()
        assert s["attempts"] == 2 and s["first_attempt_correct"] is False and s["cue_reads"] == 2
    finally:
        env.close()


def test_reverse_pattern_and_walk_away():
    env = compact("b")
    try:
        c = env.course
        op = select(env, "B10", "exchange")
        c.inventory.update(kit=1, coupling=1)
        stand_at(env, op.entities[0])
        c.interact(env.char)
        panel = select(env, "B10", "panel")
        probe = c.wm[wmp.TB10]
        stand_at(env, panel_entity(panel, probe.items[0]), (0, 0, .8))
        c.interact(env.char)
        assert len(probe.buffer) == 1
        env.char.pos += np.array([20.0, 0, 0], np.float32)
        c.tick(env.char, .03)                       # walked away: the partial entry drops back
        assert probe.buffer == [] and probe.attempts == 0
        for symbol in probe.items:                  # the cue's own order is wrong for a reverse probe
            stand_at(env, panel_entity(panel, symbol), (0, 0, .8))
            c.interact(env.char)
        assert probe.attempts == 1 and not probe.solved and not panel.complete
        for symbol in probe.expected:
            stand_at(env, panel_entity(panel, symbol), (0, 0, .8))
            c.interact(env.char)
        assert probe.solved and panel.complete
    finally:
        env.close()


def test_wrong_chest_is_empty_and_recoverable():
    env = compact("b")
    try:
        c = env.course
        op = select(env, "B09", "dive")
        probe = c.wm[wmp.TB09]
        target = probe.expected[0]
        wrong = next(e for e in op.entities if e != target)
        for eid in (wrong, wrong, target):
            stand_at(env, eid, (0, 1.2, .3))
            env.char.state = MoveState.SWIMMING
            c.interact(env.char)
            if eid == wrong:
                assert "empty" in c.last_feedback and "coupling" not in c.inventory and not op.complete
        assert op.complete and c.inventory["coupling"] == 1
        s = probe.summary()
        assert s["solved"] and s["attempts"] == 2 and s["first_attempt_correct"] is False
    finally:
        env.close()


def test_marks_are_readable_only_within_a_few_metres():
    env = compact("b")
    try:
        c = env.course
        select(env, "B09", "dive")
        op = c.op
        for eid, mark in c.wm_marks.items():
            stand_at(env, eid, (0, 1.2, 1.5))
            assert f"mark: {wmp.mark_text(mark)}" in c.local_text(env.char)
        # From the pool centre every chest is 4.5 m away: no mark is readable.
        x, z = op.state["site"]
        env.char.pos = np.array([x, -3.3, z], np.float32)
        assert "mark:" not in c.local_text(env.char)
    finally:
        env.close()


def test_matched_control_posts_the_cue_at_the_use_site():
    for control in (False, True):
        env = compact("b", wm_control=control)
        try:
            c = env.course
            panel = select(env, "B10", "panel")
            env.char.pos = env.world.ground_pos(*panel.state["site"])
            text = c.local_text(env.char)
            cue = wmp.cue_text(c.wm[wmp.TB10], compact=True)
            assert (cue in text) == control and c.wm[wmp.TB10].control == control
            dive = select(env, "B09", "dive")
            env.char.pos = env.world.ground_pos(*dive.state["site"])
            assert (wmp.cue_text(c.wm[wmp.TB09], compact=True) in c.local_text(env.char)) == control
            env.char.pos = env.world.ground_pos(0.0, 200.0)
            assert "Notice" not in c.local_text(env.char) and "Tag on" not in c.local_text(env.char)
        finally:
            env.close()


def test_probe_state_is_part_of_the_replay_digest():
    env = compact("a")
    try:
        before = env.state_digest()
        probe = env.course.wm[wmp.TA06]
        probe.read_cue(3, 1.0)
        assert env.state_digest() != before
        state = deepcopy(env.course.state()["working_memory"][wmp.TA06])
        assert state["items"] == list(probe.items) and state["cue_at"] == [3, 1.0]
    finally:
        env.close()


# -------------------------------------------------------------- full scale
@pytest.fixture(scope="module")
def expedition():
    from open_world_bench.fullscale_b_env import TeyvatExpeditionEnv
    env = TeyvatExpeditionEnv(7)
    env.reset(seed=7)
    return env


def _fs_stand(env, xz, y=None, swimming=False):
    env.char.pos[:] = (xz[0], env.fullscale.terrain.height_at(*xz) if y is None else y, xz[1])
    env.char.vel[:] = 0.0
    env.char.state = MoveState.SWIMMING if swimming else MoveState.GROUNDED


def test_fullscale_fixtures_are_clear_of_structures_and_far_enough(expedition):
    env, c = expedition, expedition.course
    summary = c.wm_summary()
    assert summary[wmp.TB10]["distance"] > 144.0      # beyond the policy renderer's entity range
    assert summary[wmp.TB09]["distance"] > 50.0
    arch = env.fullscale.architecture
    for eid in ["boatman", "winch", *c.wm_pawls, *c.wm_marks]:
        e = env.world.get(eid)
        ring = [e.pos + (np.cos(a) * 2.5, .8, np.sin(a) * 2.5) for a in np.arange(8) * np.pi / 4]
        assert not arch.blocked(np.array(ring), margin=.4).any(), eid
    for eid in c.wm_marks:
        assert env.world.get(eid).pos[1] < -6.0             # on the lake bed, below the dive threshold


def test_fullscale_chest_choice_cue_and_winch(expedition):
    env, c = expedition, expedition.course
    state = deepcopy(c.state())
    probe9, probe10 = c.wm[wmp.TB09], c.wm[wmp.TB10]
    boatman = env.world.get("boatman")
    _fs_stand(env, boatman.pos[[0, 2]] + (1.5, 0))
    c.interact(env.char)
    cue = wmp.cue_text(probe9)
    assert probe9.cue_reads == 1 and cue in c.local_text(env.char)
    # Chests: a wrong one is empty, the right one holds the coupling.
    target = probe9.expected[0]
    wrong = next(e for e in c.wm_marks if e != target)
    for eid in (wrong, target):
        e = env.world.get(eid)
        _fs_stand(env, e.pos[[0, 2]] + (1.0, 0), y=float(e.pos[1]) + 1.5, swimming=True)
        assert wmp.mark_text(c.wm_marks[eid]) in c.local_text(env.char)
        assert cue not in c.local_text(env.char)
        c.interact(env.char)
    assert c.pack.get("coupling") == 1 and probe9.solved and probe9.attempts == 2
    assert probe9.first_attempt_correct is False
    # Winch: chained until the mechanic has spoken; the pattern is entered in reverse.
    pawl = {v: k for k, v in c.wm_pawls.items()}
    _fs_stand(env, env.world.get(pawl[probe10.expected[0]]).pos[[0, 2]] + (0, 1.0))
    c.interact(env.char)
    assert "chained" in c.feedback[0] and probe10.attempts == 0 and probe10.buffer == []
    c.wm_armed.add(wmp.TB10)                         # test setup: as if the exchange had been made
    for symbol in probe10.items:
        _fs_stand(env, env.world.get(pawl[symbol]).pos[[0, 2]] + (0, 1.0))
        c.interact(env.char)
        assert "Within reach: Winch pawl" in c.local_text(env.char)
    assert probe10.attempts == 1 and not probe10.solved
    for symbol in probe10.expected:
        _fs_stand(env, env.world.get(pawl[symbol]).pos[[0, 2]] + (0, 1.0))
        c.interact(env.char)
    assert probe10.solved
    # Restore the module fixture.
    from owb.core.working_memory import WorkingMemoryProbe
    for pid, s in state["working_memory"].items():
        c.wm[pid] = WorkingMemoryProbe.from_state(s)
    c.wm_armed = set(state["wm_armed"])
    c.pack = dict(state["pack"])
    for eid in [*c.wm_marks, *c.wm_pawls]:
        env.world.get(eid).used = False
    c.wm_cue.cue = None


def test_fullscale_control_and_legacy_course(expedition):
    from open_world_bench.fullscale_b_env import COURSE_VERSION, TeyvatExpeditionEnv
    env, c = expedition, expedition.course
    _fs_stand(env, env.world.get("winch").pos[[0, 2]] + (0, 3.0))
    assert "Notice on the winch" not in c.local_text(env.char)
    control = TeyvatExpeditionEnv(7, wm_control=True)
    control.reset(seed=7)
    _fs_stand(control, control.world.get("winch").pos[[0, 2]] + (0, 3.0))
    assert wmp.cue_text(control.course.wm[wmp.TB10]) in control.course.local_text(control.char)
    assert control.course.wm[wmp.TB10].items == c.wm[wmp.TB10].items
    legacy = TeyvatExpeditionEnv(7, working_memory=False)
    legacy.reset(seed=7)
    manifest = json.loads((ROOT / "assets/demos/teyvat-fullscale-b/seed-7/manifest.json").read_text())
    assert legacy.course.version == COURSE_VERSION and legacy.state_digest() == manifest["initial_digest"]
    fresh = TeyvatExpeditionEnv(7)
    fresh.reset(seed=7)
    assert fresh.rng.bit_generator.state == legacy.rng.bit_generator.state
    assert fresh.world.rng.bit_generator.state == legacy.world.rng.bit_generator.state
    assert fresh.course.relay_elements == legacy.course.relay_elements
    assert fresh.course.vent_element == legacy.course.vent_element and fresh.course.safe_beat == legacy.course.safe_beat
