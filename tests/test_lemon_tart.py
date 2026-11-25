"""Milestone tests for the lemon tart. No MuJoCo and no kitchen assets."""
import numpy as np
import pytest

from owb.tasks.lemon_tart import (
    BAKEWARE,
    DISHWASHER,
    FRIDGE,
    FRIDGE_INGREDIENTS,
    HAND_TOOLS,
    INGREDIENTS,
    MILESTONES,
    RECIPE_INSTRUCTION,
    SHORT_INSTRUCTION,
    LemonTartTracker,
    assign_homes,
)


DRAWERS = ("d0", "d1", "d2", "d3", "d4")


def _homes():
    return assign_homes(DRAWERS, np.random.default_rng(0))


def _tracker(homes=None):
    homes = homes or _homes()
    return LemonTartTracker(homes, (FRIDGE, *DRAWERS)), homes


def _state(homes, **overrides):
    state = {
        "inside": {name: {home} for name, home in homes.items()},
        "touching": {name: set() for name in homes},
        "open": set(),
        "held": set(),
        "knobs": {"function": 0.0, "temperature": 0.0, "time": 0.0},
    }
    for key, value in overrides.items():
        state[key] = value
    return state


def _put(state, name, fixture):
    state["inside"][name] = {fixture}


def _knobs(state, value):
    state["knobs"] = {"function": value, "temperature": value, "time": value}


def test_assignment_keeps_ingredients_out_of_the_dishwasher():
    for seed in range(30):
        homes = assign_homes(DRAWERS, np.random.default_rng(seed))
        assert [homes[name] for name in FRIDGE_INGREDIENTS] == [FRIDGE] * 3
        assert homes["flour"] != homes["sugar"]
        assert {homes["flour"], homes["sugar"]} <= set(DRAWERS)
        assert homes["baking_sheet"] == DISHWASHER
        assert homes["tart_pan"] == DISHWASHER
        in_washer = [name for name in HAND_TOOLS if homes[name] == DISHWASHER]
        assert len(in_washer) == 1
        in_drawers = [homes[name] for name in HAND_TOOLS if homes[name] != DISHWASHER]
        assert len(in_drawers) == 2 and len(set(in_drawers)) == 2
        used = {homes["flour"], homes["sugar"], *in_drawers}
        assert used <= set(DRAWERS) and len(used) == 4
        assert all(homes[name] != DISHWASHER for name in INGREDIENTS)


def test_assignment_is_stable_for_a_seed():
    first = assign_homes(DRAWERS, np.random.default_rng(4))
    second = assign_homes(DRAWERS, np.random.default_rng(4))
    assert first == second


def test_needs_four_distinct_drawers():
    with pytest.raises(ValueError, match="4"):
        assign_homes(("d0", "d1", "d2"), np.random.default_rng(0))
    with pytest.raises(ValueError, match="unique"):
        assign_homes(("d0", "d0", "d1", "d2"), np.random.default_rng(0))


def test_full_recipe_latches_in_order_and_hides_the_card_after_the_crust():
    tracker, homes = _tracker()
    state = _state(homes)
    assert tracker.update(state) == []
    assert tracker.recipe_visible
    assert not tracker.succeeded

    for name in INGREDIENTS:
        _put(state, name, "counter")
    assert tracker.update(state) == ["ingredients_out"]
    assert tracker.recipe_visible

    _put(state, "tart_pan", "toaster_oven")
    _knobs(state, 0.4)
    assert tracker.update(state) == ["crust_baked"]
    assert not tracker.recipe_visible

    _knobs(state, 0.0)
    assert tracker.update(state) == ["knobs_zeroed"]

    state["touching"]["saucepan"] = {"stove"}
    assert tracker.update(state) == ["curd_started"]

    _put(state, "baking_sheet", "toaster_oven")
    _knobs(state, 0.5)
    assert tracker.update(state) == ["crumble_baked"]

    _put(state, "tart_pan", "counter")
    state["touching"]["tart_pan"] = {"counter"}
    state["touching"]["lemon"] = {"tart_pan"}
    assert tracker.update(state) == ["assembled"]

    for name in INGREDIENTS:
        _put(state, name, homes[name])
    state["touching"]["lemon"] = set()
    assert tracker.update(state) == ["ingredients_restored"]

    for name in ("bowl", "whisk", "saucepan", "baking_sheet"):
        _put(state, name, DISHWASHER)
    assert tracker.update(state) == ["tools_washed", "served"]
    assert tracker.public_state()["success"]
    assert list(tracker.latched) == list(MILESTONES)


def test_crumble_does_not_count_while_the_sheet_never_leaves_the_oven():
    tracker, homes = _tracker()
    state = _state(homes)
    for name in INGREDIENTS:
        _put(state, name, "counter")
    _put(state, "baking_sheet", "toaster_oven")
    _put(state, "tart_pan", "toaster_oven")
    _knobs(state, 0.5)
    tracker.update(state)
    assert "crust_baked" in tracker.latched

    state["touching"]["saucepan"] = {"stove"}
    _knobs(state, 0.0)
    tracker.update(state)
    _knobs(state, 0.5)
    assert tracker.update(state) == []
    assert "crumble_baked" not in tracker.latched

    _put(state, "baking_sheet", "counter")
    tracker.update(state)
    _put(state, "baking_sheet", "toaster_oven")
    assert tracker.update(state) == ["crumble_baked"]


def test_putting_the_lemon_away_without_setting_it_on_the_tart_does_not_finish():
    tracker, homes = _tracker()
    state = _state(homes)
    for name in INGREDIENTS:
        _put(state, name, "counter")
    tracker.update(state)
    _put(state, "tart_pan", "toaster_oven")
    _knobs(state, 0.4)
    tracker.update(state)
    _knobs(state, 0.0)
    tracker.update(state)
    state["touching"]["saucepan"] = {"stove"}
    tracker.update(state)
    _put(state, "baking_sheet", "toaster_oven")
    _knobs(state, 0.5)
    tracker.update(state)
    for name in INGREDIENTS:
        _put(state, name, homes[name])
    assert tracker.update(state) == []
    assert "assembled" not in tracker.latched
    assert "ingredients_restored" not in tracker.latched


def test_restore_opens_are_evaluator_only():
    homes = {
        "butter": FRIDGE, "egg": FRIDGE, "lemon": FRIDGE,
        "flour": "d0", "sugar": "d1",
        "bowl": "d2", "whisk": DISHWASHER, "saucepan": "d3",
        "baking_sheet": DISHWASHER, "tart_pan": DISHWASHER,
    }
    tracker = LemonTartTracker(homes, (FRIDGE, *DRAWERS))
    state = _state(homes)
    for name in INGREDIENTS:
        _put(state, name, "counter")
    tracker.update(state)
    state["open"] = {"d4"}
    tracker.update(state)
    state["open"] = {FRIDGE}
    tracker.update(state)
    report = tracker.evaluator_state()
    assert report["opens_during_restore"] == 2
    assert report["wrong_opens"] == 1
    assert "homes" not in tracker.public_state()
    assert "wrong_opens" not in tracker.public_state()
    assert homes["flour"] not in RECIPE_INSTRUCTION
    assert "function" not in SHORT_INSTRUCTION


def test_success_drops_if_the_ending_is_undone_but_the_reward_stays_spent():
    tracker, homes = _tracker()
    state = _state(homes)
    for name in INGREDIENTS:
        _put(state, name, "counter")
    tracker.update(state)
    _put(state, "tart_pan", "toaster_oven")
    _knobs(state, 0.4)
    tracker.update(state)
    _knobs(state, 0.0)
    tracker.update(state)
    state["touching"]["saucepan"] = {"stove"}
    tracker.update(state)
    _put(state, "baking_sheet", "toaster_oven")
    _knobs(state, 0.5)
    tracker.update(state)
    _put(state, "tart_pan", "counter")
    state["touching"]["tart_pan"] = {"counter"}
    state["touching"]["lemon"] = {"tart_pan"}
    tracker.update(state)
    for name in INGREDIENTS:
        _put(state, name, homes[name])
    tracker.update(state)
    for name in ("bowl", "whisk", "saucepan", "baking_sheet"):
        _put(state, name, DISHWASHER)
    tracker.update(state)
    assert tracker.consume_reward() == len(MILESTONES)
    assert tracker.succeeded
    _put(state, "butter", "counter")
    assert tracker.update(state) == []
    assert not tracker.succeeded
    assert tracker.consume_reward() == 0


def test_the_tart_pan_is_not_a_dish_to_wash():
    tracker, homes = _tracker()
    assert "tart_pan" not in ("bowl", "whisk", "saucepan", "baking_sheet")
    assert homes["tart_pan"] == DISHWASHER
    assert set(BAKEWARE) <= set(homes)


def test_env_class_imports_when_robocasa_is_installed():
    pytest.importorskip("robocasa")
    from owb.tasks.lemon_tart_env import LemonTart

    assert LemonTart.__name__ == "LemonTart"


def test_knob_hint_states_the_reverse_rule_without_spelling_it_out():
    from owb.tasks.lemon_tart_lock import knob_hint

    text = knob_hint(("time", "function", "temperature"))
    assert "time, then function, then temperature" in text
    assert "reverse of that order" in text
    assert "temperature, then function, then time" not in text


def test_reverse_knob_probe_scores_blocked_turns_after_the_recipe_is_withdrawn():
    from owb.tasks.lemon_tart_lock import ReverseKnobProbe

    probe = ReverseKnobProbe(("time", "function", "temperature"))
    assert probe.probe.expected == ("temperature", "function", "time")
    # blocked turns while baking (recipe still in view) are not WM attempts
    assert probe.update(["ingredients_out"], 2, 100, 5.0) == []
    events = probe.update(["ingredients_out", "crust_baked"], 2, 200, 10.0)
    assert [e["type"] for e in events] == ["wm_cue_read"]
    events = probe.update(["ingredients_out", "crust_baked"], 3, 260, 13.0)
    assert events[-1]["type"] == "wm_attempt" and events[-1]["correct"] is False
    events = probe.update(["ingredients_out", "crust_baked", "knobs_zeroed"], 3, 400, 20.0)
    assert events[-1]["correct"] is True
    s = probe.probe.summary()
    assert s["solved"] and s["attempts"] == 2 and s["first_attempt_correct"] is False
    assert s["first_attempt_delay"] == {"decisions": 60, "seconds": 3.0}
    assert s["solve_delay"] == {"decisions": 200, "seconds": 10.0}
    assert probe.update(["crust_baked", "knobs_zeroed"], 9, 500, 25.0) == []


def test_reverse_knob_probe_first_attempt_correct_without_blocked_turns():
    from owb.tasks.lemon_tart_lock import ReverseKnobProbe

    probe = ReverseKnobProbe(("function", "temperature", "time"), control=True)
    probe.update(["crust_baked"], 0, 10, 1.0)
    probe.update(["crust_baked", "knobs_zeroed"], 0, 50, 3.0)
    s = probe.probe.summary()
    assert s["first_attempt_correct"] is True and s["attempts"] == 1 and s["control"] is True
