"""Objective semantics, local information, durable replay and real architecture."""
from copy import deepcopy
import json

import numpy as np
import pytest

from owb.bench.baselines.skylands_reference import neutral
from owb.env.skylands_tempest import SkylandsTempestEnv
from owb.core import blocks as B
from owb.render.tempest import _building


@pytest.fixture(scope="module")
def world():
    with SkylandsTempestEnv(seed=19) as env:
        env.reset(seed=19)
        yield env


def test_goal_only_observation_and_no_remote_evaluator_leak(world):
    obs = world._observation()
    assert world.observation_space.contains(obs)
    assert "runaway atmospheric machine" in obs["text"]
    assert "Cooling works" not in obs["text"]  # Unseen destinations stay private.
    assert "control:Landing office" in obs["text"]  # A nearby visible plaque is fair.
    assert "B01" not in obs["text"] and "next" not in obs["text"]
    assert not {"milestones", "task_progress", "live_regions"} & world._info([]).keys()
    ids = {int(row[1]) for row in obs["nearby"] if row[0]}
    assert world.engine_targets["stop"] not in ids
    assert world._info([dict(type="platform_reversed", position=[900, 112, 900])])["events"] == []


def test_open_success_does_not_require_milestone_checklist(world, monkeypatch):
    monkeypatch.setattr(world, "engine_disabled", True)
    monkeypatch.setattr(world, "shutdown_seconds", 5.)
    monkeypatch.setattr(world, "completed", [])
    assert world.system_operational()
    monkeypatch.setattr(world, "objective", "extended")
    assert not world.system_operational()


def test_extended_checks_live_services_and_original_delivery_not_training_flags(world, monkeypatch):
    monkeypatch.setattr(world, "objective", "extended")
    monkeypatch.setattr(world, "engine_disabled", True)
    monkeypatch.setattr(world, "shutdown_seconds", 5.)
    monkeypatch.setattr(world, "completed", [])
    for mid in ("B15", "B17", "B19"):
        state = deepcopy(world.by_id[mid].state)
        state.update(installed=True, stable=5., done=True)
        if mid == "B19":
            state["station_part"] = "spindle"
        monkeypatch.setattr(world.by_id[mid], "state", state)
    cargo = deepcopy(world.cargo)
    for cid, mid in (("power_module", "B15"), ("marsh_coupling", "B17")):
        cargo[cid].update(parent=mid, anchored=True, pos=world.by_id[mid].pos.copy())
    cargo["relief"].update(parent="refuge", anchored=True, delivered_to=world.destination,
                            pos=world.refuge_receivers[world.destination].copy())
    monkeypatch.setattr(world, "cargo", cargo)
    assert world.system_operational()
    cargo["relief"]["pos"][0] += 10
    assert not world.system_operational()
    cargo["relief"]["pos"] = world.refuge_receivers[world.destination].copy()
    cargo["power_module"]["intact"] = False
    assert not world.system_operational()


def test_refuge_accepts_an_alternative_physical_delivery_without_lock_flags(world, monkeypatch):
    cargo = deepcopy(world.cargo)
    monkeypatch.setattr(world, "cargo", cargo)
    destination = world.destination
    receiver = world.refuge_receivers[destination]
    cargo["relief"].update(pos=receiver.copy(), parent=None)
    monkeypatch.setattr(world.agent, "pos", (receiver+(0, .001, -1)).astype(np.float32))
    monkeypatch.setattr(world, "engine_disabled", True)  # Delivery can follow shutdown.
    monkeypatch.setattr(world.by_id["B24"], "state", deepcopy(world.by_id["B24"].state))
    action = neutral(); action["target"] = world.refuge_targets[destination]
    events = world._dispatch("use", [], action)
    assert any(e["type"] == "cargo_delivered" for e in events)
    assert cargo["relief"]["parent"] == "refuge"
    assert not world.by_id["B21"].state["installed"] and not world.by_id["B23"].state["installed"]


def test_remote_engine_action_preserves_state(world):
    before = (world.engine_disabled, world.locks.copy(), world.capacitors.copy())
    action = neutral(); action["target"] = world.engine_targets["lock_0"]
    events = world._dispatch("use", [], action)
    assert events[0]["type"] == "action_failed"
    assert before == (world.engine_disabled, world.locks, world.capacitors)


def test_wrong_selector_and_closed_shutters_do_not_grant_progress(world, monkeypatch):
    tid = world.engine_targets["lock_0"]
    monkeypatch.setattr(world.agent, "pos", (world.targets[tid]["pos"]+(0, -1, -1)).astype(np.float32))
    monkeypatch.setattr(world, "locks", [False]*3)
    monkeypatch.setattr(world, "steps", 60-world.cycle_offset)
    old_health = world.agent.health
    action = neutral(); action["target"] = tid; action["option"] = (world.lock_settings[0]+1) % 4
    events = world._dispatch("use", [], action)
    assert not world.locks[0] and events[0]["type"] == "action_failed"
    world.agent.health = old_health
    monkeypatch.setattr(world, "steps", -world.cycle_offset)
    action["option"] = world.lock_settings[0]
    assert world._dispatch("use", [], action)[0]["type"] == "action_failed"
    assert not world.locks[0]
    monkeypatch.setattr(world, "steps", 60-world.cycle_offset)
    assert world._dispatch("use", [], action)[0]["type"] == "circuit_isolated"
    assert world.locks[0]


def test_live_grounding_rejects_missing_or_relocated_original_cargo(world, monkeypatch):
    cargo = deepcopy(world.cargo)
    c = cargo["grounding_cartridge"]
    c.update(parent="tempest_ground", anchored=True, pos=world.grounding_socket.copy())
    monkeypatch.setattr(world, "cargo", cargo)
    monkeypatch.setattr(world, "grounding_installed", True)
    assert world.grounding_live()
    c["pos"][0] += 5
    assert not world.grounding_live()
    c["pos"] = world.grounding_socket.copy(); c["intact"] = False
    assert not world.grounding_live()


def test_unloaded_cargo_target_tracks_the_physical_receiver(world, monkeypatch):
    cargo = deepcopy(world.cargo)
    targets = deepcopy(world.targets)
    monkeypatch.setattr(world, "cargo", cargo)
    monkeypatch.setattr(world, "targets", targets)
    cargo["grounding_cartridge"]["pos"] = world.rafts["T01"]["receiver"].copy()
    world._task_outcome([])
    np.testing.assert_array_equal(targets[cargo["grounding_cartridge"]["target"]]["pos"],
                                  cargo["grounding_cartridge"]["pos"]+(0, 1, 0))


def test_regional_preparation_changes_live_encounter_window(world, monkeypatch):
    monkeypatch.setattr(world, "steps", 40-world.cycle_offset)
    assert world.engine_phase() == "pressurized"
    states = deepcopy(world.by_id["B15"].state)
    states.update(installed=True, stable=5.)
    monkeypatch.setattr(world.by_id["B15"], "state", states)
    cargo = deepcopy(world.cargo)
    cargo["power_module"].update(parent="B15", anchored=True, pos=world.by_id["B15"].pos.copy())
    monkeypatch.setattr(world, "cargo", cargo)
    assert world.engine_phase() == "venting"
    cargo["power_module"]["pos"][0] += 20
    assert world.engine_phase() == "pressurized"


def test_buildings_have_real_collision_and_open_entrances(world):
    assert len(world.architecture) == 50
    assert len({record["kind"] for record in world.architecture}) == 10
    for record in world.architecture:
        x, y, z = record["pos"]
        assert B.SOLID[world.world.get(x, y-1, z)]
        if record["kind"] in ("boathouse", "workshop", "foundry", "archive", "pump_house"):
            assert B.SOLID[world.world.get(x-7, y+1, z-6)]
            assert world.world.get(x, y+1, z-6) == B.AIR


def test_asset_generation_is_stable_diverse_and_does_not_touch_sim_rng(world):
    state = deepcopy(world.np_random.bit_generator.state)
    record = world.architecture[0]
    a = _building(record).finish()
    b = _building(record).finish()
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    alternate = dict(record, appearance=record["appearance"]+1)
    c = _building(alternate).finish()
    assert not np.array_equal(a[1], c[1])
    assert world.np_random.bit_generator.state == state
    assert len(a[0]) > 3000


def test_checkpoint_rejects_corruption_before_world_generation(tmp_path):
    path = tmp_path/"checkpoint.json"
    path.write_text(json.dumps(dict(payload="{}", sha256="incorrect")))
    with pytest.raises(ValueError, match="checksum"):
        SkylandsTempestEnv.load_checkpoint(path)


def test_native_checkpoint_continuation_is_exact(tmp_path):
    with SkylandsTempestEnv(seed=23) as original:
        original.reset(seed=23)
        for _ in range(3):
            original.step(neutral())
        path = tmp_path/"checkpoint.json"
        original.save_checkpoint(path)
        with SkylandsTempestEnv.load_checkpoint(path) as restored:
            assert restored.state_digest() == original.state_digest()
            for _ in range(4):
                _, ra, ta, ua, ia = original.step(neutral())
                _, rb, tb, ub, ib = restored.step(neutral())
                assert (ra, ta, ua, ia["events"]) == (rb, tb, ub, ib["events"])
            assert restored.state_digest() == original.state_digest()
