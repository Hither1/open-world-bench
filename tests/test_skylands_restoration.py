"""Local controls, live dependencies, information boundaries and persistence.

State fixtures below isolate predicates; complete legal-action runs are collected
and independently replayed by owb.scripts.skylands_restoration.
"""
from copy import deepcopy
import json

import gymnasium as gym
import numpy as np
import pytest

from owb.env.skylands_restoration import EXTRA_CHECKPOINTS, SkylandsRestorationEnv
from owb.env.skylands_tempest import SkylandsTempestEnv
from owb.bench.baselines.skylands_reference import neutral
from owb.bench.baselines.skylands_restoration import SkylandsRestorationPolicy


@pytest.fixture(scope="module")
def world():
    with SkylandsRestorationEnv(seed=19) as e:
        e.reset(seed=19)
        yield e


@pytest.fixture
def env(world, monkeypatch):
    for name in ("cargo", "satchel", "service_dials", "service_profiles", "tool_location",
                 "manifold_manufactured", "refuge_switches", "refuge_stable_seconds", "engine_disabled",
                 "shutdown_seconds", "completed", "restoration_completed", "online", "facts_seen",
                 "last_text", "cue_until", "success", "hauling", "objective", "closed", "steps"):
        monkeypatch.setattr(world, name, deepcopy(getattr(world, name)))
    for m in world.milestones:
        monkeypatch.setattr(m, "state", deepcopy(m.state))
    monkeypatch.setattr(world.agent, "pos", world.agent.pos.copy())
    monkeypatch.setattr(world.agent, "health", world.agent.health)
    return world


def local(e, tid):
    e.agent.pos = (e.targets[tid]["pos"]+(0, -1, -1)).astype(np.float32)
    assert e._reachable(e.targets[tid]["pos"], 4.5)


def operate(e, key, command="use", option=0):
    tid = e.restoration_targets[key]
    local(e, tid)
    a = neutral(); a["target"] = tid; a["option"] = option
    return e._dispatch(command, [], a)


def services(e):
    for mid, name in (("B15", "cooling"), ("B17", "anchors"), ("B19", "monitoring")):
        e.by_id[mid].state.update(installed=True, stable=5., station_part="spindle")
        e.service_dials[name] = e.service_profiles[name].copy()
    for cid, mid in (("power_module", "B15"), ("marsh_coupling", "B17")):
        e.cargo[cid].update(parent=mid, anchored=True, pos=e.by_id[mid].pos.copy())


def restored(e):
    services(e)
    e.manifold_manufactured = True
    e.cargo["pressure_manifold"].update(parent="pressure_bus", anchored=True, pos=e.manifold_socket.copy())
    e.cargo["relief"].update(parent="refuge", anchored=True, delivered_to=e.destination,
                              pos=e.refuge_receivers[e.destination].copy())
    e.engine_disabled = True
    e.shutdown_seconds = 5.
    e.refuge_switches[e.destination] = True
    e.refuge_stable_seconds = 30.


def test_registered_objectives_and_local_observation(env):
    assert gym.spec("OpenWorldBench/Skylands-Tempest-Restoration-v0").entry_point.endswith(":SkylandsRestorationEnv")
    assert gym.spec("OpenWorldBench/Skylands-Tempest-Restoration-Open-v0").entry_point.endswith(":make_open")
    obs = env._observation()
    assert env.observation_space.contains(obs)
    assert "pressure manifold" in obs["text"]
    assert "service profile:" not in obs["text"]
    assert not {"campaign_milestones", "service_profiles", "tooling_location"} & env._info([]).keys()
    assert len(env.milestones)+len(EXTRA_CHECKPOINTS) == 32


@pytest.mark.parametrize("key", ["tooling", "forge", "socket", "cooling_0", "refuge_0"])
def test_remote_fixture_cannot_change_progress(env, key):
    action = neutral(); action["target"] = env.restoration_targets[key]
    before = (env.tool_location, env.manifold_manufactured, deepcopy(env.service_dials), list(env.refuge_switches))
    events = env._dispatch("use", [], action)
    assert events[0]["type"] == "action_failed"
    assert before == (env.tool_location, env.manifold_manufactured, env.service_dials, env.refuge_switches)


def test_tooling_has_single_identity_and_can_be_recovered_from_chosen_locker(env):
    assert operate(env, "tooling", "take")[0]["type"] == "tooling_recovered"
    assert operate(env, "tooling", "take")[0]["type"] == "action_failed"
    tid = next(t for t, o in env.targets.items() if o["role"] == "cache" and o["option"] == 3)
    local(env, tid)
    a = neutral(); a["target"] = tid; a["option"] = 2
    env._dispatch("cache", [], a)
    assert env.tool_location == "locker:3" and not env.satchel["manifold_tooling"]
    assert env._dispatch("cache", [], a)[0]["type"] == "action_failed"
    env._dispatch("retrieve", [], a)
    assert env.tool_location == "carried" and env.satchel["manifold_tooling"] == 1
    assert env._dispatch("retrieve", [], a)[0]["type"] == "action_failed"


def test_service_profile_is_only_revealed_by_local_inspection(env):
    a = neutral(); a["target"] = env.engine_targets["record_0"]
    assert env._dispatch("inspect", [], a)[0]["type"] == "action_failed"
    local(env, a["target"])
    events = env._dispatch("inspect", [], a)
    assert "Cooling service profile: dials 0, 1, 2 = " in env._observation()["text"]
    assert str(env.service_profiles["cooling"][0]) in events[0]["text"]


def test_calibration_requires_real_equipment_and_can_be_invalidated(env):
    for name in env.service_dials:
        env.service_dials[name] = env.service_profiles[name].copy()
    assert not any(env.live_regions().values())
    services(env)
    assert all(env.live_regions().values())
    wrong = (env.service_profiles["cooling"][0]+1) % 4
    operate(env, "cooling_0", option=wrong)
    assert not env.live_regions()["cooling"]
    operate(env, "cooling_0", option=env.service_profiles["cooling"][0])
    assert env.live_regions()["cooling"]
    env.cargo["power_module"]["intact"] = False
    assert not env.live_regions()["cooling"]


def test_fabrication_needs_tooling_and_all_calibrated_services(env):
    assert operate(env, "forge")[0]["type"] == "action_failed"
    operate(env, "tooling", "take")
    assert operate(env, "forge")[0]["type"] == "action_failed"
    services(env)
    assert operate(env, "forge")[0]["type"] == "manifold_fabricated"
    assert env.tool_location == "forge" and not env.satchel["manifold_tooling"]
    assert not env.cargo["pressure_manifold"]["anchored"]
    assert operate(env, "forge")[0]["type"] == "action_failed"
    assert operate(env, "tooling", "take")[0]["type"] == "action_failed"


def test_socket_requires_the_fabricated_original_and_live_position(env):
    assert operate(env, "socket")[0]["type"] == "action_failed"
    env.cargo["pressure_manifold"]["pos"] = env.manifold_socket.copy()
    assert operate(env, "socket")[0]["type"] == "action_failed"
    env.manifold_manufactured = True
    assert operate(env, "socket")[0]["type"] == "manifold_installed"
    assert env.manifold_live()
    env.cargo["pressure_manifold"]["pos"][0] += 10
    assert not env.manifold_live()


def test_historical_checkpoints_cannot_replace_live_success(env):
    env.completed = [{"id": f"B{i:02}"} for i in range(1, 25)]
    env.restoration_completed = [{"id": key} for key in EXTRA_CHECKPOINTS]
    env.engine_disabled = True; env.shutdown_seconds = 5.
    assert not env.system_operational()
    restored(env)
    env.completed = []; env.restoration_completed = []
    assert env.system_operational()
    env.cargo["relief"]["intact"] = False
    assert not env.system_operational()


def test_power_must_remain_stable_and_can_recover_from_interruption(env):
    restored(env)
    env.refuge_stable_seconds = 29.7
    env._task_outcome([])
    assert not env.success
    env._task_outcome([])
    assert env.success
    env.refuge_switches[env.destination] = False
    env._task_outcome([])
    assert env.refuge_stable_seconds == 0 and not env.success
    env.refuge_switches[env.destination] = True
    env._task_outcome([])
    assert env.refuge_stable_seconds == .2 and not env.success


def test_wrong_refuge_and_missing_manifold_do_not_power_target(env):
    restored(env)
    env.refuge_switches[env.destination] = False
    env.refuge_switches[1-env.destination] = True
    assert not env.system_operational()
    env.refuge_switches[env.destination] = True
    env.cargo["pressure_manifold"]["intact"] = False
    assert not env.system_operational()


def test_open_objective_allows_early_shutdown_without_restoration(env):
    env.objective = "open"
    env.engine_disabled = True; env.shutdown_seconds = 5.
    assert env.system_operational()
    assert not env.manifold_live() and not any(env.live_regions().values())


def test_reference_reads_profiles_from_observations_not_hidden_settings(env):
    policy = SkylandsRestorationPolicy(env)
    # Keep it at its initial safe local stage while exposing a deliberately
    # different record, proving observation-owned memory rather than state reads.
    obs = env._observation()
    obs["text"] = "Cooling service profile: dials 0, 1, 2 = 3, 2, 1."
    policy(obs)
    assert policy.profiles == {"cooling": [3, 2, 1]}


def test_reference_revisits_upper_console_via_native_ascent(env):
    policy = SkylandsRestorationPolicy(env)
    policy.index = policy.plan.index("calibrate_monitoring_0")
    policy.profiles["monitoring"] = env.service_profiles["monitoring"].copy()
    env.by_id["B19"].state["landed"] = True  # A previous visit must not bypass ascent.
    env.agent.pos = np.asarray(env.by_id["B19"].state["launch"], dtype=np.float32)
    action = policy(env._observation())
    assert env.commands[int(action["command"])] == "ascend"
    old_y = env.agent.pos[1]
    env._dispatch("ascend", [], action)
    assert env.agent.pos[1] > old_y+5
    action = policy(env._observation())
    assert int(action["target"]) == env.restoration_targets["monitoring_0"]


def test_reused_craft_stops_at_loading_dock_before_collecting_second_payload(env, monkeypatch):
    monkeypatch.setattr(env, "rafts", deepcopy(env.rafts))
    r = env.rafts["T01"]
    r["parts"] = [pid for pid, p in env.parts.items() if p["owner"] == "T01"][:4]
    r.update(charge=200., drive_vector=np.array((-1., 0., 0.)), driving=True)
    env.agent.pos = (r["pos"]+(.5, .001, .5)).astype(np.float32)
    policy = SkylandsRestorationPolicy(env, route="grounding")
    policy.index = policy.plan.index("grounding_canal")
    action = policy(env._observation())
    assert env.commands[int(action["command"])] == "drive" and int(action["option"]) == 2
    before = r["pos"].copy()
    env._dispatch("drive", [], action)
    assert not np.any(r["drive_vector"])
    np.testing.assert_array_equal(r["pos"], before)
    assert r["cargo"] is None


def test_checkpoint_restores_variant_and_continuation(tmp_path):
    with SkylandsRestorationEnv(seed=23) as original:
        original.reset(seed=23)
        original.step(neutral())
        path = tmp_path/"restoration.json"
        original.save_checkpoint(path)
        with SkylandsRestorationEnv.load_checkpoint(path) as restored_env:
            assert restored_env.state_digest() == original.state_digest()
            for _ in range(2):
                a = original.step(neutral()); b = restored_env.step(neutral())
                assert a[1:] == b[1:]
            assert restored_env.state_digest() == original.state_digest()
        with pytest.raises(ValueError, match="replay diverged"):
            SkylandsTempestEnv.load_checkpoint(path)


def test_fixture_meshes_preserve_simulation_and_are_repeatable(env):
    from open_world_bench.render.meshes import MeshBatch
    from owb.render.skylands_restoration import restoration_meshes
    before = env.state_digest()
    a, b = MeshBatch(), MeshBatch()
    restoration_meshes(env, a); restoration_meshes(env, b)
    for x, y in zip(a.finish(), b.finish()):
        np.testing.assert_array_equal(x, y)
    assert env.state_digest() == before
