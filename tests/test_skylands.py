"""Construction, ascent, layer memory, rendering and trial completion contracts."""
from copy import deepcopy

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

import owb
from owb.bench import DemoPolicy, run_episode
from owb.sims.voxel.core import blocks as B
from owb.sims.voxel.env.skylands import COMMANDS, SkylandsEnv, default_config


def action(command="none", *, part=0, delta=(0, 0, 0)):
    return dict(move=np.zeros(2, np.float32), look=np.zeros(2, np.float32),
                buttons=np.zeros(5, np.int8), command=COMMANDS.index(command), slot=0,
                ingredients=np.zeros(5, np.int64), part=part, build_move=np.array(delta, np.float32))


@pytest.fixture
def env():
    cfg = default_config(True)
    cfg.obs_rgb = False
    with SkylandsEnv(cfg) as e:
        e.reset(seed=7)
        yield e


def test_native_contract_and_registration(env):
    check_env(env, skip_render_check=True)
    assert "OpenWorldBench/Skylands-Small-v0" in gym.envs.registry
    with gym.make("OpenWorldBench/Skylands-Small-v0", obs_rgb=False) as registered:
        obs, _ = registered.reset(seed=7)
        assert registered.observation_space.contains(obs)


@pytest.mark.parametrize("seed", [0, 7, 23])
def test_demo_builds_and_crosses_without_privileged_state(seed):
    with owb.make_env("skylands", image_size=(24, 18), max_steps=45) as e:
        summary = run_episode(e, DemoPolicy(), seed=seed)
        assert summary["success"] and summary["termination_reason"] == "success"
        assert summary["event_counts"]["part_placed"] == 2
        assert summary["event_counts"]["trial_complete"] == 1
        assert summary["metrics"]["bridge_complete"]
        assert e.native.agent.deaths == 0
        assert summary["event_counts"].get("fall_damage", 0) == 0
        with pytest.raises(gym.error.ResetNeeded):
            e.step(owb.bench.neutral_action())


def test_failed_placement_preserves_parts_and_voxels(env):
    original = env.world.vox.copy()
    env.step(action("grab", part=1))
    part = env.construction.parts[1]
    assert all(env.world.get(*p) == B.AIR for p in part.cells())
    # Move into the supporting terrain. Failure cannot overwrite that terrain.
    env.step(action(delta=(0, -1, 0)))
    before = env.world.vox.copy()
    *_, info = env.step(action("place"))
    assert any(e.get("reason") == "occupied" for e in info["events"])
    assert env.construction.held == 1
    np.testing.assert_array_equal(before, env.world.vox)
    env.step(action(delta=(0, 1, 0)))
    env.step(action("place"))
    np.testing.assert_array_equal(original, env.world.vox)


def test_out_of_reach_and_occluded_parts_cannot_be_grabbed(env):
    p = env.construction.parts[1]
    env.agent.pos += (0, -25, 0)
    *_, info = env.step(action("grab", part=1))
    assert any(e.get("reason") == "no_reachable_part" for e in info["events"])
    obs = env._observation()
    assert not obs["construction"].any()
    env.reset(seed=7)
    p = env.construction.parts[1]
    x, y, z = env.agent.pos.astype(int)
    env.world.fill_box((x - 1, y, z - 1), (x + 3, y + 3, z - 1), B.BRICK)
    *_, info = env.step(action("grab", part=1))
    assert any(e.get("reason") == "no_reachable_part" for e in info["events"])
    assert all(env.world.get(*cell) == B.BRIDGE for cell in p.cells())


def test_move_is_once_per_decision_rotation_and_invalid_action_are_atomic(env):
    env.step(action("grab", part=1))
    start = env.construction.parts[1].anchor.copy()
    env.step(action(delta=(1, 0, 0)))
    np.testing.assert_array_equal(env.construction.parts[1].anchor, start + (1, 0, 0))
    env.step(action("rotate"))
    np.testing.assert_array_equal(env.construction.parts[1].size, (1, 1, 3))
    before = env.construction.parts[1].anchor.copy(), env.steps, env.elapsed
    bad = action("place", delta=(np.nan, 0, 0))
    with pytest.raises(ValueError):
        env.step(bad)
    np.testing.assert_array_equal(before[0], env.construction.parts[1].anchor)
    assert (env.steps, env.elapsed) == before[1:]


def test_ascent_reaches_sky_and_layer_maps_remain_separate(env):
    edge, sky, z, _ = env.world.meta["bridge"]
    surface = env.world.meta["layer_heights"][1]
    env.agent.pos = np.array((edge - 1.5, surface + 1.01, z + 0.5), np.float32)
    env.agent.vel[:] = 0
    env.step(action())
    surface_memory = env.map_memory.copy()
    assert env.layer == 1
    *_, info = env.step(action("ascend"))
    assert any(e["type"] == "ascended" for e in info["events"])
    assert env.agent.pos[1] == pytest.approx(sky + 1, abs=0.02)
    assert env.layer == 2
    np.testing.assert_array_equal(surface_memory, env.layer_maps[1])
    assert info["visited_cells_by_layer"]["depths"] == 0
    assert info["visited_cells_by_layer"]["surface"] > 0
    assert env.agent.stamina < env.agent.max_stamina


def test_ascent_rejects_thick_ceiling_blocked_exit_and_empty_sky(env):
    *_, info = env.step(action("ascend"))
    assert any(e.get("reason") == "no_ceiling" for e in info["events"])
    edge, _, z, _ = env.world.meta["bridge"]
    surface = env.world.meta["layer_heights"][1]
    env.agent.pos = np.array((edge + 1.5, surface + 1.01, z + 0.5), np.float32)
    env.agent.vel[:] = 0
    x, y, z = env.agent.pos.astype(int)
    env.world.fill_box((x - 1, y + 3, z - 1), (x + 1, y + 12, z + 1), B.BRICK)
    before = env.agent.pos.copy()
    *_, info = env.step(action("ascend"))
    assert any(e.get("reason") == "ceiling_too_thick" for e in info["events"])
    np.testing.assert_allclose(env.agent.pos, before, atol=0.02)
    env.world.fill_box((x - 1, y + 3, z - 1), (x + 1, env.world.size_y - 1, z + 1), B.AIR)
    env.world.set_block(x, y + 3, z, B.BRICK)
    env.world.set_block(x, y + 5, z, B.BRICK)
    *_, info = env.step(action("ascend"))
    assert any(e.get("reason") == "blocked_exit" for e in info["events"])


def test_beacon_requires_a_bridge_not_just_arrival(env):
    x, y, z = env.world.meta["beacon"]
    env.agent.pos = np.array((x - 1.5, y + 0.01, z + 0.5), np.float32)
    _, _, terminated, _, info = env.step(action("interact"))
    assert not terminated and not info["success"]
    assert any(e.get("reason") == "bridge_incomplete" for e in info["events"])


@pytest.mark.parametrize("style", ["landscape", "voxel"])
def test_render_buffers_are_deterministic_and_include_held_parts(style):
    cfg = default_config(True)
    cfg.render_cfg.style = style
    cfg.render_cfg.width, cfg.render_cfg.height = 48, 32
    cfg.render_cfg.quality = "fast"
    cfg.render_cfg.third_person = True
    cfg.obs_depth = cfg.obs_semantic = True
    with SkylandsEnv(cfg) as e:
        obs, _ = e.reset(seed=7)
        rng = deepcopy(e.np_random.bit_generator.state)
        voxels = e.world.vox.copy()
        frame = e.render()
        e._frame = None
        np.testing.assert_array_equal(frame, e.render())
        assert e.np_random.bit_generator.state == rng
        np.testing.assert_array_equal(voxels, e.world.vox)
        assert e.observation_space.contains(obs)
        obs, *_ = e.step(action("grab", part=1))
        assert e.observation_space.contains(obs)
        assert np.isfinite(obs["depth"]).all()
        assert e.construction.render_list()
        e.reset(seed=7)
        np.testing.assert_array_equal(frame, e.render())
