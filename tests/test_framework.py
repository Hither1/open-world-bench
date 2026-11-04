"""Cross-backend controls, batching, episode semantics, and recording contracts."""
from dataclasses import asdict, replace
import json

import gymnasium as gym
from gymnasium.utils.env_checker import check_env, data_equivalence
import numpy as np
import pytest

import owb
from owb.bench import (
    BenchmarkConfig, BUTTONS, COMMANDS, MODALITIES, PROPRIO_NAMES,
    available_environments, environment_specs, make_env, neutral_action,
)
from owb.bench.rollout import DemoPolicy, RandomPolicy, run_episode


def native_config(name):
    if name == "teyvat":
        from open_world_bench import EnvConfig, RenderSettings, TerrainConfig, WorldConfig
        return EnvConfig(world=WorldConfig(terrain=TerrainConfig(size=512, resolution=65), prop_density=0),
                         render=RenderSettings(march_steps=16, shadow_steps=0, draw_props=False))
    return owb.EnvConfig(
        world_cfg=owb.WorldConfig(preset="arena", size_x=32, size_z=32, size_y=40, animal_density=0),
        sim_cfg=owb.SimConfig(weather_enabled=False, wind_enabled=False),
        render_cfg=owb.RenderConfig(max_dist=40, shadows=False, max_steps=64),
    )


def build(name, **kwargs):
    return make_env(name, native_config=native_config(name), image_size=(32, 24), **kwargs)


@pytest.fixture(params=available_environments())
def env(request):
    with build(request.param, max_steps=8) as instance:
        yield instance


def flatten(env):
    """Remove procedural obstacles to verify control semantics in world space."""
    native = env.native
    if env.name == "teyvat":
        from copy import deepcopy
        terrain = deepcopy(native.world.terrain)
        terrain.height[:] = 10.0
        terrain.normals[:] = (0, 1, 0)
        native.world.terrain = native.controller.terrain = native.camera_rig.terrain = terrain
        native.world.entities.clear()
        native.char = native.controller.spawn(0, 0)
        native.cam = native.camera_rig.reset(native.char)
    else:
        from owb.core import blocks as B
        native.world.fill_box((0, 0, 0), (31, 39, 31), B.AIR)
        native.world.fill_box((0, 0, 0), (31, 7, 31), B.STONE)
        native.agent.pos = np.array((12.5, 8.001, 12.5), np.float32)
        native.agent.vel[:] = 0
        native.npcs.clear()
        native.bodies.bodies.clear()
        native.landmarks.clear()
        native.discovered.clear()
        native.elements.fires.clear()


def test_registration_lists_only_runnable_backends():
    assert available_environments() == ("teyvat", "wildlands")
    assert not next(s for s in environment_specs() if s.name == "skylands").implemented
    with pytest.raises(NotImplementedError, match="planned"):
        make_env("skylands")
    with pytest.raises(ValueError, match="unknown"):
        make_env("not-an-environment")
    assert "OpenWorldBench/Genshin-v0" in gym.envs.registry
    assert "OpenWorldBench/Wildlands-v0" in gym.envs.registry


def test_common_spaces_are_identical():
    with build("teyvat") as a, build("wildlands") as b:
        assert a.action_space == b.action_space
        assert a.observation_space == b.observation_space


def test_gym_contract_and_seeded_replay(env):
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=7)
    assert env.observation_space.contains(obs)
    policy = RandomPolicy()
    policy.reset(7)
    actions, first = [], []
    for _ in range(4):
        action = policy(obs)
        actions.append(action)
        result = env.step(action)
        first.append(result)
        obs = result[0]
    reset_obs, reset_info = env.reset(seed=7)
    assert reset_info == info
    for action, expected in zip(actions, first):
        actual = env.step(action)
        assert data_equivalence(actual, expected, exact=True)
        assert env.observation_space.contains(actual[0])
        json.dumps(actual[-1], allow_nan=False)


def test_mixed_environment_vector_batch():
    vector = gym.vector.SyncVectorEnv([lambda: build("teyvat", max_steps=4),
                                      lambda: build("wildlands", max_steps=4)])
    try:
        obs, info = vector.reset(seed=7)
        assert obs["rgb"].shape == (2, 24, 32, 3)
        np.testing.assert_array_equal(info["success_defined"], [True, False])
        np.testing.assert_array_equal(info["task_progress_defined"], [True, False])
        action = {key: np.stack([value, value]) for key, value in neutral_action().items()}
        obs, rewards, terminated, truncated, info = vector.step(action)
        assert obs["proprio"].shape == (2, len(PROPRIO_NAMES))
        assert rewards.shape == (2,) and not terminated.any() and not truncated.any()
        for _ in range(3):
            obs, rewards, terminated, truncated, info = vector.step(action)
        assert truncated.all() and not terminated.any()
        # Both reset strategies used by supported Gymnasium versions must keep
        # unsupported success distinguishable from an unsuccessful task.
        obs, rewards, terminated, truncated, info = vector.step(action)
        np.testing.assert_array_equal(info["success_defined"], [True, False])
    finally:
        vector.close()


@pytest.mark.parametrize("yaw", [0.0, np.pi / 2, -np.pi / 3])
@pytest.mark.parametrize("axis", [0, 1])
def test_forward_and_right_follow_view_at_multiple_yaws(env, yaw, axis):
    env.reset(seed=7)
    flatten(env)
    if env.name == "teyvat":
        env.native.cam.yaw = yaw
        body = env.native.char
    else:
        env.native.agent.yaw = yaw
        body = env.native.agent
    _, forward, right, *_ = env.adapter.state()
    start = body.pos.copy()
    action = neutral_action()
    action["move"][axis] = 1.0
    obs, *_ = env.step(action)
    expected, perpendicular = (forward, right) if axis == 0 else (right, forward)
    delta = body.pos - start
    assert np.dot(delta, expected) > 0.1
    assert abs(np.dot(delta, perpendicular)) < 1e-4
    assert obs["proprio"][axis] > 0.0


def test_look_right_and_up_are_consistent(env):
    env.reset(seed=7)
    flatten(env)
    action = neutral_action()
    action["look"][:] = (0.5, 0.5)
    body = env.native.cam if env.name == "teyvat" else env.native.agent
    yaw, pitch = body.yaw, body.pitch
    env.step(action)
    assert body.yaw > yaw
    assert (body.pitch < pitch) if env.name == "teyvat" else (body.pitch > pitch)


def test_timing_and_end_of_episode(env):
    with pytest.raises(gym.error.ResetNeeded):
        env.step(neutral_action())
    env.reset(seed=7)
    for i in range(env.config.max_steps):
        obs, reward, term, trunc, info = env.step(neutral_action())
        assert info["simulated_seconds"] == pytest.approx((i + 1) * 0.2)
        assert trunc == (i + 1 == env.config.max_steps)
        assert not term
    assert info["episode"]["steps"] == 8
    assert info["termination_reason"] == "time_limit"
    if env.name == "wildlands":
        assert info["success"] is None and info["task_progress"] is None
    with pytest.raises(gym.error.ResetNeeded):
        env.step(neutral_action())
    assert env.render().shape == (24, 32, 3)


def test_common_death_semantics(env):
    env.reset(seed=7)
    if env.name == "teyvat":
        env.native.world.cfg.respawn_on_death = False
        env.native.char.hp = 0.0
    else:
        env.native.agent.health = 0.0
        env.native.agent.alive = False
    _, _, term, trunc, info = env.step(neutral_action())
    assert term and not trunc and info["termination_reason"] == "death"


def test_unsupported_capabilities_are_reported(env):
    obs, _ = env.reset(seed=7)
    action = neutral_action()
    command = "cook" if env.name == "teyvat" else "skill"
    action["command"] = COMMANDS.index(command)
    action["swap"] = 4
    assert obs["command_mask"][action["command"]] == 0
    _, _, _, _, info = env.step(action)
    assert {e.get("command") for e in info["events"] if e["type"] == "action_unavailable"} >= {command}
    assert any(e.get("swap") == 4 for e in info["events"])
    assert info["steps"] == 1


def test_invalid_actions_are_atomic(env):
    env.reset(seed=7)
    action = neutral_action()
    action["move"][0] = np.nan
    with pytest.raises(ValueError):
        env.step(action)
    assert env.steps == env.native.steps == 0


def test_returned_data_does_not_mutate_simulation(env):
    obs, info = env.reset(seed=7)
    expected = env.render()
    obs["rgb"][:] = 0
    frame = env.render()
    np.testing.assert_array_equal(frame, expected)
    frame[:] = 0
    np.testing.assert_array_equal(env.render(), expected)
    obs["party"][:] = 999
    obs["inventory"][:] = 999
    again, *_ = env.step(neutral_action())
    assert not np.all(again["party"] == 999)
    assert not np.all(again["inventory"] == 999)


def test_disabled_modalities_are_zero_and_goal_guidance_is_opt_in(env):
    obs, _ = env.reset(seed=7)
    for i, field in enumerate(MODALITIES):
        if not obs["modality_mask"][i]:
            assert not obs[field].any()
    assert not obs["goal"].any()
    with build("teyvat", include_goal_hint=True) as guided:
        obs, _ = guided.reset(seed=7)
        assert obs["modality_mask"][MODALITIES.index("goal")] == 1
        assert obs["goal"].any()


@pytest.mark.parametrize("name", available_environments())
def test_shared_config_does_not_mutate_native_config(name):
    cfg = native_config(name)
    original = asdict(cfg)
    with make_env(name, native_config=cfg, image_size=(32, 24), max_steps=2) as env:
        env.reset(seed=17)
        env.step(neutral_action())
    assert asdict(cfg) == original


def test_unrepresentable_decision_duration_is_rejected():
    for name in available_environments():
        with pytest.raises(ValueError, match="positive multiple"):
            build(name, decision_seconds=0.07)


def test_runner_records_and_replays_actions(env, tmp_path):
    path = tmp_path / env.name
    summary = run_episode(env, RandomPolicy(), seed=23, output_dir=path)
    config = json.loads((path / "config.json").read_text())
    assert config["environment"] == env.name and config["seed"] == 23
    assert summary == json.loads((path / "summary.json").read_text())
    rows = [json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()]
    assert len(rows) == env.config.max_steps + 1 and rows[0]["type"] == "reset"
    env.reset(seed=23)
    for row in rows[1:]:
        action = {key: np.asarray(value, dtype=env.action_space[key].dtype)
                  if env.action_space[key].shape else value for key, value in row["action"].items()}
        observation, reward, term, trunc, info = env.step(action)
        assert reward == row["reward"] and info == row["info"]
        np.testing.assert_array_equal(observation["proprio"], row["proprio"])
    with pytest.raises(FileExistsError):
        run_episode(env, seed=23, output_dir=path)


def test_demo_policy_uses_only_supported_commands(env):
    summary = run_episode(env, DemoPolicy(), seed=7)
    assert "action_unavailable" not in summary["event_counts"]
    if env.name == "wildlands":
        assert summary["event_counts"].get("cook", 0) + summary["event_counts"].get("cooked", 0) > 0
    else:
        assert summary["event_counts"]["skill"] > 0


def test_gif_recording_works_through_common_runner(tmp_path):
    pytest.importorskip("imageio")
    image_module = pytest.importorskip("PIL.Image")
    with build("teyvat", max_steps=3) as env:
        run_episode(env, DemoPolicy(), seed=7, output_dir=tmp_path / "video", video="gif")
    with image_module.open(tmp_path / "video" / "episode.gif") as gif:
        assert gif.size == (32, 24) and gif.n_frames >= 2
        assert gif.info["duration"] == 200


def test_presentation_renderer_preserves_gameplay_and_observation_contract():
    with build("teyvat", max_steps=3) as fast, build(
        "teyvat", max_steps=3, render_quality="presentation"
    ) as pretty:
        a, ia = fast.reset(seed=7)
        b, ib = pretty.reset(seed=7)
        assert ia == ib
        assert a["rgb"].shape == b["rgb"].shape == (24, 32, 3)
        assert pretty.observation_space.contains(b)
        assert not np.array_equal(a["rgb"], b["rgb"])
        for _ in range(2):
            action = neutral_action()
            action["move"][0] = 0.7
            ra, rb = fast.step(action), pretty.step(action)
            assert ra[1:] == rb[1:]
            for key in ra[0]:
                if key != "rgb":
                    np.testing.assert_equal(ra[0][key], rb[0][key])


def test_terrain_occludes_buried_meshes_in_the_same_depth_space():
    from open_world_bench import Entity, EntityKind
    with build("teyvat") as env:
        env.reset(seed=7)
        flatten(env)
        native = env.native
        native.cam.eye = np.array([0, 14, 6], np.float32)
        native.cam.pivot = np.array([0, 10, 0], np.float32)
        native._renderer.settings.draw_character = False
        native._renderer.settings.draw_props = False
        baseline = native._renderer.render(native.world, native.char, native.cam)
        native.world.spawn(Entity("buried", EntityKind.NPC, np.array([2, 7.5, 0], np.float32)))
        hidden = native._renderer.render(native.world, native.char, native.cam)
        np.testing.assert_array_equal(hidden, baseline)


def test_smooth_triangle_colors_interpolate_inside_the_triangle():
    from open_world_bench.render.raster import draw_triangles
    screen = np.array([[[0, 0], [4, 0], [0, 4]]], np.float32)
    color = np.zeros((4, 4, 3), np.float32)
    depth = np.full((4, 4), np.inf, np.float32)
    normal = np.zeros((4, 4, 3), np.float32)
    ids = np.zeros((4, 4), np.int32)
    shaded = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], np.float32)
    draw_triangles(screen, np.ones((1, 3)), np.array([True]), shaded,
                   np.array([[0, 1, 0]]), np.array([2]), color, depth, normal, ids)
    np.testing.assert_allclose(color[0, 0], (0.75, 0.125, 0.125))
    np.testing.assert_allclose(color[1, 1], (0.25, 0.375, 0.375))
    assert np.isinf(depth[3, 3])
