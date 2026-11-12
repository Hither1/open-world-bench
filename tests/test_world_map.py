"""Clickable fog-of-war world map shared by Teyvat, Wildlands and Skylands."""
import numpy as np
import pytest

from owb.sim import world_map as WM


def multi_action(env, **overrides):
    action = {key: np.zeros(space.shape, space.dtype) if space.shape else 0
              for key, space in env.action_space.spaces.items()}
    action.update(overrides)
    return action


def logo_click(env, width, height, **overrides):
    return multi_action(env, click=1, pointer=np.array(WM.logo_center(width, height), np.float32), **overrides)


def logo_pixels(rgb):
    c0, r0, c1, r1 = WM.logo_box(rgb.shape[1], rgb.shape[0])
    return rgb[r0:r1, c0:c1]


def test_fog_reveals_a_capsule_along_the_path_but_not_across_jumps():
    fog = WM.FogOfWar((0, 0, 100, 100), 1.0, 5.0)
    fog.reveal(10, 50)
    fog.reveal(30, 50)
    assert fog.is_explored(20, 50) and not fog.is_explored(20, 60)
    fog.reveal(90, 90)  # farther than six radii: a jump, not a trail
    assert fog.is_explored(90, 90) and not fog.is_explored(60, 70)
    fog.reset_trail()
    fog.reveal(70, 90)
    assert not fog.is_explored(80, 90 - 5.5)
    assert 0 < fog.fraction < 0.2


def test_unexplored_shadow_hides_world_content():
    fog = WM.FogOfWar((0, 0, 64, 64), 2.0, 8.0)
    a = np.full((64, 64, 3), 30, np.uint8)
    b = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert np.array_equal(WM.render_map(160, 120, a, (0, 0, 64, 64), fog),
                          WM.render_map(160, 120, b, (0, 0, 64, 64), fog))
    fog.reveal(32, 32)
    assert not np.array_equal(WM.render_map(160, 120, a, (0, 0, 64, 64), fog),
                              WM.render_map(160, 120, b, (0, 0, 64, 64), fog))


def test_clicks_only_toggle_on_the_logo_and_tabs_switch_layers():
    ui = WM.WorldMapUI(("depths", "surface", "sky"))
    assert ui.click(0.5, 0.5, 128, 96, 1) == [] and not ui.open
    u, v = WM.logo_center(128, 96)
    assert ui.click(u, v, 128, 96, 1)[0] == dict(type="map_opened", layer="surface")
    c0, r0, c1, r1 = WM.tab_boxes(128, 96, 3)[0]  # top tab is the highest layer
    assert ui.click((c0 + c1) / 256, (r0 + r1) / 192, 128, 96, 1)[0]["layer"] == "sky"
    assert ui.click(u, v, 128, 96, 1)[0]["type"] == "map_closed" and not ui.open


def test_wildlands_logo_click_opens_paused_map_and_closes():
    from owb.env import make_small

    env = make_small()
    obs, info = env.reset(seed=4)
    h, w = obs["rgb"].shape[:2]
    assert not info["map_open"] and info["explored_fraction"] > 0
    closed_logo = logo_pixels(obs["rgb"]).copy()
    for _ in range(3):
        obs, *_ = env.step(multi_action(env, move=np.array([1, 0], np.float32)))
    pos, elapsed, steps = env.agent.pos.copy(), env.elapsed, env.steps
    obs, reward, term, trunc, info = env.step(logo_click(env, w, h))
    assert info["map_open"] and [e["type"] for e in info["events"]] == ["map_opened"]
    assert not np.array_equal(logo_pixels(obs["rgb"]), closed_logo)  # open-state badge
    # Paused: movement is ignored, time stands still, the decision still counts.
    obs, *_, info = env.step(multi_action(env, move=np.array([1, 0], np.float32)))
    assert np.array_equal(env.agent.pos, pos) and env.elapsed == elapsed and env.steps == steps + 2
    obs, *_, info = env.step(multi_action(env, click=1, pointer=np.array([0.3, 0.6], np.float32)))
    assert info["map_open"] and info["events"] == []
    obs, *_, info = env.step(logo_click(env, w, h, move=np.array([1, 0], np.float32)))
    assert not info["map_open"] and info["events"][0]["type"] == "map_closed"
    assert env.elapsed > elapsed
    # Actions recorded before the map existed omit the pointer fields.
    legacy = multi_action(env)
    del legacy["pointer"], legacy["click"]
    env.step(legacy)
    assert np.array_equal(env.render(), env._screen(env._render_buffers()["rgb"]))


def test_wildlands_discrete_map_action_and_disabled_map():
    from owb.core.config import EnvConfig, RenderConfig, WorldConfig
    from owb.env import WildlandsEnv

    cfg = EnvConfig(action_space="discrete", obs_rgb=False,
                    world_cfg=WorldConfig(size_x=48, size_z=48, size_y=48, relief=8, mountain_relief=8))
    env = WildlandsEnv(cfg)
    env.reset(seed=1)
    assert env.discrete_actions[-1] == "map"
    _, _, _, _, info = env.step(env.discrete_actions.index("map"))
    assert info["map_open"]
    _, _, _, _, info = env.step(env.discrete_actions.index("map"))
    assert not info["map_open"]
    cfg.world_map = False
    plain = WildlandsEnv(cfg, render_cfg=RenderConfig(width=32, height=24))
    assert "map" not in plain.discrete_actions
    _, info = plain.reset(seed=1)
    assert "map_open" in info and not info["map_open"] and "explored_fraction" not in info


def test_skylands_keeps_separate_layer_maps():
    from owb.env.skylands import make_small

    env = make_small()
    obs, info = env.reset(seed=2)
    h, w = obs["rgb"].shape[:2]
    explored = info["explored_fraction_by_layer"]
    assert explored[info["layer"]] > 0 and sum(v > 0 for v in explored.values()) == 1
    obs, *_, info = env.step(logo_click(env, w, h))
    assert info["map_open"] and env.map_ui.view_layer == env._layer()
    c0, r0, c1, r1 = WM.tab_boxes(w, h, 3)[2]  # bottom tab: depths
    obs, *_, info = env.step(multi_action(env, click=1, pointer=np.array(((c0 + c1) / 2 / w, (r0 + r1) / 2 / h),
                                                                            np.float32)))
    assert info["events"] == [dict(type="map_layer_selected", layer="depths")]
    assert env.map_ui.view_layer == 0


def test_teyvat_map_click_pauses_and_supports_flat_actions():
    from open_world_bench.env import _make_small

    env = _make_small(max_episode_steps=50)
    obs, info = env.reset(seed=3)
    h, w = obs["rgb"].shape[:2]
    assert not info["map_open"] and info["explored_fraction"] > 0
    move = dict(move=np.array([0, 1], np.float32))
    env.step(multi_action(env, **move))
    pos, elapsed = env.char.pos.copy(), env.world.elapsed
    obs, *_, info = env.step(logo_click(env, w, h))
    assert info["map_open"] and info["events"] == ["map_opened"]
    env.step(multi_action(env, **move))
    assert np.array_equal(env.char.pos, pos) and env.world.elapsed == elapsed
    obs, *_, info = env.step(logo_click(env, w, h, **move))
    assert not info["map_open"] and env.world.elapsed > elapsed

    flat = _make_small(flat_actions=True)
    flat.reset(seed=3)
    assert flat.action_space.shape == (18,)
    action = np.zeros(18, np.float32)
    action[-3] = 1
    action[-2:] = np.array(WM.logo_center(64, 48)) * 2 - 1
    assert flat.step(action)[4]["map_open"]
    flat.step(np.zeros(15, np.float32))  # pre-map flat layout is still accepted


@pytest.mark.parametrize("name", ["wildlands", "skylands"])
def test_common_schema_click_reaches_native_map(name):
    import owb
    from owb.bench.framework import SCHEMA_VERSION, neutral_action

    env = owb.make_env(name, seed=0, image_size=(64, 48))
    _, info = env.reset()
    assert SCHEMA_VERSION == 3 and info["map_open"] is False and info["explored_fraction"] > 0
    action = neutral_action()
    action.update(click=1, pointer=np.array(WM.logo_center(64, 48), np.float32))
    obs, _, _, _, info = env.step(action)
    assert info["map_open"] and obs["rgb"].shape == (48, 64, 3)
    old = neutral_action()
    del old["pointer"], old["click"]
    assert env.step(old)[4]["map_open"]  # schema-2 action: no click, map stays open
