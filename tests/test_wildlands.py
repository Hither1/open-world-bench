"""Integration coverage for traversal, survival, inventory and the agent API."""
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

import owb
from owb.sims.voxel.core import blocks as B, items as I
from owb.sims.voxel.env import COMMANDS
from owb.sims.voxel.physics import collide
from owb.sims.voxel.world.worldgen import Landmark


def config(**kwargs):
    return owb.EnvConfig(
        world_cfg=owb.WorldConfig(preset="arena", size_x=32, size_z=32, size_y=40,
                                  animal_density=0),
        sim_cfg=owb.SimConfig(weather_enabled=False, wind_enabled=False),
        render_cfg=owb.RenderConfig(width=32, height=24, max_dist=40, shadows=False),
        obs_rgb=False, **kwargs)


def action(command="none", slot=0, ingredients=(), move=(0, 0), buttons=(0, 0, 0, 0, 0)):
    return dict(move=np.array(move, np.float32), look=np.zeros(2, np.float32),
                buttons=np.array(buttons, np.int8), command=COMMANDS.index(command), slot=slot,
                ingredients=np.array([s + 1 for s in ingredients] + [0] * (5 - len(ingredients))))


@pytest.fixture
def env():
    with owb.WildlandsEnv(config()) as e:
        e.reset(seed=4)
        yield e


def flat(e):
    """Controlled terrain isolates mechanics from procedural terrain variation."""
    e.world.fill_box((0, 0, 0), (31, 39, 31), B.AIR)
    e.world.fill_box((0, 0, 0), (31, 7, 31), B.STONE)
    e.agent.pos = np.array((12.5, 8.001, 12.5), np.float32)
    e.agent.vel[:] = 0
    e.npcs.clear()
    e.bodies.bodies.clear()
    e.landmarks.clear()
    e.discovered.clear()
    e.elements.fires.clear()
    e.gameplay.plants.clear()
    e.gameplay.chests.clear()


@pytest.mark.parametrize("mode", ["multi", "discrete"])
def test_gym_contract_and_seeded_resets(mode):
    e = owb.WildlandsEnv(config(action_space=mode, obs_depth=True, obs_semantic=True, obs_text=True))
    check_env(e, skip_render_check=True)
    first, info = e.reset(seed=12)
    e.action_space.seed(33)
    actions = [e.action_space.sample() for _ in range(4)]
    trajectory = [e.step(a) for a in actions]
    again, info2 = e.reset(seed=12)
    for key in first:
        np.testing.assert_equal(first[key], again[key])
    assert info == info2
    for expected, a in zip(trajectory, actions):
        actual = e.step(a)
        assert e.observation_space.contains(actual[0])
        for key in expected[0]:
            np.testing.assert_equal(expected[0][key], actual[0][key])
        assert expected[1:] == actual[1:]
    frame = e.render()
    assert frame.shape == (24, 32, 3) and frame.dtype == np.uint8
    frame[:] = 0
    assert e.render().any()  # external consumers cannot mutate the cached image


def test_registered_environment():
    with gym.make("OpenWorldBench/Wildlands-v0", cfg=config()) as e:
        obs, _ = e.reset(seed=3)
        assert e.observation_space.contains(obs)


def test_gather_cook_eat_once_per_decision(env):
    env.cfg.frame_skip = 5
    env.step(action("interact"))
    assert env.agent.inventory.count("spicy_pepper") == 1
    slot = next(i for i, st in enumerate(env.agent.inventory.stacks) if st.item == "spicy_pepper")
    result = env.step(action("cook", ingredients=(slot,)))
    assert any(e["type"] == "cook" for e in result[-1]["events"])
    dish = env.agent.inventory.stacks[slot]
    assert dish.data["dish"] and dish.effects()[I.EFF_COLD_RESIST] > 0
    env.agent.health = 10
    env.step(action("eat", slot=slot))
    assert env.agent.health > 10
    assert env.agent.cold_resist() > 0
    assert not env.agent.inventory.has("spicy_pepper")
    assert not any(s.data.get("dish") for s in env.agent.inventory.stacks)


def test_cooking_failure_is_atomic(env):
    inv = env.agent.inventory
    inv.add("spicy_pepper", 2)
    inv.capacity = len(inv.stacks)
    before = inv.to_dict()
    env.step(action("cook", ingredients=(3,)))
    assert env.agent.inventory.to_dict() == before
    env.step(action("cook", ingredients=(3, 3, 3)))
    assert env.agent.inventory.to_dict() == before
    result = env.step(action("cook", ingredients=(3, 3)))
    assert any(e["type"] == "cook" for e in result[-1]["events"])


def test_full_inventory_preserves_resource(env):
    env.agent.inventory.capacity = len(env.agent.inventory.stacks)
    plant = next(p for p in env.gameplay.plants if p["item"] == "spicy_pepper")
    env.step(action("interact"))
    assert not plant.get("used")
    env.agent.inventory.capacity = 32
    env.step(action("interact"))
    assert plant["used"]


def test_interactions_require_line_of_sight(env):
    flat(env)
    env.gameplay.plants.append(dict(pos=(14.5, 8, 12.5), item="spicy_pepper"))
    env.world.fill_box((13, 8, 11), (13, 11, 13), B.STONE)
    env.step(action("interact"))
    assert not env.agent.inventory.has("spicy_pepper")
    env.world.fill_box((13, 8, 11), (13, 11, 13), B.AIR)
    env.step(action("interact"))
    assert env.agent.inventory.has("spicy_pepper")


def test_sprinting_rest_and_jump(env):
    flat(env)
    start = env.agent.pos.copy()
    env.step(action(move=(1, 0), buttons=(0, 1, 0, 0, 0)))
    assert env.agent.pos[0] > start[0]
    assert env.agent.stamina < env.agent.max_stamina
    stamina = env.agent.stamina
    env.step(action())
    assert env.agent.stamina > stamina
    result = env.step(action(buttons=(1, 0, 0, 0, 0)))
    assert env.agent.pos[1] > start[1]
    assert sum(e["type"] == "jump" for e in result[-1]["events"]) == 1


def test_climbing_and_rain_cost(env):
    flat(env)
    env.world.fill_box((13, 8, 10), (13, 20, 15), B.STONE)
    env.agent.pos[0] = 12.65
    cmd = action(move=(1, 0), buttons=(0, 0, 1, 0, 0))
    env.step(cmd)
    dry_cost = 100 - env.agent.stamina
    assert env.agent.climbing and env.agent.pos[1] > 8.001
    env.cfg.sim_cfg.slip_chance_in_rain = 0
    env.elements.climate.weather = "rain"
    before = env.agent.stamina
    env.step(cmd)
    assert before - env.agent.stamina > dry_cost
    env.agent.stamina = 0
    env.step(cmd)
    assert not env.agent.climbing


def test_gliding_slows_descent_and_costs_stamina(env):
    flat(env)
    env.agent.pos[1] = 20
    env.agent.vel[1] = -10
    env.step(action(buttons=(0, 0, 0, 1, 0)))
    assert env.agent.gliding
    assert env.agent.vel[1] >= env.cfg.sim_cfg.glide_sink - 0.01
    assert env.agent.stamina < 100


def test_idle_swimming_exhausts_and_drowns(env):
    flat(env)
    env.world.fill_box((8, 8, 8), (18, 13, 18), B.WATER)
    env.agent.pos[1] = 10
    env.agent.stamina = 0.1
    env.step(action())
    assert env.agent.stamina == 0
    assert env.agent.submerged
    assert env.agent.health < env.agent.max_health


def test_temperature_and_cooked_protection(env):
    flat(env)
    # Armor's physical defense must not cancel all small environmental ticks.
    env.agent.inventory.add("heat_cloak")
    env.agent.inventory.equip(env.agent.inventory.find("heat_cloak"))
    env.agent.body_temp = -1.5
    env.step(action())
    assert env.agent.health < env.agent.max_health
    env.agent.add_effect(I.EFF_COLD_RESIST, 3, 5)
    before = env.agent.health
    env.step(action())
    assert env.agent.health == before
    env.agent.effects[0].ttl = 0.05
    env.agent.body_temp = -1.5
    env.step(action())
    assert env.agent.health < before


def test_death_and_time_limit_are_distinct(env):
    flat(env)
    env.cfg.max_steps = 1
    _, _, term, trunc, _ = env.step(action())
    assert trunc and not term
    with pytest.raises(gym.error.ResetNeeded):
        env.step(action())
    env.reset(seed=4)
    flat(env)
    env.agent.health = 0.01
    env.agent.body_temp = -1.5
    _, _, term, trunc, info = env.step(action())
    assert term and not trunc
    assert any(e["type"] == "death" for e in info["events"])


def test_discovery_reward_is_once_and_map_hides_unvisited(env):
    flat(env)
    env.map_memory[:] = 0
    env.discovered.clear()
    env.landmarks.append(Landmark("ruin", "Test ruin", tuple(env.agent.pos), radius=2))
    obs, reward, _, _, _ = env.step(action())
    assert reward == 1
    assert np.count_nonzero(obs["minimap"]) == 1
    assert env.step(action())[1] == 0
    env.cfg.reward_mode = "dense"
    env.cfg.exploration_bonus = 0.2
    env.agent.pos[0] += 5
    assert env.step(action())[1] == pytest.approx(0.2)


def test_melee_durability_and_drops(env):
    flat(env)
    npc = env.npcs.spawn("frog", env.agent.pos + (1, 0, 0))
    sword = env.agent.inventory.equipped_stack("weapon")
    sword.durability = 1
    result = env.step(action("attack"))
    assert not npc.alive
    assert not env.agent.inventory.has("sword_wood")
    assert any(e["type"] == "npc_died" for e in result[-1]["events"])


def test_fire_warmth_and_stowing_conductive_gear(env):
    flat(env)
    env.world.set_block(11, 8, 12, B.CAMPFIRE)
    result = env.step(action("interact"))
    assert any(e["type"] == "light_campfire" for e in result[-1]["events"])
    assert env.elements.heat_at(env.world, env.agent.pos) > 0
    env.agent.inventory.add("sword_iron")
    env.agent.inventory.equip(env.agent.inventory.find("sword_iron"))
    env.step(action("stow"))
    assert not env.agent.inventory.has_metal_equipped()


@pytest.mark.parametrize("seed", [0, 1, 17])
def test_starter_camp_is_safe_and_pot_reachable(seed):
    e = owb.WildlandsEnv(config())
    e.reset(seed=seed)
    center = e.agent.pos + (0, e.cfg.sim_cfg.agent_height / 2, 0)
    assert not collide.box_collides(e.world, center, e.controller.half)
    assert e.gameplay.nearby_blocks({B.COOKPOT})


def test_invalid_actions_do_not_advance(env):
    bad = action()
    bad["move"][0] = np.nan
    with pytest.raises(ValueError):
        env.step(bad)
    assert env.steps == 0
    before = env.cfg.world_cfg.seed
    env.reset(seed=9)
    assert env.cfg.world_cfg.seed == before


def test_fall_damage_on_landing(env):
    flat(env)
    env.agent.pos[1] = 22
    events = []
    for _ in range(20):
        events.extend(env.step(action())[-1]["events"])
        if env.agent.on_ground:
            break
    assert env.agent.on_ground
    assert env.agent.health < env.agent.max_health
    assert any(e["type"] == "fall_damage" for e in events)


def test_weather_replays_across_independent_environments():
    cfg = config()
    cfg.sim_cfg.weather_enabled = True
    cfg.sim_cfg.wind_enabled = True
    a, b = owb.WildlandsEnv(cfg), owb.WildlandsEnv(cfg)
    for e in (a, b):
        e.reset(seed=42)
        e.elements.climate.weather = "cloud"
        e.elements.climate.weather_ttl = 0
        # These used to get process-global IDs, which broke seeded replay.
        e.npcs.spawn("frog", e.agent.pos + (8, 0, 0))
        e.bodies.add("boulder", e.agent.pos + (10, 3, 0))
    events = []
    for _ in range(20):
        ra, rb = a.step(action()), b.step(action())
        events.extend(ra[-1]["events"])
        assert ra[1:] == rb[1:]
        for key in ra[0]:
            np.testing.assert_equal(ra[0][key], rb[0][key])
        assert a.npcs.snapshot() == b.npcs.snapshot()
        assert a.bodies.snapshot() == b.bodies.snapshot()
    assert any(e["type"] == "weather" for e in events)


def test_info_cannot_mutate_cooked_effects(env):
    env.step(action("interact"))
    env.step(action("cook", ingredients=(3,)))
    _, _, _, _, info = env.step(action())
    info["inventory"]["stacks"][3]["data"]["effects"][I.EFF_COLD_RESIST] = 999
    assert env.agent.inventory.stacks[3].effects()[I.EFF_COLD_RESIST] != 999
