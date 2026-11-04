"""Gameplay regressions for party combat and the Gymnasium contract."""
from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env, data_equivalence

from open_world_bench import (
    CharacterConfig, CharacterController, Element, Entity, EntityKind, EnvConfig,
    OpenWorldEnv, Party, PartyConfig, RenderSettings, Terrain, TerrainConfig, World, WorldConfig,
)
from open_world_bench.character import MoveState
from open_world_bench.elements import Aura
from open_world_bench.party import MAX_ENERGY
from open_world_bench.quests import QuestEngine
from open_world_bench.quests.schema import ObjKind, Objective, Quest, QuestGraph, Stage
from open_world_bench.scatter import BOULDER, PropField


@pytest.fixture
def arena():
    def make(members=("kestrel", "maren", "wren"), **world_options):
        terrain = Terrain(TerrainConfig(size=64.0, resolution=33))
        terrain.height[:] = 10.0
        terrain.normals[:] = (0.0, 1.0, 0.0)
        party = Party(PartyConfig(members=members, swap_cooldown=0.0))
        world = World(WorldConfig(prop_density=0.0, **world_options), terrain=terrain, party=party)
        controller = CharacterController(terrain, world.props, CharacterConfig(max_hp=party.max_hp))
        return world, controller, controller.spawn(0.0, 0.0)
    return make


def enemy(world, id="target", pos=(0.0, 10.0, -2.0), **kwargs):
    return world.spawn(Entity(id, EntityKind.ENEMY, np.array(pos, np.float32),
                              hp=kwargs.pop("hp", 1000.0), **kwargs))


def action(*, skill=False, burst=False, attack=False, swap=0):
    return {"move": np.zeros(2, np.float32), "look": np.zeros(2, np.float32),
            "buttons": np.array([0, 0, 0, attack, 0, skill, burst], np.int8), "swap": swap}


@pytest.fixture
def env():
    cfg = EnvConfig(
        world=WorldConfig(terrain=TerrainConfig(size=512.0, resolution=65), prop_density=0.0),
        render=RenderSettings(width=32, height=24, march_steps=16, shadow_steps=0, draw_props=False),
        backend="soft", text_obs=True,
    )
    instance = OpenWorldEnv(cfg)
    instance.reset(seed=7)
    yield instance
    instance.close()


def flatten_env(env, arena, **kwargs):
    world, controller, char = arena(**kwargs)
    env.world, env.controller, env.char, env.party = world, controller, char, world.party
    env.camera_rig.terrain, env.camera_rig.props = world.terrain, world.props
    env.cam = env.camera_rig.reset(char)


def test_swap_preserves_health_position_and_bench_cooldowns(arena):
    world, controller, char = arena()
    party = world.party
    party.cfg.swap_cooldown = 1.0
    char.hp = 43.0
    world.combat.cast(char, controller)
    position = char.pos.copy()
    assert world.swap_character(char, 1)[0].type == "party_swap"
    assert char.hp == 125.0 and party.members[0].hp == 43.0
    np.testing.assert_array_equal(position, char.pos)
    assert world.swap_character(char, 0) == []
    world.update(char, 1.1, controller.cfg)
    assert party.members[0].skill_cd == pytest.approx(4.9)
    assert party.members[0].energy > 0.0
    world.swap_character(char, 0)
    assert char.hp == 43.0


@pytest.mark.parametrize("first,second,reaction", [
    ("maren", "kestrel", "steamburst"), ("nym", "kestrel", "thaw"),
    ("maren", "nym", "glaciate"), ("maren", "vex", "conduit"),
    ("nym", "vex", "shattercharge"), ("kestrel", "vex", "overpressure"),
    ("kestrel", "wren", "squall"), ("maren", "boro", "encase"),
])
def test_all_reactions_are_reachable_with_party_skills(arena, first, second, reaction):
    world, controller, char = arena((first, second))
    target = enemy(world)
    world.combat.cast(char, controller)
    hp = target.hp
    assert hp < 1000.0
    world.swap_character(char, 1)
    events = world.combat.cast(char, controller)
    assert [e.get("reaction") for e in events if e.type == "reaction"] == [reaction]
    assert target.hp < hp and target.elemental.aura.element == Element.NONE
    assert all(member.energy > 0.0 for member in world.party.members)


def test_freeze_stops_attacks_and_expires(arena):
    world, controller, char = arena(("maren", "nym"))
    target = enemy(world, pos=(0.0, 10.0, -1.0))
    world.combat.cast(char, controller)
    world.swap_character(char, 1)
    world.combat.cast(char, controller)
    hp, position = char.hp, target.pos.copy()
    world.update(char, 1.0, controller.cfg)
    assert char.hp == hp
    np.testing.assert_array_equal(target.pos, position)
    world.update(char, 2.1, controller.cfg)
    assert char.hp < hp


def test_conduit_only_hits_wet_neighbours(arena):
    world, controller, char = arena(("maren", "vex"))
    enemy(world)
    wet = enemy(world, "wet", pos=(6.0, 10.0, -2.0))
    dry = enemy(world, "dry", pos=(-6.0, 10.0, -2.0))
    world.combat.cast(char, controller)
    dry.elemental.aura = Aura()
    wet_hp, dry_hp = wet.hp, dry.hp
    world.swap_character(char, 1)
    world.combat.cast(char, controller)
    assert wet.hp < wet_hp and dry.hp == dry_hp


def test_squall_spreads_aura_without_recursive_reactions(arena):
    world, controller, char = arena(("kestrel", "wren"))
    enemy(world)
    neighbour = enemy(world, "neighbour", pos=(6.0, 10.0, -2.0))
    neighbour.elemental.aura.apply(Element.TIDE)
    world.combat.cast(char, controller)
    world.swap_character(char, 1)
    events = world.combat.cast(char, controller)
    assert [e.get("reaction") for e in events if e.type == "reaction"] == ["squall"]
    assert neighbour.elemental.aura.element == Element.EMBER and neighbour.hp < 1000.0


def test_bolt_targets_nearest_forward_enemy_and_respects_height(arena):
    world, controller, char = arena(("nym",))
    near = enemy(world, "near", pos=(0.0, 10.0, -5.0))
    far = enemy(world, "far", pos=(0.0, 10.0, -12.0))
    behind = enemy(world, "behind", pos=(0.0, 10.0, 3.0))
    above = enemy(world, "above", pos=(0.0, 35.0, -1.0))
    world.combat.cast(char, controller)
    assert near.hp == 948.0 and all(e.hp == 1000.0 for e in (far, behind, above))


def test_burst_requires_energy_heals_party_and_cannot_repeat(arena):
    world, controller, char = arena(("maren", "boro"))
    party = world.party
    char.hp = party.active.hp = 20.0
    party.members[1].hp = 30.0
    assert world.combat.cast(char, controller, burst=True) == []
    party.active.energy = MAX_ENERGY
    events = world.combat.cast(char, controller, burst=True)
    assert events[0].type == "burst"
    assert char.hp == party.active.hp == 65.0 and party.members[1].hp == 75.0
    assert party.active.energy == 0.0
    party.active.energy = MAX_ENERGY
    assert world.combat.cast(char, controller, burst=True) == []


def test_ward_absorbs_damage_then_health_takes_over(arena):
    world, controller, char = arena(("boro",))
    world.combat.cast(char, controller)
    enemy(world, pos=(0.0, 10.0, -1.0), damage=60.0)
    events = world.update(char, 0.1, controller.cfg)
    assert char.hp == world.party.active.hp == 140.0 and world.party.shield == 0.0
    hit = next(e for e in events if e.type == "player_hit")
    assert hit.get("damage") == 5.0 and hit.get("absorbed") == 55.0


def test_elemental_shield_break_and_resistance(arena):
    world, controller, char = arena(("kestrel",))
    target = enemy(world)
    target.elemental.shield = 50.0
    target.elemental.shield_element = Element.TIDE
    events = world.combat.cast(char, controller)
    assert any(e.type == "shield_broken" for e in events)
    assert target.hp == pytest.approx(974.0)
    target.elemental.resists = (Element.EMBER,)
    world.party.active.skill_cd = 0.0
    world.combat.cast(char, controller)
    assert target.hp == pytest.approx(974.0 - 46.0 * 0.4)


def test_updraft_launches_and_refunds_stamina(arena):
    world, controller, char = arena(("wren",))
    char.stamina = 20.0
    world.combat.cast(char, controller)
    assert char.state == MoveState.AIRBORNE and not char.grounded
    assert char.vel[1] == 13.0 and char.stamina == 45.0


def test_dash_sweeps_targets_and_stops_at_solid_props(arena):
    world, controller, char = arena(("vex",))
    target = enemy(world, pos=(0.0, 10.0, -2.0))
    controller.props = PropField(np.array([[0.0, 10.0, -5.0]], np.float32),
                                 np.zeros(1), np.ones(1), np.array([BOULDER], np.int8))
    world.combat.cast(char, controller)
    assert -3.4 < char.pos[2] < -1.0 and target.hp == 960.0


def test_resonance_health_is_consistent_after_healing_and_revive():
    party = Party(PartyConfig(members=("boro", "boro")))
    assert party.max_hp == 174.0
    party.active.hp = 160.0
    party.heal(50.0)
    assert party.active.hp == 174.0 and party.as_vector()[1] == 1.0
    party.active.hp, party.active.downed = 0.0, True
    party.revive()
    assert party.active.hp == 174.0 and party.active.alive


@pytest.mark.parametrize("respawn", [True, False])
def test_downed_members_auto_swap_and_party_wipe(arena, respawn):
    world, controller, char = arena(("kestrel", "maren"), respawn_on_death=respawn)
    target = enemy(world, pos=(0.0, 10.0, -1.0), damage=1000.0)
    first = world.update(char, 0.1, controller.cfg)
    assert [e.type for e in first] == ["player_hit", "member_downed", "party_swap"]
    assert world.party.active_index == 1 and char.hp == 125.0
    assert not world.party.members[0].alive and world.swap_character(char, 0) == []
    target.attack_cd = 0.0
    second = world.update(char, 0.1, controller.cfg)
    assert [e.type for e in second] == ["player_hit", "member_downed", "player_defeated"]
    assert world.party.any_alive == respawn and (char.hp > 0.0) == respawn
    if respawn:
        assert all(m.alive and m.energy == 0.0 for m in world.party.members)


def test_gym_contract_and_seeded_replay(env):
    check_env(env, skip_render_check=True)
    actions = [action(skill=True), action(swap=2, skill=True), action(burst=True)]
    env.reset(seed=13)
    first = [env.step(a) for a in actions]
    env.reset(seed=13)
    second = [env.step(a) for a in actions]
    for left, right in zip(first, second):
        assert data_equivalence(left, right, exact=True)
        assert env.observation_space.contains(left[0])


def test_agent_skill_kill_completes_quest_once(env, arena):
    flatten_env(env, arena)
    enemy(env.world, hp=10.0, aggro_radius=0.0)
    env.graph = QuestGraph([Quest("fight", "Fight", "", [Stage([
        Objective("kill", ObjKind.DEFEAT, "Defeat target", targets=("target",))
    ])])])
    env.quests = QuestEngine(env.graph)
    env.cfg.action_repeat = 4
    obs, reward, terminated, truncated, info = env.step(action(skill=True))
    assert info["events"].count("skill") == info["events"].count("defeat") == 1
    assert info["quests_done"] == 1 and terminated and not truncated and reward > 25.0
    assert env.observation_space.contains(obs) and env.render().shape == (24, 32, 3)


def test_swap_does_not_create_damage_penalty(env, arena):
    flatten_env(env, arena)
    env.cfg.reward.shaping = 0.0
    env.party.members[1].hp = 10.0
    _, reward, _, _, _ = env.step(action(swap=2))
    assert reward == -env.cfg.reward.time_penalty and env.char.hp == env.party.active.hp == 10.0


def test_party_wipe_terminates_without_finish_bonus(env, arena):
    flatten_env(env, arena, members=("kestrel",), respawn_on_death=False)
    enemy(env.world, pos=(0.0, 10.0, -1.0), damage=1000.0)
    env.cfg.reward.shaping = 0.0
    _, reward, terminated, _, info = env.step(action())
    assert terminated and reward < -5.0
    assert info["events"].count("player_defeated") == 1 and info["quests_done"] == 0


@pytest.mark.parametrize("flat,party", [(False, None), (True, None), (True, PartyConfig())])
def test_legacy_and_flat_action_spaces(env, flat, party):
    other = OpenWorldEnv(replace(env.cfg, flat_actions=flat, party=party))
    try:
        check_env(other, skip_render_check=True)
        if party:
            other.reset(seed=7)
            a = np.zeros(15, np.float32)
            a[12], a[9] = 1.0, 1.0
            _, _, _, _, info = other.step(a)
            assert info["active_character"] == "maren" and "skill" in info["events"]
            other.step(np.zeros(9, np.float32))
        else:
            obs, _ = other.reset()
            assert "party" not in obs
    finally:
        other.close()


def test_reset_clears_combat_state(env):
    env.step(action(skill=True))
    env.party.shield, env.party.active.energy, env.party.members[1].hp = 50.0, 60.0, 1.0
    obs, info = env.reset(seed=7)
    assert env.party.active.spec.id == "kestrel" and env.party.shield == 0.0
    assert env.world.combat.effects == []
    assert all(m.hp == env.party.member_max_hp(m) and m.energy == 0.0 and m.skill_cd == 0.0
               for m in env.party.members)
    assert env.observation_space.contains(obs) and info["events"] == []


@pytest.mark.parametrize("bad", [
    {**action(), "move": [np.nan, 0.0]}, {**action(), "buttons": [0, 1]},
    {**action(), "swap": 4}, {**action(), "look": [0.0, 1.0, 2.0]},
])
def test_invalid_actions_fail_before_mutating_state(env, bad):
    position = env.char.pos.copy()
    with pytest.raises(ValueError):
        env.step(bad)
    assert env.steps == 0
    np.testing.assert_array_equal(env.char.pos, position)
