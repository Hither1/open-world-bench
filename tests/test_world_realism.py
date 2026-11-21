"""Organic terrain, contained water, creatures, riding and orchards (Wildlands)."""
import numpy as np
import pytest

from owb.core import blocks as B
from owb.env.wildlands_reference import WildlandsReferenceEnv, STABLE, FLOOD, RIVER
from owb.sim import riding as R
from owb.world.organic import uncontained_water, can_reach, walk_heights, Road, rounded_corners, simplify


@pytest.fixture(scope="module")
def env():
    e = WildlandsReferenceEnv("b", 7)
    e.reset()
    return e


def act(env, command="none", move=(0, 0), buttons=(0, 0, 0, 0, 0), slot=0, target=0):
    return env.step(dict(move=np.array(move, np.float32), look=np.zeros(2, np.float32),
                         buttons=np.array(buttons, np.int8), command=env.commands.index(command), slot=slot,
                         ingredients=np.zeros(5, np.int64), target=target, vector=np.zeros(3, np.float32)))


def test_all_water_is_held_by_ground(env):
    assert not uncontained_water(env.world.vox).any()
    # The flooded road lies flush with the road surface, never above it.
    (x0, y, z0), (x1, _, z1) = FLOOD
    assert (env.world.vox[x0:x1 + 1, y, z0:z1 + 1] == B.WATER).all()
    assert not (env.world.vox[x0:x1 + 1, y + 1:, z0:z1 + 1] == B.WATER).any()


def test_river_holds_water_and_only_descends(env):
    channel = RIVER.mask(env.world.vox.shape[::2])
    wet = (env.world.vox == B.WATER).any(axis=1)
    assert (channel & wet).sum() > 0.6 * channel.sum()
    tops = [t for t in env.world.meta["river"]["water_top"] if t >= 0]
    # Pools step down over sills; natural sea water at the mouth may vary by a block.
    assert all(b <= a + 1 for a, b in zip(tops, tops[1:])) and tops[0] > tops[-1]


def test_course_has_no_axis_aligned_road_grid(env):
    roads = env.world.meta["course_roads"]
    bent = 0
    for road in roads:
        p = np.asarray(road["points"])
        chord = p[-1] - p[0]
        normal = np.array((-chord[1], chord[0])) / max(np.linalg.norm(chord), 1e-9)
        bent += np.abs((p - p[0]) @ normal).max() > 1.0
    assert bent >= len(roads) // 2


def test_monsters_cannot_walk_onto_the_course(env):
    from owb.env.wildlands_reference import course_ground, B_TASK
    heights = walk_heights(env.world)
    course, _, _ = course_ground(env.world.vox.shape[::2], {p for _, _, p in B_TASK}, env._seed)
    reach = can_reach(heights, course, climb=0)
    dens = [n for n in env.npcs.npcs if n.data.get("den")]
    assert dens
    for n in dens:
        x, z = int(n.data["den"][0]), int(n.data["den"][2])
        assert not reach[x, z]


def test_monsters_rest_by_day_and_come_out_at_night():
    e = WildlandsReferenceEnv("a", 7)
    e.reset()
    monsters = [n for n in e.npcs.npcs if n.data.get("den")]
    e.elements.climate.time = 0.35
    for _ in range(150):
        act(e)
    day = sum(n.present for n in monsters)
    e.elements.climate.time = 0.85
    for _ in range(3):
        act(e)
    night = sum(n.present for n in monsters)
    assert night == len(monsters) > day
    assert not any(n.present for n in monsters if n.kind == "stalker") or e.elements.climate.is_night


def test_riding_is_a_skill_with_handling_and_proficiency():
    e = WildlandsReferenceEnv("a", 7)
    e.reset()
    a = e.agent
    a.pos[:] = (STABLE[0] + 1.6, STABLE[1], STABLE[2]); a.yaw = np.pi
    info = act(e, "ride")[4]
    assert any(ev["type"] == "horse_spooked" for ev in info["events"])
    horse = e.npcs.by_id(1)
    a.pos[:] = horse.pos + (1.6, 0, 0); a.yaw = np.pi
    slot = next(i for i, s in enumerate(a.inventory.stacks) if s.spec.id == "apple")
    assert act(e, "feed", slot=slot)[4]["events"][0]["type"] == "horse_calmed"
    assert act(e, "ride")[4]["events"][0]["type"] == "mounted"
    assert act(e, "inspect", target=1)[4]["events"][0]["reason"] == "dismount first"
    assert R.gait(0)[0] == "walk" and R.gait(100)[0] == "trot" and R.gait(300)[0] == "canter"
    e.riding.distance = 300
    e.riding.before_tick(type("I", (), dict(sprint=True, climb=False, glide=False, crouch=False))())
    assert a.flags["ride_mult"] == R.GALLOP
    assert act(e, "dismount")[4]["events"][0]["type"] == "dismounted"
    assert "ride_mult" not in a.flags


def test_striking_an_apple_tree_shakes_apples_loose(env):
    tree = max(env.orchard.trees, key=lambda t: t["apples"])
    before = tree["apples"]
    a = env.agent
    x, y, z = tree["trunk"]
    a.pos[:] = (x - 1.5, y, z); a.yaw = 0.0; a.cooldown = 0
    events = act(env, "attack")[4]["events"]
    fell = [ev for ev in events if ev["type"] == "apple_fell"]
    assert fell and tree["apples"] == before - len(fell)
    assert any(p.get("source") == "apple_tree" for p in env.gameplay.plants)


def test_presentation_creatures_do_not_change_simulation(env):
    from owb.render.scene_export import capture_scene
    env.reference_view_id = env.milestones[0].id
    before = env.state_digest()
    scene = capture_scene(env)
    assert env.state_digest() == before
    assert scene.metadata["water_volume"] is True


def test_road_geometry_helpers():
    pts, s = rounded_corners([(0, 0), (20, 0), (20, 20)], 5)
    assert np.allclose(pts[0], (0, 0)) and np.allclose(pts[-1], (20, 20))
    assert np.min(np.linalg.norm(pts - (20, 0), axis=1)) > 1.0       # the corner is rounded off
    stair = [(i // 2, (i + 1) // 2) for i in range(40)]
    assert len(simplify(stair, 1.2)) == 2
    road = Road(np.stack((np.linspace(5, 25, 41), np.full(41, 5.0)), 1), 2.0)
    assert (road.mask((32, 32)) == road.stamp((32, 32))).mean() > .99


def test_bonded_horse_answers_a_whistle():
    e = WildlandsReferenceEnv("a", 7)
    e.reset()
    a = e.agent
    assert act(e, "whistle")[4]["events"][0]["reason"] == "no bonded horse within earshot"
    horse = e.npcs.by_id(1)
    horse.data["bonded"] = True
    a.pos[:] = horse.pos + (12.0, 0, 0)
    events = act(e, "whistle")[4]["events"]
    assert events[0]["type"] == "whistled" and events[0]["answered"]
    arrived = []
    for _ in range(60):
        arrived += [ev for ev in act(e)[4]["events"] if ev["type"] in ("horse_arrived", "horse_stuck")]
        if arrived:
            break
    assert arrived and arrived[0]["type"] == "horse_arrived"
    assert np.hypot(*(horse.pos - a.pos)[[0, 2]]) < 2.0 and "route" not in horse.data
