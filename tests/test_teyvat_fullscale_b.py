"""Full-scale Teyvat Task B: mechanism rules, locks, observation privacy, and a replayed reference run."""
import json

import numpy as np
import pytest

from open_world_bench.character import MoveState
from open_world_bench.elements import Element
from open_world_bench.fullscale_b_env import GOAL, TeyvatExpeditionEnv
from open_world_bench.fullscale_b_world import RESERVE_POWER, SKYLIFT_POWER
from open_world_bench.fullscale_mechanics import (EXPOSED_TIME, WINDUP_TIME, RelayUnit, SentinelUnit, VentState)
from owb.scripts import teyvat_fullscale_b as script

E, T, G = Element.EMBER, Element.TIDE, Element.GALE
RULES = dict(source=E, vane=T, drain=G)


@pytest.fixture(scope="module")
def env():
    e = TeyvatExpeditionEnv(7)
    e.reset(seed=7)
    return e


def _stand_at(env, xz):
    """Test setup only: place the character on the ground."""
    env.char.pos[:] = (xz[0], env.fullscale.terrain.height_at(*xz), xz[1])
    env.char.vel[:] = 0.0
    env.char.state = MoveState.GROUNDED


def test_course_and_observation_contract(env):
    c = env.course
    assert [m.id for m in c.milestones] == [f"B{k:02d}" for k in range(1, 25)]
    assert sorted(e.name for e in c.relay_elements.values()) == ["EMBER", "GALE", "TIDE"]
    assert c.guard_elements[0] != c.guard_elements[1]
    obs = env._observe()
    assert obs["instruction"] == GOAL and not np.any(obs["task"])
    text = obs["local_text"]
    assert "Step" not in text and "B01" not in text
    for value in env.char.pos[[0, 2]]:
        assert f"{value:.0f}" not in text.split("Nearby")[0]


def test_relay_rules_single_and_double_ring():
    r = RelayUnit("r", "Relay", "field", 1, {}, overloaded=True, interrupt="prime")
    assert not r.apply("source", E, RULES)[0]               # overloaded: nothing takes
    assert r.apply("drain", G, RULES)[0] and not r.overloaded
    ok, text, payload = r.apply("source", E, RULES)
    assert ok and payload.get("interrupted") and r.overloaded and r.progress == 1   # surge keeps priming
    assert r.apply("drain", G, RULES)[0] and r.progress == 1
    assert not r.apply("vane", E, RULES)[0]                 # wrong element
    assert r.apply("vane", T, RULES)[0] and r.confirm()[0] and r.active
    clean = RelayUnit("c", "Old", "field", 1, {})
    assert clean.apply("drain", G, RULES)[0] and clean.unnecessary_drains == 1
    double = RelayUnit("d", "Double", "shrine", 2, {})
    ok, text, payload = double.apply("source", E, RULES)
    assert not ok and payload["reason"] == "residual" and "residual" in text
    assert not double.confirm()[0]
    assert double.apply("drain", G, RULES)[0] and double.unnecessary_drains == 0
    assert double.apply("source", E, RULES)[0] and double.apply("vane", T, RULES)[0] and double.confirm()[0]


def test_sentinel_phases_and_wrong_elements():
    s = SentinelUnit("s", "e", "Sentinel", 100.0)
    assert s.element_hit(T, (E, G)) == [("sentinel_pulse", dict(reason="wrong_element"))]
    assert s.element_hit(E, (E, G))[0][0] == "sentinel_exposed"
    assert s.tick(EXPOSED_TIME + 0.01, True, 3.0)[0][0] == "sentinel_windup"
    assert s.element_hit(E, (E, G))[0][0] == "sentinel_slam"            # wrong interrupter
    s.set_phase("windup")
    assert s.element_hit(G, (E, G))[0][0] == "sentinel_interrupted" and s.phase == "vulnerable"
    late = SentinelUnit("l", "e", "Late", 100.0, phase="windup", hold=True)
    assert late.tick(WINDUP_TIME + 1, False, 30.0) == [] and late.phase == "windup"     # held until near
    assert late.tick(WINDUP_TIME + 0.01, True, 5.0)[0] == ("sentinel_slam", dict(reason="window_missed"))
    reform = SentinelUnit("r", "e", "Reform", 100.0, phase="windup", reform_once=True)
    assert reform.element_hit(G, (E, G))[0][0] == "sentinel_reformed" and reform.phase == "shield"


def test_vent_beat_and_column():
    v = VentState("v", "r", 1, (0.0, 0.0), 10.0, 50.0, None, offset=0.0)
    assert v.beat(0.5) == "bright" and v.beat(3.0) == "dark" and v.beat(4.2) == "bright"
    assert v.in_column(np.array([1.0, 10.5, 1.0])) and not v.in_column(np.array([3.0, 10.0, 0.0]))


def test_locks_and_unpowered_vents(env):
    c = env.course
    chest = env.world.get("reserve")
    _stand_at(env, chest.pos[[0, 2]])
    c.interact(env.char)
    assert not chest.used and "reserve" not in c.pack and RESERVE_POWER not in c.facts
    assert "seal" in c.feedback[0]
    sky = next(v for v in env.fullscale.vents if v.requires == SKYLIFT_POWER and v.leg == 1)
    _stand_at(env, sky.pad)
    c.apply_element(env.char, c.vent_element, 5.0)
    assert not c.vents[sky.id].charged and "no power" in c.feedback[0]
    outpost = c.sentinels["b24_sentinel"]
    assert not c.live_sentinel("b24_sentinel") and env.world.get(outpost.entity).alive


def test_reference_completes_and_replays(tmp_path):
    run = tmp_path / "seed-7"
    manifest = script.collect(7, run)
    assert manifest["success"] and manifest["deaths"] == 0 and not manifest["falls"]
    times = [m["completed"] for m in manifest["milestones"]]
    assert None not in times and times == sorted(times)
    result = script.verify(run)
    assert result["ok"] and result["actions"] == manifest["actions"]
    rows = script.read_rows(run)
    facts = [e["fact"] for r in rows for e in r["events"] if e["type"] == "fact_opened"]
    assert facts == ["causeway_open", "reserve_power", "skylift_power", "crater_gate_open", "crown_signal",
                     "summit_power", "outpost_clear"]
    kinds = {e["type"] for r in rows for e in r["events"]}
    assert {"vent_launch", "sentinel_interrupted", "relay_interrupted", "housing_unlocked",
            "designed_context_boundary"} <= kinds
    assert all(v["launches"] >= 1 for v in manifest["vents"].values())
    assert manifest["relays"]["relay_b19"]["version"] == 2
    plan = script.frame_plan(rows)
    assert plan[0]["step"] == 0 and plan[-1]["step"] == rows[-1]["step"]
    json.dumps(manifest)


def test_world_gates_and_vent_proofs():
    from open_world_bench.fullscale_b_check import check_world_parallel
    report = check_world_parallel(7, workers=9)
    assert [s["name"] for s in report["stages"]] == ["start", "causeway_open", "skylift_power", "crater_gate_open",
                                                     "summit_power", "outpost_clear"]
    for stage in report["stages"]:
        assert all(v == 0 for v in stage["sealed"].values()), stage["name"]
        assert all(stage["sites"].values()), stage["name"]
    assert all(c["ok"] for c in report["no_vents"])
    assert all(f["on_shelf"] and f["fall_damage"] == 0.0 for f in report["flights"])
    assert report["ok"]


# ------------------------------------------------------------- review fixes
NEUTRAL = dict(move=np.zeros(2, np.float32), look=np.zeros(2, np.float32), buttons=np.zeros(7, np.int8), swap=0)
JUMP, SKILL = 0, 5


def _action(*buttons, swap=0):
    a = {k: v.copy() if hasattr(v, "copy") else v for k, v in NEUTRAL.items()}
    for b in buttons:
        a["buttons"][b] = 1
    a["swap"] = swap
    return a


def test_milestones_completed_in_one_step_still_finish():
    env = TeyvatExpeditionEnv(7)
    env.reset(seed=7)
    c = env.course
    c.finish_op()                                   # test setup: the keeper lesson is heard
    c.relays["relay_b01"].active = c.relays["relay_b02"].active = True
    obs, reward, terminated, truncated, info = env.step(_action())
    done = [e["milestone"] for e in info["event_details"] if e["type"] == "milestone_complete"]
    assert done == ["B01", "B02"]
    assert env.quests.progress["fullscale_b"].stage == c.index == 2
    # The engine follows any number of milestones in one step, through the finish.
    from open_world_bench.events import Event
    engine = type(env.quests)(env.graph)
    engine.consume([Event("activate", {"entity": m.id}) for m in c.milestones], {})
    assert engine.all_complete and engine.progress["fullscale_b"].stage == 24


def test_sentinel_and_cast_damage_down_a_member_without_ending_the_episode():
    env = TeyvatExpeditionEnv(7)
    env.reset(seed=7)
    c = env.course
    tick, hits = c.tick, []

    def tick_with_pulse(char, dt):                   # test setup: a pulse lands after the base defeat check
        tick(char, dt)
        if not hits:
            hits.append(1)
            c._hurt(char, 1e4, "test_pulse")
    c.tick = tick_with_pulse
    first = env.party.active.spec.id
    obs, reward, terminated, truncated, info = env.step(_action())
    assert not terminated and env.char.hp > 0.0
    assert {"member_downed", "party_swap"} <= set(info["events"]) and env.party.active.spec.id != first
    c.tick = tick
    apply = c.apply_element

    def apply_with_pulse(char, element, reach):      # test setup: a wrong-element cast is punished
        apply(char, element, reach)
        c._hurt(char, 1e4, "test_pulse")
    c.apply_element = apply_with_pulse
    for _ in range(40):                              # the swapped-in member's skill may be cooling down
        obs, reward, terminated, truncated, info = env.step(_action(SKILL))
        if "member_downed" in info["events"]:
            break
    assert "member_downed" in info["events"] and not terminated and env.char.hp > 0.0


def test_reaction_splash_never_damages_a_sentinel(env):
    c = env.course
    world, combat = env.world, env.world.combat
    sentinel = world.get(c.sentinels["b13_sentinel"].entity)
    hp = float(sentinel.hp)
    raiders = [e for e in world.entities.values() if e.alive and e.id.startswith("raider")][:2]
    assert len(raiders) == 2
    saved = [(r.pos.copy(), r.hp, r.home.copy()) for r in raiders]
    splashed = False
    for first, second in ((E, T), (T, E), (G, E), (E, G), (G, T), (T, G)):
        for r in raiders:
            r.pos = sentinel.pos + np.array([1.0, 0.0, 0.0], np.float32)
            r.hp = r.max_hp
            r.elemental.aura.element, r.elemental.aura.charge = Element.NONE, 0.0
        combat._hit(raiders[0], env.char, 1.0, first)
        events = combat._hit(raiders[0], env.char, 1.0, second)
        splashed |= any(e.type == "enemy_hit" and e.get("entity") == raiders[1].id for e in events)
    assert splashed                                  # the splash path ran
    assert sentinel.alive and float(sentinel.hp) == hp
    assert combat._damage(sentinel, 50.0, E) == [] and float(sentinel.hp) == hp
    for r, (pos, rhp, home) in zip(raiders, saved):
        r.pos, r.hp, r.home = pos, rhp, home


def test_vent_launches_only_on_a_jump_in_the_safe_beat():
    env = TeyvatExpeditionEnv(7)
    env.reset(seed=7)
    c = env.course
    vent = next(v for v in c.vents.values() if v.requires is None)
    dt = env.cfg.dt * env.cfg.action_repeat

    def ready(safe):
        _stand_at(env, vent.pad)
        env.char.stamina = 100.0
        c._was_grounded, c.launch = True, None
        t = env.world.elapsed
        while not all(vent.safe(t + k * 0.05, c.safe_beat) == safe for k in range(int(dt / 0.05) + 3)):
            t += 0.05
        env.world.elapsed = t                         # test setup: wait for the beat
        c.apply_element(env.char, c.vent_element, 5.0)
        assert vent.charged

    def kinds(info):
        return {e["type"] for e in info["event_details"]}

    gale = next(i for i, m in enumerate(env.party.members) if m.spec.element == Element.GALE)
    env.step(_action(swap=gale + 1))
    ready(safe=True)
    obs, reward, terminated, truncated, info = env.step(_action(SKILL))      # Updraft is not a launch
    assert "vent_launch" not in kinds(info) and vent.charged
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(_action())
    ready(safe=False)
    obs, reward, terminated, truncated, info = env.step(_action(JUMP))
    assert "vent_sputter" in kinds(info) and not vent.charged and c.launch is None
    for _ in range(10):
        env.step(_action())
    ready(safe=True)
    obs, reward, terminated, truncated, info = env.step(_action(JUMP))
    assert "vent_launch" in kinds(info) and not vent.charged


def test_info_carries_no_step_pointer(env):
    info = env._info([])
    assert info["instruction"] == GOAL
    titles = [m.title for m in env.course.milestones]
    assert not any(t in env.course.local_text(env.char) for t in titles)


def test_strict_carry_is_sound_against_the_real_controller():
    """Climbs from proven-reachable ground never stand on unreached ground or rise above a face ceiling."""
    from open_world_bench.character import CharacterConfig, Command
    from open_world_bench.fullscale_b_world import build_world
    from open_world_bench.fullscale_physics import FullScaleController
    from open_world_bench.fullscale_reach import _shift, over_model, patch_centres, reach, strict_face_slack

    world = build_world(7)
    cell = world.terrain._cell
    model = over_model(cell)
    assert strict_face_slack(model, cell) > 3.0
    proof = reach(world, (), model, strict_carry=True)
    g = proof.graph
    px, pz = patch_centres(world)
    x0, z0 = float(px[0, 0]), float(pz[0, 0])
    along_x_first = abs(float(px[1, 0] - px[0, 0])) > abs(float(px[0, 1] - px[0, 0]))

    def patch(pos):
        a = int(np.clip(np.floor((pos[0] - x0) / cell + 0.5), 0, g.rest.shape[0] - 1))
        b = int(np.clip(np.floor((pos[2] - z0) / cell + 0.5), 0, g.rest.shape[0] - 1))
        return (a, b) if along_x_first else (b, a)

    stand = proof.rest & ~g.water
    pairs = []
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ii, jj = np.nonzero(stand & _shift(g.steep, di, dj, False))
        pairs += [((float(px[i, j]), float(pz[i, j])), (float(px[i + di, j + dj]), float(pz[i + di, j + dj])))
                  for i, j in zip(ii, jj)]
    rng = np.random.default_rng(5)
    picks = [pairs[k] for k in rng.choice(len(pairs), size=36, replace=False)]
    ctl = FullScaleController(world.terrain, world.props, CharacterConfig())
    ctl.architecture = world.architecture
    dt = 1 / 30
    violations, gains = [], []
    for n, ((sx, sz), (fx, fz)) in enumerate(picks):
        into = np.array([fx - sx, fz - sz]) / np.hypot(fx - sx, fz - sz)
        angle, stick, seconds = ((0.0, 1.0, 13.0), (1.15, 0.09, 30.0), (-0.7, 1.0, 13.0))[n % 3]
        turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ into
        st = ctl.spawn(sx, sz)
        base = float(st.pos[1])
        for _ in range(12):
            ctl.step(st, Command(move=np.array([0.0, 1.0], np.float32), sprint=True),
                     float(np.arctan2(into[0], -into[1])), dt)
        if ctl.can_launch(st):                        # Updraft: launch speed plus its stamina refund
            st.vel[1], st.grounded, st.state = 13.0, False, MoveState.AIRBORNE
            st.stamina = min(100.0, st.stamina + 25.0)
        yaw = float(np.arctan2(turn[0], -turn[1]))
        for k in range(int(seconds / dt)):
            regrab = n % 3 == 2 and k > 120 and (k % 20) < 6
            ctl.step(st, Command(move=np.array([0.0, stick], np.float32), climb=not regrab), yaw, dt)
            i, j = patch(st.pos)
            y = float(st.pos[1])
            if st.state == MoveState.GROUNDED and ctl.walkable(st.pos[0], st.pos[2]) and not proof.rest[i, j]:
                violations.append(("stand", n, st.pos.tolist()))
            if st.state == MoveState.CLIMBING and g.steep[i, j] and y > proof.ceiling[i, j] + 0.02:
                violations.append(("ceiling", n, st.pos.tolist(), float(proof.ceiling[i, j])))
            gains.append(y - base)
    assert not violations, violations[:5]
    assert max(gains) > 15.0                          # the climbs really climbed


def test_sites_have_compounds_and_groves(env):
    """Every site has its landmark pieces, they collide, and the groves leave sites and trails open."""
    from open_world_bench.fullscale_b_world import _TRAILS, place_structures
    from open_world_bench.fullscale_sites import PIECE_RADII, _segment_distance
    world = env.fullscale
    buildings, pieces, skipped = place_structures(world.terrain, world.layout)
    assert len(pieces) > 180 and len(buildings) > 18 and len(skipped) < 10
    kinds = {(p.site, p.kind) for p in pieces}
    assert {("depot", "granary"), ("relay_b11", "signal_mast"), ("relay_b19", "shrine_hall"),
            ("crown_landing", "beacon_tower"), ("camp", "palisade"), ("arena_b03", "standing_stone")} <= kinds
    assert world.digest() == env.world_digest
    granary = next(p for p in pieces if p.kind == "granary")
    inside = np.array([granary.pos[0], granary.pos[1] + 1.0, granary.pos[2]])
    assert world.architecture.blocked(inside[None])[0]
    props = world.props
    solid = props.radius > 0
    assert solid.sum() > 3000 and len(set(props.kind.tolist())) >= 7
    anchors = np.array(list(world.layout.anchors.values()))
    near = np.min(np.hypot(props.pos[solid][:, None, 0] - anchors[None, :, 0],
                           props.pos[solid][:, None, 2] - anchors[None, :, 1]), axis=1)
    assert near.min() > 15.0
    for a, b in _TRAILS:
        d = _segment_distance(props.pos[solid][:, 0], props.pos[solid][:, 2], world.layout.anchors[a], world.layout.anchors[b])
        assert d.min() > 5.0, (a, b)
    for p in pieces:                                   # no trunk inside a structure
        d = np.hypot(props.pos[solid][:, 0] - p.pos[0], props.pos[solid][:, 2] - p.pos[2])
        assert d.min() > PIECE_RADII[p.kind] + 2.0, p.id


def test_observations_show_the_site_pieces():
    from open_world_bench.render.meshes import MeshBatch
    e = TeyvatExpeditionEnv(7, render_enabled=True, width=96, height=54)
    e.reset(seed=7)
    eye = e.char.pos + np.array([0.0, 2.0, 0.0])
    with_pieces = e._renderer._build_meshes(e.world, e.char, eye, np.array([0.0, 0.0, -1.0]), batch=MeshBatch())
    ids = np.concatenate(with_pieces.ids)
    assert (ids >= 24000).any()                         # site pieces carry ids from 24000
    obs, *_ = e.step(_action())
    assert obs["rgb"].shape == (54, 96, 3) and obs["rgb"].std() > 1.0
