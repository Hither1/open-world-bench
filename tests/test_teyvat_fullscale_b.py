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
