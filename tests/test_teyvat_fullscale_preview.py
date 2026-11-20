"""Full-scale Teyvat Task A route preview: env lifts and checkpoints, planning, and a replayed run."""
import json

import numpy as np
import pytest

from open_world_bench.character import MoveState
from open_world_bench.fullscale_env import LIFT_REACH, ROUTE, TeyvatFullScaleEnv
from open_world_bench.fullscale_nav import NavGrid
from open_world_bench.fullscale_world import CORRIDOR_OPEN, FERRY_OPEN
from owb.scripts import teyvat_fullscale_preview as preview

NEUTRAL = {"move": np.zeros(2, np.float32), "look": np.zeros(2, np.float32), "buttons": np.zeros(7, np.int8),
           "swap": 0, "pointer": np.zeros(2, np.float32), "click": 0}


def _press(button):
    action = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in NEUTRAL.items()}
    action["buttons"][button] = 1
    return action


@pytest.fixture(scope="module")
def env():
    e = TeyvatFullScaleEnv(7)
    e.reset(seed=7)
    return e


def _stand_at(env, xyz):
    """Test setup only: place the character on the ground at a lift end."""
    env.char.pos[:] = (xyz[0], env.fullscale.terrain.height_at(xyz[0], xyz[2]), xyz[2])
    env.char.vel[:] = 0.0
    env.char.state = MoveState.GROUNDED


def test_reset_is_deterministic(env):
    other = TeyvatFullScaleEnv(7)
    other.reset(seed=7)
    assert other.state_digest() == TeyvatFullScaleEnv(7).state_digest()
    assert len(ROUTE) == 12 and [m.id for m in ROUTE] == [f"A{k:02d}" for k in range(1, 13)]


def test_lifts_need_their_fact_and_carry_the_character(env):
    env.reset(seed=7)
    lift = next(l for l in env.world.lifts if l.id == "ferry_lift")
    bottom, top = lift.point(0.0), lift.point(lift.length)
    _stand_at(env, bottom)
    _, _, _, _, info = env.step(_press(2))
    assert [e["type"] for e in info["event_details"]] == ["lift_locked"]
    env.world.open_fact(FERRY_OPEN)
    _stand_at(env, bottom)
    _, _, _, _, info = env.step(_press(2))
    assert "lift_board" in [e["type"] for e in info["event_details"]]
    heights = []
    for _ in range(400):
        _, _, _, _, info = env.step(NEUTRAL)
        heights.append(float(env.char.pos[1]))
        if "lift_arrived" in [e["type"] for e in info["event_details"]]:
            break
    assert np.linalg.norm((env.char.pos - top)[[0, 2]]) < LIFT_REACH
    assert max(heights) <= top[1] + 3.5 and env.char.hp == env.controller.cfg.max_hp
    assert top[1] - bottom[1] > 45.0


def test_ravine_route_opens_with_the_gate(env):
    fw = env.fullscale
    plateau, ridge = fw.layout.anchors["outpost"], fw.layout.anchors["relay_1"]
    closed = NavGrid(fw, {FERRY_OPEN})
    opened = NavGrid(fw, {FERRY_OPEN, CORRIDOR_OPEN})
    assert not closed.connected(plateau, ridge)
    path = opened.plan(plateau, ridge)
    assert path is not None and len(path) > 2
    lowland = fw.layout.anchors["spawn"]
    assert not opened.connected(lowland, plateau)      # only the lift joins the tiers


def test_route_preview_completes_and_replays_exactly(tmp_path):
    run = tmp_path / "seed-7"
    manifest = preview.collect(7, run)
    assert manifest["success"] and manifest["min_hp"] == manifest["final_hp"]
    assert [m["completed"] is not None for m in manifest["milestones"]] == [True] * 12
    times = [m["completed"] for m in manifest["milestones"]]
    assert times == sorted(times)
    result = preview.verify(run)
    assert result["ok"] and result["actions"] == manifest["actions"]
    rows = preview.read_rows(run)
    facts = [e["fact"] for r in rows for e in r["events"] if e["type"] == "fact_opened"]
    assert facts == ["ferry_open", "corridor_open", "cap_installed"]
    assert any(r["move_state"] == "GLIDING" for r in rows)
    plan = preview.frame_plan(rows)
    assert plan[0]["step"] == 0 and plan[-1]["step"] == rows[-1]["step"]
    assert {e for p in plan for e in p["events"]} >= {"milestone_complete", "fact_opened", "lift_board"}
    json.dumps(manifest)
