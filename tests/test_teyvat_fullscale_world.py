"""Full-scale Teyvat Task A, phase 1: solid-terrain physics, world, and gate proofs."""
from dataclasses import dataclass

import numpy as np
import pytest

from open_world_bench.architecture import Architecture
from open_world_bench.campaign_env import GateCollisions
from open_world_bench.character import CharacterConfig, Command, MoveState
from open_world_bench.fullscale_physics import FullScaleCombat, FullScaleController, climb_height_bound
from open_world_bench.fullscale_reach import check_world, over_model, reach, region_patches, under_model
from open_world_bench.fullscale_routes import climb_attempt
from open_world_bench.fullscale_world import (FACT_ORDER, PILLAR_TOP, PLATEAU, RIDGE, FactState, build_world,
                                              make_layout)
from open_world_bench.party import Party, PartyConfig
from open_world_bench.scatter import PropField
from open_world_bench.terrain import Terrain, TerrainConfig
from open_world_bench.world import World, WorldConfig

DT = 1 / 30
EMPTY = PropField(np.empty((0, 3), np.float32), np.empty(0, np.float32), np.empty(0, np.float32), np.empty(0, np.int8))


def _wall_terrain(rise, run=0.0, base=5.0, size=256, resolution=103):
    """Flat ground at ``base`` with ground ``rise`` metres higher north of z = 0."""
    t = Terrain(TerrainConfig(size=size, resolution=resolution, max_height=200))
    axis = np.linspace(-size / 2, size / 2, resolution)
    _, z = np.meshgrid(axis, axis, indexing="ij")
    frac = np.clip(-z / run, 0, 1) if run > 0 else (z < 0).astype(float)
    t.height[:] = (base + rise * frac).astype(np.float32)
    t._compute_normals()
    t._compute_biomes()
    return t


def _drive(terrain, seconds, command, launch=False):
    party = Party(PartyConfig())
    party.swap(2)
    party.swap_timer = 0.0
    world = World(WorldConfig(seed=0, terrain=terrain.cfg), terrain=terrain, props=EMPTY, party=party)
    world.combat = FullScaleCombat(world, party)
    ctl = FullScaleController(terrain, EMPTY, CharacterConfig())
    st = ctl.spawn(0.0, 3.0)
    best, casts, refused, rising = float(st.pos[1]), 0, 0, False
    for frame in range(int(seconds / DT)):
        party.tick(DT)
        cmd = command(st, frame)
        if launch and frame < 60:
            cmd = Command(move=cmd.move)                 # walk to the foot first
        if launch and frame == 60 and st.state == MoveState.GROUNDED:
            casts += bool(world.combat.cast(st, ctl))
            rising = True
        if rising:
            cmd = Command(move=cmd.move, climb=st.vel[1] <= 0.5)
            rising = st.vel[1] > 0.5
        if st.state == MoveState.CLIMBING and party.ready_skill() is not None and not launch:
            refused += not world.combat.cast(st, ctl)
        ctl.step(st, cmd, 0.0, DT)
        best = max(best, float(st.pos[1]))
    return best - 5.0, casts, refused


FORWARD = np.array([0.0, 1.0], np.float32)


@pytest.mark.parametrize("run", [0.0, 10.0, 22.0, 45.0])
def test_steep_slopes_block_walking_and_jumping(run):
    terrain = _wall_terrain(60.0, run)
    walked, _, _ = _drive(terrain, 20, lambda st, f: Command(move=FORWARD))
    jumped, _, _ = _drive(terrain, 20, lambda st, f: Command(move=FORWARD, jump=f % 2 == 0))
    assert walked < 1.0
    assert jumped < 2.0


@pytest.mark.parametrize("run", [0.0, 10.0, 22.0])
def test_climbing_is_limited_by_stamina_and_launches_need_ground(run):
    terrain = _wall_terrain(60.0, run)
    bound = climb_height_bound()
    climbed, _, refused = _drive(terrain, 60, lambda st, f: Command(move=FORWARD, climb=True))
    launched, casts, _ = _drive(terrain, 60, lambda st, f: Command(move=FORWARD, climb=True), launch=True)
    assert 10.0 < climbed <= bound.unmargined
    assert climbed < launched <= bound.unmargined and casts == 1
    assert refused > 0          # Updraft was tried while hanging and refused every time


def test_walkable_ramps_still_walk():
    terrain = _wall_terrain(60.0, 100.0)        # 31 degrees
    walked, _, _ = _drive(terrain, 40, lambda st, f: Command(move=FORWARD))
    assert walked > 59.0


def test_climb_bound_follows_party():
    default = climb_height_bound()
    drift = climb_height_bound(party=PartyConfig(members=("wren", "wren", "kestrel")))
    no_launch = climb_height_bound(party=PartyConfig(members=("kestrel", "maren")))
    assert drift.climb > default.climb                     # two gale members lower climbing cost
    assert no_launch.start < default.start == pytest.approx(13.0 ** 2 / 48.0)


# ----------------------------------------------------------------- the world
@pytest.fixture(scope="module")
def world7():
    return build_world(7)


def test_world_is_deterministic_and_seeded(world7):
    assert build_world(7).digest() == world7.digest()
    assert build_world(19).digest() != world7.digest()
    a, b = make_layout(7).anchors, make_layout(19).anchors
    assert all(np.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1]) <= 2 * 18.0 * np.sqrt(2) + 1e-6 for k in a)


def test_sites_sit_on_their_tiers(world7):
    h = {name: world7.anchor(name)[1] for name in world7.layout.anchors}
    for name in ("spawn", "keeper", "yard_1", "junction_1", "cache", "camp", "ferry_low", "ruin"):
        assert 0.5 < h[name] < 20.0, name
    for name in ("ferry_high", "outpost", "west_edge", "corridor_mouth"):
        assert abs(h[name] - PLATEAU) < 3.0, name
    for name in ("relay_1", "relay_2", "ravine_top"):
        assert abs(h[name] - RIDGE) < 3.0, name
    assert h["summit_top"] == pytest.approx(PILLAR_TOP, abs=0.5)


def test_scenery_stays_off_cliffs_and_water(world7):
    t, props = world7.terrain, world7.props
    assert len(props.pos) > 500 and len(world7.buildings) >= 12
    assert (t.slope_at(props.pos[:, 0], props.pos[:, 2]) < np.radians(35.0)).all()
    assert (props.pos[:, 1] > 1.0).all()
    for b in world7.buildings:
        x, _, z = b.pos
        assert t.height_at(x, z) > 0.8 and t.slope_at(x, z) < np.radians(30.0), b.id


# ----------------------------------------------------------------- the proofs
@dataclass
class _Layout:
    anchors: dict


@dataclass
class _ToyWorld:
    """Just enough of FullScaleWorld for ``reach`` on a single wall."""
    terrain: Terrain
    layout: _Layout
    seed: int = 0
    lifts: tuple = ()

    def __post_init__(self):
        self.props = EMPTY
        self.state = FactState()
        self.architecture = GateCollisions(Architecture(()), self.state)

    @property
    def gates(self):
        return self.state.gates

    def grid(self):
        axis = np.linspace(-self.terrain._half, self.terrain._half, self.terrain.cfg.resolution)
        return np.meshgrid(axis, axis, indexing="ij")


@pytest.mark.parametrize("rise,climbable", [(15.0, True), (45.0, False)])
def test_proof_models_agree_with_the_controller(rise, climbable):
    """The validator is not vacuous: it opens a short wall and seals a tall one,
    and the real controller agrees in both cases."""
    terrain = _wall_terrain(rise, size=256, resolution=103)
    toy = _ToyWorld(terrain, _Layout({"spawn": (0.0, 40.0), "top": (0.0, -40.0)}))
    over = reach(toy, (), over_model(terrain._cell))
    under = reach(toy, (), under_model(terrain._cell))
    assert over.at(toy, "top") == climbable
    assert not under.at(toy, "top")                     # walking alone never scales a wall
    attempt = climb_attempt(toy, (0.0, 6.0), (0.0, -20.0))
    assert attempt.launched and attempt.top_reached == climbable


@pytest.mark.parametrize("seed", [7, 19])
def test_every_gate_holds_and_every_route_opens(seed):
    report = check_world(build_world(seed))
    for stage in report["stages"]:
        assert stage["ok"], stage
    assert report["west_glide"]["ok"], report["west_glide"]
    assert report["climbs"]["ok"] and report["climbs"]["max_gained"] <= climb_height_bound().unmargined
    assert report["ok"]


def test_regions_and_facts_line_up(world7):
    regions = region_patches(world7)
    assert all(regions[k].sum() > 20 for k in ("lowland", "plateau", "ridge", "summit_top"))
    assert FACT_ORDER == ("ferry_open", "corridor_open", "cap_installed")
    assert [lift.opens for lift in world7.lifts] == ["ferry_open", "cap_installed"]
    assert [gate["opens"] for gate in world7.gates] == ["corridor_open"]
