"""Gate proofs and real-controller vent flights for the Task B island.

Stages follow the fact order. At each stage the ``over`` model must not reach
any sealed region or stack top, and the ``under`` model must reach every site
the stage opens. Vents count as launch sources in both models (see
:func:`open_world_bench.fullscale_reach.reach`). Three further proofs disable
vents to show that every vent-only place really needs a vent. Finally the real
controller flies every leg from its column top and must land on the shelf.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .character import CharacterConfig, Command, MoveState
from .fullscale_b_world import (CAUSEWAY_OPEN, CRATER_GATE_OPEN, CROWN_SIGNAL, FACT_ORDER, OUTPOST_CLEAR,
                                RESERVE_POWER, SKYLIFT_POWER, STACKS, SUITE_VERSION, SUMMIT_POWER)
from .fullscale_physics import PHYSICS_VERSION, FullScaleController, climb_height_bound
from .fullscale_reach import over_model, reach, region_patches, under_model

DT = 1.0 / 30.0
LANDING_SLACK = 1.5             # metres inside a shelf's edge that count as on it

LOWLAND_SITES = ("spawn", "keeper", "relay_b01", "relay_b02", "warden", "arena_b03", "arena_b04", "guide",
                 "pad_b05", "pad_b06", "depot", "depot_statue", "camp", "lake", "cut_mouth")
LOWLAND_TOPS = ("stack_b05a", "overlook_1", "stack_b06a", "stack_b06b", "overlook_2")
PLATEAU_SITES = ("cut_top", "relay_b11", "pad_b12", "arena_b13", "relay_b14", "relay_b16",
                 "pad_b16", "stack_b12a", "post")
HIGHLAND_SITES = ("stack_k1", "highland_rim", "crater_gate", "arena_b17", "relay_b17")
CRATER_SITES = ("crater_inside", "relay_b19", "pad_b18", "stack_b18a", "shrine", "pad_b20", "stack_b20a",
                "stack_b20b", "crown_landing", "relay_b21", "relay_b22", "relay_b23", "spire_pad", "outpost_pad",
                "arena_b24")
UPPER_TOPS = ("stack_b12a", "post", "stack_k1", "stack_b18a", "shrine", "stack_b20a", "stack_b20b", "outpost")


def _stages():
    f = FACT_ORDER
    return [
        # name, facts, sites the under model must reach, regions/tops over must not reach
        ("start", (), LOWLAND_SITES + LOWLAND_TOPS,
         ("plateau", "highland", "crater_floor", "crown", "spire_top") + UPPER_TOPS),
        (CAUSEWAY_OPEN, f[:1], PLATEAU_SITES,
         ("highland", "crater_floor", "crown", "spire_top", "stack_k1", "stack_b18a", "shrine", "stack_b20a",
          "stack_b20b", "outpost")),
        (SKYLIFT_POWER, f[:3], HIGHLAND_SITES,
         ("crater_floor", "crown", "spire_top", "stack_b18a", "shrine", "stack_b20a", "stack_b20b", "outpost")),
        (CRATER_GATE_OPEN, f[:5], CRATER_SITES, ("spire_top", "outpost")),
        (SUMMIT_POWER, f[:6], ("spire_top",), ("outpost",)),
        (OUTPOST_CLEAR, f[:7], ("stack_b24a", "outpost"), ()),
    ]


def _sealed(o, regions, names):
    return {n: int((o.any & regions[n]).sum()) for n in names}


_NO_VENT_CHECKS = (
    ("lowland, no vents", "spawn", tuple(STACKS) + ("highland", "crater_floor", "crown", "spire_top")),
    ("upper valley, no vents", "cut_top", tuple(STACKS) + ("highland", "crater_floor", "crown", "spire_top")),
    ("crater floor, no vents", "crater_inside",
     ("stack_b18a", "shrine", "stack_b20a", "stack_b20b", "stack_b24a", "outpost", "crown", "spire_top")),
)


def _no_vent_proofs(world, over, regions):
    """With every fact open but no vents, each vent-only place stays out of reach."""
    out = []
    for name, start, sealed in _NO_VENT_CHECKS:
        o = reach(world, FACT_ORDER, over, start=start, use_vents=False, strict_carry=True)
        entry = dict(name=name, start=start, sealed=_sealed(o, regions, sealed))
        entry["ok"] = all(v == 0 for v in entry["sealed"].values())
        out.append(entry)
    return out


@dataclass
class Flight:
    vent: str
    landed: bool
    on_shelf: bool
    landing: tuple
    seconds: float
    fall_damage: float
    stamina_left: float
    distance_to_centre: float


def vent_flight(world, vent, char_cfg=None, seconds: float = 60.0) -> Flight:
    """Start at the column top as a launch leaves the character, then glide to the shelf.

    The launch costs the jump's stamina; the rise itself is free (kinematic)."""
    cfg = char_cfg or CharacterConfig()
    controller = FullScaleController(world.terrain, world.props, cfg)
    controller.architecture = world.architecture
    st = controller.spawn(*vent.pad)
    st.pos[:] = (vent.pad[0], vent.apex, vent.pad[1])
    st.vel[:] = 0.0
    st.state, st.grounded = MoveState.AIRBORNE, False
    st.stamina = cfg.stamina_max - cfg.jump_cost
    hp, time = st.hp, 0.0
    tx, tz = vent.target_xz
    radius = STACKS[vent.target][0] if vent.target in STACKS else 8.0
    while time < seconds:
        dx, dz = tx - st.pos[0], tz - st.pos[2]
        dist = float(np.hypot(dx, dz))
        move = np.array([0.0, 1.0 if dist > 2.5 else 0.0], np.float32)
        controller.step(st, Command(move=move, jump=True), float(np.arctan2(dx, -dz)), DT)
        time += DT
        if st.state in (MoveState.GROUNDED, MoveState.SWIMMING):
            break
    dist = float(np.hypot(tx - st.pos[0], tz - st.pos[2]))
    landed = st.state == MoveState.GROUNDED
    on_shelf = landed and dist < radius - LANDING_SLACK and st.pos[1] > vent.target_top - 1.0
    return Flight(vent=vent.id, landed=bool(landed), on_shelf=bool(on_shelf),
                  landing=(round(float(st.pos[0]), 2), round(float(st.pos[1]), 2), round(float(st.pos[2]), 2)),
                  seconds=round(time, 3), fall_damage=round(float(hp - st.hp), 3),
                  stamina_left=round(float(st.stamina), 2), distance_to_centre=round(dist, 2))


def _stage_job(seed, name, facts, sites, sealed):
    """One proof stage in a worker process (the world is rebuilt from its seed)."""
    from .fullscale_b_world import build_world
    world = build_world(seed)
    cell = world.terrain._cell
    o = reach(world, facts, over_model(cell), strict_carry=True)
    u = reach(world, facts, under_model(cell))
    stage = dict(name=name, facts=list(facts), sealed=_sealed(o, region_patches(world), sealed),
                 over_max_height=round(o.max_height(), 2),
                 sites={s: u.at(world, s, radius=4.0) for s in sites},
                 iterations=dict(over=o.iterations, under=u.iterations))
    stage["ok"] = all(v == 0 for v in stage["sealed"].values()) and all(stage["sites"].values())
    return stage


def _no_vent_job(seed, name, start, sealed):
    from .fullscale_b_world import build_world
    world = build_world(seed)
    o = reach(world, FACT_ORDER, over_model(world.terrain._cell), start=start, use_vents=False, strict_carry=True)
    entry = dict(name=name, start=start, sealed=_sealed(o, region_patches(world), sealed))
    entry["ok"] = all(v == 0 for v in entry["sealed"].values())
    return entry


def check_world_parallel(seed, workers=8, controller_checks=True) -> dict:
    """``check_world`` for the default character, with proofs spread over processes."""
    from concurrent.futures import ProcessPoolExecutor
    from .fullscale_b_world import build_world
    world = build_world(seed)
    report = check_world(world, controller_checks=controller_checks, proofs=False)
    with ProcessPoolExecutor(workers) as pool:
        stages = [pool.submit(_stage_job, seed, *args) for args in _stages()]
        no_vents = [pool.submit(_no_vent_job, seed, *args) for args in _NO_VENT_CHECKS]
        report["stages"] = [f.result() for f in stages]
        report["no_vents"] = [f.result() for f in no_vents]
    report["ok"] &= all(s["ok"] for s in report["stages"]) and all(c["ok"] for c in report["no_vents"])
    return report


def check_world(world, char_cfg=None, party=None, controller_checks: bool = True, proofs: bool = True) -> dict:
    cell = world.terrain._cell
    over = over_model(cell, char_cfg, party)
    under = under_model(cell, char_cfg)
    regions = region_patches(world)
    bound = climb_height_bound(char_cfg, party)
    report = dict(suite=SUITE_VERSION, physics=PHYSICS_VERSION, seed=world.seed, world_digest=world.digest(),
                  climb_bound=dict(bound.__dict__),
                  models={m.name: {k: v for k, v in m.__dict__.items()} for m in (over, under)},
                  vents=[asdict(v) for v in world.vents], stages=[], ok=True)
    for name, facts, sites, sealed in (_stages() if proofs else ()):
        o = reach(world, facts, over, strict_carry=True)
        u = reach(world, facts, under)
        stage = dict(name=name, facts=list(facts), sealed=_sealed(o, regions, sealed),
                     over_max_height=round(o.max_height(), 2),
                     sites={s: u.at(world, s, radius=4.0) for s in sites},
                     iterations=dict(over=o.iterations, under=u.iterations))
        stage["ok"] = all(v == 0 for v in stage["sealed"].values()) and all(stage["sites"].values())
        report["ok"] &= stage["ok"]
        report["stages"].append(stage)
    if proofs:
        report["no_vents"] = _no_vent_proofs(world, over, regions)
        report["ok"] &= all(c["ok"] for c in report["no_vents"])
    if controller_checks:
        flights = [vent_flight(world, v, char_cfg) for v in world.vents]
        report["flights"] = [asdict(f) for f in flights]
        report["flights_ok"] = all(f.on_shelf and f.fall_damage == 0.0 and f.stamina_left > 0.0 for f in flights)
        report["ok"] &= report["flights_ok"]
    return report
