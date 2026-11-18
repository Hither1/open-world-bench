"""Checks that drive the real full-scale controller through the world.

``fullscale_reach`` reasons about the heightfield with bracketing models. These
checks run the actual :class:`FullScaleController` instead:

* :func:`glide` walks off an edge toward a target and glides until landing;
* :func:`climb_attempt` tries the strongest single climb (a launch skill from
  the ground, then climbing straight up) at the foot of a face.

They confirm the designed glide works and that the climb bound used by the
proofs is not exceeded on the actual gate walls.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .character import CharacterConfig, Command, MoveState
from .fullscale_physics import FullScaleCombat, FullScaleController
from .party import Party, PartyConfig
from .world import World, WorldConfig

DT = 1.0 / 30.0


def _yaw_toward(dx: float, dz: float) -> float:
    """Camera yaw for which a forward move points along (dx, dz)."""
    return float(np.arctan2(dx, -dz))


def _controller(world, char_cfg=None):
    controller = FullScaleController(world.terrain, world.props, char_cfg or CharacterConfig())
    controller.architecture = world.architecture
    return controller


@dataclass
class GlideResult:
    landed: bool
    landing: tuple
    seconds: float
    glide_seconds: float
    fall_damage: float
    distance_to_target: float


def glide(world, start: str, target: str, seconds: float = 60.0, char_cfg=None) -> GlideResult:
    """Walk from ``start`` toward ``target``, glide off the first drop, and land."""
    controller = _controller(world, char_cfg)
    sx, sz = world.layout.anchors[start]
    tx, tz = world.layout.anchors[target]
    st = controller.spawn(sx, sz)
    hp, time, gliding, airborne = st.hp, 0.0, 0.0, False
    while time < seconds:
        yaw = _yaw_toward(tx - st.pos[0], tz - st.pos[2])
        controller.step(st, Command(move=np.array([0.0, 1.0], np.float32), jump=True), yaw, DT)
        time += DT
        gliding += DT if st.state == MoveState.GLIDING else 0.0
        airborne |= st.state in (MoveState.AIRBORNE, MoveState.GLIDING) and gliding > 0.0
        if airborne and st.state in (MoveState.GROUNDED, MoveState.SWIMMING):
            break
    landed = airborne and st.state in (MoveState.GROUNDED, MoveState.SWIMMING)
    return GlideResult(landed=bool(landed), landing=(float(st.pos[0]), float(st.pos[2])), seconds=time,
                       glide_seconds=gliding, fall_damage=float(hp - st.hp),
                       distance_to_target=float(np.hypot(tx - st.pos[0], tz - st.pos[2])))


@dataclass
class ClimbResult:
    foot: tuple
    foot_height: float
    gained: float
    launched: bool
    top_reached: bool       # stood on walkable ground more than 2 m above the foot


def climb_attempt(world, foot_xz, toward_xz, seconds: float = 45.0, char_cfg=None,
                  party: PartyConfig | None = None) -> ClimbResult:
    """Strongest single climb: walk to the face, launch, then climb straight up."""
    controller = _controller(world, char_cfg)
    party_obj = Party(party or PartyConfig())
    launcher = next((i for i, m in enumerate(party_obj.members) if m.spec.skill.launch_speed), None)
    if launcher is not None:
        party_obj.swap(launcher)
        party_obj.swap_timer = 0.0
    sim = World(WorldConfig(seed=world.seed, terrain=world.terrain.cfg), terrain=world.terrain,
                props=world.props, party=party_obj)
    sim.combat = FullScaleCombat(sim, party_obj)
    st = controller.spawn(float(foot_xz[0]), float(foot_xz[1]))
    foot_height = float(st.pos[1])
    yaw = _yaw_toward(toward_xz[0] - foot_xz[0], toward_xz[1] - foot_xz[1])
    forward = np.array([0.0, 1.0], np.float32)
    best, time, launched, phase = foot_height, 0.0, False, "approach"
    still, stood_above = 0, False
    while time < seconds:
        party_obj.tick(DT)
        climb = True
        if phase == "approach":
            before = st.pos.copy()
            controller.step(st, Command(move=forward), yaw, DT)
            still = still + 1 if np.linalg.norm((st.pos - before)[[0, 2]]) < 1e-3 else 0
            if still > 5:
                phase = "launch"
        else:
            if phase == "launch" and st.state == MoveState.GROUNDED:
                launched = bool(sim.combat.cast(st, controller)) or launched
                phase = "rise"
            if phase == "rise":
                climb = st.vel[1] <= 0.5
                if climb:
                    phase = "climb"
            controller.step(st, Command(move=forward, climb=climb), yaw, DT)
        time += DT
        best = max(best, float(st.pos[1]))
        stood_above |= (st.state == MoveState.GROUNDED and st.pos[1] > foot_height + 2.0
                        and controller.walkable(st.pos[0], st.pos[2]))
    return ClimbResult(foot=(float(foot_xz[0]), float(foot_xz[1])), foot_height=foot_height,
                       gained=best - foot_height, launched=launched, top_reached=bool(stood_above))


def wall_feet(world):
    """Climb start points at the foot of the ridge wall and the summit pillar."""
    from .fullscale_world import PILLAR_RADIUS, RAVINE_HALF_WIDTH, ridge_z
    feet = []
    for x in np.arange(-450.0, 451.0, 100.0):
        if abs(x - world.layout.corridor_x) < RAVINE_HALF_WIDTH + 30.0:
            continue
        line = float(ridge_z(x))
        feet.append(("ridge", (float(x), line + 6.0), (float(x), line - 20.0)))
    sx, sz = world.layout.summit
    for angle in np.arange(0.0, 2 * np.pi, np.pi / 4):
        d = np.array((np.cos(angle), np.sin(angle)))
        foot = np.array((sx, sz)) + d * (PILLAR_RADIUS + 5.0)
        feet.append(("pillar", (float(foot[0]), float(foot[1])), (sx, sz)))
    return feet
