"""Solid-terrain traversal for the full-scale Teyvat tasks.

The shared :class:`CharacterController` treats the heightfield only as a floor:
each frame it snaps the feet to the terrain height under the character. Its
slide on over-steep ground also pushes along ``-normal``, which points uphill,
so a character walking into any slope, even a vertical cliff, rides up it
without spending stamina. A climber who runs out of stamina lands on the face,
recovers stamina there, and keeps climbing. A launch skill (Updraft) can also
be cast while hanging, and the climber can re-grab the wall afterwards.

The full-scale tasks rely on cliffs as real barriers. This module adds the
following rules without changing the shared controller or any existing task:

* Terrain is solid sideways. A move is rejected when the ground ahead rises
  more than the step height above the feet, or when the ground ahead is
  steeper than ``max_walk_slope_deg`` and higher than the feet.
* Ground steeper than the walkable limit cannot be stood on. The character
  slides downhill, cannot jump, and does not recover stamina there.
* Climbing presses against the face without sinking into it and costs stamina
  as before. The hold lasts until the face tops out within arm's reach; then the
  climber pulls up onto walkable ground no more than ``VAULT_RISE`` above the
  feet, or lets go if there is none.
* Launch skills fire only while standing on walkable ground.

:func:`climb_height_bound` gives an upper bound on the height one climb can
gain under these rules. ``fullscale_reach`` uses that bound to prove gates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import math3d as mth
from .character import CharacterConfig, CharacterController, CharacterState, Command, MoveState
from .party import DEFAULT_PARTY, ROSTER, PartyConfig, Resonance, _RESONANCE_BY_ELEMENT
from .world import CombatSystem

PHYSICS_VERSION = "teyvat-fullscale-physics-v1"
CLIMB_CONTACT_EPSILON = 0.0    # a climber's feet never sink below the face it is pressed against
SLIDE_ACCEL = 11.0             # m/s^2 downhill on unwalkable ground, as in the shared controller
VAULT_REACH = 3.0              # metres searched for walkable ground when a face ends
VAULT_RISE = 1.0               # metres a climber can pull up onto the top of a face


class SolidSurface:
    """Terrain view whose normals and slopes match the bilinear ground height.

    ``Terrain.normal_at`` interpolates per-vertex normals, so flat ground within
    one cell of a cliff reports a steep slope while ``height_at`` is still flat.
    The controller would then refuse to stand at a cliff's foot. Here normals
    come from the gradient of the same bilinear surface that ``height_at``
    samples. All other attributes delegate to the wrapped terrain.
    """

    def __init__(self, terrain):
        self._terrain = terrain

    def __getattr__(self, name):
        return getattr(self._terrain, name)

    def gradient(self, x, z):
        t = self._terrain
        gi, gj = t._to_grid(x, z)
        i0 = np.floor(gi).astype(np.int32)
        j0 = np.floor(gj).astype(np.int32)
        tx, tz = gi - i0, gj - j0
        h = t.height
        f00, f10, f01, f11 = h[i0, j0], h[i0 + 1, j0], h[i0, j0 + 1], h[i0 + 1, j0 + 1]
        dx = ((f10 - f00) * (1 - tz) + (f11 - f01) * tz) / t._cell
        dz = ((f01 - f00) * (1 - tx) + (f11 - f10) * tx) / t._cell
        return dx, dz

    def normal_at(self, x, z):
        dx, dz = self.gradient(x, z)
        n = np.stack([-np.asarray(dx, np.float32), np.ones_like(dx, np.float32), -np.asarray(dz, np.float32)], -1)
        return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)

    def slope_at(self, x, z):
        dx, dz = self.gradient(x, z)
        return np.arctan(np.hypot(dx, dz))


class FullScaleController(CharacterController):
    """CharacterController with solid, unwalkable cliffs."""

    def __init__(self, terrain, props, cfg: CharacterConfig | None = None):
        super().__init__(SolidSurface(terrain), props, cfg)

    def step(self, st: CharacterState, cmd: Command, cam_yaw: float, dt: float) -> list[str]:
        if st.state == MoveState.GROUNDED and not self.walkable(st.pos[0], st.pos[2]):
            # Slide downhill. Handing the shared step an airborne state skips its
            # uphill slide; jumping off unwalkable ground is not allowed.
            n = np.asarray(self.terrain.normal_at(st.pos[0], st.pos[2]), dtype=np.float32)
            flat = np.array([n[0], 0.0, n[2]], dtype=np.float32)
            length = float(np.linalg.norm(flat))
            if length > 1e-6:
                st.vel += flat / length * SLIDE_ACCEL * dt
            st.state = MoveState.AIRBORNE
            st.grounded = False
            if cmd.jump:
                cmd = Command(move=cmd.move, jump=False, sprint=cmd.sprint, climb=cmd.climb,
                              interact=cmd.interact, attack=cmd.attack, skill=cmd.skill,
                              burst=cmd.burst, swap=cmd.swap)
        stamina = float(st.stamina)
        self._before = st.pos.copy()
        self._blocking = True
        try:
            events = super().step(st, cmd, cam_yaw, dt)
        finally:
            self._blocking = False
        if st.state == MoveState.CLIMBING:
            self._vault_if_face_ended(st, cmd, cam_yaw)
        if st.state == MoveState.GROUNDED and not self.walkable(st.pos[0], st.pos[2]):
            st.stamina = min(float(st.stamina), stamina)   # no rest on a slope you cannot stand on
        return events

    def walkable(self, x: float, z: float) -> bool:
        return float(self.terrain.slope_at(x, z)) <= np.radians(self.cfg.max_walk_slope_deg)

    def _wall_contact(self, st: CharacterState, wish_dir: np.ndarray):
        """Climbable face directly ahead, probed at several reaches.

        A single probe at arm's length lands on the flat top of a face that is
        narrower than the reach, while the top is still far above the climber,
        so the climber would let go mid-face. Probing closer reaches keeps the
        hold until the face is actually below the feet.
        """
        c = self.cfg
        probe_dir = wish_dir if float(np.linalg.norm(wish_dir)) > 1e-3 else mth.yaw_pitch_to_dir(st.yaw, 0.0)
        steep = np.radians(c.min_climb_slope_deg)
        on_face = float(self.terrain.slope_at(st.pos[0], st.pos[2])) >= steep
        for reach in (0.15, 0.3, 0.45, 0.6, c.radius + 0.35):
            px, pz = st.pos[0] + probe_dir[0] * reach, st.pos[2] + probe_dir[2] * reach
            if float(self.terrain.height_at(px, pz)) <= st.pos[1] + 0.35:
                continue
            # Ahead is higher: a hold if the ground there is steep, or if the
            # climber is still on a face that tops out within this reach.
            sx, sz = (px, pz) if float(self.terrain.slope_at(px, pz)) >= steep else (
                (float(st.pos[0]), float(st.pos[2])) if on_face else (None, None))
            if sx is None:
                continue
            n = np.asarray(self.terrain.normal_at(sx, sz), dtype=np.float32)
            flat = np.array([n[0], 0.0, n[2]], dtype=np.float32)
            length = float(np.linalg.norm(flat))
            return flat / length if length > 1e-3 else -probe_dir
        return super()._wall_contact(st, wish_dir)

    def can_launch(self, st: CharacterState) -> bool:
        return st.state == MoveState.GROUNDED and self.walkable(st.pos[0], st.pos[2])

    # ------------------------------------------------------------ internals
    def _resolve_props(self, st: CharacterState) -> None:
        if getattr(self, "_blocking", False):
            self._block_terrain(st)
        super()._resolve_props(st)

    def _allowed(self, st: CharacterState, x: float, z: float) -> bool:
        ground = float(self.terrain.height_at(x, z))
        feet = float(st.pos[1])
        if st.state == MoveState.CLIMBING:
            return ground <= feet + CLIMB_CONTACT_EPSILON
        if ground > feet + self.cfg.step_height:
            return False
        if ground > feet + 1e-3 and not self.walkable(x, z):
            return False
        return True

    def _block_terrain(self, st: CharacterState) -> None:
        old, new = self._before, st.pos
        if abs(float(new[0] - old[0])) < 1e-7 and abs(float(new[2] - old[2])) < 1e-7:
            return
        if self._allowed(st, float(new[0]), float(new[2])):
            return
        # Slide along whichever axis remains open, otherwise stop sideways.
        for x, z, axis in ((float(new[0]), float(old[2]), 2), (float(old[0]), float(new[2]), 0)):
            if self._allowed(st, x, z):
                st.pos[0], st.pos[2] = x, z
                st.vel[axis] = 0.0
                return
        st.pos[0], st.pos[2] = old[0], old[2]
        st.vel[0] = st.vel[2] = 0.0

    def _vault_if_face_ended(self, st: CharacterState, cmd: Command, cam_yaw: float) -> None:
        move = np.asarray(cmd.move, dtype=np.float32)
        cy, sy = np.cos(cam_yaw), np.sin(cam_yaw)
        wish = np.array([move[0] * cy + move[1] * sy, 0.0, move[0] * sy - move[1] * cy], dtype=np.float32)
        norm = float(np.linalg.norm(wish))
        wish_dir = wish / norm if norm > 1e-3 else np.zeros(3, dtype=np.float32)
        if self._wall_contact(st, wish_dir) is not None:
            return
        into = -np.asarray(st.climb_normal, dtype=np.float32)
        into[1] = 0.0
        length = float(np.linalg.norm(into))
        feet = float(st.pos[1])
        if length > 1e-6:
            into /= length
            for reach in np.arange(0.25, VAULT_REACH + 1e-6, 0.25):
                x, z = float(st.pos[0] + into[0] * reach), float(st.pos[2] + into[2] * reach)
                ground = float(self.terrain.height_at(x, z))
                if ground <= feet + VAULT_RISE and self.walkable(x, z):
                    st.pos[0], st.pos[1], st.pos[2] = x, ground, z
                    st.state = MoveState.GROUNDED
                    st.grounded = True
                    st.vel[:] = 0.0
                    return
        st.state = MoveState.AIRBORNE
        st.grounded = False
        st.vel[:] = 0.0


class FullScaleCombat(CombatSystem):
    """Combat whose launch skills require walkable ground under the caster."""

    def cast(self, char: CharacterState, controller, *, burst: bool = False):
        party = self.party
        if party is not None and hasattr(controller, "can_launch"):
            ability = party.ready_burst() if burst else party.ready_skill()
            if ability is not None and ability.launch_speed and not controller.can_launch(char):
                return []
        return super().cast(char, controller, burst=burst)


@dataclass(frozen=True)
class ClimbBound:
    """Upper bound on height gained by one climb from rest, with its parts (m)."""
    climb: float
    start: float
    unmargined: float
    margin: float
    total: float


def climb_height_bound(char_cfg: CharacterConfig | None = None, party: PartyConfig | None = None,
                       margin: float = 1.25) -> ClimbBound:
    """Upper bound on the height one climb can gain under FullScaleController.

    Climbing spends ``climb_cost`` stamina per second from a full bar, and
    stamina never recovers on unwalkable ground, so the face can be climbed for
    at most ``stamina_max / climb_cost`` seconds at no more than
    ``climb_speed`` (the controller moves at 0.85 of it along the face).
    Before grabbing the face the climber may jump or fire a launch skill from
    walkable ground, whichever starts higher. Gliding never gains height.
    Party resonance that lowers stamina costs is included.
    """
    cfg = char_cfg or CharacterConfig()
    party = party or PartyConfig()
    members = [ROSTER[m] for m in party.members]
    elements = [m.element for m in members]
    resonance = Resonance.NONE
    for element in set(elements):
        if elements.count(element) >= 2:
            resonance = _RESONANCE_BY_ELEMENT[element]
    cost = cfg.climb_cost * (0.8 if resonance == Resonance.DRIFT else 1.0)
    climb = cfg.stamina_max / cost * cfg.climb_speed
    gravity = abs(cfg.gravity)
    jump = cfg.jump_speed ** 2 / (2 * gravity)
    launch = max((m.skill.launch_speed for m in members), default=0.0) ** 2 / (2 * gravity)
    start = max(jump, launch)
    raw = climb + start
    return ClimbBound(climb=climb, start=start, unmargined=raw, margin=margin, total=raw * margin)


DEFAULT_PARTY_CONFIG = PartyConfig(members=DEFAULT_PARTY)
