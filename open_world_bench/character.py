"""Kinematic third-person character controller.

Movement set is deliberately Genshin-like, because that is what makes the
long-horizon traversal interesting: run, sprint, jump, climb any steep face,
glide from height, and swim -- all drawing from one stamina pool.

The controller is kinematic (no rigid-body solver): it integrates velocity,
then resolves the terrain surface and prop cylinders analytically. That keeps
a step cheap enough for RL rollouts and keeps behaviour reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from . import math3d as mth
from .scatter import PropField
from .terrain import Terrain


class MoveState(IntEnum):
    GROUNDED = 0
    AIRBORNE = 1
    GLIDING = 2
    CLIMBING = 3
    SWIMMING = 4


@dataclass
class CharacterConfig:
    radius: float = 0.4
    height: float = 1.75
    eye_height: float = 1.55

    gravity: float = -24.0
    walk_speed: float = 2.2
    run_speed: float = 5.4
    sprint_speed: float = 8.2
    swim_speed: float = 2.7
    climb_speed: float = 1.9
    glide_speed: float = 8.5
    glide_fall_speed: float = -3.2
    terminal_velocity: float = -55.0

    jump_speed: float = 8.5
    ground_accel: float = 34.0
    air_accel: float = 9.0
    friction: float = 12.0
    turn_rate: float = 11.0
    max_walk_slope_deg: float = 50.0
    min_climb_slope_deg: float = 52.0
    step_height: float = 0.55

    stamina_max: float = 100.0
    stamina_regen: float = 21.0
    stamina_regen_delay: float = 0.45
    sprint_cost: float = 19.0
    climb_cost: float = 9.0
    glide_cost: float = 5.5
    swim_cost: float = 4.0
    jump_cost: float = 7.0

    fall_damage_speed: float = -22.0   # impact speed that starts hurting
    fall_damage_scale: float = 2.2     # hp per (m/s) past the threshold
    max_hp: float = 100.0
    glide_min_clearance: float = 1.8   # metres above ground needed to deploy


@dataclass
class Command:
    """One frame of control input, in camera space."""

    move: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    jump: bool = False
    sprint: bool = False
    climb: bool = False
    interact: bool = False
    attack: bool = False
    skill: bool = False
    burst: bool = False
    swap: int = -1              # zero-based party slot; -1 keeps the current member

    @staticmethod
    def zero() -> "Command":
        return Command()


@dataclass
class CharacterState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    yaw: float = 0.0
    state: MoveState = MoveState.GROUNDED
    stamina: float = 100.0
    hp: float = 100.0
    grounded: bool = True
    anim_phase: float = 0.0
    air_time: float = 0.0
    climb_normal: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    _regen_timer: float = 0.0
    _jump_was_down: bool = False
    _peak_fall_speed: float = 0.0

    def copy(self) -> "CharacterState":
        return CharacterState(
            pos=self.pos.copy(), vel=self.vel.copy(), yaw=self.yaw, state=self.state,
            stamina=self.stamina, hp=self.hp, grounded=self.grounded,
            anim_phase=self.anim_phase, air_time=self.air_time,
            climb_normal=self.climb_normal.copy(), _regen_timer=self._regen_timer,
            _jump_was_down=self._jump_was_down, _peak_fall_speed=self._peak_fall_speed,
        )


class CharacterController:
    """Steps a :class:`CharacterState` against a terrain and a prop field."""

    def __init__(self, terrain: Terrain, props: PropField | None = None, cfg: CharacterConfig | None = None):
        self.terrain = terrain
        self.props = props
        self.cfg = cfg or CharacterConfig()

    # ------------------------------------------------------------------ spawn
    def spawn(self, x: float, z: float, yaw: float = 0.0) -> CharacterState:
        y = float(self.terrain.height_at(x, z))
        st = CharacterState(pos=np.array([x, y, z], dtype=np.float32), yaw=float(yaw))
        st.stamina = self.cfg.stamina_max
        st.hp = self.cfg.max_hp
        return st

    # ------------------------------------------------------------------- step
    def step(self, st: CharacterState, cmd: Command, cam_yaw: float, dt: float) -> list[str]:
        """Advance one frame. Returns a list of event names that fired."""
        c = self.cfg
        events: list[str] = []
        move = np.asarray(cmd.move, dtype=np.float32)
        mag = float(np.linalg.norm(move))
        if mag > 1.0:
            move = move / mag
            mag = 1.0

        # Desired horizontal direction, rotated out of camera space into world.
        cy, sy = np.cos(cam_yaw), np.sin(cam_yaw)
        wish = np.array([move[0] * cy - move[1] * sy, 0.0, -(move[0] * sy + move[1] * cy)], dtype=np.float32)
        wish_dir = wish / max(np.linalg.norm(wish), 1e-6) if mag > 1e-3 else np.zeros(3, dtype=np.float32)

        jump_pressed = bool(cmd.jump) and not st._jump_was_down
        st._jump_was_down = bool(cmd.jump)

        ground_h = float(self.terrain.height_at(st.pos[0], st.pos[2]))
        ground_n = np.asarray(self.terrain.normal_at(st.pos[0], st.pos[2]), dtype=np.float32)
        slope = float(np.arccos(np.clip(ground_n[1], -1.0, 1.0)))
        clearance = float(st.pos[1] - ground_h)
        water_here = ground_h < self.terrain.cfg.sea_level
        in_water = water_here and st.pos[1] < self.terrain.cfg.sea_level + 0.25

        # ------------------------------------------------------- state machine
        prev_state = st.state
        if in_water:
            st.state = MoveState.SWIMMING
        elif st.state == MoveState.SWIMMING:
            st.state = MoveState.GROUNDED if clearance <= 0.3 else MoveState.AIRBORNE

        if st.state == MoveState.GROUNDED and clearance > c.step_height and not in_water:
            st.state = MoveState.AIRBORNE      # we ran off a ledge

        if st.state == MoveState.CLIMBING:
            if not cmd.climb or st.stamina <= 0.0:
                st.state = MoveState.AIRBORNE
                if st.stamina <= 0.0:
                    events.append("stamina_depleted")
        elif st.state in (MoveState.GROUNDED, MoveState.AIRBORNE):
            wall_n = self._wall_contact(st, wish_dir)
            if cmd.climb and wall_n is not None and st.stamina > 1.0:
                st.state = MoveState.CLIMBING
                st.climb_normal = wall_n
                st.vel[:] = 0.0
                if prev_state != MoveState.CLIMBING:
                    events.append("climb_start")

        if st.state == MoveState.GLIDING:
            if st.grounded or in_water or not cmd.jump or st.stamina <= 0.0:
                st.state = MoveState.SWIMMING if in_water else (MoveState.GROUNDED if st.grounded else MoveState.AIRBORNE)
                if st.stamina <= 0.0:
                    events.append("stamina_depleted")
        elif st.state == MoveState.AIRBORNE and cmd.jump and st.vel[1] < 0.0:
            # Hold jump while falling to deploy the glider, like the real game.
            if clearance > c.glide_min_clearance and st.stamina > 2.0:
                st.state = MoveState.GLIDING
                st.vel[1] = max(st.vel[1], c.glide_fall_speed)
                events.append("glide_start")

        # ------------------------------------------------------------ dynamics
        if st.state == MoveState.GROUNDED:
            sprinting = bool(cmd.sprint) and st.stamina > 0.0 and mag > 0.3
            target_speed = c.sprint_speed if sprinting else (c.run_speed if mag > 0.55 else c.walk_speed)
            self._accelerate(st, wish_dir, target_speed * max(mag, 0.0), c.ground_accel, dt)
            if mag < 1e-3:
                self._apply_friction(st, dt)
            if sprinting:
                st.stamina -= c.sprint_cost * dt
                st._regen_timer = c.stamina_regen_delay
            if slope > np.radians(c.max_walk_slope_deg):
                # Too steep to stand on: slide along the surface gradient.
                downhill = mth.norm(np.array([-ground_n[0], 0.0, -ground_n[2]], dtype=np.float32))
                st.vel += downhill * 11.0 * dt
                st.vel[1] += c.gravity * dt
                st.state = MoveState.AIRBORNE
            else:
                st.vel[1] = 0.0
            if jump_pressed and st.stamina >= 1.0:
                st.vel[1] = c.jump_speed
                st.stamina -= c.jump_cost
                st._regen_timer = c.stamina_regen_delay
                st.state = MoveState.AIRBORNE
                events.append("jump")

        elif st.state == MoveState.AIRBORNE:
            self._accelerate(st, wish_dir, c.run_speed * mag, c.air_accel, dt)
            st.vel[1] = max(st.vel[1] + c.gravity * dt, c.terminal_velocity)

        elif st.state == MoveState.GLIDING:
            self._accelerate(st, wish_dir, c.glide_speed * mag, c.air_accel * 1.6, dt)
            st.vel[1] = mth.damp(st.vel[1], c.glide_fall_speed, 0.02, dt)
            st.stamina -= c.glide_cost * dt
            st._regen_timer = c.stamina_regen_delay

        elif st.state == MoveState.CLIMBING:
            n = st.climb_normal
            # Build a tangent frame on the wall so input maps to up/sideways.
            side = mth.norm(np.cross(mth.UP, n))
            up_t = mth.norm(np.cross(n, side))
            st.vel[:] = (side * float(np.dot(wish_dir, side)) + up_t * max(mag, 0.0) * 0.85) * c.climb_speed
            if mag < 1e-3:
                st.vel[:] = 0.0
            st.vel -= n * 0.4           # stay pressed against the face
            st.stamina -= c.climb_cost * dt * (1.0 if mag > 0.1 else 0.35)
            st._regen_timer = c.stamina_regen_delay

        elif st.state == MoveState.SWIMMING:
            self._accelerate(st, wish_dir, c.swim_speed * mag, 10.0, dt)
            surface = self.terrain.cfg.sea_level - 0.15
            if cmd.jump and st.stamina > 1.0:
                st.vel[1] = -2.4                         # hold jump to dive
                st.stamina -= c.swim_cost * dt
            else:
                st.vel[1] = (surface - st.pos[1]) * 4.0  # bob back to the surface
            if mag > 0.1:
                st.stamina -= c.swim_cost * dt
                st._regen_timer = c.stamina_regen_delay
            if prev_state != MoveState.SWIMMING:
                events.append("enter_water")

        # ------------------------------------------------------------ integrate
        st.pos = st.pos + st.vel * dt
        self._resolve_props(st)
        landed_speed = self._resolve_ground(st)
        x, z = self.terrain.clamp_to_bounds(float(st.pos[0]), float(st.pos[2]), margin=3.0)
        if x != float(st.pos[0]) or z != float(st.pos[2]):
            st.vel[0] *= 0.0
            st.vel[2] *= 0.0
        st.pos[0], st.pos[2] = x, z

        if landed_speed is not None:
            events.append("land")
            st.state = MoveState.GROUNDED
            impact = min(landed_speed, 0.0)
            if impact < c.fall_damage_speed:
                dmg = (c.fall_damage_speed - impact) * c.fall_damage_scale
                st.hp = max(0.0, st.hp - float(dmg))
                events.append("fall_damage")

        # ------------------------------------------------------ stamina & anim
        st._regen_timer = max(0.0, st._regen_timer - dt)
        if st._regen_timer <= 0.0 and st.state in (MoveState.GROUNDED, MoveState.SWIMMING):
            st.stamina += c.stamina_regen * dt
        st.stamina = float(np.clip(st.stamina, 0.0, c.stamina_max))

        st.grounded = st.state == MoveState.GROUNDED
        st.air_time = 0.0 if st.grounded else st.air_time + dt
        speed_xz = float(np.linalg.norm(st.vel[[0, 2]]))
        st.anim_phase = float((st.anim_phase + dt * (1.4 + speed_xz * 0.9)) % 1.0)

        # Face the direction of travel (or the wall when climbing).
        if st.state == MoveState.CLIMBING:
            target_yaw = float(np.arctan2(-st.climb_normal[0], st.climb_normal[2]))
            st.yaw = self._turn_toward(st.yaw, target_yaw, c.turn_rate * 1.5, dt)
        elif speed_xz > 0.4:
            target_yaw = float(np.arctan2(st.vel[0], -st.vel[2]))
            st.yaw = self._turn_toward(st.yaw, target_yaw, c.turn_rate, dt)

        if st.stamina <= 0.0 and st.state == MoveState.SWIMMING:
            events.append("drowning")
        return events

    def dash(self, st: CharacterState, distance: float) -> tuple[np.ndarray, np.ndarray]:
        """Sweep forward along walkable terrain, stopping at cliffs and props."""
        start = st.pos.copy()
        direction = mth.yaw_pitch_to_dir(st.yaw, 0.0)
        steps = max(1, int(np.ceil(distance / max(self.cfg.radius * 0.5, 0.05))))
        stride = direction * (distance / steps)
        for _ in range(steps):
            previous = st.pos.copy()
            candidate = previous + stride
            x, z = self.terrain.clamp_to_bounds(float(candidate[0]), float(candidate[2]), margin=3.0)
            candidate[0], candidate[2] = x, z
            ground = float(self.terrain.height_at(x, z))
            if st.grounded:
                if (ground < self.terrain.cfg.sea_level
                        or abs(ground - float(previous[1])) > self.cfg.step_height
                        or self.terrain.slope_at(x, z) > np.radians(self.cfg.max_walk_slope_deg)):
                    break
                candidate[1] = ground
            elif ground > float(candidate[1]):
                break
            st.pos = candidate.copy()
            self._resolve_props(st)
            if not np.allclose(st.pos, candidate, atol=1e-5):
                st.pos = previous
                break
        return start, st.pos.copy()

    # -------------------------------------------------------------- internals
    def _accelerate(self, st: CharacterState, wish_dir: np.ndarray, target_speed: float, accel: float, dt: float) -> None:
        if target_speed <= 1e-4:
            return
        horiz = np.array([st.vel[0], 0.0, st.vel[2]], dtype=np.float32)
        current = float(np.dot(horiz, wish_dir))
        add = min(target_speed - current, accel * dt)
        if add > 0.0:
            st.vel[0] += wish_dir[0] * add
            st.vel[2] += wish_dir[2] * add
        # Clamp so diagonal input can never exceed the state's top speed.
        sp = float(np.linalg.norm(st.vel[[0, 2]]))
        if sp > target_speed > 0.0:
            scale = max(target_speed, sp - self.cfg.friction * dt) / sp
            st.vel[0] *= scale
            st.vel[2] *= scale

    def _apply_friction(self, st: CharacterState, dt: float) -> None:
        sp = float(np.linalg.norm(st.vel[[0, 2]]))
        if sp < 1e-4:
            st.vel[0] = st.vel[2] = 0.0
            return
        scale = max(0.0, sp - self.cfg.friction * dt) / sp
        st.vel[0] *= scale
        st.vel[2] *= scale

    def _wall_contact(self, st: CharacterState, wish_dir: np.ndarray) -> np.ndarray | None:
        """Return the normal of a climbable surface right in front, if any."""
        c = self.cfg
        probe_dir = wish_dir if float(np.linalg.norm(wish_dir)) > 1e-3 else mth.yaw_pitch_to_dir(st.yaw, 0.0)
        probe = st.pos + probe_dir * (c.radius + 0.35)
        gh = float(self.terrain.height_at(probe[0], probe[2]))
        if gh > st.pos[1] + 0.35:
            n = np.asarray(self.terrain.normal_at(probe[0], probe[2]), dtype=np.float32)
            slope = float(np.arccos(np.clip(n[1], -1.0, 1.0)))
            if slope >= np.radians(c.min_climb_slope_deg):
                flat = np.array([n[0], 0.0, n[2]], dtype=np.float32)
                if float(np.linalg.norm(flat)) > 1e-3:
                    return mth.norm(flat)
                return -probe_dir
        if self.props is not None:
            # Tree trunks and boulders are climbable too.
            idx = self.props.solid_near(float(probe[0]), float(probe[2]), c.radius)
            for i in idx:
                if st.pos[1] < self.props.pos[i, 1] + self.props.height[i] - 0.5:
                    d = st.pos - self.props.pos[i]
                    flat = np.array([d[0], 0.0, d[2]], dtype=np.float32)
                    if float(np.linalg.norm(flat)) > 1e-3:
                        return mth.norm(flat)
        return None

    def _resolve_props(self, st: CharacterState) -> None:
        if self.props is None or len(self.props) == 0:
            return
        c = self.cfg
        idx = self.props.solid_near(float(st.pos[0]), float(st.pos[2]), c.radius)
        for i in idx:
            p = self.props.pos[i]
            if st.pos[1] > p[1] + self.props.height[i]:
                continue                     # we are above the canopy
            d = np.array([st.pos[0] - p[0], 0.0, st.pos[2] - p[2]], dtype=np.float32)
            dist = float(np.linalg.norm(d))
            want = float(self.props.radius[i]) + c.radius
            if dist < 1e-4:
                d = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                dist = 1e-4
            push = (want - dist) / dist
            if push > 0.0:
                st.pos[0] += d[0] * push
                st.pos[2] += d[2] * push
                n = mth.norm(d)
                into = float(np.dot(st.vel, n))
                if into < 0.0 and st.state != MoveState.CLIMBING:
                    st.vel -= n * into       # cancel the velocity into the trunk

    def _resolve_ground(self, st: CharacterState) -> float | None:
        """Push the character out of the ground. Returns impact speed on landing."""
        c = self.cfg
        gh = float(self.terrain.height_at(st.pos[0], st.pos[2]))
        if st.state == MoveState.SWIMMING:
            st.pos[1] = max(st.pos[1], gh + 0.1)
            return None
        if st.state == MoveState.CLIMBING:
            st.pos[1] = max(st.pos[1], gh)
            return None
        if st.pos[1] <= gh + 1e-4:
            impact = float(st.vel[1])
            st.pos[1] = gh
            st.vel[1] = 0.0
            was_air = st.state != MoveState.GROUNDED
            st.state = MoveState.GROUNDED
            return impact if was_air else None
        # Snap down small drops so running over terraces does not launch us.
        if st.state == MoveState.AIRBORNE and st.vel[1] <= 0.0 and st.pos[1] - gh < c.step_height:
            st.pos[1] = gh
            impact = float(st.vel[1])
            st.vel[1] = 0.0
            st.state = MoveState.GROUNDED
            return impact if impact < -4.0 else None
        return None

    @staticmethod
    def _turn_toward(current: float, target: float, rate: float, dt: float) -> float:
        diff = (target - current + np.pi) % (2.0 * np.pi) - np.pi
        step = np.clip(diff, -rate * dt, rate * dt)
        return float((current + step + np.pi) % (2.0 * np.pi) - np.pi)
