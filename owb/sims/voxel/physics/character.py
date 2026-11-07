"""The agent's character controller.

This is where the game *feels* like what it is modelled on: climbing any
surface at the cost of stamina, gliding from height, swimming, sprinting, and
the way rain, cold and a dead stamina bar turn a trivial route into a real
problem.  Every mechanic here is something a task can be built around.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import blocks as B
from ..core import items as I
from ..core.config import SimConfig
from ..entities.agent import AgentState
from . import collide as C


@dataclass
class MoveIntent:
    forward: float = 0.0       # -1 back .. +1 forward, agent-local
    strafe: float = 0.0        # -1 left .. +1 right
    sprint: bool = False
    jump: bool = False
    climb: bool = False
    glide: bool = False
    crouch: bool = False


class CharacterController:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.half = np.array([cfg.agent_radius, cfg.agent_height * 0.5,
                              cfg.agent_radius], dtype=np.float64)

    # ------------------------------------------------------------------
    def step(self, world, agent: AgentState, intent: MoveIntent, dt: float,
             ctx: dict | None = None) -> list[dict]:
        """Advance the agent one tick. Returns a list of event dicts."""
        cfg = self.cfg
        ctx = ctx or {}
        ev: list[dict] = []
        if not agent.alive:
            return ev
        agent.sprinting = False

        center = np.asarray(agent.pos, dtype=np.float64) + \
            np.array([0.0, cfg.agent_height * 0.5, 0.0])
        vel = np.asarray(agent.vel, dtype=np.float64).copy()
        half = self.half

        # ---------------- medium ------------------------------------------
        feet = np.array([agent.pos[0], agent.pos[1] + 0.2, agent.pos[2]])
        head = np.array([agent.pos[0], agent.pos[1] + cfg.agent_height - 0.2,
                         agent.pos[2]])
        feet_blk = world.get_f(*feet)
        head_blk = world.get_f(*head)
        in_water = feet_blk == B.WATER or head_blk == B.WATER
        submerged = head_blk == B.WATER
        grounded = C.on_ground(world, center, half)

        raining = bool(ctx.get("raining", False))
        wind = np.asarray(ctx.get("wind", (0.0, 0.0, 0.0)), dtype=np.float64)

        # ---------------- local frame -------------------------------------
        fwd = np.array([np.cos(agent.yaw), 0.0, np.sin(agent.yaw)])
        right = np.array([-np.sin(agent.yaw), 0.0, np.cos(agent.yaw)])
        wish = fwd * float(intent.forward) + right * float(intent.strafe)
        wmag = float(np.linalg.norm(wish))
        if wmag > 1.0:
            wish /= wmag
            wmag = 1.0

        # ---------------- climbing ----------------------------------------
        can_climb, wall_n, wall_blk = C.wall_contact(world, center, half)
        climbing = False
        if (intent.climb and can_climb and not in_water and not agent.exhausted
                and agent.stamina > 0.0):
            climbing = True
            cost = cfg.climb_cost * agent.climb_efficiency()
            if raining and wall_blk != B.VINE:
                cost *= cfg.rain_climb_penalty
            up = float(intent.forward)
            if up <= 0.0:
                cost *= 0.35          # hanging on is cheaper than climbing
            if not agent.spend_stamina(cost * dt):
                climbing = False
                ev.append(dict(type="climb_fail", reason="stamina"))
            else:
                vel[:] = 0.0
                vel[1] = cfg.climb_speed * up
                lateral = right * float(intent.strafe) * cfg.climb_speed * 0.6
                vel[0] += lateral[0]
                vel[2] += lateral[2]
                # stick to the wall
                vel -= wall_n * 0.6
                if raining and wall_blk != B.VINE:
                    rng = ctx.get("rng")
                    p = cfg.slip_chance_in_rain * dt * 60.0
                    if rng is not None and float(rng.random()) < p:
                        climbing = False
                        vel[1] = -2.0
                        ev.append(dict(type="slip", wet=True))

        # ---------------- gliding -----------------------------------------
        gliding = False
        if (not climbing and not grounded and not in_water and intent.glide
                and agent.has_glider() and agent.stamina > 0.0
                and not agent.exhausted and vel[1] < 1.5):
            if agent.spend_stamina(cfg.glide_cost * dt):
                gliding = True
                vel[1] = max(vel[1], cfg.glide_sink)
                glide_dir = fwd if wmag < 0.1 else wish
                speed = cfg.glide_speed * agent.speed_mult()
                vel[0] = glide_dir[0] * speed + wind[0] * 1.4
                vel[2] = glide_dir[2] * speed + wind[2] * 1.4
                lift = float(ctx.get("updraft", 0.0))
                if lift > 0:
                    vel[1] += lift
                    ev.append(dict(type="updraft", strength=lift))
            else:
                ev.append(dict(type="glide_fail", reason="stamina"))

        # ---------------- swimming ----------------------------------------
        if in_water and not climbing:
            speed = cfg.swim_speed * agent.speed_mult()
            vel[0] += (wish[0] * speed - vel[0]) * min(1.0, 6.0 * dt)
            vel[2] += (wish[2] * speed - vel[2]) * min(1.0, 6.0 * dt)
            target_y = 0.9 if submerged else 0.0
            if intent.jump:
                target_y = 2.2
            elif intent.crouch:
                target_y = -1.6
            vel[1] += (target_y - vel[1]) * min(1.0, 5.0 * dt)
            if agent.spend_stamina(cfg.swim_cost * dt * (1.0 + wmag)):
                pass
            else:
                vel[1] -= 1.2
                if submerged:
                    agent.hurt(cfg.drown_damage * dt, "drown")
                    ev.append(dict(type="drowning"))
            agent.wetness = 1.0

        # ---------------- walking / falling --------------------------------
        elif not climbing and not gliding:
            if grounded:
                sprint = (intent.sprint and wmag > 0.1 and not agent.exhausted
                          and agent.stamina > 0)
                speed = (cfg.run_speed if sprint else cfg.walk_speed) \
                    * agent.speed_mult()
                if sprint and not agent.spend_stamina(cfg.sprint_cost * dt):
                    speed = cfg.walk_speed
                    sprint = False
                agent.sprinting = sprint
                ground_blk = world.get(int(np.floor(agent.pos[0])),
                                       int(np.floor(agent.pos[1] - 0.2)),
                                       int(np.floor(agent.pos[2])))
                slip = float(B.SLIPPERY[ground_blk])
                accel = min(1.0, (14.0 * (1.0 - slip) + 1.2) * dt)
                vel[0] += (wish[0] * speed - vel[0]) * accel
                vel[2] += (wish[2] * speed - vel[2]) * accel
                if intent.jump:
                    vel[1] = cfg.jump_speed
                    grounded = False
                    ev.append(dict(type="jump"))
            else:
                agent.sprinting = False
                air = min(1.0, 2.6 * dt)
                speed = cfg.walk_speed * agent.speed_mult()
                vel[0] += (wish[0] * speed - vel[0]) * air
                vel[2] += (wish[2] * speed - vel[2]) * air
                vel[0] += wind[0] * 0.25 * dt
                vel[2] += wind[2] * 0.25 * dt

        # ---------------- gravity ------------------------------------------
        if not climbing and not in_water:
            if gliding:
                vel[1] = max(vel[1] + cfg.gravity * 0.08 * dt, cfg.glide_sink)
            else:
                vel[1] += cfg.gravity * dt
            vel[1] = max(vel[1], cfg.max_fall_speed)
        elif in_water:
            vel[1] = max(vel[1], -3.0)

        # ---------------- integrate ----------------------------------------
        before_y = vel[1]
        new_center, hit = C.move_and_collide(world, center, half, vel * dt,
                                             step_height=cfg.step_height)
        actual = new_center - center
        if dt > 1e-6:
            for ax in range(3):
                if hit[ax]:
                    vel[ax] = 0.0
        moved = float(np.linalg.norm(actual[[0, 2]]))
        agent.distance_travelled += moved

        now_grounded = C.on_ground(world, new_center, half)

        # ---------------- landing ------------------------------------------
        # water at the *landing* point, not at the start of the tick: diving
        # off a cliff into a lake is supposed to be free
        landed_wet = in_water or world.get_f(
            new_center[0], new_center[1] - cfg.agent_height * 0.5 + 0.2,
            new_center[2]) == B.WATER
        if now_grounded and not grounded and before_y < 0:
            if before_y < cfg.fall_damage_threshold and not landed_wet:
                dmg = (abs(before_y) - abs(cfg.fall_damage_threshold)) \
                    * cfg.fall_damage_scale
                taken = agent.hurt(dmg, "fall")
                ev.append(dict(type="fall_damage", amount=float(taken),
                               speed=float(before_y)))
            ev.append(dict(type="land", speed=float(before_y)))
            agent.air_time = 0.0
        if not now_grounded:
            agent.air_time += dt
        else:
            agent.air_time = 0.0

        # ---------------- write back ----------------------------------------
        agent.pos = (new_center - np.array([0.0, cfg.agent_height * 0.5, 0.0])
                     ).astype(np.float32)
        agent.vel = vel.astype(np.float32)
        agent.on_ground = bool(now_grounded)
        agent.in_water = bool(in_water)
        agent.submerged = bool(submerged)
        agent.climbing = bool(climbing)
        agent.gliding = bool(gliding)

        # ---------------- stamina regen -------------------------------------
        if now_grounded and not climbing and not gliding and not in_water \
                and not agent.sprinting:
            agent.regen_stamina(cfg.stamina_regen * dt)

        # ---------------- contact hazards ------------------------------------
        touching = C.box_block_ids(world, new_center, half)
        if B.LAVA in touching:
            agent.hurt(cfg.lava_damage * dt, "lava")
            ev.append(dict(type="burning", source="lava"))
        if B.FIRE in touching or B.CAMPFIRE_LIT in touching:
            agent.hurt(cfg.fire_damage * dt, "fire")
            agent.wetness = max(0.0, agent.wetness - 0.5 * dt)
            ev.append(dict(type="burning", source="fire"))
        if B.CACTUS in touching and moved > 0.01:
            agent.hurt(1.2 * dt, "spike")

        # ---------------- wetness ---------------------------------------------
        if raining and not submerged:
            agent.wetness = min(1.0, agent.wetness + 0.5 * dt)
        elif not in_water:
            agent.wetness = max(0.0, agent.wetness - 0.12 * dt)

        if not agent.alive:
            ev.append(dict(type="death"))
        return ev

    # ------------------------------------------------------------------
    def can_climb_here(self, world, agent: AgentState) -> bool:
        center = np.asarray(agent.pos, dtype=np.float64) + \
            np.array([0.0, self.cfg.agent_height * 0.5, 0.0])
        found, _, _ = C.wall_contact(world, center, self.half)
        return found

    def place_on_ground(self, world, agent: AgentState) -> None:
        """Drop the agent to the surface of its column (used at reset)."""
        x, z = float(agent.pos[0]), float(agent.pos[2])
        y = world.spawn_y(x, z)
        agent.pos = np.array([x, y + 0.05, z], dtype=np.float32)
        agent.vel[:] = 0.0
