"""Creatures: monsters, wildlife and the few friendly NPCs.

Behaviour is a small finite state machine (idle / patrol / chase / attack /
flee / stunned) over line-of-sight perception.  It is deliberately legible:
the point of combat here is resource and durability management -- which weapon
you brought, whether you are holding metal in a storm, whether you remembered
to cook before picking a fight -- not twitch reflexes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from ..core import blocks as B
from ..physics import collide as C

IDLE, PATROL, CHASE, ATTACK, FLEE, STUNNED, DEAD = (
    "idle", "patrol", "chase", "attack", "flee", "stunned", "dead")


@dataclass
class NpcKind:
    name: str
    health: float
    attack: float
    speed: float
    aggro: float                   # detection radius
    reach: float                   # melee range
    hostile: bool = True
    color: tuple[int, int, int] = (190, 90, 80)
    radius: float = 0.45
    height: float = 1.6
    drops: tuple[tuple[str, int, float], ...] = ()
    conductive: bool = False
    flammable: bool = True
    nocturnal: bool = False
    ranged: bool = False
    element: str = ""              # "", "fire", "ice", "electric"
    flees: bool = False
    semantic: int = B.SEM_MONSTER
    xp: float = 1.0


KINDS: dict[str, NpcKind] = {}


def _reg(k: NpcKind) -> NpcKind:
    KINDS[k.name] = k
    return k


# ----------------------------- monsters -----------------------------------
_reg(NpcKind("bokoblin", 12.0, 3.0, 3.1, 16.0, 1.9, color=(176, 84, 64),
             drops=(("monster_horn", 1, 0.8), ("monster_fang", 1, 0.4),
                    ("club_boko", 1, 0.35), ("meat_raw", 1, 0.2)), xp=1.0))
_reg(NpcKind("moblin", 28.0, 6.0, 2.8, 18.0, 2.4, color=(140, 96, 58),
             radius=0.6, height=2.4,
             drops=(("monster_horn", 2, 0.9), ("monster_guts", 1, 0.5),
                    ("spear_wood", 1, 0.3)), xp=2.5))
_reg(NpcKind("stalker", 16.0, 4.5, 4.2, 20.0, 2.0, color=(216, 212, 198),
             nocturnal=True, flammable=False,
             drops=(("monster_fang", 2, 0.8), ("flint", 1, 0.3)), xp=1.8))
_reg(NpcKind("chuchu", 8.0, 2.0, 1.5, 11.0, 1.6, color=(126, 196, 150),
             radius=0.5, height=1.1,
             drops=(("monster_guts", 1, 0.9),), xp=0.6))
_reg(NpcKind("fire_chuchu", 10.0, 3.5, 1.5, 12.0, 1.6, color=(224, 118, 62),
             radius=0.5, height=1.1, element="fire", flammable=False,
             drops=(("fire_gland", 1, 0.85), ("monster_guts", 1, 0.4)), xp=1.2))
_reg(NpcKind("ice_chuchu", 10.0, 3.0, 1.4, 12.0, 1.6, color=(140, 204, 232),
             radius=0.5, height=1.1, element="ice",
             drops=(("ice_gland", 1, 0.85), ("monster_guts", 1, 0.4)), xp=1.2))
_reg(NpcKind("electro_chuchu", 10.0, 3.5, 1.5, 12.0, 1.6, color=(226, 206, 92),
             radius=0.5, height=1.1, element="electric", conductive=True,
             drops=(("electro_gland", 1, 0.85), ("monster_guts", 1, 0.4)),
             xp=1.2))
_reg(NpcKind("guardian", 70.0, 9.0, 1.6, 26.0, 3.0, color=(126, 118, 150),
             radius=0.9, height=2.6, conductive=True, flammable=False,
             ranged=True,
             drops=(("guardian_core", 1, 1.0), ("metal_scrap", 3, 0.8),
                    ("crystal", 1, 0.5)), xp=10.0))

# ----------------------------- wildlife ------------------------------------
_reg(NpcKind("boar", 9.0, 2.0, 3.6, 13.0, 1.6, hostile=False, flees=True,
             color=(122, 88, 68), semantic=B.SEM_ANIMAL,
             drops=(("meat_raw", 2, 0.95), ("hide", 1, 0.4)), xp=0.5))
_reg(NpcKind("deer", 7.0, 0.0, 4.6, 18.0, 1.4, hostile=False, flees=True,
             color=(168, 134, 94), semantic=B.SEM_ANIMAL, height=1.8,
             drops=(("meat_raw", 2, 0.95), ("hide", 1, 0.6)), xp=0.5))
_reg(NpcKind("goat", 8.0, 1.0, 3.8, 14.0, 1.4, hostile=False, flees=True,
             color=(206, 198, 184), semantic=B.SEM_ANIMAL,
             drops=(("meat_raw", 1, 0.9), ("hide", 1, 0.5)), xp=0.5))
_reg(NpcKind("bird", 3.0, 0.0, 5.0, 14.0, 1.0, hostile=False, flees=True,
             color=(230, 226, 214), radius=0.25, height=0.5,
             semantic=B.SEM_ANIMAL,
             drops=(("feather", 2, 0.95), ("egg", 1, 0.3)), xp=0.3))
_reg(NpcKind("fish", 3.0, 0.0, 3.0, 9.0, 1.0, hostile=False, flees=True,
             color=(120, 170, 200), radius=0.25, height=0.4,
             semantic=B.SEM_ANIMAL, drops=(("fish_raw", 1, 1.0),), xp=0.3))
_reg(NpcKind("frog", 3.0, 0.0, 2.4, 8.0, 1.0, hostile=False, flees=True,
             color=(110, 170, 110), radius=0.25, height=0.4,
             semantic=B.SEM_ANIMAL, drops=(("monster_guts", 1, 0.5),), xp=0.2))
_reg(NpcKind("crab", 4.0, 1.0, 1.8, 7.0, 1.2, hostile=False, flees=True,
             color=(206, 112, 88), radius=0.3, height=0.5,
             semantic=B.SEM_ANIMAL, drops=(("meat_raw", 1, 0.7),), xp=0.2))

_reg(NpcKind("horse", 30.0, 0.0, 4.2, 9.0, 1.6, hostile=False, flees=False,
             color=(120, 84, 56), semantic=B.SEM_ANIMAL, radius=0.55, height=1.7,
             xp=0.0))

# ----------------------------- friendly ------------------------------------
_reg(NpcKind("merchant", 20.0, 0.0, 1.4, 10.0, 2.0, hostile=False,
             color=(96, 132, 196), semantic=B.SEM_NPC, height=1.8, xp=0.0))
_reg(NpcKind("traveller", 18.0, 0.0, 1.6, 10.0, 2.0, hostile=False,
             color=(132, 164, 96), semantic=B.SEM_NPC, height=1.8, xp=0.0))


@dataclass
class Npc:
    id: int
    kind: str
    pos: np.ndarray
    vel: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))
    yaw: float = 0.0
    health: float = 10.0
    state: str = IDLE
    home: tuple[float, float] = (0.0, 0.0)
    alive: bool = True
    cooldown: float = 0.0
    burning: float = 0.0
    frozen: float = 0.0
    stun: float = 0.0
    patrol_target: np.ndarray | None = None
    aggro_timer: float = 0.0
    data: dict = field(default_factory=dict)

    @property
    def spec(self) -> NpcKind:
        return KINDS[self.kind]

    @property
    def hostile(self) -> bool:
        return self.spec.hostile

    @property
    def present(self) -> bool:
        """Alive, out of its den, and not carrying a rider."""
        return self.alive and not self.data.get("denned") and self.data.get("rider") is None

    @property
    def conductive(self) -> bool:
        return self.spec.conductive

    def hurt(self, amount: float) -> bool:
        """Apply damage; returns True if this killed it."""
        self.health -= float(amount)
        self.stun = max(self.stun, 0.35)
        if self.health <= 0:
            self.health = 0.0
            self.alive = False
            self.state = DEAD
            return True
        if self.spec.flees or not self.spec.hostile:
            self.state = FLEE
            self.aggro_timer = 6.0
        else:
            self.state = CHASE
            self.aggro_timer = 10.0
        return False


class NpcSystem:
    """Spawns and ticks every creature in the world."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.npcs: list[Npc] = []
        self._next_id = itertools.count(1)

    # ------------------------------------------------------------------
    def spawn(self, kind: str, pos, home=None, **data) -> Npc:
        spec = KINDS[kind]
        n = Npc(id=next(self._next_id), kind=kind,
                pos=np.asarray(pos, dtype=np.float64).copy(),
                health=spec.health,
                home=tuple(home) if home is not None else (float(pos[0]),
                                                           float(pos[2])),
                data=dict(data))
        n.yaw = float(self.rng.uniform(0, 2 * np.pi))
        self.npcs.append(n)
        return n

    def clear(self) -> None:
        self.npcs.clear()

    @property
    def alive(self) -> list[Npc]:
        """Creatures present in the world (not resting in a den or ridden)."""
        return [n for n in self.npcs if n.present]

    def near(self, pos, radius: float, hostile_only: bool = False) -> list[Npc]:
        p = np.asarray(pos, dtype=np.float64)
        r2 = radius * radius
        out = []
        for n in self.npcs:
            if not n.present or (hostile_only and not n.hostile):
                continue
            if float(((n.pos - p) ** 2).sum()) <= r2:
                out.append(n)
        return out

    def by_id(self, nid: int) -> Npc | None:
        for n in self.npcs:
            if n.id == nid:
                return n
        return None

    # ------------------------------------------------------------------
    def step(self, world, agent, dt: float, ctx: dict | None = None
             ) -> list[dict]:
        ctx = ctx or {}
        is_night = bool(ctx.get("is_night", False))
        stealth = float(ctx.get("stealth", 0.0))
        ev: list[dict] = []
        ap = np.asarray(agent.pos, dtype=np.float64)

        for n in self.npcs:
            if not n.alive or n.data.get("rider") is not None:
                continue
            spec = n.spec
            den = n.data.get("den")
            if den is not None:
                # Monsters keep to their dens by day; all come out after dusk.
                out = is_night or bool(n.data.get("day_active"))
                if not out:
                    if n.data.get("denned"):
                        continue
                    home = np.asarray(den, float) - n.pos
                    home[1] = 0.0
                    far = float(np.linalg.norm(home))
                    if far < 1.5:
                        n.data["denned"] = True
                        n.state, n.patrol_target, n.aggro_timer = IDLE, None, 0.0
                        n.vel[:] = 0.0
                        ev.append(dict(type="npc_denned", npc=n.id, kind=n.kind))
                    else:
                        self._integrate(world, n, home / far, spec.speed * 0.8, dt)
                    continue
                if n.data.get("denned"):
                    n.data["denned"] = False
                    n.pos = np.asarray(den, float).copy()
                    n.state, n.patrol_target = IDLE, None
                    ev.append(dict(type="npc_emerged", npc=n.id, kind=n.kind))
            if n.data.get("route"):
                ev.extend(self._answer_whistle(world, n, dt))
                continue
            if n.frozen > 0:
                n.frozen -= dt
                continue
            if n.stun > 0:
                n.stun -= dt
            if n.burning > 0:
                n.burning -= dt
                if self.rng.random() < dt * 3.0:
                    if n.hurt(1.4):
                        ev.extend(self._die(n, world, cause="fire"))
                        continue
            # standing in fire or lava hurts
            blk = world.get_f(n.pos[0], n.pos[1] + 0.3, n.pos[2])
            if blk == B.LAVA:
                if n.hurt(9.0 * dt):
                    ev.extend(self._die(n, world, cause="lava"))
                    continue
            elif blk == B.FIRE and spec.flammable and n.burning <= 0:
                n.burning = 4.0

            d = ap - n.pos
            dist = float(np.linalg.norm(d))
            aggro = spec.aggro * (1.35 if (is_night and spec.nocturnal) else 1.0)
            aggro *= max(0.25, 1.0 - 0.6 * stealth)
            sees = (dist < aggro and agent.alive
                    and world.line_of_sight(n.pos + np.array([0, 1.0, 0]),
                                            ap + np.array([0, 1.2, 0])))

            if n.aggro_timer > 0:
                n.aggro_timer -= dt
            if sees:
                if spec.hostile:
                    n.state = ATTACK if dist <= spec.reach else CHASE
                    n.aggro_timer = 8.0
                elif spec.flees:
                    n.state = FLEE
                    n.aggro_timer = 5.0
            elif n.aggro_timer <= 0 and n.state in (CHASE, ATTACK, FLEE):
                n.state = PATROL

            move = np.zeros(3)
            speed = spec.speed
            if n.state == CHASE:
                move = d / max(dist, 1e-5)
            elif n.state == ATTACK:
                # close in, but keep out of the agent's own volume
                sign = 0.25 if dist > spec.reach * 0.75 else -0.45
                move = d / max(dist, 1e-5) * sign
                if n.cooldown <= 0 and dist <= spec.reach + 0.4:
                    ev.extend(self._attack(n, agent, world, ctx))
                    n.cooldown = 1.3
            elif n.state == FLEE:
                move = -d / max(dist, 1e-5)
                speed *= 1.25
                leash = n.data.get("leash")
                if leash is not None:
                    # Tethered animals bolt only as far as their lead allows.
                    back = np.array([n.home[0] - n.pos[0], 0.0, n.home[1] - n.pos[2]])
                    if float(np.linalg.norm(back)) > leash:
                        move = back / float(np.linalg.norm(back))
            elif n.state == PATROL or n.state == IDLE:
                if n.patrol_target is None or \
                        float(np.linalg.norm(n.patrol_target - n.pos)) < 1.2:
                    if self.rng.random() < 0.02 or n.patrol_target is None:
                        hx, hz = n.home
                        roam = float(n.data.get("roam", 7.0))
                        n.patrol_target = np.array(
                            [hx + self.rng.uniform(-roam, roam), n.pos[1],
                             hz + self.rng.uniform(-roam, roam)])
                    else:
                        n.patrol_target = None
                if n.patrol_target is not None:
                    pd = n.patrol_target - n.pos
                    pd[1] = 0
                    nn = float(np.linalg.norm(pd))
                    if nn > 0.5:
                        move = pd / nn
                        speed *= 0.45
            if n.cooldown > 0:
                n.cooldown -= dt
            if n.stun > 0:
                move *= 0.1

            self._integrate(world, n, move, speed, dt)
        # reap
        self.npcs = [n for n in self.npcs if n.alive or n.data.get("keep")]
        return ev

    # ------------------------------------------------------------------
    def _answer_whistle(self, world, n: Npc, dt: float) -> list[dict]:
        """A summoned horse canters along its walking route to the rider."""
        route = n.data["route"]
        while route and float(np.hypot(route[0][0] + .5 - n.pos[0], route[0][1] + .5 - n.pos[2])) < 0.6:
            route.pop(0)
        if not route:
            return self._settle_horse(n, "horse_arrived")
        x, z, _ = route[0]
        move = np.array([x + .5 - n.pos[0], 0.0, z + .5 - n.pos[2]])
        move /= max(float(np.linalg.norm(move)), 1e-6)
        speed = n.spec.speed * 1.6
        before = n.pos.copy()
        self._integrate(world, n, move, speed, dt, step_height=1.1, radius=0.4)
        moved = float(np.hypot(*(n.pos - before)[[0, 2]]))
        n.data["stall"] = n.data.get("stall", 0.0) + dt if moved < 0.15 * speed * dt else 0.0
        if n.data["stall"] > 3.0:
            return self._settle_horse(n, "horse_stuck")
        return []

    @staticmethod
    def _settle_horse(n: Npc, kind: str) -> list[dict]:
        n.data.pop("route", None)
        n.data.pop("stall", None)
        n.home = (float(n.pos[0]), float(n.pos[2]))
        n.data.update(roam=1.5, leash=3.0)
        n.state, n.patrol_target = IDLE, None
        n.vel[0] = n.vel[2] = 0.0
        return [dict(type=kind, npc=n.id, pos=[round(float(v), 3) for v in n.pos])]

    def _integrate(self, world, n: Npc, move: np.ndarray, speed: float,
                   dt: float, step_height: float = 0.7, radius: float | None = None) -> None:
        spec = n.spec
        # A route-following animal threads one-block trails with a slimmer box.
        r = spec.radius if radius is None else radius
        half = np.array([r, spec.height * 0.5, r])
        center = n.pos + np.array([0.0, spec.height * 0.5, 0.0])
        swim = world.get_f(n.pos[0], n.pos[1] + 0.3, n.pos[2]) == B.WATER
        if float(np.linalg.norm(move)) > 1e-4:
            n.yaw = float(np.arctan2(move[2], move[0]))
        n.vel[0] += (move[0] * speed - n.vel[0]) * min(1.0, 7.0 * dt)
        n.vel[2] += (move[2] * speed - n.vel[2]) * min(1.0, 7.0 * dt)
        if n.data.get("aquatic"):
            ahead = n.pos + n.vel * dt + np.sign(n.vel) * np.array([spec.radius, 0.0, spec.radius])
            if world.get_f(ahead[0], n.pos[1] + 0.3, ahead[2]) != B.WATER:
                n.vel[0] = n.vel[2] = 0.0
                n.patrol_target = None
        if spec.name == "fish":
            n.vel[1] += ((0.0 if swim else -2.0) - n.vel[1]) * min(1.0, 4.0 * dt)
        elif swim:
            n.vel[1] += (1.2 - n.vel[1]) * min(1.0, 4.0 * dt)
        else:
            n.vel[1] += self.cfg.gravity * dt
            n.vel[1] = max(n.vel[1], self.cfg.max_fall_speed)
        new, hit = C.move_and_collide(world, center, half, n.vel * dt,
                                      step_height=step_height)
        if hit[1]:
            n.vel[1] = 0.0
        if hit[0]:
            n.vel[0] = 0.0
        if hit[2]:
            n.vel[2] = 0.0
        n.pos = new - np.array([0.0, spec.height * 0.5, 0.0])

    # ------------------------------------------------------------------
    def _attack(self, n: Npc, agent, world, ctx) -> list[dict]:
        spec = n.spec
        dmg = spec.attack
        if ctx.get("is_night"):
            dmg *= 1.15
        taken = agent.hurt(dmg, "melee")
        ev = [dict(type="npc_attack", npc=n.id, kind=n.kind,
                   damage=float(taken))]
        if spec.element == "fire":
            ev.append(dict(type="ignite_agent"))
        elif spec.element == "electric":
            resist = agent.shock_resist()
            extra = self.cfg.shock_damage * max(0.0, 1.0 - 0.8 * resist)
            if agent.inventory.has_metal_equipped():
                extra *= 1.8
                dropped = agent.inventory.unequip_metal()
                ev.append(dict(type="shocked", target="agent", dropped=dropped,
                               damage=float(extra)))
            if extra > 0:
                agent.hurt(extra, "shock")
        elif spec.element == "ice":
            agent.add_effect("chilled", 1.0, 3.0)
            ev.append(dict(type="chilled"))
        if not agent.alive:
            ev.append(dict(type="death", cause=f"killed by {n.kind}"))
        return ev

    def _die(self, n: Npc, world, cause: str = "") -> list[dict]:
        n.alive = False
        n.state = DEAD
        drops: list[tuple[str, int]] = []
        for item, count, prob in n.spec.drops:
            if self.rng.random() < prob:
                drops.append((item, int(count)))
        return [dict(type="npc_died", npc=n.id, kind=n.kind, cause=cause,
                     pos=tuple(float(v) for v in n.pos), drops=drops,
                     xp=n.spec.xp)]

    def kill(self, n: Npc, world, cause: str = "") -> list[dict]:
        return self._die(n, world, cause)

    # ------------------------------------------------------------------
    def render_list(self, near=None, radius: float = 90.0) -> list[dict]:
        out = []
        p = np.asarray(near, dtype=np.float64) if near is not None else None
        for n in self.npcs:
            if not n.alive or n.data.get("denned"):
                continue
            if p is not None and float(np.linalg.norm(n.pos - p)) > radius:
                continue
            spec = n.spec
            col = spec.color
            if n.burning > 0:
                col = (250, 150, 60)
            elif n.frozen > 0:
                col = (160, 212, 240)
            out.append(dict(
                pos=(float(n.pos[0]), float(n.pos[1] + spec.height * 0.5),
                     float(n.pos[2])),
                radius=spec.radius, height=spec.height, color=col,
                kind=n.kind, yaw=n.yaw, npc=n.id,
                speed=float(np.hypot(n.vel[0], n.vel[2])),
                ridden=n.data.get("rider") is not None,
                state=n.state,
                semantic=spec.semantic,
                emissive=spec.element == "fire" or n.burning > 0))
        return out

    def snapshot(self, near=None, radius: float = 40.0) -> list[dict]:
        p = np.asarray(near, dtype=np.float64) if near is not None else None
        out = []
        for n in self.npcs:
            if not n.present:
                continue
            if p is not None and float(np.linalg.norm(n.pos - p)) > radius:
                continue
            out.append(dict(id=n.id, kind=n.kind, state=n.state,
                            pos=tuple(float(v) for v in n.pos),
                            health=float(n.health),
                            hostile=bool(n.spec.hostile),
                            dist=float(np.linalg.norm(n.pos - p)) if p is not None
                            else 0.0))
        out.sort(key=lambda d: d["dist"])
        return out
