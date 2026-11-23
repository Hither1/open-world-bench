"""Dynamic rigid bodies: boulders, crates, logs, metal slabs, bombs, arrows
and dropped items.

Bodies are simple AABBs integrated against the voxel grid.  They exist so the
world can be *acted on* rather than only walked through: roll a boulder onto a
pressure plate, float a log across a river, drag a metal slab with the magnet
rune, freeze one in stasis and beat stored momentum into it, or blow a cracked
wall open with a bomb.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from ..core import blocks as B
from . import collide as C


@dataclass
class BodyKind:
    name: str
    half: tuple[float, float, float]
    mass: float
    color: tuple[int, int, int]
    restitution: float = 0.1
    friction: float = 0.86
    floats: bool = False
    conductive: bool = False
    flammable: bool = False
    pushable: bool = True
    magnetic: bool = False
    drops: tuple[tuple[str, int], ...] = ()
    semantic: int = B.SEM_STRUCTURE


# Cubic bodies render as box impostors in the policy view, not discs.
BOX_KINDS = frozenset(("crate", "metal_slab"))

KINDS: dict[str, BodyKind] = {
    "boulder": BodyKind("boulder", (0.55, 0.55, 0.55), 420.0, (124, 122, 118),
                        restitution=0.05, friction=0.80,
                        drops=(("stone", 3), ("flint", 1)), semantic=B.SEM_ROCK),
    "crate": BodyKind("crate", (0.45, 0.45, 0.45), 55.0, (172, 132, 82),
                      restitution=0.12, friction=0.90, floats=True,
                      flammable=True, drops=(("wood", 2),)),
    "log": BodyKind("log", (0.55, 0.35, 0.55), 70.0, (118, 84, 52),
                    restitution=0.08, friction=0.88, floats=True,
                    flammable=True, drops=(("wood", 3),), semantic=B.SEM_TREE),
    "metal_slab": BodyKind("metal_slab", (0.5, 0.5, 0.5), 320.0, (158, 164, 172),
                           restitution=0.05, friction=0.82, conductive=True,
                           magnetic=True, drops=(("metal_scrap", 2),),
                           semantic=B.SEM_METAL),
    "bomb": BodyKind("bomb", (0.25, 0.25, 0.25), 9.0, (48, 48, 56),
                     restitution=0.35, friction=0.9, magnetic=True,
                     semantic=B.SEM_HAZARD),
    "arrow": BodyKind("arrow", (0.1, 0.1, 0.1), 0.6, (216, 208, 186),
                      restitution=0.0, friction=0.0, semantic=B.SEM_ITEM),
    "item": BodyKind("item", (0.22, 0.22, 0.22), 2.0, (236, 206, 92),
                     restitution=0.2, friction=0.9, floats=True,
                     semantic=B.SEM_ITEM),
}


@dataclass
class Body:
    id: int
    kind: str
    pos: np.ndarray
    vel: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))
    ttl: float = -1.0
    burning: float = 0.0
    frozen: float = 0.0                 # stasis time remaining
    stored: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))
    held: bool = False                  # carried by the magnet rune
    resting: bool = False
    alive: bool = True
    data: dict = field(default_factory=dict)

    @property
    def spec(self) -> BodyKind:
        return KINDS[self.kind]

    @property
    def half(self) -> np.ndarray:
        return np.array(self.spec.half, dtype=np.float64)

    def center(self) -> np.ndarray:
        return self.pos

    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))


class BodySystem:
    """Fixed-capacity pool of dynamic bodies."""

    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng
        self.bodies: dict[int, Body] = {}
        self._next_id = itertools.count(1)

    # ------------------------------------------------------------------
    def add(self, kind: str, pos, vel=(0, 0, 0), **data) -> int:
        if kind not in KINDS:
            raise KeyError(f"unknown body kind {kind!r}")
        if len(self.bodies) >= self.cfg.max_bodies:
            self._evict()
        bid = next(self._next_id)
        self.bodies[bid] = Body(id=bid, kind=kind,
                                pos=np.asarray(pos, dtype=np.float64).copy(),
                                vel=np.asarray(vel, dtype=np.float64).copy(),
                                ttl=float(data.pop("ttl", -1.0)),
                                data=dict(data))
        return bid

    def _evict(self) -> None:
        """Drop the least interesting body (a resting arrow or dropped item)."""
        for bid, b in self.bodies.items():
            if b.kind in ("arrow", "item") and b.resting:
                del self.bodies[bid]
                return
        del self.bodies[next(iter(self.bodies))]

    def get(self, bid: int | None) -> Body | None:
        return self.bodies.get(int(bid)) if bid is not None else None

    def remove(self, bid: int) -> None:
        self.bodies.pop(int(bid), None)

    def near(self, pos, radius: float, kinds: tuple[str, ...] | None = None
             ) -> list[Body]:
        p = np.asarray(pos, dtype=np.float64)
        out = []
        r2 = radius * radius
        for b in self.bodies.values():
            if kinds and b.kind not in kinds:
                continue
            if float(((b.pos - p) ** 2).sum()) <= r2:
                out.append(b)
        return out

    # ------------------------------------------------------------------
    def step(self, world, dt: float, ctx: dict | None = None) -> list[dict]:
        """Integrate every body one tick; returns events (explosions, hits...)."""
        ctx = ctx or {}
        cfg = self.cfg
        events: list[dict] = []
        for bid, b in list(self.bodies.items()):
            if not b.alive:
                del self.bodies[bid]
                continue
            if b.ttl > 0:
                b.ttl -= dt
                if b.ttl <= 0:
                    events.extend(self._expire(world, b))
                    continue
            if b.frozen > 0:
                b.frozen -= dt
                if b.frozen <= 0:
                    # stasis releases everything it absorbed at once
                    b.vel += b.stored
                    b.stored[:] = 0.0
                    events.append(dict(type="stasis_release", body=b.id,
                                       impulse=float(np.linalg.norm(b.vel))))
                continue
            if b.held:
                continue
            if b.burning > 0:
                b.burning -= dt
                if b.burning <= 0 and b.spec.flammable:
                    b.alive = False
                    events.append(dict(type="body_burned", body=b.id,
                                       pos=tuple(b.pos)))
                    continue

            half = b.half
            blk_here = world.get_f(b.pos[0], b.pos[1], b.pos[2])
            in_water = blk_here == B.WATER
            in_lava = blk_here == B.LAVA

            if in_water:
                if b.spec.floats:
                    # buoyancy toward the surface with heavy drag
                    surf = world.water_height[
                        int(np.clip(b.pos[0], 0, world.size_x - 1)),
                        int(np.clip(b.pos[2], 0, world.size_z - 1))] + 1.0
                    b.vel[1] += np.clip(surf - b.pos[1], -1.0, 1.0) * 9.0 * dt
                    b.vel *= (1.0 - 2.4 * dt)
                else:
                    b.vel[1] += cfg.gravity * 0.25 * dt
                    b.vel *= (1.0 - 1.8 * dt)
            else:
                b.vel[1] += cfg.gravity * dt
                b.vel[1] = max(b.vel[1], cfg.max_fall_speed)
            if in_lava and b.spec.flammable and b.burning <= 0:
                b.burning = 4.0

            old = b.pos.copy()
            new, hit = C.move_and_collide(world, b.pos, half, b.vel * dt, 0.0)
            b.pos = new
            if hit[1]:
                if b.vel[1] < -4.0 and b.spec.restitution > 0:
                    b.vel[1] = -b.vel[1] * b.spec.restitution
                else:
                    b.vel[1] = 0.0
                b.vel[0] *= b.spec.friction
                b.vel[2] *= b.spec.friction
            if hit[0]:
                b.vel[0] = -b.vel[0] * b.spec.restitution
            if hit[2]:
                b.vel[2] = -b.vel[2] * b.spec.restitution

            if b.kind == "arrow" and (hit.any()):
                events.extend(self._arrow_impact(world, b, old))
                continue

            moved = float(np.linalg.norm(b.pos - old))
            b.resting = moved < 1e-3 and abs(b.vel[1]) < 0.05
            if b.resting:
                b.vel[0] *= 0.6
                b.vel[2] *= 0.6
                if abs(b.vel[0]) < 0.02:
                    b.vel[0] = 0.0
                if abs(b.vel[2]) < 0.02:
                    b.vel[2] = 0.0
        return events

    # ------------------------------------------------------------------
    def _expire(self, world, b: Body) -> list[dict]:
        if b.kind == "bomb":
            return self.explode(world, b.pos, radius=float(b.data.get("radius", 3.2)),
                                power=float(b.data.get("power", 14.0)), source=b.id)
        b.alive = False
        self.bodies.pop(b.id, None)
        return []

    def _arrow_impact(self, world, b: Body, old: np.ndarray) -> list[dict]:
        ev = [dict(type="arrow_hit_world", pos=tuple(b.pos),
                   ammo=b.data.get("ammo", "arrow"), body=b.id)]
        ammo = b.data.get("ammo", "arrow")
        vx, vy, vz = (int(np.floor(b.pos[0])), int(np.floor(b.pos[1])),
                      int(np.floor(b.pos[2])))
        if ammo == "arrow_fire":
            ev.append(dict(type="ignite", pos=(vx, vy, vz)))
        elif ammo == "arrow_ice":
            ev.append(dict(type="freeze", pos=(vx, vy, vz)))
        elif ammo == "arrow_shock":
            ev.append(dict(type="shock", pos=(vx, vy, vz), power=6.0))
        b.alive = False
        self.bodies.pop(b.id, None)
        return ev

    # ------------------------------------------------------------------
    def explode(self, world, pos, radius: float = 3.2, power: float = 14.0,
                source: int | None = None) -> list[dict]:
        """Blast: carve blocks, launch bodies, and report for damage handling."""
        p = np.asarray(pos, dtype=np.float64)
        removed = world.destroy_sphere(p, radius)
        for b in self.bodies.values():
            if b.id == source or not b.alive:
                continue
            d = b.pos - p
            dist = float(np.linalg.norm(d))
            if dist > radius * 2.4:
                continue
            dirv = d / max(dist, 0.3)
            mag = power * 18.0 / max(b.spec.mass, 1.0) \
                * max(0.0, 1.0 - dist / (radius * 2.4))
            if b.frozen > 0:
                b.stored += dirv * mag
            else:
                b.vel += dirv * mag
                b.vel[1] += mag * 0.4
            if b.spec.flammable:
                b.burning = max(b.burning, 3.0)
        if source is not None:
            self.bodies.pop(source, None)
        return [dict(type="explosion", pos=tuple(p), radius=float(radius),
                     power=float(power), removed=len(removed),
                     removed_blocks=removed)]

    def impulse(self, pos, radius: float, power: float) -> None:
        p = np.asarray(pos, dtype=np.float64)
        for b in self.bodies.values():
            d = b.pos - p
            dist = float(np.linalg.norm(d))
            if dist > radius:
                continue
            dirv = d / max(dist, 0.25)
            mag = power * 10.0 / max(b.spec.mass, 1.0) * (1.0 - dist / radius)
            if b.frozen > 0:
                b.stored += dirv * mag
            else:
                b.vel += dirv * mag

    # ------------------------------------------------------------------
    # rune interactions
    # ------------------------------------------------------------------
    def freeze(self, bid: int, duration: float = 8.0) -> bool:
        b = self.get(bid)
        if b is None:
            return False
        b.frozen = float(duration)
        b.stored[:] = 0.0
        b.vel[:] = 0.0
        return True

    def add_stored_velocity(self, bid: int, dv) -> bool:
        """Feed velocity straight into a body, or into its stasis buffer.

        Used by melee hits: the rune's whole point is that blows landed on a
        frozen object are banked and released together, so a boulder far too
        heavy to budge can be launched by hitting it a few times.
        """
        b = self.get(bid)
        if b is None:
            return False
        dv = np.asarray(dv, dtype=np.float64)
        if b.frozen > 0:
            b.stored += dv
        else:
            b.vel += dv * (60.0 / max(b.spec.mass, 1.0))
        return True

    def hold(self, bid: int) -> bool:
        b = self.get(bid)
        if b is None or not b.spec.magnetic:
            return False
        b.held = True
        b.vel[:] = 0.0
        return True

    def move_held(self, world, bid: int, target: np.ndarray,
                  max_speed: float = 7.0, dt: float = 0.1) -> None:
        b = self.get(bid)
        if b is None or not b.held:
            return
        d = np.asarray(target, dtype=np.float64) - b.pos
        dist = float(np.linalg.norm(d))
        if dist < 1e-4:
            return
        stepv = d / dist * min(max_speed * dt, dist)
        new, _ = C.move_and_collide(world, b.pos, b.half, stepv, 0.0)
        b.vel = (new - b.pos) / max(dt, 1e-5)
        b.pos = new

    def release(self, bid: int, velocity=(0, 0, 0)) -> None:
        b = self.get(bid)
        if b is None:
            return
        b.held = False
        b.vel = b.vel * 0.3 + np.asarray(velocity, dtype=np.float64)

    # ------------------------------------------------------------------
    def render_list(self) -> list[dict]:
        """Impostor descriptors for the renderer."""
        out = []
        for b in self.bodies.values():
            sp = b.spec
            col = sp.color
            if b.burning > 0:
                col = (248, 150, 60)
            elif b.frozen > 0:
                col = (150, 205, 240)
            r = float(max(sp.half[0], sp.half[2]))
            out.append(dict(pos=(float(b.pos[0]), float(b.pos[1]),
                                 float(b.pos[2])),
                            radius=r, height=float(sp.half[1] * 2.0),
                            color=col, semantic=sp.semantic,
                            emissive=b.burning > 0,
                            **({"shape": "box"} if sp.name in BOX_KINDS else {})))
        return out

    def snapshot(self) -> list[dict]:
        return [dict(id=b.id, kind=b.kind, pos=tuple(float(v) for v in b.pos),
                     vel=tuple(float(v) for v in b.vel), frozen=b.frozen,
                     burning=b.burning, held=b.held)
                for b in self.bodies.values()]
