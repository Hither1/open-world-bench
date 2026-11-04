"""Elemental systems: fire, wind, weather, lightning, conduction, temperature.

These are the systemic couplings that make the world feel reactive rather than
scripted.  Fire spreads downwind through grass and wood, burns out in rain, and
throws an updraft you can ride with the glider.  Thunderstorms hunt the tallest
conductive thing around -- which is you, if you are carrying a metal sword.
Electricity conducts through connected water.  Cold and heat are survivable
only with the right food or clothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import blocks as B
from ..core import items as I
from ..core.config import SimConfig
from ..world import biomes as Bi

WEATHERS = ("clear", "cloud", "rain", "storm", "snow", "fog")


@dataclass
class Climate:
    """Global time of day, weather and wind."""

    time: float = 0.3                 # 0..1 within the day
    day: int = 0
    weather: str = "clear"
    weather_ttl: float = 120.0
    wind: np.ndarray = field(
        default_factory=lambda: np.array([0.6, 0.0, 0.2], dtype=np.float32))
    wind_target: np.ndarray = field(
        default_factory=lambda: np.array([0.6, 0.0, 0.2], dtype=np.float32))
    flash: float = 0.0                # lightning screen flash, 0..1

    @property
    def is_night(self) -> bool:
        return self.time < 0.23 or self.time > 0.77

    @property
    def raining(self) -> bool:
        return self.weather in ("rain", "storm")

    def wind_speed(self) -> float:
        return float(np.linalg.norm(self.wind))


class ElementSystem:
    """Fire propagation, weather, lightning and body temperature."""

    MAX_FIRES = 420

    def __init__(self, cfg: SimConfig, rng):
        self.cfg = cfg
        self.rng = rng
        self.climate = Climate(time=cfg.start_time)
        #: (x, y, z) -> remaining burn time
        self.fires: dict[tuple[int, int, int], float] = {}
        self.shocked_cells: dict[tuple[int, int, int], float] = {}
        self.strike_cooldown = 6.0

    # ------------------------------------------------------------------
    def reset(self, time: float | None = None) -> None:
        self.fires.clear()
        self.shocked_cells.clear()
        self.climate = Climate(time=self.cfg.start_time if time is None else time)
        self.strike_cooldown = 6.0

    # ------------------------------------------------------------------
    # weather & time
    # ------------------------------------------------------------------
    def step_climate(self, dt: float, biome_climate: float = 0.0) -> list[dict]:
        cl = self.climate
        ev: list[dict] = []
        prev = cl.time
        cl.time = (cl.time + dt / max(self.cfg.day_length, 1e-3)) % 1.0
        if cl.time < prev:
            cl.day += 1
            ev.append(dict(type="new_day", day=cl.day))
        cl.flash = max(0.0, cl.flash - dt * 4.0)

        if self.cfg.wind_enabled:
            if self.rng.random() < dt * 0.05:
                ang = self.rng.uniform(0, 2 * np.pi)
                mag = float(self.rng.uniform(0.2, 2.4))
                if cl.weather == "storm":
                    mag *= 2.0
                cl.wind_target = np.array([np.cos(ang) * mag, 0.0,
                                           np.sin(ang) * mag], dtype=np.float32)
            cl.wind += (cl.wind_target - cl.wind) * min(1.0, 0.5 * dt)

        if not self.cfg.weather_enabled:
            return ev
        cl.weather_ttl -= dt
        if cl.weather_ttl <= 0:
            cold = biome_climate < -0.4
            if cl.weather in ("rain", "storm", "snow"):
                nxt = "cloud" if self.rng.random() < 0.7 else "clear"
            elif cl.weather == "clear":
                nxt = "cloud" if self.rng.random() < 0.55 else "clear"
            else:
                r = self.rng.random()
                if r < 0.38:
                    nxt = "snow" if cold else (
                        "storm" if self.rng.random() < self.cfg.thunder_prob
                        else "rain")
                elif r < 0.5:
                    nxt = "fog"
                else:
                    nxt = "clear"
            if nxt != cl.weather:
                ev.append(dict(type="weather", weather=nxt, was=cl.weather))
            cl.weather = nxt
            cl.weather_ttl = float(self.rng.uniform(90.0, 320.0))
        return ev

    # ------------------------------------------------------------------
    # fire
    # ------------------------------------------------------------------
    def ignite(self, world, x: int, y: int, z: int, force: bool = False) -> bool:
        """Set a voxel alight if it will burn."""
        if not self.cfg.fire_enabled:
            return False
        if len(self.fires) >= self.MAX_FIRES:
            return False
        x, y, z = int(x), int(y), int(z)
        blk = world.get(x, y, z)
        if blk == B.BEACON:
            world.set_block(x, y, z, B.BEACON_LIT)
            return True
        if blk == B.CAMPFIRE:
            world.set_block(x, y, z, B.CAMPFIRE_LIT)
            self.fires[(x, y, z)] = 1e6          # campfires do not burn out
            return True
        if blk == B.AIR:
            # look for fuel underneath, otherwise a short-lived flame
            below = world.get(x, y - 1, z)
            if B.FLAMMABILITY[below] <= 0 and not force:
                return False
            world.set_block(x, y, z, B.FIRE)
            self.fires[(x, y, z)] = self.cfg.fire_lifetime
            return True
        if B.FLAMMABILITY[blk] > 0 or force:
            world.set_block(x, y, z, B.FIRE)
            self.fires[(x, y, z)] = self.cfg.fire_lifetime
            return True
        return False

    def extinguish(self, world, x: int, y: int, z: int) -> None:
        key = (int(x), int(y), int(z))
        if key in self.fires:
            del self.fires[key]
            if world.get(*key) in (B.FIRE, B.CAMPFIRE_LIT):
                world.set_block(*key, B.ASH if world.get(*key) == B.FIRE
                                else B.CAMPFIRE)

    def step_fire(self, world, dt: float) -> list[dict]:
        if not self.cfg.fire_enabled or not self.fires:
            return []
        ev: list[dict] = []
        cl = self.climate
        wind = cl.wind
        wet = 1.0 if cl.raining else 0.0
        spread_p = self.cfg.fire_spread_rate * dt * (0.25 if wet else 1.0)
        rng = self.rng
        burned_out: list[tuple[int, int, int]] = []

        for cell, ttl in list(self.fires.items()):
            x, y, z = cell
            if world.get(x, y, z) not in (B.FIRE, B.CAMPFIRE_LIT):
                burned_out.append(cell)
                continue
            ttl -= dt * (3.0 if wet else 1.0)
            if ttl <= 0:
                burned_out.append(cell)
                ev.append(dict(type="fire_out", pos=cell))
                continue
            self.fires[cell] = ttl
            if len(self.fires) >= self.MAX_FIRES:
                continue
            # candidate neighbours, biased downwind and upward
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
                               (0, 1, 0), (0, -1, 0), (1, 1, 0), (-1, 1, 0),
                               (0, 1, 1), (0, 1, -1)):
                nx, ny, nz = x + dx, y + dy, z + dz
                if (nx, ny, nz) in self.fires:
                    continue
                blk = world.get(nx, ny, nz)
                flam = float(B.FLAMMABILITY[blk])
                if flam <= 0:
                    continue
                bias = 1.0
                if dx or dz:
                    w = wind[0] * dx + wind[2] * dz
                    bias = float(np.clip(1.0 + 0.9 * w, 0.15, 3.2))
                if dy > 0:
                    bias *= 1.5
                if rng.random() < spread_p * flam * bias:
                    world.set_block(nx, ny, nz, B.FIRE)
                    self.fires[(nx, ny, nz)] = self.cfg.fire_lifetime
                    ev.append(dict(type="fire_spread", pos=(nx, ny, nz)))

        for cell in burned_out:
            self.fires.pop(cell, None)
            if world.get(*cell) == B.FIRE:
                world.set_block(*cell, B.AIR)
                bx, by, bz = cell[0], cell[1] - 1, cell[2]
                below = world.get(bx, by, bz)
                if below in (B.GRASS, B.LEAVES, B.HAY, B.LOG, B.PLANK, B.BUSH):
                    world.set_block(bx, by, bz, B.ASH)
        if cl.raining and self.fires and rng.random() < dt * 0.9:
            for cell in list(self.fires)[:8]:
                if world.get(*cell) == B.FIRE:
                    self.fires.pop(cell, None)
                    world.set_block(*cell, B.AIR)
                    ev.append(dict(type="fire_doused", pos=cell))
        return ev

    def updraft_at(self, pos) -> float:
        """Thermal lift above a fire -- ride it with the glider."""
        if not self.fires:
            return 0.0
        p = np.asarray(pos, dtype=np.float32)
        best = 0.0
        for (x, y, z) in self.fires:
            dx, dz = p[0] - (x + 0.5), p[2] - (z + 0.5)
            dy = p[1] - y
            if abs(dx) > 3.0 or abs(dz) > 3.0 or dy < -1.0 or dy > 26.0:
                continue
            horiz = float(np.hypot(dx, dz))
            lift = self.cfg.updraft_strength * max(0.0, 1.0 - horiz / 3.0) \
                * max(0.25, 1.0 - dy / 26.0)
            best = max(best, lift)
        return float(best)

    def heat_at(self, world, pos, radius: float = 4.0) -> float:
        """Warmth from nearby flame and lava, 0..1."""
        p = np.asarray(pos, dtype=np.float32)
        warm = 0.0
        for (x, y, z) in self.fires:
            d = float(np.linalg.norm(p - np.array([x + 0.5, y + 0.5, z + 0.5])))
            if d < radius:
                warm = max(warm, 1.0 - d / radius)
        xi, zi = int(p[0]), int(p[2])
        for dx in (-2, 0, 2):
            for dz in (-2, 0, 2):
                if world.get(xi + dx, int(p[1]), zi + dz) == B.LAVA:
                    warm = max(warm, 0.9)
        return float(warm)

    def light_sources(self, world, near, radius: float = 26.0) -> list[dict]:
        """Point lights for the renderer: fires, lava pools, crystals."""
        p = np.asarray(near, dtype=np.float32)
        out: list[dict] = []
        for (x, y, z) in list(self.fires)[:40]:
            c = np.array([x + 0.5, y + 0.6, z + 0.5], dtype=np.float32)
            if float(np.linalg.norm(c - p)) < radius:
                out.append(dict(pos=tuple(c), radius=9.0, intensity=1.1,
                                color=(1.0, 0.68, 0.32)))
            if len(out) >= 8:
                break
        return out

    # ------------------------------------------------------------------
    # lightning
    # ------------------------------------------------------------------
    def step_lightning(self, world, agent, bodies, npcs, dt: float) -> list[dict]:
        """During a thunderstorm, strike the tallest conductive thing nearby.

        Carrying or wearing metal makes the agent the preferred target, which
        is the whole reason the game teaches you to stow your sword in a storm.
        """
        cl = self.climate
        if cl.weather != "storm":
            return []
        self.strike_cooldown -= dt
        if self.strike_cooldown > 0:
            return []
        if self.rng.random() > dt * 0.55:
            return []
        self.strike_cooldown = float(self.rng.uniform(4.0, 11.0))

        ap = np.asarray(agent.pos, dtype=np.float64)
        candidates: list[tuple[float, str, object, np.ndarray]] = []
        if agent.inventory.has_metal_equipped():
            candidates.append((2.4 + ap[1] * 0.02, "agent", agent, ap.copy()))
        for b in bodies.bodies.values():
            if not b.spec.conductive:
                continue
            if float(np.linalg.norm(b.pos[[0, 2]] - ap[[0, 2]])) < 26.0:
                candidates.append((1.0 + b.pos[1] * 0.02, "body", b, b.pos.copy()))
        for npc in npcs:
            if getattr(npc, "conductive", False) and npc.alive:
                p = np.asarray(npc.pos, dtype=np.float64)
                if float(np.linalg.norm(p[[0, 2]] - ap[[0, 2]])) < 26.0:
                    candidates.append((0.8, "npc", npc, p))
        # bare metal blocks in the open also attract strikes
        xi, zi = int(ap[0]), int(ap[2])
        for _ in range(6):
            tx = int(np.clip(xi + self.rng.integers(-20, 21), 1, world.size_x - 2))
            tz = int(np.clip(zi + self.rng.integers(-20, 21), 1, world.size_z - 2))
            ty = world.surface_y(tx, tz)
            if world.get(tx, ty, tz) in (B.METAL, B.COOKPOT):
                candidates.append((1.6, "block", (tx, ty, tz),
                                   np.array([tx + 0.5, ty + 1.0, tz + 0.5])))
        if not candidates:
            if self.rng.random() < 0.5:
                cl.flash = 0.6
                return [dict(type="thunder", distant=True)]
            return []

        candidates.sort(key=lambda c: -c[0])
        _, kind, target, pos = candidates[0]
        cl.flash = 1.0
        ev = [dict(type="lightning", kind=kind, pos=tuple(float(v) for v in pos))]
        for i in range(3):
            self.ignite(world, int(pos[0]), int(pos[1]) + i, int(pos[2]))
        if kind == "agent":
            resist = agent.shock_resist()
            dmg = self.cfg.shock_damage * 2.4 * max(0.0, 1.0 - 0.75 * resist)
            if dmg > 0:
                agent.hurt(dmg, "shock")
            dropped = agent.inventory.unequip_metal()
            ev.append(dict(type="shocked", target="agent", damage=float(dmg),
                           dropped=dropped))
        elif kind == "npc":
            target.hurt(self.cfg.shock_damage * 3.0)
            ev.append(dict(type="shocked", target="npc"))
        ev.extend(self.shock_water(world, pos, agent, npcs))
        return ev

    # ------------------------------------------------------------------
    def shock_water(self, world, pos, agent, npcs, power: float = 1.0) -> list[dict]:
        """Electrify a connected body of water and everything standing in it."""
        p = np.asarray(pos, dtype=np.float64)
        xi, yi, zi = int(p[0]), int(p[1]), int(p[2])
        cell = None
        for dy in (0, -1, -2, 1):
            if world.get(xi, yi + dy, zi) == B.WATER:
                cell = (xi, yi + dy, zi)
                break
        if cell is None:
            return []
        body = world.water_body(cell, limit=2500)
        if not body:
            return []
        ev = [dict(type="water_shock", cells=len(body))]
        ap = np.asarray(agent.pos, dtype=np.float64)
        if agent.in_water and (int(ap[0]), int(ap[1]), int(ap[2])) in body or \
                any((int(ap[0]), int(ap[1]) + d, int(ap[2])) in body
                    for d in (0, 1, -1)):
            resist = agent.shock_resist()
            dmg = self.cfg.shock_damage * power * max(0.0, 1.0 - 0.8 * resist)
            if dmg > 0:
                agent.hurt(dmg, "shock")
                ev.append(dict(type="shocked", target="agent", damage=float(dmg)))
        for npc in npcs:
            if not npc.alive:
                continue
            q = np.asarray(npc.pos, dtype=np.float64)
            if (int(q[0]), int(q[1]), int(q[2])) in body:
                npc.hurt(self.cfg.shock_damage * power * 1.5)
                ev.append(dict(type="shocked", target="npc", npc=npc.id))
        return ev

    # ------------------------------------------------------------------
    # temperature
    # ------------------------------------------------------------------
    def ambient_temp(self, world, pos) -> float:
        """Environmental temperature in comfort units (-1 freezing .. +1 hot)."""
        x = int(np.clip(pos[0], 0, world.size_x - 1))
        z = int(np.clip(pos[2], 0, world.size_z - 1))
        base = float(Bi.CLIMATE[int(world.biome[x, z])])
        base += 0.35 * float(world.temperature[x, z])
        # altitude and night both bite
        base -= max(0.0, float(pos[1]) - world.sea_level) / 90.0
        cl = self.climate
        if cl.is_night:
            base -= 0.25
        if cl.weather in ("rain", "storm"):
            base -= 0.15
        elif cl.weather == "snow":
            base -= 0.35
        return float(np.clip(base, -1.6, 1.6))

    def step_temperature(self, world, agent, dt: float) -> list[dict]:
        ev: list[dict] = []
        amb = self.ambient_temp(world, agent.pos)
        warm = self.heat_at(world, agent.pos)
        amb += 0.9 * warm
        amb -= 0.25 * agent.wetness
        torch = agent.inventory.equipped_stack("weapon")
        if torch is not None and "flame" in torch.spec.tags:
            amb += 0.45
        agent.body_temp += (amb - agent.body_temp) * min(1.0, 0.35 * dt)

        cold = -agent.body_temp - agent.cold_resist()
        hot = agent.body_temp - 0.55 - agent.heat_resist()
        if cold > 0.35:
            dmg = self.cfg.cold_damage * dt * min(cold, 2.0)
            agent.hurt(dmg, "cold")
            ev.append(dict(type="freezing", severity=float(cold)))
        elif hot > 0.0:
            dmg = self.cfg.heat_damage * dt * min(hot + 0.3, 2.0)
            agent.hurt(dmg, "heat")
            ev.append(dict(type="overheating", severity=float(hot)))
            # wooden gear catches light in extreme heat
            w = agent.inventory.equipped_stack("weapon")
            if w is not None and "wood" in w.spec.tags and hot > 0.5 \
                    and self.rng.random() < dt * 0.25:
                broke = agent.inventory.damage("weapon", 10)
                ev.append(dict(type="gear_burned", item=w.item, broke=broke))
        return ev
