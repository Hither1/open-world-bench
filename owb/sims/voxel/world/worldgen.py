"""Procedural terrain generation.

Pipeline (all vectorised over the whole map):

1. elevation  -- warped fBm continents + ridged mountain crests + plateau
   terracing, which is what produces the climbable mesas and cliff bands the
   traversal tasks depend on;
2. hydrology  -- river channels carved by a thin |fBm| ridge, oceans/lakes from
   anything below sea level;
3. climate    -- temperature (latitude + lapse rate) and moisture fields;
4. biomes     -- classified from (height, slope, temperature, moisture);
5. voxels     -- columns filled from the heightmap, then caves carved, ore
   veins injected, ground cover / trees / ingredient plants scattered.

Set-piece landmarks (towers, villages, monster camps, trial chambers, forges)
are added afterwards by :mod:`owb.sims.voxel.world.structures`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import blocks as B
from ..core import noise as N
from ..core.config import WorldConfig
from ..core.rng import derive, make_rng
from . import biomes as Bi
from .voxel import VoxelWorld


@dataclass
class Landmark:
    kind: str
    name: str
    pos: tuple[float, float, float]
    radius: float = 8.0
    data: dict = field(default_factory=dict)


@dataclass
class WorldGenResult:
    world: VoxelWorld
    spawn: tuple[float, float, float]
    landmarks: list[Landmark] = field(default_factory=list)
    plants: list[dict] = field(default_factory=list)
    npc_spawns: list[dict] = field(default_factory=list)
    chests: list[dict] = field(default_factory=list)
    bodies: list[dict] = field(default_factory=list)
    heightmap: np.ndarray | None = None

    def landmarks_of(self, kind: str) -> list[Landmark]:
        return [l for l in self.landmarks if l.kind == kind]


# ==========================================================================
# elevation + climate fields
# ==========================================================================
def rank_normalize(a: np.ndarray) -> np.ndarray:
    """Map a field to an exactly uniform [0, 1] by rank.

    Raw fBm has a strong low-frequency DC component, so at map scale one seed
    comes out mostly ocean and the next has no mountains at all.  Normalising
    by rank fixes the *budget* of each terrain class while leaving the spatial
    layout entirely up to the noise -- which is what makes biome fractions and
    therefore task difficulty comparable across seeds.
    """
    flat = np.asarray(a, dtype=np.float32).ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    return ranks.reshape(np.shape(a))


def _fields(cfg: WorldConfig):
    sx, sz = cfg.size_x, cfg.size_z
    seed = cfg.seed
    xs = np.arange(sx, dtype=np.float32)[:, None]
    zs = np.arange(sz, dtype=np.float32)[None, :]
    X = np.broadcast_to(xs, (sx, sz)).astype(np.float32)
    Z = np.broadcast_to(zs, (sx, sz)).astype(np.float32)
    sea = float(cfg.sea_level)

    # --- elevation -------------------------------------------------------
    wx, wz = N.warp2(X / cfg.continent_scale, Z / cfg.continent_scale,
                     derive(seed, 1), strength=cfg.warp_strength / 10.0, freq=1.4)
    continent = N.fbm2(wx, wz, derive(seed, 2), octaves=5)
    hills = N.fbm2(X / cfg.hill_scale, Z / cfg.hill_scale, derive(seed, 3),
                   octaves=4)
    detail = N.fbm2(X / cfg.detail_scale, Z / cfg.detail_scale, derive(seed, 4),
                    octaves=3)

    # hypsometric curve: uniform rank -> metres.  The exponent keeps lowland
    # broad and walkable while the top decile climbs fast into cliff country.
    elev = rank_normalize(0.72 * continent + 0.28 * hills)
    h = (sea - 4.5) + (cfg.relief * 1.9) * np.power(elev, 1.25) + 2.0 * detail

    # --- mountains -------------------------------------------------------
    mraw = rank_normalize(N.fbm2(X / (cfg.continent_scale * 1.3),
                                 Z / (cfg.continent_scale * 1.3),
                                 derive(seed, 5), octaves=3))
    mountain_mask = N.smoothstep(0.66, 0.97, mraw) * N.smoothstep(
        sea - 2.0, sea + 12.0, h)
    ridges = N.ridged2(X / (cfg.hill_scale * 1.25), Z / (cfg.hill_scale * 1.25),
                       derive(seed, 6), octaves=5)
    h = h + cfg.mountain_relief * mountain_mask * (ridges ** 1.3)

    # plateau terracing in rocky country -> the ledges and mesas that make
    # stamina-limited climbing interesting rather than impossible
    terrace_amt = np.clip(0.5 * mountain_mask * cfg.cliff_sharpness, 0.0, 0.85)
    step = 4.0
    h = h * (1.0 - terrace_amt) + np.round(h / step) * step * terrace_amt

    # --- rivers ----------------------------------------------------------
    if cfg.river_density > 0:
        rr = np.abs(N.fbm2(X / 88.0, Z / 88.0, derive(seed, 7), octaves=3))
        chan = N.smoothstep(0.045 * cfg.river_density, 0.004, rr)
        chan = chan * (1.0 - 0.65 * mountain_mask)
        h = h - 13.0 * chan
        river_field = chan
    else:
        river_field = np.zeros_like(h)

    # compress depth below the waterline: shallow, wadeable lake and sea beds
    depth = np.maximum(sea - h, 0.0)
    h = np.where(h < sea, sea - np.power(depth, 0.72), h)
    h = N.box_blur(h, 1, 1)
    h = np.clip(h, 2.0, cfg.size_y - 10.0)

    # --- volcanic province ----------------------------------------------
    rng = make_rng(derive(seed, 10))
    ang = float(rng.uniform(0, 2 * np.pi))
    vr = 0.30 * min(sx, sz)
    vcx = sx * 0.5 + np.cos(ang) * vr
    vcz = sz * 0.5 + np.sin(ang) * vr
    vdist = np.sqrt((X - vcx) ** 2 + (Z - vcz) ** 2)
    volcanic = N.smoothstep(min(sx, sz) * 0.145, min(sx, sz) * 0.05, vdist)
    volcanic = volcanic * np.clip(
        0.9 + 1.4 * N.fbm2(X / 24.0, Z / 24.0, derive(seed, 11), octaves=2),
        0.0, 1.25)
    volcanic = np.clip(volcanic, 0.0, 1.0).astype(np.float32)

    # --- climate ---------------------------------------------------------
    lat = (Z / max(sz - 1, 1)) * 2.0 - 1.0                # -1 north .. +1 south
    temp_raw = (-1.25 * lat
                + 2.2 * N.fbm2(X / cfg.temperature_scale, Z / cfg.temperature_scale,
                               derive(seed, 8), octaves=3))
    temp = 1.9 * rank_normalize(temp_raw) - 0.95           # uniform [-0.95, 0.95]
    # Lapse rate redistributes temperature with altitude but is mean-removed,
    # so the map-wide climate budget stays exactly the uniform above and the
    # frozen fraction does not drift with how mountainous a seed happens to be.
    lapse = np.maximum(h - sea, 0.0) / 48.0
    temp = temp - (lapse - float(lapse.mean()))
    temp = np.clip(temp + 0.9 * volcanic, -1.6, 1.6).astype(np.float32)

    # aridity province: one broad dry belt, so the desert is a *place* you can
    # be told to cross rather than scattered dry pixels
    arid = rank_normalize(N.fbm2(X / (cfg.moisture_scale * 1.45),
                                 Z / (cfg.moisture_scale * 1.45),
                                 derive(seed, 12), octaves=3))

    moist_raw = N.fbm2(X / cfg.moisture_scale, Z / cfg.moisture_scale,
                       derive(seed, 9), octaves=4)
    moisture = rank_normalize(moist_raw)
    moisture = np.clip(moisture - 0.55 * N.smoothstep(0.62, 0.95, arid)
                       - 0.12 * np.maximum(temp, 0.0)
                       + 0.30 * river_field, 0.0, 1.0).astype(np.float32)

    gx, gz = np.gradient(h)
    slope = np.sqrt(gx * gx + gz * gz).astype(np.float32)

    return dict(h=h.astype(np.float32), temp=temp, moisture=moisture,
                slope=slope, river=river_field.astype(np.float32),
                volcanic=volcanic, mountain=mountain_mask.astype(np.float32),
                elev=elev.astype(np.float32), arid=arid.astype(np.float32),
                volcano_center=(float(vcx), float(vcz)))


def _classify_biomes(cfg: WorldConfig, f: dict) -> np.ndarray:
    """Assign a biome per column from (height, slope, temperature, moisture).

    Rules are applied in order of increasing priority, so the last matching
    rule wins: water and shoreline always beat the climate rules above them.
    """
    h, temp, moist, slope = f["h"], f["temp"], f["moisture"], f["slope"]
    sea = cfg.sea_level
    biome = np.full(h.shape, Bi.PLAINS, dtype=np.uint8)

    biome = np.where(moist > 0.52, Bi.FOREST, biome)
    biome = np.where(h > sea + 20, Bi.HILLS, biome)
    biome = np.where((h > sea + 20) & (moist > 0.68), Bi.FOREST, biome)
    arid = f["arid"]
    biome = np.where((arid > 0.74) & (temp > -0.45), Bi.SAVANNA, biome)
    biome = np.where((arid > 0.89) & (temp > -0.30), Bi.DESERT, biome)
    biome = np.where((moist > 0.80) & (h < sea + 7), Bi.SWAMP, biome)
    biome = np.where((slope > 1.5) & (h > sea + 8) & (temp > 0.25),
                     Bi.CANYON, biome)
    biome = np.where((slope > 1.7) & (h > sea + 14) & (temp <= 0.25),
                     Bi.MOUNTAIN, biome)
    biome = np.where(f["volcanic"] > 0.55, Bi.VOLCANIC, biome)
    biome = np.where((h > cfg.snow_line) | (temp < -0.86), Bi.SNOWFIELD, biome)
    biome = np.where((h <= sea + 1.6) & (h >= sea - 1.0), Bi.BEACH, biome)
    biome = np.where(h < sea - 0.5, Bi.LAKE, biome)
    return biome


# ==========================================================================
# voxel filling
# ==========================================================================
def _fill_columns(cfg: WorldConfig, world: VoxelWorld, f: dict,
                  biome: np.ndarray) -> None:
    sx, sy, sz = cfg.size_x, cfg.size_y, cfg.size_z
    hi = np.clip(np.round(f["h"]).astype(np.int16), 1, sy - 2)
    surf = Bi.SURFACE[biome].astype(np.uint8)
    sub = Bi.SUBSURFACE[biome].astype(np.uint8)

    # rock type varies by province
    rock = np.full((sx, sz), B.STONE, dtype=np.uint8)
    rock[f["volcanic"] > 0.5] = B.BASALT
    rock[biome == Bi.SWAMP] = B.MOSS_STONE

    # snow caps on anything high, whatever the biome says
    high = f["h"] > cfg.snow_line - 3
    surf = np.where(high, np.uint8(B.SNOW), surf)
    sub = np.where(high, np.uint8(B.SNOW_PACK), sub)
    # steep ground is bare rock, not soil (cliffs you must climb)
    steep = f["slope"] > 1.9
    surf = np.where(steep & ~high, rock, surf)
    sub = np.where(steep, rock, sub)
    # river/lake beds
    wet = f["h"] < cfg.sea_level + 0.4
    surf = np.where(wet, np.uint8(B.SAND), surf)

    ys = np.arange(sy, dtype=np.int16)[None, :, None]
    hi3 = hi[:, None, :]
    vox = np.where(ys < hi3 - 3, rock[:, None, :], np.uint8(B.AIR)).astype(np.uint8)
    band = (ys >= hi3 - 3) & (ys < hi3)
    vox = np.where(band, sub[:, None, :], vox)
    vox = np.where(ys == hi3, surf[:, None, :], vox)
    water = (ys > hi3) & (ys <= cfg.sea_level)
    vox = np.where(water, np.uint8(B.WATER), vox)
    vox[:, 0, :] = B.RUNE_WALL          # indestructible bedrock
    world.vox = np.ascontiguousarray(vox)


def _carve_caves(cfg: WorldConfig, world: VoxelWorld, f: dict,
                 rng: np.random.Generator) -> None:
    if cfg.cave_density <= 0:
        return
    sx, sy, sz = cfg.size_x, cfg.size_y, cfg.size_z
    hi = np.clip(np.round(f["h"]).astype(np.int32), 1, sy - 2)
    xs = np.arange(sx, dtype=np.float32)[:, None, None]
    ys = np.arange(sy, dtype=np.float32)[None, :, None]
    zs = np.arange(sz, dtype=np.float32)[None, None, :]
    # two warped 3-D fields intersecting -> tunnels rather than swiss cheese
    a = N.perlin3(xs / 26.0, ys / 15.0, zs / 26.0, derive(cfg.seed, 21))
    b = N.perlin3(xs / 22.0 + 11.0, ys / 13.0 - 5.0, zs / 22.0 + 3.0,
                  derive(cfg.seed, 22))
    thr = 0.11 * cfg.cave_density
    tunnel = (np.abs(a) < thr) & (np.abs(b) < thr)
    depth_ok = (ys < (hi[:, None, :] - 4)) & (ys > 2)
    carve = tunnel & depth_ok & B.SOLID[world.vox]
    world.vox[carve] = B.AIR

    # sprinkle glow crystals on cave walls so caves are navigable in the dark
    cav = np.argwhere(carve)
    if len(cav):
        pick = cav[rng.integers(0, len(cav), size=max(1, len(cav) // 900))]
        for x, y, z in pick:
            if world.get(int(x), int(y) - 1, int(z)) != B.AIR:
                world.vox[int(x), int(y), int(z)] = B.CRYSTAL


def _place_ore(cfg: WorldConfig, world: VoxelWorld, f: dict, biome: np.ndarray,
               rng: np.random.Generator) -> None:
    if cfg.ore_density <= 0:
        return
    sx, sz = cfg.size_x, cfg.size_z
    n = int(0.0035 * sx * sz * cfg.ore_density)
    xs = rng.integers(2, sx - 2, size=n)
    zs = rng.integers(2, sz - 2, size=n)
    for x, z in zip(xs, zs):
        bio = Bi.get(int(biome[x, z]))
        if not bio.ore:
            continue
        kinds = [k for k, _ in bio.ore]
        wts = np.array([w for _, w in bio.ore], dtype=np.float64)
        blk = int(kinds[int(rng.choice(len(kinds), p=wts / wts.sum()))])
        top = world.surface_y(int(x), int(z))
        if top < 2:
            continue
        # outcrops sit at the surface on steep ground, veins sit just below
        exposed = f["slope"][x, z] > 1.4 or rng.random() < 0.35
        y = top if exposed else top - int(rng.integers(1, 5))
        y = int(np.clip(y, 2, world.size_y - 2))
        r = 1 if rng.random() < 0.7 else 2
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx * dx + dy * dy + dz * dz > r * r + 1:
                        continue
                    px, py, pz = int(x) + dx, y + dy, int(z) + dz
                    cur = world.get(px, py, pz)
                    if cur in (B.STONE, B.BASALT, B.MOSS_STONE, B.GRAVEL,
                               B.SNOW_PACK, B.DIRT):
                        world.vox[px, py, pz] = blk


# ==========================================================================
# vegetation
# ==========================================================================
def _tree(world: VoxelWorld, x: int, y: int, z: int, kind: str,
          rng: np.random.Generator) -> None:
    sy = world.size_y
    if kind == "pine":
        th = int(rng.integers(7, 12))
        for i in range(th):
            if y + i < sy:
                world.vox[x, y + i, z] = B.LOG
        for i in range(th - 2):
            r = max(0, int(round((th - 3 - i) * 0.42)))
            yy = y + 3 + i
            if yy >= sy or r == 0:
                continue
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if abs(dx) + abs(dz) > r:
                        continue
                    px, pz = x + dx, z + dz
                    if world.in_bounds_xz(px, pz) and world.vox[px, yy, pz] == B.AIR:
                        world.vox[px, yy, pz] = B.LEAVES
        if y + th < sy:
            world.vox[x, y + th, z] = B.LEAVES
    elif kind == "palm":
        th = int(rng.integers(5, 9))
        lean = int(rng.integers(-1, 2))
        for i in range(th):
            px = x + (lean if i > th // 2 else 0)
            if y + i < sy and world.in_bounds_xz(px, z):
                world.vox[px, y + i, z] = B.LOG
        tx, ty = x + lean, min(y + th, sy - 1)
        for dx, dz in ((2, 0), (-2, 0), (0, 2), (0, -2), (1, 1), (-1, -1),
                       (1, -1), (-1, 1), (1, 0), (-1, 0), (0, 1), (0, -1)):
            px, pz = tx + dx, z + dz
            if world.in_bounds_xz(px, pz) and world.vox[px, ty, pz] == B.AIR:
                world.vox[px, ty, pz] = B.LEAVES
    elif kind == "dead":
        th = int(rng.integers(3, 7))
        for i in range(th):
            if y + i < sy:
                world.vox[x, y + i, z] = B.LOG
        for _ in range(int(rng.integers(1, 4))):
            dx, dz = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
            yy = min(y + th - 1, sy - 1)
            px, pz = x + dx, z + dz
            if world.in_bounds_xz(px, pz) and world.vox[px, yy, pz] == B.AIR:
                world.vox[px, yy, pz] = B.LOG
    elif kind == "mushroom":
        th = int(rng.integers(2, 5))
        for i in range(th):
            if y + i < sy:
                world.vox[x, y + i, z] = B.LOG
        yy = min(y + th, sy - 1)
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                if abs(dx) + abs(dz) > 2:
                    continue
                px, pz = x + dx, z + dz
                if world.in_bounds_xz(px, pz) and world.vox[px, yy, pz] == B.AIR:
                    world.vox[px, yy, pz] = B.LEAVES
    else:  # oak / broadleaf
        th = int(rng.integers(4, 8))
        for i in range(th):
            if y + i < sy:
                world.vox[x, y + i, z] = B.LOG
        cy = y + th
        r = int(rng.integers(2, 4))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx * dx + dy * dy * 1.6 + dz * dz > r * r + 0.8:
                        continue
                    px, py, pz = x + dx, cy + dy, z + dz
                    if (world.in_bounds_xz(px, pz) and 0 <= py < sy
                            and world.vox[px, py, pz] == B.AIR):
                        world.vox[px, py, pz] = B.LEAVES
    # climbable vines sometimes hang from big trees
    if rng.random() < 0.18:
        for i in range(int(rng.integers(1, 3))):
            yy = y + 2 + i
            px = x + int(rng.choice([-1, 1]))
            if world.in_bounds_xz(px, z) and 0 <= yy < sy and world.vox[px, yy, z] == B.AIR:
                world.vox[px, yy, z] = B.VINE


def _vegetate(cfg: WorldConfig, world: VoxelWorld, f: dict, biome: np.ndarray,
              rng: np.random.Generator) -> list[dict]:
    sx, sz = cfg.size_x, cfg.size_z
    slope = f["slope"]
    plants: list[dict] = []

    # --- ground cover (vectorised per block type) ------------------------
    hi = world.height
    top = world.vox[np.arange(sx)[:, None], np.clip(hi, 0, world.size_y - 1),
                    np.arange(sz)[None, :]]
    plantable = np.isin(top, np.array([B.GRASS, B.DIRT, B.SAND, B.MUD, B.SNOW,
                                       B.BASALT, B.GRAVEL, B.SNOW_PACK],
                                      dtype=np.uint8))
    plantable &= (hi > cfg.sea_level) & (slope < 2.2) & (hi < world.size_y - 3)

    for bid, bio in enumerate(Bi.BIOMES):
        if not bio.cover:
            continue
        here = plantable & (biome == bid)
        if not here.any():
            continue
        for blk, prob in bio.cover:
            r = rng.random((sx, sz))
            m = here & (r < prob * cfg.grass_density)
            xs, zs = np.nonzero(m)
            ys = np.clip(hi[xs, zs] + 1, 0, world.size_y - 1)
            keep = world.vox[xs, ys, zs] == B.AIR
            world.vox[xs[keep], ys[keep], zs[keep]] = blk

    # --- trees -----------------------------------------------------------
    for bid, bio in enumerate(Bi.BIOMES):
        if bio.tree_density <= 0 or bio.tree_kind == "none":
            continue
        dens = bio.tree_density * cfg.tree_density
        here = plantable & (biome == bid) & (slope < 1.5)
        cand = np.nonzero(here & (rng.random((sx, sz)) < dens))
        for x, z in zip(*cand):
            x, z = int(x), int(z)
            if x < 3 or z < 3 or x > sx - 4 or z > sz - 4:
                continue
            y = int(hi[x, z]) + 1
            _tree(world, x, y, z, bio.tree_kind, rng)

    # --- ingredient plants (harvestable entities, not voxels) -------------
    for bid, bio in enumerate(Bi.BIOMES):
        if not bio.flora:
            continue
        here = plantable & (biome == bid)
        dens = 0.0022 * cfg.grass_density
        cand = np.nonzero(here & (rng.random((sx, sz)) < dens))
        names = [n for n, _ in bio.flora]
        wts = np.array([w for _, w in bio.flora], dtype=np.float64)
        wts = wts / wts.sum()
        for x, z in zip(*cand):
            x, z = int(x), int(z)
            item = names[int(rng.choice(len(names), p=wts))]
            y = float(world.height[x, z]) + 1.0
            plants.append(dict(pos=(x + 0.5, y, z + 0.5), item=item,
                               biome=int(bid)))
    world.recompute_heightmaps()
    return plants


# ==========================================================================
# fauna & monster placement
# ==========================================================================
def _populate(cfg: WorldConfig, world: VoxelWorld, biome: np.ndarray,
              rng: np.random.Generator) -> list[dict]:
    sx, sz = cfg.size_x, cfg.size_z
    out: list[dict] = []
    n_sites = int(0.00035 * sx * sz)
    for _ in range(max(4, n_sites)):
        x = int(rng.integers(4, sx - 4))
        z = int(rng.integers(4, sz - 4))
        bio = Bi.get(int(biome[x, z]))
        y = world.spawn_y(x + 0.5, z + 0.5)
        if y <= cfg.sea_level + 0.5:
            continue
        if bio.monsters and rng.random() < 0.55 * cfg.monster_density:
            kinds = [k for k, _ in bio.monsters]
            w = np.array([v for _, v in bio.monsters], dtype=np.float64)
            kind = kinds[int(rng.choice(len(kinds), p=w / w.sum()))]
            for _ in range(int(rng.integers(1, 4))):
                jx = x + float(rng.normal(0, 2.0))
                jz = z + float(rng.normal(0, 2.0))
                if not world.in_bounds_xz(int(jx), int(jz)):
                    continue
                out.append(dict(kind=kind, pos=(jx, world.spawn_y(jx, jz), jz),
                                home=(jx, jz)))
        if bio.animals and rng.random() < 0.6 * cfg.animal_density:
            kinds = [k for k, _ in bio.animals]
            w = np.array([v for _, v in bio.animals], dtype=np.float64)
            kind = kinds[int(rng.choice(len(kinds), p=w / w.sum()))]
            for _ in range(int(rng.integers(1, 3))):
                jx = x + float(rng.normal(0, 3.0))
                jz = z + float(rng.normal(0, 3.0))
                if not world.in_bounds_xz(int(jx), int(jz)):
                    continue
                out.append(dict(kind=kind, pos=(jx, world.spawn_y(jx, jz), jz),
                                home=(jx, jz)))
    return out


# ==========================================================================
# spawn selection
# ==========================================================================
def pick_spawn(world: VoxelWorld, f: dict, biome: np.ndarray,
               cfg: WorldConfig, rng: np.random.Generator) -> tuple[float, float, float]:
    """A flat, dry, temperate column near the middle of the map."""
    sx, sz = cfg.size_x, cfg.size_z
    good = ((f["slope"] < 0.7) & (world.height > cfg.sea_level + 2)
            & (world.height < cfg.snow_line - 6)
            & np.isin(biome, np.array([Bi.PLAINS, Bi.FOREST, Bi.HILLS,
                                       Bi.SAVANNA], dtype=np.uint8)))
    xs, zs = np.nonzero(good)
    if len(xs) == 0:
        x, z = sx // 2, sz // 2
        return (x + 0.5, world.spawn_y(x + 0.5, z + 0.5) + 0.1, z + 0.5)
    d = (xs - sx / 2) ** 2 + (zs - sz / 2) ** 2
    order = np.argsort(d)[:max(1, len(xs) // 40)]
    pick = int(rng.choice(order))
    x, z = int(xs[pick]), int(zs[pick])
    return (x + 0.5, world.spawn_y(x + 0.5, z + 0.5) + 0.1, z + 0.5)


# ==========================================================================
# entry point
# ==========================================================================
def generate(cfg: WorldConfig) -> WorldGenResult:
    """Generate a complete world from ``cfg`` (deterministic in ``cfg.seed``)."""
    from .presets import apply_preset
    from .structures import add_landmarks

    cfg = apply_preset(cfg)
    rng = make_rng(derive(cfg.seed, 99))

    f = _fields(cfg)
    biome = _classify_biomes(cfg, f)

    world = VoxelWorld(cfg.size_x, cfg.size_y, cfg.size_z, cfg.sea_level)
    _fill_columns(cfg, world, f, biome)
    world.recompute_heightmaps()
    _carve_caves(cfg, world, f, rng)
    world.recompute_heightmaps()
    _place_ore(cfg, world, f, biome, rng)
    world.recompute_heightmaps()

    world.biome = biome
    world.temperature = f["temp"]
    world.moisture = f["moisture"]
    world.slope = f["slope"]
    world.set_region_names(Bi.region_names())
    world.meta.update(sea_level=cfg.sea_level, snow_line=cfg.snow_line,
                      volcano_center=f["volcano_center"], seed=cfg.seed,
                      preset=cfg.preset)

    plants = _vegetate(cfg, world, f, biome, rng)
    npcs = _populate(cfg, world, biome, rng)

    res = WorldGenResult(world=world, spawn=(0.0, 0.0, 0.0), plants=plants,
                         npc_spawns=npcs, heightmap=f["h"])
    if cfg.landmarks and cfg.structure_density > 0:
        add_landmarks(cfg, res, f, biome, rng)
    world.recompute_heightmaps()
    res.spawn = pick_spawn(world, f, biome, cfg, rng)
    return res
