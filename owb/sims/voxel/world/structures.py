"""Set-piece landmarks: villages, towers, monster camps, trial chambers,
forges, beacons, ruins and blastable caves.

Landmarks are what turn procedural terrain into a *game*: they are the
destinations quests point at, the stations crafting needs (cooking pot, forge),
and the physics set-ups the puzzle tasks are built on.  Each one registers a
:class:`~owb.sims.voxel.world.worldgen.Landmark` so tasks and the text observation can
refer to it by name.
"""

from __future__ import annotations

import numpy as np

from ..core import blocks as B
from ..core.config import WorldConfig
from . import biomes as Bi
from .voxel import VoxelWorld
from .worldgen import Landmark, WorldGenResult

# --------------------------------------------------------------------------
# loot tables
# --------------------------------------------------------------------------
LOOT: dict[str, list[tuple[str, int, float]]] = {
    # item, count, probability
    "camp": [("club_boko", 1, 0.5), ("arrow", 6, 0.6), ("meat_raw", 2, 0.5),
             ("monster_horn", 2, 0.7), ("shield_wood", 1, 0.3),
             ("bow_crude", 1, 0.3)],
    "village": [("apple", 3, 0.7), ("hearty_radish", 2, 0.4),
                ("axe_stone", 1, 0.4), ("pickaxe_stone", 1, 0.4),
                ("warm_doublet", 1, 0.15), ("rope", 2, 0.3)],
    "ruin": [("iron_ingot", 1, 0.35), ("key_bronze", 1, 0.3),
             ("sword_iron", 1, 0.2), ("crystal", 1, 0.25),
             ("flint", 3, 0.6)],
    "trial": [("trial_token", 1, 1.0), ("rune_stasis", 1, 0.25),
              ("rune_cryo", 1, 0.25), ("rune_magnet", 1, 0.25),
              ("sword_iron", 1, 0.3), ("climb_gear", 1, 0.2)],
    "cave": [("crystal", 2, 0.5), ("iron_ore", 3, 0.6), ("bomb", 3, 0.4),
             ("guardian_core", 1, 0.12), ("heat_cloak", 1, 0.15)],
    "tower": [("glider", 1, 0.5), ("arrow", 10, 0.5), ("stambulb", 2, 0.5)],
    "forge": [("iron_ingot", 2, 0.6), ("hammer", 1, 0.35),
              ("pickaxe_iron", 1, 0.25), ("charcoal", 3, 0.5)],
}


def roll_loot(kind: str, rng: np.random.Generator,
              bonus: list[tuple[str, int]] | None = None) -> list[list]:
    out: list[list] = []
    for item, count, prob in LOOT.get(kind, []):
        if rng.random() < prob:
            out.append([item, int(count)])
    if bonus:
        out.extend([[i, int(c)] for i, c in bonus])
    if not out:
        out.append(["wood", 2])
    return out


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _box(w: VoxelWorld, x0, y0, z0, x1, y1, z1, block, only_air=False) -> None:
    x0, x1 = sorted((int(x0), int(x1)))
    y0, y1 = sorted((int(y0), int(y1)))
    z0, z1 = sorted((int(z0), int(z1)))
    x0, z0, y0 = max(0, x0), max(0, z0), max(0, y0)
    x1 = min(w.size_x - 1, x1)
    z1 = min(w.size_z - 1, z1)
    y1 = min(w.size_y - 1, y1)
    if x0 > x1 or y0 > y1 or z0 > z1:
        return
    sub = w.vox[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
    if only_air:
        sub[sub == B.AIR] = block
    else:
        sub[...] = block


def _hollow(w: VoxelWorld, x0, y0, z0, x1, y1, z1, wall, inner=B.AIR) -> None:
    _box(w, x0, y0, z0, x1, y1, z1, wall)
    _box(w, x0 + 1, y0 + 1, z0 + 1, x1 - 1, y1 - 1, z1 - 1, inner)


def _flatten(w: VoxelWorld, cx: int, cz: int, r: int, y: int,
             surface: int = B.DIRT) -> None:
    """Level a pad to height ``y``: fill holes, shave anything above."""
    x0, x1 = max(0, cx - r), min(w.size_x - 1, cx + r)
    z0, z1 = max(0, cz - r), min(w.size_z - 1, cz + r)
    if x0 > x1 or z0 > z1:
        return
    _box(w, x0, y + 1, z0, x1, min(w.size_y - 1, y + 14), z1, B.AIR)
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            col = w.vox[x, :, z]
            for yy in range(1, y):
                if col[yy] == B.AIR or col[yy] == B.WATER:
                    col[yy] = B.DIRT
            col[y] = surface
    w._refresh_region(x0, x1, z0, z1)


def _ladder(w: VoxelWorld, x: int, z: int, y0: int, y1: int,
            block: int = B.VINE) -> None:
    for y in range(int(y0), int(y1) + 1):
        if w.in_bounds(x, y, z) and w.vox[x, y, z] == B.AIR:
            w.vox[x, y, z] = block


# --------------------------------------------------------------------------
# site selection
# --------------------------------------------------------------------------
def _sites(cfg: WorldConfig, world: VoxelWorld, f: dict, biome: np.ndarray,
           rng: np.random.Generator, n: int, *, min_sep: float = 24.0,
           max_slope: float = 1.1, dry: bool = True, margin: int = 12,
           allow_biomes: tuple[int, ...] | None = None,
           min_height: int | None = None,
           taken: list[tuple[float, float]] | None = None) -> list[tuple[int, int]]:
    """Rejection-sample ``n`` well-separated, buildable sites."""
    out: list[tuple[int, int]] = []
    taken = taken if taken is not None else []
    tries = 0
    sea = cfg.sea_level
    while len(out) < n and tries < n * 400:
        tries += 1
        x = int(rng.integers(margin, max(margin + 1, cfg.size_x - margin)))
        z = int(rng.integers(margin, max(margin + 1, cfg.size_z - margin)))
        h = int(world.height[x, z])
        if dry and h <= sea + 1:
            continue
        if min_height is not None and h < min_height:
            continue
        if f["slope"][x, z] > max_slope:
            continue
        if allow_biomes is not None and int(biome[x, z]) not in allow_biomes:
            continue
        ok = True
        for (tx, tz) in list(taken) + [(a, b) for a, b in out]:
            if (tx - x) ** 2 + (tz - z) ** 2 < min_sep * min_sep:
                ok = False
                break
        if ok:
            out.append((x, z))
    return out


# --------------------------------------------------------------------------
# individual structures
# --------------------------------------------------------------------------
def _build_tower(w: VoxelWorld, x: int, z: int, rng, res: WorldGenResult) -> None:
    """A tall brick tower with a vine-covered flank and a pedestal on top.

    Towers are the map's survey points: climbing one (or gliding from one) is
    the intended route for several traversal tasks.
    """
    base = int(w.height[x, z])
    hgt = int(rng.integers(20, 30))
    top = min(w.size_y - 3, base + hgt)
    _flatten(w, x, z, 5, base, B.BRICK)
    _box(w, x - 3, base, z - 3, x + 3, top, z + 3, B.BRICK)
    _box(w, x - 2, base + 1, z - 2, x + 2, top - 1, z + 2, B.AIR)
    # internal spiral of plank steps
    y = base + 1
    ring = [(-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2), (2, -1), (2, 0),
            (2, 1), (2, 2), (1, 2), (0, 2), (-1, 2), (-2, 2), (-2, 1),
            (-2, 0), (-2, -1)]
    i = 0
    while y < top - 1:
        dx, dz = ring[i % len(ring)]
        _box(w, x + dx, y, z + dz, x + dx, y, z + dz, B.PLANK)
        i += 1
        if i % 2 == 0:
            y += 1
    # roof deck with a hole, pedestal in the middle, torches at the corners
    _box(w, x - 3, top, z - 3, x + 3, top, z + 3, B.BRICK)
    _box(w, x - 1, top, z - 1, x + 1, top, z + 1, B.AIR)
    _box(w, x, top + 1, z, x, top + 1, z, B.PEDESTAL)
    for dx, dz in ((-3, -3), (3, -3), (-3, 3), (3, 3)):
        _box(w, x + dx, top + 1, z + dz, x + dx, top + 1, z + dz, B.TORCH_WALL)
    # climbable outside flank
    _ladder(w, x + 4, z, base + 1, top - 2)
    _box(w, x + 4, base, z, x + 4, top - 2, z, B.VINE)
    res.chests.append(dict(pos=(x + 2, top + 1, z + 2),
                           items=roll_loot("tower", rng), kind="tower"))
    res.landmarks.append(Landmark("tower", f"survey tower", (x + 0.5, top + 2.0, z + 0.5),
                                  radius=6.0,
                                  data=dict(top=int(top + 1), base=int(base),
                                            pedestal=(x, top + 1, z))))


def _build_village(w: VoxelWorld, x: int, z: int, rng, res: WorldGenResult) -> None:
    """A small stable/village: huts, a cooking pot, a forge and a merchant."""
    base = int(w.height[x, z])
    _flatten(w, x, z, 9, base, B.DIRT)
    n_huts = int(rng.integers(2, 5))
    spots = [(-6, -6), (6, -6), (-6, 6), (6, 6), (0, -7), (7, 0)]
    rng.shuffle(spots)
    for (dx, dz) in spots[:n_huts]:
        hx, hz = x + dx, z + dz
        wx, wz = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        hh = int(rng.integers(3, 5))
        _hollow(w, hx - wx, base + 1, hz - wz, hx + wx, base + hh, hz + wz, B.PLANK)
        _box(w, hx - wx, base + hh + 1, hz - wz, hx + wx, base + hh + 1, hz + wz,
             B.PLANK)
        # doorway + window
        _box(w, hx, base + 1, hz - wz, hx, base + 2, hz - wz, B.AIR)
        _box(w, hx + wx, base + 2, hz, hx + wx, base + 2, hz, B.GLASS)
        if rng.random() < 0.5:
            res.chests.append(dict(pos=(hx, base + 1, hz),
                                   items=roll_loot("village", rng),
                                   kind="village"))
    # village square: cooking pot, forge, signpost, campfire
    _box(w, x, base + 1, z, x, base + 1, z, B.COOKPOT)
    _box(w, x + 2, base + 1, z, x + 2, base + 1, z, B.CAMPFIRE)
    _box(w, x - 2, base + 1, z + 1, x - 2, base + 1, z + 1, B.FORGE)
    _box(w, x, base + 1, z + 3, x, base + 1, z + 3, B.SIGNPOST)
    res.npc_spawns.append(dict(kind="merchant", pos=(x + 1.5, base + 1.0, z + 1.5),
                               home=(x + 1.5, z + 1.5)))
    if rng.random() < 0.7:
        res.npc_spawns.append(dict(kind="traveller",
                                   pos=(x - 1.5, base + 1.0, z - 1.5),
                                   home=(x - 1.5, z - 1.5)))
    res.landmarks.append(Landmark("village", "stable", (x + 0.5, base + 1.0, z + 0.5),
                                  radius=12.0,
                                  data=dict(cookpot=(x, base + 1, z),
                                            forge=(x - 2, base + 1, z + 1),
                                            campfire=(x + 2, base + 1, z))))


def _build_camp(w: VoxelWorld, x: int, z: int, rng, res: WorldGenResult,
                biome_id: int) -> None:
    """A monster camp: hay huts, a campfire, a weapon chest and 2-4 monsters."""
    base = int(w.height[x, z])
    _flatten(w, x, z, 6, base, B.DIRT)
    _box(w, x, base + 1, z, x, base + 1, z, B.CAMPFIRE)
    for _ in range(int(rng.integers(2, 5))):
        dx, dz = int(rng.integers(-5, 6)), int(rng.integers(-5, 6))
        if abs(dx) < 2 and abs(dz) < 2:
            continue
        hh = int(rng.integers(2, 4))
        _box(w, x + dx, base + 1, z + dz, x + dx, base + hh, z + dz, B.HAY)
    # spiked wooden fence segments
    for _ in range(int(rng.integers(3, 7))):
        fx, fz = int(rng.integers(-6, 7)), int(rng.integers(-6, 7))
        _box(w, x + fx, base + 1, z + fz, x + fx, base + 2, z + fz, B.PLANK)
    res.chests.append(dict(pos=(x + 1, base + 1, z + 1),
                           items=roll_loot("camp", rng), kind="camp"))
    bio = Bi.get(biome_id)
    kinds = [k for k, _ in bio.monsters] or ["bokoblin"]
    wts = np.array([v for _, v in bio.monsters], dtype=np.float64) \
        if bio.monsters else np.array([1.0])
    n_mon = int(rng.integers(2, 5))
    for _ in range(n_mon):
        kind = kinds[int(rng.choice(len(kinds), p=wts / wts.sum()))]
        jx = x + float(rng.normal(0, 2.2))
        jz = z + float(rng.normal(0, 2.2))
        res.npc_spawns.append(dict(kind=kind, pos=(jx, base + 1.0, jz),
                                   home=(x + 0.5, z + 0.5), camp=True))
    res.landmarks.append(Landmark("camp", "monster camp",
                                  (x + 0.5, base + 1.0, z + 0.5), radius=9.0,
                                  data=dict(campfire=(x, base + 1, z),
                                            monsters=n_mon)))


def _build_trial(w: VoxelWorld, x: int, z: int, rng, res: WorldGenResult) -> None:
    """A sealed trial chamber: a physics puzzle behind an indestructible shell.

    Layout: a pressure plate wired to a gate, a boulder or metal slab to move
    onto it, and a chest with a trial token (plus, often, a rune gadget).
    """
    base = int(w.height[x, z])
    rx, rz, ry = 7, 7, 6
    _flatten(w, x, z, rx + 2, base, B.BRICK)
    # shell
    _hollow(w, x - rx, base, z - rz, x + rx, base + ry, z + rz, B.RUNE_WALL)
    _box(w, x - rx + 1, base + 1, z - rz + 1, x + rx - 1, base + 1, z + rz - 1,
         B.BRICK)   # floor
    # entrance corridor with a gate
    _box(w, x, base + 2, z - rz, x, base + 3, z - rz, B.GATE)
    _box(w, x, base + 2, z - rz - 2, x, base + 3, z - rz + 1, B.AIR)
    # puzzle: plate on the far side, mover object in the middle
    plate = (x + int(rng.integers(-3, 4)), base + 2, z + rz - 3)
    _box(w, plate[0], plate[1], plate[2], plate[0], plate[1], plate[2], B.PLATE)
    inner_gate = (x + 3, base + 2, z + 2)
    _box(w, inner_gate[0], inner_gate[1], inner_gate[2],
         inner_gate[0], inner_gate[1] + 1, inner_gate[2], B.GATE)
    kind = str(rng.choice(["boulder", "metal_slab"]))
    res.bodies.append(dict(kind=kind, pos=(x - 2.5, base + 3.0, z - 1.5)))
    # reward
    chest_pos = (x + 4, base + 2, z + 4)
    res.chests.append(dict(pos=chest_pos, items=roll_loot("trial", rng),
                           kind="trial", locked_by_plate=plate))
    # a little light
    for dx, dz in ((-rx + 1, -rz + 1), (rx - 1, -rz + 1), (-rx + 1, rz - 1),
                   (rx - 1, rz - 1)):
        _box(w, x + dx, base + ry - 1, z + dz, x + dx, base + ry - 1, z + dz,
             B.CRYSTAL)
    res.landmarks.append(Landmark("trial", "trial chamber",
                                  (x + 0.5, base + 2.0, z - rz - 1.5),
                                  radius=rx + 2.0,
                                  data=dict(plate=plate, gate=inner_gate,
                                            chest=chest_pos,
                                            inside=(x + 0.5, base + 2.0, z + 0.5))))


def _build_beacon(w: VoxelWorld, x: int, z: int, rng, res: WorldGenResult) -> None:
    """An unlit hilltop beacon -- the 'light the beacons' quest objective."""
    base = int(w.height[x, z])
    _flatten(w, x, z, 3, base, B.BRICK)
    _box(w, x - 1, base + 1, z - 1, x + 1, base + 2, z + 1, B.BRICK)
    _box(w, x, base + 3, z, x, base + 3, z, B.BEACON)
    # a bundle of firewood at the foot, so the puzzle is solvable locally
    res.landmarks.append(Landmark("beacon", "signal beacon",
                                  (x + 0.5, base + 3.0, z + 0.5), radius=5.0,
                                  data=dict(block=(x, base + 3, z))))


def _build_ruin(w: VoxelWorld, x: int, z: int, rng, res: WorldGenResult) -> None:
    """Broken brick walls plus a cracked wall hiding a loot cave (bomb task)."""
    base = int(w.height[x, z])
    _flatten(w, x, z, 7, base, B.GRAVEL)
    for _ in range(int(rng.integers(3, 7))):
        dx, dz = int(rng.integers(-6, 7)), int(rng.integers(-6, 7))
        ln = int(rng.integers(2, 6))
        hh = int(rng.integers(1, 4))
        if rng.random() < 0.5:
            _box(w, x + dx, base + 1, z + dz, x + dx + ln, base + hh, z + dz, B.BRICK)
        else:
            _box(w, x + dx, base + 1, z + dz, x + dx, base + hh, z + dz + ln, B.BRICK)
    # sealed vault: cracked wall -> chamber
    vx, vz = x + 5, z
    _box(w, vx, base + 1, vz - 2, vx + 6, base + 4, vz + 2, B.BRICK)
    _box(w, vx + 1, base + 1, vz - 1, vx + 5, base + 3, vz + 1, B.AIR)
    _box(w, vx, base + 1, vz, vx, base + 3, vz, B.CRACKED_WALL)
    res.chests.append(dict(pos=(vx + 4, base + 1, vz),
                           items=roll_loot("ruin", rng), kind="ruin"))
    res.landmarks.append(Landmark("ruin", "ruins", (x + 0.5, base + 1.0, z + 0.5),
                                  radius=9.0,
                                  data=dict(cracked=(vx, base + 2, vz),
                                            vault=(vx + 4, base + 1, vz))))


def _build_forge(w: VoxelWorld, x: int, z: int, rng, res: WorldGenResult) -> None:
    """An open-air forge: smelting and sword-forging station plus an ore pile."""
    base = int(w.height[x, z])
    _flatten(w, x, z, 5, base, B.BRICK)
    _box(w, x, base + 1, z, x, base + 1, z, B.FORGE)
    _box(w, x + 2, base + 1, z, x + 2, base + 1, z, B.COOKPOT)
    _box(w, x - 2, base + 1, z, x - 1, base + 1, z, B.ORE_IRON)
    _box(w, x - 2, base + 2, z, x - 2, base + 2, z, B.ORE_IRON)
    _box(w, x, base + 1, z + 3, x, base + 2, z + 3, B.METAL)
    for dx in (-3, 3):
        _box(w, x + dx, base + 1, z - 3, x + dx, base + 2, z - 3, B.TORCH_WALL)
    res.chests.append(dict(pos=(x + 1, base + 1, z - 2),
                           items=roll_loot("forge", rng), kind="forge"))
    res.bodies.append(dict(kind="log", pos=(x - 0.5, base + 2.0, z - 1.5)))
    res.landmarks.append(Landmark("forge", "wayside forge",
                                  (x + 0.5, base + 1.0, z + 0.5), radius=7.0,
                                  data=dict(forge=(x, base + 1, z),
                                            cookpot=(x + 2, base + 1, z))))


def _build_cook_camp(w: VoxelWorld, x: int, z: int, rng, res: WorldGenResult) -> None:
    """A traveller's camp: cooking pot, campfire, bedroll-ish planks."""
    base = int(w.height[x, z])
    _flatten(w, x, z, 4, base, B.DIRT)
    _box(w, x, base + 1, z, x, base + 1, z, B.COOKPOT)
    _box(w, x + 2, base + 1, z + 1, x + 2, base + 1, z + 1, B.CAMPFIRE)
    _box(w, x - 2, base + 1, z - 1, x - 1, base + 1, z - 1, B.PLANK)
    _box(w, x - 2, base + 1, z + 2, x - 2, base + 1, z + 2, B.SIGNPOST)
    res.landmarks.append(Landmark("cook_camp", "traveller's camp",
                                  (x + 0.5, base + 1.0, z + 0.5), radius=6.0,
                                  data=dict(cookpot=(x, base + 1, z),
                                            campfire=(x + 2, base + 1, z + 1))))


def _build_cave_mouth(w: VoxelWorld, x: int, z: int, rng, res: WorldGenResult) -> None:
    """A short loot cave dug into a hillside, sealed by a cracked wall."""
    base = int(w.height[x, z])
    depth = int(rng.integers(6, 12))
    _box(w, x - 1, base - 2, z, x + 1, base + 2, z + depth, B.AIR)
    _box(w, x - 1, base - 3, z, x + 1, base - 3, z + depth, B.STONE)
    _box(w, x - 1, base - 2, z, x + 1, base + 2, z, B.CRACKED_WALL)
    _box(w, x, base + 1, z + depth - 1, x, base + 1, z + depth - 1, B.CRYSTAL)
    res.chests.append(dict(pos=(x, base - 2, z + depth - 2),
                           items=roll_loot("cave", rng), kind="cave"))
    res.landmarks.append(Landmark("cave", "sealed cave",
                                  (x + 0.5, base - 1.0, z - 0.5), radius=6.0,
                                  data=dict(cracked=(x, base, z),
                                            inside=(x + 0.5, base - 1.0,
                                                    z + depth - 2.5))))


def _build_target_range(w: VoxelWorld, x: int, z: int, rng,
                        res: WorldGenResult) -> None:
    """Three wooden targets on posts -- the archery task's set-up."""
    base = int(w.height[x, z])
    _flatten(w, x, z, 8, base, B.GRASS)
    targets = []
    for i, dz in enumerate((-4, 0, 4)):
        tx = x + 7
        _box(w, tx, base + 1, z + dz, tx, base + 2, z + dz, B.PLANK)
        _box(w, tx, base + 3, z + dz, tx, base + 3, z + dz, B.TARGET)
        targets.append((tx, base + 3, z + dz))
    _box(w, x - 6, base + 1, z, x - 6, base + 1, z, B.SIGNPOST)
    res.chests.append(dict(pos=(x - 5, base + 1, z),
                           items=[["bow_crude", 1], ["arrow", 5], ["feather", 4]],
                           kind="range"))
    res.landmarks.append(Landmark("range", "archery range",
                                  (x - 5.5, base + 1.0, z + 0.5), radius=10.0,
                                  data=dict(targets=targets,
                                            line=(x - 5.5, base + 1.0, z + 0.5))))


def _build_lava_vent(w: VoxelWorld, x: int, z: int, rng, res: WorldGenResult) -> None:
    base = int(w.height[x, z])
    r = int(rng.integers(2, 5))
    _box(w, x - r, base - 2, z - r, x + r, base, z + r, B.LAVA)
    _box(w, x - r - 1, base - 3, z - r - 1, x + r + 1, base - 3, z + r + 1, B.BASALT)
    res.landmarks.append(Landmark("lava_vent", "lava vent",
                                  (x + 0.5, base + 1.0, z + 0.5), radius=r + 2.0))


def _scatter_bodies(cfg: WorldConfig, w: VoxelWorld, f: dict, rng,
                    res: WorldGenResult) -> None:
    """Loose dynamic objects: boulders on slopes, crates, logs, metal slabs."""
    n = int(0.0008 * cfg.size_x * cfg.size_z * cfg.structure_density)
    for _ in range(max(2, n)):
        x = int(rng.integers(4, cfg.size_x - 4))
        z = int(rng.integers(4, cfg.size_z - 4))
        y = w.spawn_y(x + 0.5, z + 0.5)
        if y <= cfg.sea_level + 0.5:
            continue
        slope = float(f["slope"][x, z])
        kind = ("boulder" if slope > 1.2 else
                str(rng.choice(["crate", "log", "boulder", "metal_slab"],
                               p=[0.3, 0.3, 0.25, 0.15])))
        res.bodies.append(dict(kind=kind, pos=(x + 0.5, y + 0.4, z + 0.5)))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def add_landmarks(cfg: WorldConfig, res: WorldGenResult, f: dict,
                  biome: np.ndarray, rng: np.random.Generator) -> None:
    """Place every landmark class, scaled to map area and density settings."""
    w = res.world
    area = cfg.size_x * cfg.size_z
    d = cfg.structure_density
    taken: list[tuple[float, float]] = []

    def take(sites):
        taken.extend([(float(a), float(b)) for a, b in sites])
        return sites

    counts = {
        "tower": max(1, int(area / 14000 * d)),
        "village": max(1, int(area / 22000 * d)),
        "camp": max(2, int(area / 7000 * d)),
        "trial": max(1, int(area / 20000 * d)),
        "beacon": max(2, int(area / 16000 * d)),
        "ruin": max(1, int(area / 18000 * d)),
        "forge": max(1, int(area / 26000 * d)),
        "cook_camp": max(1, int(area / 15000 * d)),
        "cave": max(1, int(area / 20000 * d)),
        "range": max(1, int(area / 40000 * d)),
    }

    for x, z in take(_sites(cfg, w, f, biome, rng, counts["tower"],
                            min_sep=46, max_slope=1.4, taken=taken)):
        _build_tower(w, x, z, rng, res)
    for x, z in take(_sites(cfg, w, f, biome, rng, counts["village"],
                            min_sep=44, max_slope=0.6, taken=taken,
                            allow_biomes=(Bi.PLAINS, Bi.FOREST, Bi.HILLS,
                                          Bi.SAVANNA, Bi.BEACH))):
        _build_village(w, x, z, rng, res)
    for x, z in take(_sites(cfg, w, f, biome, rng, counts["forge"],
                            min_sep=40, max_slope=0.8, taken=taken)):
        _build_forge(w, x, z, rng, res)
    for x, z in take(_sites(cfg, w, f, biome, rng, counts["cook_camp"],
                            min_sep=30, max_slope=0.8, taken=taken)):
        _build_cook_camp(w, x, z, rng, res)
    for x, z in take(_sites(cfg, w, f, biome, rng, counts["camp"],
                            min_sep=26, max_slope=0.9, taken=taken)):
        _build_camp(w, x, z, rng, res, int(biome[x, z]))
    for x, z in take(_sites(cfg, w, f, biome, rng, counts["trial"],
                            min_sep=40, max_slope=0.7, taken=taken)):
        _build_trial(w, x, z, rng, res)
    # beacons want high ground
    hi_thresh = int(np.percentile(np.maximum(w.height, 0), 78))
    for x, z in take(_sites(cfg, w, f, biome, rng, counts["beacon"],
                            min_sep=36, max_slope=1.2, taken=taken,
                            min_height=hi_thresh)):
        _build_beacon(w, x, z, rng, res)
    for x, z in take(_sites(cfg, w, f, biome, rng, counts["ruin"],
                            min_sep=32, max_slope=0.9, taken=taken)):
        _build_ruin(w, x, z, rng, res)
    for x, z in take(_sites(cfg, w, f, biome, rng, counts["cave"],
                            min_sep=30, max_slope=1.6, taken=taken)):
        _build_cave_mouth(w, x, z, rng, res)
    for x, z in take(_sites(cfg, w, f, biome, rng, counts["range"],
                            min_sep=30, max_slope=0.5, taken=taken)):
        _build_target_range(w, x, z, rng, res)

    volc = np.argwhere(f["volcanic"] > 0.6)
    if len(volc):
        for _ in range(max(1, int(area / 30000 * d))):
            x, z = volc[int(rng.integers(len(volc)))]
            _build_lava_vent(w, int(x), int(z), rng, res)

    _scatter_bodies(cfg, w, f, rng, res)
    w.recompute_heightmaps()
