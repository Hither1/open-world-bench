"""Original industrial settlements for Tempest; structural geometry is collision.

Each building owns a stable identity, material palette and voxel envelope. Small
trim and machinery live in the presentation layer; walls, roofs, floors and
doorways are authored here before any reference trajectory is collected.
"""
from __future__ import annotations

import hashlib
import numpy as np

from ..core import blocks as B


KINDS = ("boathouse", "pump_house", "foundry", "gantry", "observatory",
         "archive", "workshop", "shelter", "cistern", "signal_tower")


def identity(seed, name):
    return int.from_bytes(hashlib.sha256(f"tempest:{seed}:{name}".encode()).digest()[:4], "little")


def envelope(kind):
    """Integer closed boxes, relative to a walkable floor at y=0."""
    parts = []

    def box(lo, hi, block):
        parts.append((tuple(lo), tuple(hi), block))

    box((-8, -2, -7), (8, -1, 7), B.BRICK)
    if kind in ("gantry", "signal_tower"):
        height = 17 if kind == "signal_tower" else 12
        for x in (-6, 6):
            for z in (-5, 5):
                box((x, 0, z), (x, height, z), B.METAL)
        box((-6, height, -5), (6, height, 5), B.METAL)
        if kind == "signal_tower":
            for x in (-6, 6):
                box((x, height+1, -5), (x, height+2, 5), B.PLANK)
            box((-6, height+1, 5), (6, height+2, 5), B.PLANK)
        return parts
    if kind == "cistern":
        for x in (-6, 6):
            box((x, 0, -5), (x, 5, 5), B.BRICK)
        for z in (-5, 5):
            box((-6, 0, z), (6, 5, z), B.BRICK)
        box((-5, 0, -4), (5, 3, 4), B.WATER)
        return parts
    height = 8 if kind in ("foundry", "observatory", "archive") else 5
    wall = B.BRICK if kind in ("foundry", "pump_house", "archive", "observatory") else B.PLANK
    # Wide open entrances and real windows; no painted doors over solid walls.
    for x in (-7, 7):
        box((x, 0, -6), (x, height, 6), wall)
    for z in (-6, 6):
        box((-7, 0, z), (-3, height, z), wall)
        box((3, 0, z), (7, height, z), wall)
        box((-2, 4, z), (2, height, z), wall)
    for x in (-7, 7):
        box((x, 2, -2), (x, 3, 2), B.AIR)
    if kind == "observatory":
        box((-7, height, -6), (7, height, 6), B.BRICK)
        for x, z in ((-5, -4), (-5, 4), (5, -4), (5, 4)):
            box((x, height+1, z), (x, height+5, z), B.METAL)
    else:
        for x in range(-8, 9):
            y = height+1+(8-abs(x))//2
            box((x, y, -7), (x, y, 7), B.METAL if kind == "foundry" else B.PLANK)
        if kind in ("foundry", "pump_house"):
            box((4, 0, 3), (5, height+9, 4), B.BRICK)
    return parts


def install_building(env, name, kind, pos):
    p = np.asarray(pos, int)
    env.world.fill_box(p+(-9, 0, -8), p+(9, 24, 8), B.AIR)
    for lo, hi, block in envelope(kind):
        env.world.fill_box(p+lo, p+hi, block)
    record = dict(id=name, kind=kind, pos=p.tolist(), appearance=identity(env._seed, name))
    env.architecture.append(record)
    return record


def author_settlements(env):
    """Choose clear side plots; never replace an existing mechanism or road."""
    env.architecture = []
    occupied = []
    themes = {
        "B01": ("archive", "shelter"), "B02": ("boathouse", "workshop"),
        "B03": ("boathouse", "gantry"), "B04": ("workshop", "signal_tower"),
        "B05": ("archive", "shelter"), "B06": ("gantry", "workshop"),
        "B07": ("signal_tower", "archive"), "B08": ("observatory", "archive"),
        "B09": ("shelter", "workshop"), "B10": ("foundry", "gantry"),
        "B11": ("foundry", "cistern"), "B12": ("pump_house", "shelter"),
        "B13": ("boathouse", "gantry"), "B14": ("pump_house", "archive"),
        "B15": ("pump_house", "cistern"), "B16": ("boathouse", "cistern"),
        "B17": ("foundry", "gantry"), "B18": ("signal_tower", "workshop"),
        "B19": ("observatory", "archive"), "B20": ("signal_tower", "shelter"),
        "B21": ("foundry", "workshop"), "B22": ("gantry", "signal_tower"),
        "B23": ("boathouse", "workshop"), "B24": ("shelter", "archive"),
    }
    targets = [obj["pos"] for obj in env.targets.values()]
    for m in env.milestones:
        for k, kind in enumerate(themes.get(m.id, ("workshop",))):
            placed = False
            for radius in (35, 50, 65, 82):
                for index in range(12):
                    angle = (index+k*5)*np.pi/6
                    x, z = np.rint(m.pos[[0, 2]]+radius*np.array((np.cos(angle), np.sin(angle)))).astype(int)
                    if not (12 < x < env.world.size_x-12 and 12 < z < env.world.size_z-12):
                        continue
                    if env._road_mask[x-11:x+12, z-10:z+11].any():
                        continue
                    if any(abs(x-p[0]) < 16 and abs(z-p[2]) < 16 and abs(m.y-p[1]) < 20 for p in targets):
                        continue
                    if any(y == m.y and abs(x-px) < 24 and abs(z-pz) < 23 for px, y, pz in occupied):
                        continue
                    if (env.world.vox[x-10:x+11, m.y-2:m.y, z-9:z+10] == B.WATER).any():
                        continue
                    # Side plots have their own foundations and supporting pillars.
                    install_building(env, f"{m.id}-{kind}-{k}", kind, (x, m.y, z))
                    if m.layer == 2:
                        for dx, dz in ((-7, -6), (-7, 6), (7, -6), (7, 6)):
                            env.world.fill_box((x+dx, m.y-10, z+dz), (x+dx, m.y-3, z+dz), B.BRICK)
                    occupied.append((x, m.y, z))
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                raise RuntimeError(f"No safe plot for {m.id} {kind}")
    env.world.recompute_heightmaps()


def author_engine(env):
    """An accessible industrial sky island, with collision-backed engine and cover."""
    from .skylands import _island

    w = env.world
    x, z = env.point(550, 600)
    y = env.levels[2]
    env.engine_pos = np.array((x, y, z), float)
    fx, fz = x-200, z+10
    _island(w, x-80, z, 230, 72, y-1, 34, .71)
    # Separate service road passes north of the watercourse and all machinery.
    env._road(w, 2, [(x-220, z-28), (x, z-28), (x, z-40)])
    env._road(w, 2, [(x-220, z-28), (x-220, env.point(60, 180)[1]), env.point(60, 180)])
    env._pad(w, (x, y, z), radius=34)
    env._pad(w, (fx, y, fz), radius=12)
    # Reactor foundation and four heat-exchanger towers. The ring is walkable.
    w.fill_box((x-5, y, z-5), (x+5, y+12, z+5), B.METAL)
    w.fill_box((x-7, y+13, z-7), (x+7, y+14, z+7), B.METAL)
    for dx, dz in ((-23, -23), (-23, 23), (23, -23), (23, 23)):
        w.fill_box((x+dx-2, y, z+dz-2), (x+dx+2, y+16, z+dz+2), B.BRICK)
    # Three isolation desks and continuous sheltered service bays.
    env.lock_positions = [env.engine_pos+off for off in ((-15, 1, -10), (15, 1, -10), (0, 1, 20))]
    for p in env.lock_positions:
        px, py, pz = np.rint(p).astype(int)
        w.fill_box((px-4, y, pz+4), (px+4, y+3, pz+4), B.BRICK)
        w.fill_box((px-4, y+5, pz-3), (px+4, y+5, pz+4), B.METAL)
    return (fx, y, fz)
