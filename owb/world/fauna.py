"""Creature placement for authored voxel courses (simulation spawns).

Monsters live in camps around a den in lowland they cannot walk out of onto
the course (creatures step at most 0.7 m and do not jump). By day most rest in
their dens and one per camp keeps watch; all come out after dusk, and
stalkers appear only at night (see ``NpcSystem.step``). Wildlife grazes in
the same lowlands, frogs keep to shores and fish to deeper water.

Placement uses a dedicated generator seeded from the world identity, not the
environment's random generator.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core import blocks as B
from .organic import can_reach, stable_seed, walk_heights, water_top, top_solid

CAMPS = (("bokoblin", "bokoblin", "moblin"), ("bokoblin", "chuchu", "bokoblin"),
         ("chuchu", "chuchu"), ("bokoblin", "moblin"))
NIGHT_ONLY = ("stalker", "stalker", "stalker")
HERDS = (("deer", 3, 1.0), ("boar", 2, 1.0), ("goat", 2, 1.0), ("bird", 3, 1.0))


def _spread(rng, cells, count, spacing, taken=()):
    chosen = list(taken)
    order = rng.permutation(len(cells))
    out = []
    for i in order:
        c = cells[i]
        if all(np.hypot(*(c - o)) >= spacing for o in chosen):
            chosen.append(c)
            out.append(c)
            if len(out) == count:
                break
    return out


def populate(world, course, identity, *, sea_level, heights=None, monster_clearance=None, scale=1.0,
             wildlife_band=None, wildlife_scale=None):
    """Return ``npc_spawns`` dictionaries for the course world.

    By default creatures are kept to ground from which they cannot walk onto
    the course. Open worlds pass ``monster_clearance`` instead: dens are then
    placed at least that many metres from any trail, beyond monster sight, and
    wildlife at least six metres away. ``heights`` overrides the standing
    height map (layered worlds); ``scale`` multiplies the number of camps and herds.

    ``wildlife_band=(near, far)`` grazes the (non-hostile, fleeing) herds in
    meadows that many metres from the course, where travellers see them, on
    any standing ground; ``wildlife_scale`` then sets the number of herd sets
    independently of the monster camps.
    """
    rng = np.random.default_rng(stable_seed("fauna", *identity))
    heights = walk_heights(world) if heights is None else heights
    sx, sz = heights.shape
    distance = ndimage.distance_transform_edt(~course)
    if monster_clearance is None:
        # Columns from which a creature could walk onto the course.
        reach = can_reach(heights, course, climb=0)
    else:
        reach = distance < 6
    local_max = ndimage.maximum_filter(np.where(heights >= 0, heights, 999), size=3)
    local_min = ndimage.minimum_filter(np.where(heights >= 0, heights, -999), size=3)
    flat = (local_max - local_min) <= 1
    vox = world.vox
    ys = np.clip(heights, 0, vox.shape[1] - 4)
    headroom = np.ones_like(flat)
    for dy in (1, 2, 3):
        headroom &= ~B.SOLID[np.take_along_axis(vox, (ys + dy)[:, None, :], axis=1)[:, 0, :]]
    standing = (heights > sea_level) & ~course & flat & headroom
    border = np.zeros_like(standing)
    border[4:-4, 4:-4] = True
    standing &= border
    land = standing & ~reach
    spawns = []

    def at(cell, **data):
        x, z = map(int, cell)
        return (x + .5, float(heights[x, z]) + 1.0, z + .5)

    camp_cells = np.argwhere(land & (distance >= (8 if monster_clearance is None else monster_clearance)))
    copies = max(1, int(round(scale)))
    camp_kinds = list(CAMPS) * copies + [(k,) for k in NIGHT_ONLY] * copies
    camps = _spread(rng, camp_cells, len(camp_kinds), 18.0 if scale <= 1 else 40.0)
    for k, den in enumerate(camps):
        kinds = camp_kinds[k]
        den_pos = at(den)
        near = [c for c in camp_cells if 1.5 <= np.hypot(*(c - den)) <= 4.5]
        for j, kind in enumerate(kinds):
            cell = near[int(rng.integers(len(near)))] if near else den
            spawns.append(dict(kind=kind, pos=at(cell), home=(den_pos[0], den_pos[2]), den=den_pos,
                               day_active=(j == 0 and kind != "stalker"), roam=6.0))
    taken = list(camps)
    herd_copies = copies
    if wildlife_band is None:
        graze = np.argwhere(land & (distance >= 4))
        herd_spacing = 14.0 if scale <= 1 else 35.0
    else:
        graze = np.argwhere(standing & (distance >= wildlife_band[0]) & (distance <= wildlife_band[1]))
        herd_copies = max(1, int(round(wildlife_scale if wildlife_scale is not None else scale)))
        herd_spacing = max(12.0, float(wildlife_band[1]))
    order = ([h for h in HERDS for _ in range(herd_copies)] if wildlife_band is None
             else [h for _ in range(herd_copies) for h in HERDS])
    for kind, count, _ in order:
        if kind == "goat" and wildlife_band is None:
            pool = graze[heights[graze[:, 0], graze[:, 1]] >= np.percentile(heights[land], 60)] if len(graze) else graze
        elif kind == "goat" and len(graze):
            pool = graze[heights[graze[:, 0], graze[:, 1]] >= np.percentile(heights[graze[:, 0], graze[:, 1]], 60)]
        else:
            pool = graze
        if not len(pool):
            continue
        centre = _spread(rng, pool, 1, herd_spacing, taken)
        if not centre:
            continue
        taken += centre
        members = pool[np.hypot(*(pool - centre[0]).T) <= 6]
        for j in range(count):
            cell = members[int(rng.integers(len(members)))] if len(members) else centre[0]
            spawns.append(dict(kind=kind, pos=at(cell), home=(cell[0] + .5, cell[1] + .5), roam=8.0))
    # Frogs on shores next to water; fish in water at least two blocks deep.
    wet = water_top(world)
    ground = top_solid(world)
    shore = land & ndimage.binary_dilation(wet > ground, iterations=1)
    for cell in _spread(rng, np.argwhere(shore), 3 * copies, 10.0):
        spawns.append(dict(kind="frog", pos=at(cell), home=(cell[0] + .5, cell[1] + .5), roam=3.0))
    deep = (wet - ground >= 2) & (wet > 0)
    deep[:4] = deep[-4:] = False
    deep[:, :4] = deep[:, -4:] = False
    for cell in _spread(rng, np.argwhere(deep), 8 * copies, 7.0):
        x, z = map(int, cell)
        y = float(wet[x, z]) - 0.6
        spawns.append(dict(kind="fish", pos=(x + .5, y, z + .5), home=(x + .5, z + .5), roam=4.0, aquatic=True))
    return spawns
