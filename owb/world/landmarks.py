"""Landmark architecture placed on the relief: a hilltop castle and traveller
camps beside the trails.

The castle's walls, towers, gatehouse and keep are collision voxels on a
levelled plateau; banners, roofs and lamps are presentation trim. Camps are a
campfire block (collision) with presentation-only tents around it. Placement
uses stable world identity, never the simulation RNG.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter

from ..core import blocks as B
from . import organic as O

CASTLE_HALF = 13          # curtain wall half-size; towers overhang by 3
CASTLE_FOOT = CASTLE_HALF+5


def castle_parts(record):
    """Integer closed boxes relative to the courtyard floor at y=0, gate on -z,
    before rotation (see settlements.local_parts)."""
    h, parts = CASTLE_HALF, []

    def box(lo, hi, block=B.BRICK):
        parts.append((tuple(lo), tuple(hi), block))

    box((-h-3, -3, -h-3), (h+3, -1, h+3))                       # levelled plinth
    for s in (-1, 1):                                             # curtain walls, 2 thick, 9 high
        box((s*h-(s > 0), 0, -h), (s*h+(s < 0), 8, h))
        box((-h, 0, s*h-(s > 0)), (h, 8, s*h+(s < 0)))
    for x in range(-h, h+1, 2):                                   # merlons
        for z in (-h, h):
            box((x, 9, z), (x, 9, z))
    for z in range(-h, h+1, 2):
        for x in (-h, h):
            box((x, 9, z), (x, 9, z))
    for sx in (-1, 1):                                            # corner towers 7x7, 15 high
        for sz in (-1, 1):
            cx, cz = sx*h, sz*h
            box((cx-3, 0, cz-3), (cx+3, 14, cz+3))
            box((cx-2, 1, cz-2), (cx+2, 13, cz+2), B.AIR)
            for k in range(-3, 4, 2):
                box((cx+k, 15, cz-3), (cx+k, 15, cz-3)); box((cx+k, 15, cz+3), (cx+k, 15, cz+3))
                box((cx-3, 15, cz+k), (cx-3, 15, cz+k)); box((cx+3, 15, cz+k), (cx+3, 15, cz+k))
    for sx in (-1, 1):                                            # gatehouse towers
        box((sx*4-2, 0, -h-3), (sx*4+2, 12, -h+2))
    box((-2, 0, -h-3), (2, 5, -h+2), B.AIR)                       # gate passage, 5 wide, 6 high
    box((-5, 6, -h-3), (5, 7, -h-3))                              # arch lintel over the gate
    box((-5, 0, 3), (5, 19, 11))                                  # keep
    box((-4, 1, 4), (4, 18, 10), B.AIR)
    box((-1, 0, 3), (1, 3, 3), B.AIR)                             # keep door
    for y in (6, 11, 16):                                         # keep windows
        for x in (-3, 3):
            box((x, y, 3), (x, y+1, 3), B.AIR); box((x, y, 11), (x, y+1, 11), B.AIR)
        for z in (5, 9):
            box((-5, y, z), (-5, y+1, z), B.AIR); box((5, y, z), (5, y+1, z), B.AIR)
    for k in range(-5, 6, 2):
        box((k, 20, 3), (k, 20, 3)); box((k, 20, 11), (k, 20, 11))
    for z in range(3, 12, 2):
        box((-5, 20, z), (-5, 20, z)); box((5, 20, z), (5, 20, z))
    return parts


def place_castle(env, T, G, keep, identity, *, level, low_cap=None, min_rise=12, view=(40.0, 140.0)):
    """Find the highest level-able plateau near (but not on) the trails and
    build the castle there. Returns the record or None."""
    from .settlements import local_parts
    w = env.world
    S = 2*CASTLE_FOOT+1
    flat = (maximum_filter(T, size=S)-minimum_filter(T, size=S)) <= 3
    d = distance_transform_edt(~keep)
    ok = flat & (T >= level+min_rise) & (d >= CASTLE_FOOT+4) & (d <= view[1])
    if low_cap is not None:
        ok &= ~low_cap
    ok[:CASTLE_FOOT+2] = ok[-CASTLE_FOOT-2:] = False
    ok[:, :CASTLE_FOOT+2] = ok[:, -CASTLE_FOOT-2:] = False
    cells = np.argwhere(ok)
    if not len(cells):
        return None
    score = T[cells[:, 0], cells[:, 1]]-.08*np.abs(d[cells[:, 0], cells[:, 1]]-70)
    x, z = map(int, cells[int(np.argmax(score))])
    lo = (x-CASTLE_FOOT, z-CASTLE_FOOT); hi = (x+CASTLE_FOOT+1, z+CASTLE_FOOT+1)
    floor = int(np.median(T[lo[0]:hi[0], lo[1]:hi[1]]))+1
    if floor+26 >= w.size_y:
        return None
    # Gate toward the nearest trail.
    gy, gx = np.gradient(d.astype(np.float32))
    toward = -np.array((gy[x, z], gx[x, z]))
    if not np.any(toward):
        toward = np.array((0., -1.))
    from .settlements import door_normal
    k = int(max(range(4), key=lambda r: float(door_normal(r) @ toward)))
    record = dict(id="castle", kind="castle", layer=1, pos=[x, floor, z], rotation=k,
                  appearance=O.stable_seed("castle", *identity) & 0xffffffff)
    # Level the plateau: cut natural ground above the floor, fill below it.
    vox = w.vox
    for i in range(lo[0], hi[0]):
        for j in range(lo[1], hi[1]):
            col = vox[i, :, j]
            above = col[floor:floor+28]
            col[floor:floor+28] = np.where(O.CUTTABLE[above], B.AIR, above)
            g = int(T[i, j])
            if g < floor-1:
                col[g+1:floor-1] = B.STONE
            col[floor-1] = B.GRASS
    p = np.array((x, floor, z))
    for a, b, block in local_parts(record):
        w.fill_box(p+a, p+b, block)
    return record


def place_camps(env, T, G, keep, road, identity, *, count, level, spacing=110.0, avoid=None):
    """Traveller camps 6-12 m off the trails on level natural ground."""
    rng = np.random.default_rng(O.stable_seed("camps", *identity))
    d_road = distance_transform_edt(~road)
    flat = (maximum_filter(T, size=9)-minimum_filter(T, size=9)) <= 1
    ok = flat & (T == G) & (np.abs(T-(level-1)) <= 1) & (d_road >= 6) & (d_road <= 12)
    ok &= distance_transform_edt(~(keep & ~road)) >= 6
    if avoid is not None:
        ok &= ~avoid
    cells = np.argwhere(ok)
    camps = []
    vox = env.world.vox
    for i in rng.permutation(len(cells)):
        if len(camps) >= count:
            break
        x, z = map(int, cells[i])
        if any(np.hypot(x-c["pos"][0], z-c["pos"][2]) < spacing for c in camps):
            continue
        y = int(T[x, z])+1
        if (vox[x-4:x+5, y:y+4, z-4:z+5] != B.AIR).any():
            continue
        vox[x, y, z] = B.CAMPFIRE_LIT
        n = int(rng.integers(2, 4))
        base = float(rng.uniform(0, 2*np.pi))
        tents = []
        for t in range(n):
            a = base+t*2*np.pi/n+float(rng.uniform(-.3, .3))
            r = float(rng.uniform(3.6, 4.6))
            tents.append(dict(pos=[x+.5+r*np.cos(a), float(y), z+.5+r*np.sin(a)], yaw=float(a+np.pi),
                              style=int(rng.integers(0, 2)), size=float(rng.uniform(.9, 1.2))))
        camps.append(dict(id=f"camp-{len(camps):02}", kind="camp", layer=1, pos=[x, y, z], rotation=0,
                          appearance=O.stable_seed("camp", *identity, x, z) & 0xffffffff, tents=tents))
    return camps
