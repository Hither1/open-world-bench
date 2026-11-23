"""Significant natural relief for authored courses: ridges, massifs, mesas with
cliff faces, valleys and gorges, basin lakes and caves.

Relief is applied after every route, pad, waterway and fixture exists. Natural
ground only rises, and the uplift ramps from zero beside anything ``keep``
protects to full height further away, so routes become valley floors and
gorges and no authored cell changes. Lakes are built up, not dug into cavern
roofs, and are held by a rim above their surface. Every field is keyed to the
world identity, never to the simulation RNG.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from ..core import blocks as B, noise as N
from . import organic as O

VEGETATION = (B.LOG, B.LEAVES, B.BUSH, B.TALL_GRASS, B.FLOWER, B.VINE)


def ground_top(vox, ylo, yhi):
    """Highest solid, non-vegetation block per column within [ylo, yhi]
    (``ylo-1`` where there is none)."""
    band = vox[:, ylo:yhi+1, :]
    solid = B.SOLID[band] & ~np.isin(band, VEGETATION)
    top = ylo+band.shape[1]-1-np.argmax(solid[:, ::-1, :], axis=1)
    return np.where(solid.any(axis=1), top, ylo-1)


class _Band:
    """A height band of a world, for vegetation clean-up that must not reach
    sky islands or caverns."""

    def __init__(self, world, ylo, yhi):
        self.vox = world.vox[:, ylo:yhi, :]


def clear_trees(world, changed, ylo, yhi):
    O.clear_disturbed_vegetation(_Band(world, ylo, yhi), changed)


def _grid(shape):
    return np.meshgrid(np.arange(shape[0], dtype=np.float32), np.arange(shape[1], dtype=np.float32), indexing="ij")


def landforms(shape, seed, *, wavelength=1.0, height=1.0):
    """Uplift in metres before ramping, and the mesa weight."""
    gx, gz = _grid(shape)
    L = wavelength

    def f(scale, k, octaves=4):
        return N.fbm2(gx/(scale*L), gz/(scale*L), seed+k, octaves)

    ridge = N.ridged2(gx/(240*L), gz/(240*L), seed+2, 5)
    ridge = np.clip((ridge-.42)/.58, 0, 1)**1.5
    massif = np.clip((.5+.5*f(320, 3))-.5, 0, .5)*2
    hills = .5+.5*f(90, 5)
    mountains = height*(30*ridge+20*massif**1.2+6*hills)
    m = f(150, 4)+.18*f(28, 6, 2)
    mesa = np.clip((m-.2)/.035, 0, 1)
    tier = np.clip(np.floor(3*(.5+.5*f(420, 7, 2))), 0, 2)
    plateau = height*(13+7*tier)*mesa+1.5*hills
    return np.maximum(mountains, plateau), mesa


def ramp(keep, seed, *, start=8.0, width=(14.0, 36.0), wavelength=1.0):
    """0 within ``start`` metres of ``keep``, rising smoothly to 1 over a
    width that varies by region: narrow gorges in some, wide valleys in others."""
    gx, gz = _grid(keep.shape)
    d = distance_transform_edt(~keep)
    w = width[0]+(width[1]-width[0])*(.5+.5*N.fbm2(gx/(170*wavelength), gz/(170*wavelength), seed+1, 3))
    t = np.clip((d-start)/w, 0, 1)
    return t*t*(3-2*t), d


def basin_lakes(world, keep, identity, *, count, level, ylo, yhi, distance=30.0, radii=(13, 21)):
    """Lakes whose surface is ``level``, built up from low natural ground and
    held by a grassy rim one block higher. Returns the lake+rim mask."""
    vox = world.vox
    seed = O.stable_seed("lakes", *identity)
    rng = np.random.default_rng(seed)
    G = ground_top(vox, ylo, yhi)
    d = distance_transform_edt(~keep)
    shape = G.shape
    lakes = np.zeros(shape, bool)
    made = []
    cells = np.argwhere((d >= distance+radii[1]) & (G <= level-2) & (G >= ylo+2))
    for i in rng.permutation(len(cells)):
        if len(made) >= count:
            break
        x, z = map(int, cells[i])
        r = float(rng.uniform(*radii))
        if any(np.hypot(x-a, z-b) < 2.5*radii[1] for a, b in made):
            continue
        R = int(r*1.4)+5
        if not (R < x < shape[0]-R and R < z < shape[1]-R):
            continue
        lo0, lo1 = x-R, z-R
        local = O.blob((2*R+1, 2*R+1), (R+.5, R+.5), (r, r*float(rng.uniform(.65, 1.0))), identity+(x, z),
                       exponent=2.0, amplitude=.22)
        mask = np.zeros(shape, bool); mask[lo0:lo0+2*R+1, lo1:lo1+2*R+1] = local
        rim = binary_dilation(mask, iterations=3) & ~mask
        area = mask | rim
        if keep[area].any() or lakes[binary_dilation(area, iterations=6)].any():
            continue
        if (G[mask] > level-1).any() or (G[area] < ylo+2).any():
            continue
        # Bed: deepest in the middle; built up from natural ground, never below
        # its own column's ground minus one (cavern roofs stay intact).
        dist = distance_transform_edt(mask)
        depth = np.clip(1+dist/2.5, 1, 4).astype(int)
        xs, zs = np.nonzero(mask)
        bed = np.maximum(level-depth[xs, zs], G[xs, zs]-1)
        for x_, z_, b, g in zip(xs, zs, bed, G[xs, zs]):
            column = vox[x_, :, z_]
            if b > g:
                column[g+1:b+1] = B.DIRT
            column[b] = B.SAND if b >= level-1 else B.MUD
            column[b+1:level+1] = B.WATER
            column[level+1:level+6] = np.where(np.isin(column[level+1:level+6], VEGETATION), B.AIR, column[level+1:level+6])
        rx, rz = np.nonzero(rim)
        for x_, z_ in zip(rx, rz):
            g = G[x_, z_]
            if g < level+1:
                vox[x_, g+1:level+1, z_] = B.DIRT
                vox[x_, level+1, z_] = B.GRASS
        # A natural embankment: ground falls away from the rim at about 30
        # degrees instead of leaving the lake on a vertical pedestal.
        bank = distance_transform_edt(~area)
        near = (bank > 0) & (bank <= 14) & ~keep
        target = np.floor(level+1-.6*bank).astype(int)
        ex, ez = np.nonzero(near & (target > G))
        for x_, z_ in zip(ex, ez):
            g, t = int(G[x_, z_]), int(target[x_, z_])
            if np.isin(vox[x_, g+1:t+1, z_], VEGETATION).any() or (vox[x_, g+1:t+1, z_] == B.WATER).any():
                continue
            vox[x_, g+1:t, z_] = B.DIRT
            vox[x_, t, z_] = B.GRASS
            if vox[x_, g, z_] == B.GRASS:
                vox[x_, g, z_] = B.DIRT
            G[x_, z_] = t
        lakes |= area | (near & (target >= G-1) & (bank <= 6))
        made.append((x, z))
    return lakes, made


def raise_ground(world, keep, identity, *, ylo, yhi, cap, low_cap=None, low_cap_value=None,
                 wavelength=1.0, height=1.0, start=8.0, width=(14.0, 36.0)):
    """Raise natural ground away from ``keep``. Returns (new ground, old ground, mesa weight)."""
    vox = world.vox
    seed = O.stable_seed("relief", *identity) % 1000003
    U, mesa = landforms(keep.shape, seed, wavelength=wavelength, height=height)
    r, _ = ramp(keep, seed, start=start, width=width, wavelength=wavelength)
    G = ground_top(vox, ylo, yhi)
    T = np.rint(G+U*r).astype(int)
    T = np.minimum(T, cap)
    if low_cap is not None:
        T = np.where(low_cap, np.minimum(T, low_cap_value), T)
    T = np.where(keep | (G < ylo), G, np.maximum(T, G))
    changed = T > G
    clear_trees(world, binary_dilation(changed, iterations=1), ylo, min(yhi+20, vox.shape[1]))
    gy, gx = np.gradient(T.astype(np.float32))
    slope = np.hypot(gx, gy)
    hashv = N.hash_unit(*np.indices(T.shape), seed % 100003)
    meadow = (slope < .9) & (T < cap-4)
    top_block = np.where(meadow, B.GRASS, np.where(slope < 1.4, B.DIRT, np.where(hashv < .25, B.MOSS_STONE, B.STONE)))
    strata = mesa > .5
    for y in range(max(ylo, int(G[changed].min())+1) if changed.any() else ylo, int(T.max())+1):
        sel = changed & (y > G) & (y <= T) & (vox[:, y, :] == B.AIR)
        if not sel.any():
            continue
        depth = T-y
        block = np.where(depth == 0, top_block,
                         np.where((depth <= 2) & (top_block == B.GRASS), B.DIRT,
                                  np.where(strata & ((y//2) % 3 == 1), B.SAND, B.STONE)))
        vox[:, y, :] = np.where(sel, block, vox[:, y, :])
    xs, zs = np.nonzero(changed)
    buried = vox[xs, G[xs, zs], zs] == B.GRASS
    vox[xs[buried], G[xs[buried], zs[buried]], zs[buried]] = B.DIRT
    return T, G, mesa


def caves(world, T, G, keep, identity, *, count, ylo, length=16, min_cover=3):
    """Short arched caves driven into steep relief from natural ground at its
    foot, clear of ``keep``. Returns the cave mouths."""
    vox = world.vox
    rng = np.random.default_rng(O.stable_seed("caves", *identity))
    d = distance_transform_edt(~keep)
    shape = T.shape
    from scipy.ndimage import maximum_filter, minimum_filter
    rise = maximum_filter(T, size=7)-T
    base = minimum_filter(T, size=3)
    cells = np.argwhere((rise >= 12) & (d >= 10) & (T-base <= 1) & (T >= ylo+2) & (T-G <= 8))
    mouths = []
    for i in rng.permutation(len(cells)):
        if len(mouths) >= count:
            break
        x, z = map(int, cells[i])
        if any(np.hypot(x-a, z-b) < 90 for a, b, _ in mouths):
            continue
        win = T[x-4:x+5, z-4:z+5]
        if win.shape != (9, 9):
            continue
        k = np.unravel_index(np.argmax(win), win.shape)
        direction = np.array((k[0]-4, k[1]-4), float)
        if not direction.any():
            continue
        direction /= np.linalg.norm(direction)
        side = np.array((-direction[1], direction[0]))
        floor = int(T[x, z])+1
        path = [np.array((x+.5, z+.5))+direction*s for s in np.arange(0, length, .5)]
        cols = [tuple(np.floor(p).astype(int)) for p in path]
        if not all(0 < c[0] < shape[0]-1 and 0 < c[1] < shape[1]-1 for c in cols):
            continue
        if any(T[c] < floor+4+min_cover for c in cols[8:]) or any(keep[c] for c in cols):
            continue
        carved = 0
        for s, p in zip(np.arange(0, length, .5), path):
            taper = 1.0 if s < length-4 else max(.35, (length-s)/4)
            half, tall = 2.3*taper, 4.4*taper
            for u in np.arange(-half-.5, half+.6, .5):
                q = np.floor(p+side*u).astype(int)
                arch = tall*np.sqrt(max(0., 1-(u/max(half, 1e-3))**2))
                for y in range(floor, floor+int(np.ceil(arch))):
                    if O.CUTTABLE[vox[q[0], y, q[1]]] and vox[q[0], y, q[1]] != B.WATER:
                        vox[q[0], y, q[1]] = B.AIR; carved += 1
                if arch > 0 and B.SOLID[vox[q[0], floor-1, q[1]]]:
                    vox[q[0], floor-1, q[1]] = B.GRAVEL
        mouths.append((x, z, dict(floor=floor, direction=direction.tolist(), carved=carved)))
    return [dict(pos=[a+.5, m["floor"], b+.5], direction=m["direction"], carved=m["carved"]) for a, b, m in mouths]
