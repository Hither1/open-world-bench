"""Organic course geometry for authored voxel worlds.

This is simulation geometry (collision and walkable ground), not presentation.
Curved roads of varying width, irregular pads and graded earth banks replace
axis-aligned boxes. Shape variation is derived from stable identities and
seeded noise only; it never draws from an environment's random generator.
"""
from __future__ import annotations

from collections import deque
import hashlib

import numpy as np
from scipy import ndimage

from ..core import blocks as B
from ..core.noise import fbm2

TREE_BLOCKS = (B.LOG, B.LEAVES)
LOOSE_BLOCKS = (B.TALL_GRASS, B.FLOWER, B.BUSH, B.VINE)
#: Ground that grading may cut away; built blocks (brick, planks, moss stone,
#: metal, glass ...) are never removed by terrain shaping.
CUTTABLE = np.zeros(B.NUM_BLOCKS, bool)
CUTTABLE[[B.STONE, B.DIRT, B.GRASS, B.SAND, B.SNOW, B.ICE, B.MUD, B.GRAVEL, B.BASALT, B.SNOW_PACK,
          B.ORE_IRON, B.ORE_FLINT, B.LOG, B.LEAVES, B.TALL_GRASS, B.FLOWER, B.BUSH, B.VINE]] = True


def stable_seed(*identity) -> int:
    return int.from_bytes(hashlib.sha256(repr(identity).encode()).digest()[:4], "little") & 0x7FFFFFFF


def unit(*identity) -> float:
    return int.from_bytes(hashlib.sha256(repr(identity).encode()).digest()[:8], "little") / 2.0 ** 64


def wobble(t, *identity, terms=3):
    """Smooth noise in [-1, 1] along a parameter, from hashed sine terms."""
    t = np.asarray(t, float)
    out, total = np.zeros_like(t), 0.0
    for k in range(terms):
        frequency = 0.6 + 0.7 * k + 0.5 * unit(*identity, "f", k)
        phase = 2 * np.pi * unit(*identity, "p", k)
        weight = 1.0 / (k + 1)
        out += weight * np.sin(2 * np.pi * frequency * t + phase)
        total += weight
    return out / total


def ring(theta, *identity, terms=4):
    """Periodic noise in [-1, 1] around a closed outline."""
    theta = np.asarray(theta, float)
    out, total = np.zeros_like(theta), 0.0
    for k in range(2, 2 + terms):
        weight = 1.0 / (k - 1)
        out += weight * np.sin(k * theta + 2 * np.pi * unit(*identity, "ring", k))
        total += weight
    return out / total


def _window(t, t0, t1, ramp=0.1):
    """1 inside [t0, t1], easing to 0 over ``ramp`` outside it."""
    a = np.clip((t - (t0 - ramp)) / ramp, 0, 1)
    b = np.clip(((t1 + ramp) - t) / ramp, 0, 1)
    a, b = a * a * (3 - 2 * a), b * b * (3 - 2 * b)
    return np.minimum(a, b)


class Road:
    """A curved centreline with a varying half-width, in X/Z metres."""

    def __init__(self, points, halfwidth):
        self.points = np.asarray(points, float)
        self.halfwidth = np.broadcast_to(np.asarray(halfwidth, float), (len(self.points),)).copy()

    @classmethod
    def meander(cls, a, b, identity, *, amplitude, width=(1.6, 2.6), step=0.5, pins=()):
        """Chord from ``a`` to ``b`` with a lateral meander that vanishes at both
        ends. ``pins`` are ``(t0, t1, halfwidth)`` windows held straight with a
        fixed width, for mechanisms authored on exact cells."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        chord = b - a
        length = float(np.linalg.norm(chord))
        t = np.linspace(0, 1, max(3, int(length / step) + 1))
        normal = np.array((-chord[1], chord[0])) / max(length, 1e-9)
        offset = amplitude * np.sin(np.pi * t) * wobble(t, *identity, "offset")
        lo, hi = width
        half = lo + (hi - lo) * 0.5 * (1 + wobble(t * length / 18.0, *identity, "width"))
        for t0, t1, fixed in pins:
            hold = _window(t, t0, t1)
            offset *= 1 - hold
            inside = (t >= t0) & (t <= t1)
            half = np.where(inside, fixed, half * (1 - hold) + fixed * hold)
        return cls(a + t[:, None] * chord + offset[:, None] * normal, half)

    @classmethod
    def smooth(cls, controls, halfwidth, step=0.5, iterations=3):
        """Chaikin-smoothed polyline through authored control points."""
        p = np.asarray(controls, float)
        for _ in range(iterations):
            q = 0.75 * p[:-1] + 0.25 * p[1:]
            r = 0.25 * p[:-1] + 0.75 * p[1:]
            p = np.vstack([p[:1], np.stack((q, r), 1).reshape(-1, 2), p[-1:]])
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        samples = np.linspace(0, s[-1], max(3, int(s[-1] / step) + 1))
        points = np.stack([np.interp(samples, s, p[:, 0]), np.interp(samples, s, p[:, 1])], 1)
        return cls(points, halfwidth)

    def stamp(self, shape):
        """Fast rasterisation for long roads: union of discs of the local
        half-width centred on the (densely sampled) centreline."""
        sx, sz = shape
        out = np.zeros(shape, bool)
        r = int(np.ceil(self.halfwidth.max())) + 1
        oy, ox = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
        for (px, pz), half in zip(self.points, self.halfwidth):
            cx, cz = int(np.floor(px)), int(np.floor(pz))
            gx, gz = cx + ox, cz + oy
            keep = (np.hypot(gx + .5 - px, gz + .5 - pz) <= half) & (gx >= 0) & (gz >= 0) & (gx < sx) & (gz < sz)
            out[gx[keep], gz[keep]] = True
        return out

    def mask(self, shape):
        """Cells whose centres lie within the local half-width of the centreline."""
        sx, sz = shape
        out = np.zeros(shape, bool)
        reach = float(self.halfwidth.max()) + 1
        x0, z0 = np.floor(self.points.min(0) - reach).astype(int)
        x1, z1 = np.ceil(self.points.max(0) + reach).astype(int)
        x0, z0, x1, z1 = max(0, x0), max(0, z0), min(sx, x1 + 1), min(sz, z1 + 1)
        if x0 >= x1 or z0 >= z1:
            return out
        gx, gz = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(z0, z1) + 0.5, indexing="ij")
        cells = np.stack((gx.ravel(), gz.ravel()), 1)
        a, b = self.points[:-1], self.points[1:]
        ab = b - a
        best = np.full(len(cells), np.inf)
        width = np.zeros(len(cells))
        for i in range(len(a)):
            denom = max(float(ab[i] @ ab[i]), 1e-12)
            t = np.clip(((cells - a[i]) @ ab[i]) / denom, 0, 1)
            d = np.linalg.norm(cells - (a[i] + t[:, None] * ab[i]), axis=1)
            closer = d < best
            best[closer] = d[closer]
            width[closer] = self.halfwidth[i] * (1 - t[closer]) + self.halfwidth[i + 1] * t[closer]
        out[x0:x1, z0:z1] = (best <= width).reshape(gx.shape)
        return out


def blob(shape, center, half, identity, *, exponent=2.6, amplitude=0.13):
    """Rounded, irregular pad outline: a noisy superellipse in cell centres."""
    sx, sz = shape
    gx, gz = np.meshgrid(np.arange(sx) + 0.5, np.arange(sz) + 0.5, indexing="ij")
    dx, dz = (gx - center[0]) / half[0], (gz - center[1]) / half[1]
    theta = np.arctan2(dz, dx)
    radius = (np.abs(dx) ** exponent + np.abs(dz) ** exponent) ** (1 / exponent)
    return radius <= 1 + amplitude * ring(theta, *identity)


def rounded_box(shape, lo, hi, corner):
    """Cells inside an axis box whose corners are rounded by ``corner`` metres."""
    sx, sz = shape
    gx, gz = np.meshgrid(np.arange(sx) + 0.5, np.arange(sz) + 0.5, indexing="ij")
    cx = np.clip(gx, lo[0] + corner, hi[0] - corner)
    cz = np.clip(gz, lo[1] + corner, hi[1] - corner)
    return np.hypot(gx - cx, gz - cz) <= corner


def top_solid(world):
    """Highest solid block per column (-1 for an empty column)."""
    solid = B.SOLID[world.vox]
    flipped = solid[:, ::-1, :]
    top = solid.shape[1] - 1 - np.argmax(flipped, axis=1)
    return np.where(solid.any(axis=1), top, -1)


def shape_course(world, surface, paving, level, identity, *, slope=(2.0, 3.1), jitter=0.8):
    """Make ``surface`` cells flat walkable ground at ``level`` and blend them
    into the natural terrain with graded banks instead of vertical walls.

    Banks fall (or rise) at least two blocks per metre beyond the first cell,
    so creatures cannot walk up them and the agent cannot climb out of the
    lowland, matching the access of the former vertical causeway walls.
    Returns the mask of columns whose ground changed.
    """
    sx, sz = surface.shape
    natural = top_solid(world)
    seed = stable_seed(*identity, "banks")
    gx, gz = np.meshgrid(np.arange(sx, dtype=float), np.arange(sz, dtype=float), indexing="ij")
    steep = slope[0] + (slope[1] - slope[0]) * 0.5 * (1 + fbm2(gx / 23.0, gz / 23.0, seed, octaves=3))
    rough = jitter * fbm2(gx / 6.0, gz / 6.0, seed + 1, octaves=2)
    distance = ndimage.distance_transform_edt(~surface)
    fill = np.minimum(np.floor(level - steep * (distance - 0.5) + rough), level - 1).astype(int)
    # A bank that meets a lake below its surface leaves the lake as it is.
    wet = water_top(world)
    fill = np.where((wet > natural) & (fill <= wet), -1, fill)
    cut = np.maximum(np.floor(level + steep * (distance - 0.5) + rough), level + 1).astype(int)
    changed = surface.copy()
    vox = world.vox
    for x, z in np.argwhere(surface):
        vox[x, :level, z] = B.DIRT
        vox[x, level, z] = B.GRAVEL if paving[x, z] else B.GRASS
        vox[x, level + 1:, z] = B.AIR
    for x, z in np.argwhere(~surface & (natural < fill) & (fill >= 0)):
        h, f = natural[x, z], fill[x, z]
        vox[x, h + 1:f, z] = B.DIRT
        vox[x, f, z] = B.SAND if wet[x, z] >= f else B.GRASS
        changed[x, z] = True
    for x, z in np.argwhere(~surface & (natural > cut)):
        c = cut[x, z]
        vox[x, c + 1:, z] = B.AIR
        if vox[x, c, z] in (B.DIRT, B.GRASS):
            vox[x, c, z] = B.GRASS
        changed[x, z] = True
    clear_disturbed_vegetation(world, changed)
    return changed


def clear_disturbed_vegetation(world, changed):
    """Remove whole trees rooted in or overhanging regraded columns, and loose
    plants left floating or buried, so no foliage hangs in the air."""
    vox = world.vox
    tree = np.isin(vox, TREE_BLOCKS)
    if tree.any():
        labels, count = ndimage.label(tree, structure=np.ones((3, 3, 3), bool))
        touched = np.unique(labels[:, :, :][np.broadcast_to(changed[:, None, :], labels.shape)])
        touched = touched[touched > 0]
        if len(touched):
            vox[np.isin(labels, touched)] = B.AIR
    loose = np.isin(vox, LOOSE_BLOCKS)
    below = np.zeros_like(loose)
    below[:, 1:, :] = B.SOLID[vox[:, :-1, :]]
    vox[loose & (~below | changed[:, None, :])] = B.AIR


def water_top(world):
    """Highest water block per column (-1 where there is none)."""
    water = world.vox == B.WATER
    top = water.shape[1] - 1 - np.argmax(water[:, ::-1, :], axis=1)
    return np.where(water.any(axis=1), top, -1)


def walk_heights(world):
    """Standing height per column, or -1 where the top is water or unusable."""
    top = top_solid(world)
    sx, sy, sz = world.vox.shape
    above = np.take_along_axis(world.vox, np.clip(top + 1, 0, sy - 1)[:, None, :], axis=1)[:, 0, :]
    return np.where((top >= 0) & (above != B.WATER), top, -1)


def can_reach(heights, goal, climb=0):
    """Columns from which ``goal`` is reachable by 4-neighbour moves that rise
    by at most ``climb`` blocks (falls are allowed)."""
    sx, sz = heights.shape
    reach = goal.copy()
    queue = deque(map(tuple, np.argwhere(goal)))
    while queue:
        x, z = queue.popleft()
        h = heights[x, z]
        for nx, nz in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
            if 0 <= nx < sx and 0 <= nz < sz and not reach[nx, nz]:
                hn = heights[nx, nz]
                # A move from (nx, nz) up to (x, z) must rise by <= climb.
                if hn >= 0 and h - hn <= climb:
                    reach[nx, nz] = True
                    queue.append((nx, nz))
    return reach


def carve_river(world, river, surface, identity, *, sea_level, depth=2, valley=1.6, deck=3):
    """Cut a river along ``river`` whose water surface only ever descends and
    always lies at least one block below the ground beside it.

    Where the channel passes under ``surface`` (roads), the top ``deck``
    blocks of ground are kept as an earthen bridge. Valley sides are lowered
    to a slope instead of vertical walls. Returns the water-surface block
    index per centreline sample and the channel mask.
    """
    sx, sz = surface.shape
    ground = top_solid(world)
    wet = water_top(world)
    holding = np.maximum(ground, wet)
    # The river ends where it first reaches standing water (a lake or the sea).
    end, mouth, entered = len(river.points), None, False
    for i, (px, pz) in enumerate(river.points):
        x, z = int(np.floor(px)), int(np.floor(pz))
        if not (0 <= x < sx and 0 <= z < sz):
            if entered:
                end = i; break
            continue
        entered = True
        if wet[x, z] > ground[x, z] and not surface[x, z]:
            end, mouth = i + 1, int(wet[x, z]); break
    river = Road(river.points[:end], river.halfwidth[:end])
    channel = river.mask((sx, sz))
    # The water must sit below every bank it touches.
    bank = ndimage.binary_dilation(channel, iterations=2) & ~channel
    levels = []
    for (px, pz), half in zip(river.points, river.halfwidth):
        x0, x1 = int(max(0, px - half - 3)), int(min(sx, px + half + 4))
        z0, z1 = int(max(0, pz - half - 3)), int(min(sz, pz + half + 4))
        local = bank[x0:x1, z0:z1] & ~surface[x0:x1, z0:z1]
        heights = holding[x0:x1, z0:z1][local]
        levels.append(int(heights.min()) - 1 if heights.size else sea_level)
    floor = sea_level if mouth is None else mouth
    levels = np.maximum(np.minimum.accumulate(np.asarray(levels)), floor)
    # Nearest centreline sample per channel cell gives its level and depth.
    cells = np.argwhere(ndimage.binary_dilation(channel, iterations=7))
    centres = cells + 0.5
    best = np.full(len(cells), np.inf)
    index = np.zeros(len(cells), int)
    for i, p in enumerate(river.points):
        d = np.hypot(*(centres - p).T)
        closer = d < best
        best[closer], index[closer] = d[closer], i
    vox = world.vox
    seed = stable_seed(*identity, "river")
    for (x, z), d, i in zip(cells, best, index):
        top, half = levels[i], river.halfwidth[i]
        roof = ground[x, z] - deck if surface[x, z] else world.size_y
        if channel[x, z]:
            bed = top - depth - (1 if d < half * 0.5 else 0)
            vox[x, bed + 1:min(top + 1, roof + 1), z] = B.WATER
            vox[x, top + 1:roof + 1, z] = B.AIR
            vox[x, bed, z] = B.GRAVEL if unit(seed, int(x), int(z)) < 0.6 else B.SAND
        elif not surface[x, z]:
            # Sloped valley side, never raised.
            side = int(np.floor(top + 1 + valley * (d - half)))
            if ground[x, z] > side:
                vox[x, side + 1:, z] = B.AIR
                if vox[x, side, z] in (B.DIRT, B.STONE):
                    vox[x, side, z] = B.GRASS
    # Cascades: where the level drops, the first cells of the lower pool become
    # a rock sill as high as the pool above, so each pool is held by its lip and
    # water spills over the sill top instead of standing as a wall.
    level = np.full((sx, sz), -1)
    for (x, z), i in zip(cells, index):
        if channel[x, z]:
            level[x, z] = levels[i]
    higher = np.full((sx, sz), -1)
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        higher = np.maximum(higher, np.roll(level, shift, axis=axis))
    sills = channel & (higher > level)
    for x, z in np.argwhere(sills):
        bed = level[x, z] - depth - 1
        vox[x, max(0, bed):higher[x, z] + 1, z] = B.STONE
    # Levees where a neighbouring bank is lower than the pool it must hold.
    pools = channel & ~sills
    ground = top_solid(world)
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        need = np.roll(np.where(pools, level, -1), shift, axis=axis)
        for x, z in np.argwhere((need >= 0) & ~channel):
            column = vox[x, :need[x, z] + 1, z]
            # Seal cave openings and raise low banks up to the pool level.
            column[~B.SOLID[column] & (column != B.WATER)] = B.DIRT
            if ground[x, z] < need[x, z]:
                vox[x, need[x, z], z] = B.GRASS
    clear_disturbed_vegetation(world, ndimage.binary_dilation(channel, iterations=7))
    return levels, pools


def settle_water(world, max_rounds=200, band=None):
    """Drain water that is not held by ground: a water block needs solid or
    water below it and on all four sides, as real still water would.
    ``band=(ylo, yhi)`` limits the work to heights that can hold water."""
    ylo, yhi = (0, world.vox.shape[1]) if band is None else band
    vox = world.vox[:, ylo:yhi, :]
    for _ in range(max_rounds):
        loose = uncontained_water(vox)
        if ylo > 0:
            # The band floor rests on the world below it.
            loose[:, 0, :] &= ~(B.SOLID[world.vox[:, ylo - 1, :]] | (world.vox[:, ylo - 1, :] == B.WATER))
        if not loose.any():
            return
        vox[loose] = B.AIR
    raise RuntimeError("water did not settle")


def uncontained_water(vox):
    water = vox == B.WATER
    held = B.SOLID[vox] | water
    ok = np.ones_like(water)
    ok[:, 0, :] &= True
    ok[:, 1:, :] &= held[:, :-1, :]
    for axis in (0, 2):
        for shift in (1, -1):
            neighbour = np.roll(held, shift, axis=axis)
            edge = [slice(None)] * 3
            edge[axis] = 0 if shift == 1 else -1
            neighbour[tuple(edge)] = True  # world borders hold water
            ok &= neighbour
    return water & ~ok


def assert_water_contained(world, band=None):
    ylo, yhi = (0, world.vox.shape[1]) if band is None else band
    loose = uncontained_water(world.vox[:, ylo:yhi, :])
    if ylo > 0:
        loose[:, 0, :] &= ~(B.SOLID[world.vox[:, ylo - 1, :]] | (world.vox[:, ylo - 1, :] == B.WATER))
    if loose.any():
        raise AssertionError(f"{int(loose.sum())} water blocks are not held by ground, e.g. {np.argwhere(loose)[0].tolist()}")


def rounded_corners(corners, radius, step=0.5):
    """Polyline through ``corners`` with each interior corner replaced by a
    circular arc of up to ``radius`` metres (limited by the adjacent legs), so
    a route keeps its authored legs but has no right angles."""
    p = [np.asarray(c, float) for c in corners]
    p = [q for i, q in enumerate(p) if i == 0 or np.linalg.norm(q - p[i - 1]) > 1e-6]
    if len(p) < 3:
        out = p
    else:
        out = [p[0]]
        for a, b, c in zip(p, p[1:], p[2:]):
            d1, d2 = b - a, c - b
            l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
            d1, d2 = d1 / l1, d2 / l2
            turn = np.arccos(np.clip(d1 @ d2, -1, 1))
            if turn < 1e-3 or turn > 2.6:
                out.append(b)          # straight on, or doubling back: keep the corner
                continue
            r = min(radius, 0.45 * l1, 0.45 * l2)
            cut = min(r * np.tan(turn / 2), 0.45 * l1, 0.45 * l2)
            s, e = b - d1 * cut, b + d2 * cut
            # Quadratic Bezier through the corner approximates the arc.
            for t in np.linspace(0, 1, max(3, int(turn * r / step) + 2)):
                out.append((1 - t) ** 2 * s + 2 * (1 - t) * t * b + t * t * e)
        out.append(p[-1])
    pts = np.asarray(out)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    samples = np.linspace(0, s[-1], max(2, int(s[-1] / step) + 1))
    return np.stack([np.interp(samples, s, pts[:, 0]), np.interp(samples, s, pts[:, 1])], 1), samples


def simplify(points, tolerance):
    """Ramer-Douglas-Peucker: staircase grid paths become straight runs."""
    p = np.asarray(points, float)
    if len(p) < 3:
        return p
    a, b = p[0], p[-1]
    ab = b - a
    length = np.linalg.norm(ab)
    if length < 1e-9:
        d = np.linalg.norm(p - a, axis=1)
    else:
        d = np.abs(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) / length
    i = int(np.argmax(d))
    if d[i] <= tolerance:
        return np.vstack([a, b])
    return np.vstack([simplify(p[:i + 1], tolerance)[:-1], simplify(p[i:], tolerance)])


def winding_road(corners, identity, *, radius=6.0, meander=1.2, width=(2.5, 3.3), tolerance=1.2):
    """Road along authored legs: rounded corners, a gentle lateral meander that
    vanishes at the ends, and a varying half-width."""
    pts, s = rounded_corners(simplify(corners, tolerance), radius)
    if len(pts) < 2:
        return Road(np.vstack([pts, pts]), width[0])
    total = max(float(s[-1]), 1e-6)
    tangent = np.gradient(pts, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), 1)
    taper = np.clip(np.minimum(s, total - s) / 8.0, 0, 1)
    offset = meander * taper * wobble(s / 30.0, *identity, "meander")
    lo, hi = width
    half = lo + (hi - lo) * 0.5 * (1 + wobble(s / 14.0, *identity, "width"))
    return Road(pts + offset[:, None] * normal, half)


def surface_top(world, ylo, yhi):
    """Highest solid block per column within [ylo, yhi] (-1 where none)."""
    band = B.SOLID[world.vox[:, ylo:yhi + 1, :]]
    top = ylo + band.shape[1] - 1 - np.argmax(band[:, ::-1, :], axis=1)
    return np.where(band.any(axis=1), top, -1)


def grade_band(world, mask, level, identity, *, ylo, yhi, protect=None, slope=(1.1, 1.8), reach=14,
               jitter=0.6, fill_block=B.DIRT, rim=0.0):
    """Blend a flat band (``mask`` cells at ground ``level``) into surrounding
    terrain between ``ylo`` and ``yhi``: cut slopes where the ground is higher,
    embankments where lower. Only columns within ``reach`` are touched; water
    surfaces count as ground; ``protect`` columns are never changed."""
    if not mask.any():
        return
    sx, sz = mask.shape
    idx = np.argwhere(mask)
    x0, z0 = np.maximum(idx.min(0) - reach, 0)
    x1, z1 = np.minimum(idx.max(0) + reach + 1, (sx, sz))
    sub = mask[x0:x1, z0:z1]
    vox = world.vox[x0:x1, :, z0:z1]
    band = B.SOLID[vox[:, ylo:yhi + 1, :]] | (vox[:, ylo:yhi + 1, :] == B.WATER)
    ground = np.where(band.any(axis=1), ylo + band.shape[1] - 1 - np.argmax(band[:, ::-1, :], axis=1), -1)
    distance = ndimage.distance_transform_edt(~sub)
    seed = stable_seed(*identity, "grade")
    gx, gz = np.meshgrid(np.arange(x0, x1, dtype=float), np.arange(z0, z1, dtype=float), indexing="ij")
    steep = slope[0] + (slope[1] - slope[0]) * 0.5 * (1 + fbm2(gx / 19.0, gz / 19.0, seed, octaves=3))
    rough = jitter * fbm2(gx / 5.0, gz / 5.0, seed + 1, octaves=2)
    cut = np.maximum(np.floor(level + steep * (distance - 0.5) + rough), level).astype(int)
    fill = np.minimum(np.floor(level - steep * (distance - 0.5) + rough), level).astype(int)
    # A rim (levee) at least as high as ``level`` holds water in the band.
    fill = np.where(distance <= rim, level, fill)
    near = (~sub) & (distance <= reach) & (ground >= 0)
    if protect is not None:
        near &= ~protect[x0:x1, z0:z1]
    for i, j in np.argwhere(near & (ground > cut)):
        c = cut[i, j]
        column = vox[i, c + 1:yhi + 1, j]
        column[CUTTABLE[column]] = B.AIR
        if vox[i, c, j] in (B.DIRT, B.STONE, B.GRAVEL):
            vox[i, c, j] = B.GRASS
    for i, j in np.argwhere(near & (ground < fill) & (ground >= ylo)):
        f = fill[i, j]
        vox[i, ground[i, j] + 1:f, j] = fill_block
        vox[i, f, j] = B.GRASS
