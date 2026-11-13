"""Presentation landscape: horizon ridges, situated set pieces, and ground detail.

This is presentation only. None of this geometry collides, enters policy
observations, or changes simulator state; the exported task scene and action
trace are unchanged. Set pieces are placed outside protected corridors around
every task route, entity, and building, so a recorded trajectory never passes
through one. Every choice uses a local generator seeded from a stable
world/entity identity (never the simulation RNG), so appearance does not change
as the camera moves.

The pieces are organised as locations rather than a uniform scatter:

* each task site gets a role-specific vignette (ruined walls framing a relay
  court, a cairn at a survey point, a stair-and-terrace ruin by a lift, ...);
* ambient clusters group trees, bushes and grass around rock outcrops, with
  regional species so different parts of a map are distinguishable;
* worn paths, paved courts, soil, and exposed rock are painted into a
  high-resolution ground-detail texture instead of extra geometry;
* a ring of ridges beyond the walkable world frames the horizon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np

from open_world_bench.render.assets import (
    BARK, FOLIAGE, STONE, _prop_template, _rock, appearance_seed)
from open_world_bench.render.meshes import MeshBatch

TERRAIN = 7  # MATERIAL_NAMES index of "terrain"
LAYOUT_VERSION = 1


def _rng(*identity):
    return np.random.default_rng(appearance_seed(identity))


# ---------------------------------------------------------------------------
# Deterministic coordinate noise (no generator state at all)
# ---------------------------------------------------------------------------

def _hash(ix, iz, salt):
    ix = np.asarray(ix, np.int64) & 0xFFFFFFFF
    iz = np.asarray(iz, np.int64) & 0xFFFFFFFF
    h = (ix * 374761393 + iz * 668265263 + (salt & 0xFFFF) * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF) / float(0xFFFFFF)


def value_noise(x, z, scale, salt=0):
    """Smooth lattice noise in [0, 1], a pure function of world position."""
    x = np.asarray(x, np.float64) / scale
    z = np.asarray(z, np.float64) / scale
    x0, z0 = np.floor(x), np.floor(z)
    fx, fz = x - x0, z - z0
    fx, fz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)
    a, b = _hash(x0, z0, salt), _hash(x0 + 1, z0, salt)
    c, d = _hash(x0, z0 + 1, salt), _hash(x0 + 1, z0 + 1, salt)
    return (a * (1 - fx) + b * fx) * (1 - fz) + (c * (1 - fx) + d * fx) * fz


def fbm(x, z, scale, octaves=4, salt=0, ridged=False):
    total, weight, norm = 0.0, 1.0, 0.0
    for octave in range(octaves):
        n = value_noise(x, z, scale / 2 ** octave, salt + 17 * octave)
        if ridged:
            n = 1 - np.abs(2 * n - 1)
            n = n * n
        total = total + weight * n
        norm += weight
        weight *= .5
    return total / norm


def _smooth(edge0, edge1, x):
    t = np.clip((np.asarray(x, np.float64) - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------
# Templates (object coordinates, cached by identity-derived seed)
# ---------------------------------------------------------------------------

def _pack(batch):
    """Compound template arrays with outward faces and authored normals."""
    vertices, colors, emission, _ = batch.finish()
    normals = np.concatenate(getattr(batch, "smooth_normals", None) or batch.normals).copy()
    vertices = vertices.copy()
    start = 0
    for part in batch.verts:
        end = start + len(part)
        face = np.cross(part[:, 1] - part[:, 0], part[:, 2] - part[:, 0])
        authored = normals[start:end].mean(axis=1)
        outward = part.mean(axis=1) - part.reshape(-1, 3).mean(axis=0)
        reference = np.where(np.linalg.norm(authored, axis=1, keepdims=True) > 0, authored, outward)
        flip = (face * reference).sum(axis=1) < -1e-7
        vertices[start:end][flip] = part[flip][:, [0, 2, 1]]
        normals[start:end][flip] = normals[start:end][flip][:, [0, 2, 1]]
        face[flip] *= -1
        face /= np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-8)
        missing = np.linalg.norm(normals[start:end], axis=-1, keepdims=True) == 0
        normals[start:end] = np.where(missing, face[:, None, :], normals[start:end])
        start = end
    for array in (vertices, colors, normals, emission):
        array.setflags(write=False)
    materials = np.concatenate(batch.material_ids)
    materials.setflags(write=False)
    return vertices, colors, normals, materials, emission


def _stone(rng, base=(.50, .48, .43), amount=.10):
    return np.clip(np.asarray(base) * rng.uniform(1 - amount, 1 + amount), 0, 1)


def _block(batch, rng, center, size, yaw, base, moss=0.0):
    """A dressed stone block with slightly irregular proportions."""
    size = np.asarray(size) * rng.uniform(.93, 1.04, 3)
    color = _stone(rng, base)
    if moss > 0 and rng.random() < moss:
        color = color * .55 + np.array((.27, .34, .16)) * .45
    batch.box(center, size, color, yaw)


def _wall(batch, rng, length, height, thickness=.8, base=(.53, .50, .44)):
    """Coursed rubble wall along +X with a broken, weathered top profile."""
    batch.current_material = STONE
    course = rng.uniform(.34, .46)
    profile_seed = rng.uniform(0, 100)
    breaks = 2 + int(rng.integers(3))
    for row in range(int(np.ceil(height / course))):
        y = row * course
        x = -length / 2 + (rng.uniform(0, .5) if row % 2 else 0)
        while x < length / 2:
            width = min(rng.uniform(.55, 1.15), length / 2 - x)
            if width < .18:
                break
            mid = x + width / 2
            t = (mid + length / 2) / length
            top = height * (.35 + .65 * (.5 + .5 * np.sin(profile_seed + t * np.pi * breaks)) ** .7)
            top = min(height, top + (.25 if t < .12 or t > .88 else 0))
            if y < top:
                h = min(course, top - y) * rng.uniform(.92, 1)
                _block(batch, rng, (mid, y + h / 2, rng.uniform(-.04, .04)),
                       (width - .05, h - .03, thickness), rng.uniform(-.04, .04), base,
                       moss=.6 if y + h >= top - course else .12)
            x += width
    # Coping fragments and fallen rubble at the foot.
    for _ in range(int(length * 1.1)):
        side = rng.choice((-1, 1))
        p = (rng.uniform(-length / 2, length / 2), .1, side * rng.uniform(.6, 1.9))
        _rock(batch, rng, p, rng.uniform(.12, .32, 3) * (1, .7, 1), _stone(rng, base, .14), detail=0)
        batch.current_material = STONE


def _crag(batch, rng, center, radii, tone):
    """A weathered rock mass: fractured facets, bedding ledges, mossy top."""
    batch.current_material = STONE
    sides, rings = 18, 10
    latitude = np.linspace(0, np.pi, rings + 1)
    longitude = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    unit = np.stack(np.broadcast_arrays(np.sin(latitude[:, None]) * np.cos(longitude),
                                        np.cos(latitude[:, None]) * np.ones(sides),
                                        np.sin(latitude[:, None]) * np.sin(longitude)), axis=-1)
    salt = int(rng.integers(1, 30000))
    # Low-frequency mass, mid-frequency fracture planes, terraced bedding.
    mass = .78 + .38 * value_noise(unit[..., 0] * 1.6 + 7, unit[..., 2] * 1.6 + unit[..., 1], 1, salt)
    facets = np.floor(value_noise(unit[..., 0] * 4 + 3, unit[..., 2] * 4 + unit[..., 1] * 2, 1, salt + 1) * 4) / 4
    deformation = mass * (.9 + .14 * facets)
    deformation[0], deformation[-1] = deformation[0].mean(), deformation[-1].mean()
    vertices = unit * deformation[..., None] * np.asarray(radii)
    layers = max(radii[1] / 3.5, .18)
    vertices[..., 1] = np.round(vertices[..., 1] / layers) * layers * .35 + vertices[..., 1] * .65
    vertices[..., 1] = np.maximum(vertices[..., 1], -radii[1] * .55)
    vertices[..., 0] += vertices[..., 1] * rng.uniform(-.25, .25)
    vertices += np.asarray(center)
    triangles = []
    for row in range(rings):
        for col in range(sides):
            nxt = (col + 1) % sides
            if row:
                triangles.append([vertices[row, col], vertices[row + 1, col], vertices[row, nxt]])
            if row < rings - 1:
                triangles.append([vertices[row, nxt], vertices[row + 1, col], vertices[row + 1, nxt]])
    triangles = np.asarray(triangles)
    centre = triangles.mean(axis=1)
    colors = np.asarray(tone) * rng.uniform(.9, 1.08, (len(triangles), 1))
    colors *= (.85 + .3 * value_noise(centre[:, 0] * 3, centre[:, 2] * 3 + centre[:, 1] * 2, 1, salt + 2))[:, None]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    outward = centre - np.asarray(center)
    normal *= np.where((normal * outward).sum(1, keepdims=True) < 0, -1, 1)
    moss = (normal[:, 1] > .72) & (value_noise(centre[:, 0] * 2, centre[:, 2] * 2, 1, salt + 3) > .35)
    colors[moss] = colors[moss] * .45 + np.array((.25, .32, .15)) * .55
    batch.add(triangles, np.clip(colors, 0, 1))


def _column(batch, rng, height, radius=.38, base=(.58, .55, .48), broken=True):
    batch.current_material = STONE
    batch.box((0, .18, 0), (radius * 3.2, .36, radius * 3.2), _stone(rng, base))
    batch.box((0, .46, 0), (radius * 2.6, .2, radius * 2.6), _stone(rng, base))
    y, drum = .56, rng.uniform(.55, .8)
    while y < height - .05:
        h = min(drum, height - y)
        r = radius * (1 - .04 * y / max(height, 1))
        batch.tube((0, y, 0), (0, y + h - .02, 0), r, _stone(rng, base, .06), end_radius=r * .99, sides=14)
        y += h
    if not broken:
        batch.box((0, y + .14, 0), (radius * 2.8, .28, radius * 2.8), _stone(rng, base))


@lru_cache(maxsize=512)
def structure_template(kind, seed):
    """Ruins and stone landmarks in object coordinates (+Z faces the site)."""
    rng = np.random.default_rng(seed)
    batch = MeshBatch()
    batch.current_material = STONE
    if kind == "ruin_wall":
        length, height = rng.uniform(5.5, 9.0), rng.uniform(1.4, 2.9)
        _wall(batch, rng, length, height)
        if rng.random() < .6:  # a return wall forming a corner
            sub = MeshBatch(); sub.current_material = STONE
            _wall(sub, rng, rng.uniform(2.5, 4.5), height * rng.uniform(.5, .85))
            vertices, colors, normals, materials, emission = _pack(sub)
            yaw = np.pi / 2
            c, s = np.cos(yaw), np.sin(yaw)
            rotation = np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)))
            offset = np.array((-length / 2 + .4, 0, -1.6))
            batch.add(vertices @ rotation + offset, colors, vertex_normals=normals @ rotation, material=materials)
    elif kind == "ruin_stair":
        # A terrace with a stair flight, retaining walls, and a column stump.
        width, depth, height = rng.uniform(4.5, 6), rng.uniform(4, 5.5), rng.uniform(1.1, 1.6)
        base = (.55, .52, .45)
        batch.box((0, height / 2 - .15, -depth / 2), (width, height + .3, depth), _stone(rng, base))
        for k in range(10):
            batch.box((rng.uniform(-width / 2, width / 2), height + .02, -rng.uniform(.3, depth - .3)),
                      (rng.uniform(.7, 1.3), .08, rng.uniform(.6, 1.1)), _stone(rng, (.62, .59, .52)), rng.uniform(-.2, .2))
        steps = int(np.ceil(height / .24))
        for k in range(steps):
            rise = height * (k + 1) / steps
            batch.box((0, rise / 2 - .1, .32 * (steps - k) - .1), (2.3, rise + .2, .34),
                      _stone(rng, (.60, .57, .50)), rng.uniform(-.02, .02))
        for side in (-1, 1):
            batch.box((side * 1.35, height * .55, .32 * steps / 2), (.35, height * 1.1, .32 * steps + .1),
                      _stone(rng, base))
        sub = MeshBatch(); sub.current_material = STONE
        _wall(sub, rng, depth, rng.uniform(1.0, 2.0), .6, base)
        vertices, colors, normals, materials, emission = _pack(sub)
        rotation = np.array(((0, 0, -1), (0, 1, 0), (1, 0, 0)), float)
        batch.add(vertices @ rotation + (width / 2 - .3, height, -depth / 2), colors,
                  vertex_normals=normals @ rotation, material=materials)
        column = MeshBatch(); _column(column, rng, rng.uniform(1.4, 2.6))
        vertices, colors, normals, materials, emission = _pack(column)
        batch.add(vertices + (-width / 2 + .7, height, -depth + .7), colors, vertex_normals=normals, material=materials)
    elif kind == "column_pair":
        span = rng.uniform(4.2, 5.6)
        for side, h in ((-1, rng.uniform(3.6, 4.6)), (1, rng.uniform(1.3, 2.4))):
            column = MeshBatch(); _column(column, rng, h, broken=side > 0)
            vertices, colors, normals, materials, emission = _pack(column)
            batch.add(vertices + (side * span / 2, 0, 0), colors, vertex_normals=normals, material=materials)
        batch.current_material = STONE
        # A fallen drum line and a lintel fragment.
        for k in range(3):
            p = np.array((span / 2 + 1.2 + k * .72, .36, 1.4 + k * .1))
            batch.tube(p - (.33, 0, 0), p + (.33, 0, 0), .37, _stone(rng, (.58, .55, .48), .06), sides=14)
        batch.box((-.4, .3, -1.4), (3.2, .55, .7), _stone(rng, (.56, .53, .47)), rng.uniform(-.4, .4))
    elif kind == "cairn":
        # Standing stones in an arc around a stacked cairn: a survey landmark.
        count = int(rng.integers(4, 6))
        for k in range(count):
            a = -1.1 + 2.2 * k / (count - 1) + rng.uniform(-.1, .1)
            p = np.array((np.sin(a) * 3.4, 0, -np.cos(a) * 3.4))
            h = rng.uniform(1.6, 3.0)
            _rock(batch, rng, p + (0, h * .42, 0), (rng.uniform(.35, .5), h * .52, rng.uniform(.25, .38)),
                  _stone(rng, (.47, .46, .42)), detail=1)
        y = 0.0
        for k in range(5):
            r = .62 - k * .1
            _rock(batch, rng, (rng.uniform(-.05, .05), y + r * .35, rng.uniform(-.05, .05)),
                  (r, r * .5, r * .9), _stone(rng, (.52, .50, .45)), detail=0)
            y += r * .6
    elif kind == "outcrop":
        main = np.array((rng.uniform(2.0, 3.2), rng.uniform(1.4, 2.5), rng.uniform(1.7, 2.8)))
        tone = (.47, .46, .42) if rng.random() < .6 else (.53, .49, .42)
        _crag(batch, rng, (0, -main[1] * .25, 0), main, tone)
        for k in range(int(rng.integers(2, 5))):
            a = k * 2.2 + rng.uniform(-.4, .4)
            r = rng.uniform(.6, 1.35)
            p = np.array((np.cos(a), 0, np.sin(a))) * (main[[0, 2]].mean() + r * .3)
            _crag(batch, rng, p + (0, -r * .3, 0), (r, r * rng.uniform(.55, .95), r * rng.uniform(.7, 1)), tone)
        if rng.random() < .5:  # a tilted slab
            _rock(batch, rng, (main[0] * .9, .35, -main[2] * .6), (1.4, .35, .9), _stone(rng, tone), detail=1)
    elif kind == "boulders":
        for k in range(int(rng.integers(2, 4))):
            a = k * 2.4 + rng.uniform(-.5, .5)
            r = rng.uniform(.7, 1.4)
            _rock(batch, rng, (np.cos(a) * 1.4 * k, r * .38, np.sin(a) * 1.4 * k), (r, r * .75, r * .9),
                  _stone(rng, (.48, .47, .43)), detail=1)
    elif kind == "plinth":
        # A low foundation ring shared by a relay court corner.
        for tier in range(2):
            s = 2.2 - tier * .5
            batch.box((0, .12 + tier * .22, 0), (s, .24, s), _stone(rng, (.55, .53, .47)))
    else:
        raise ValueError(f"unknown structure template {kind!r}")
    batch.current_material = 0
    return _pack(batch)


@lru_cache(maxsize=96)
def tuft_template(variant, flowers):
    """A soft, curved grass tuft (optionally with a few small flowers)."""
    rng = np.random.default_rng(9103 + variant * 31 + int(flowers))
    batch = MeshBatch()
    batch.current_material = FOLIAGE
    base_color = np.array((.24, .33, .13)) * rng.uniform(.9, 1.1)
    tip_color = np.array((.46, .53, .26)) * rng.uniform(.9, 1.1)
    for _ in range(int(rng.integers(18, 28))):
        a = rng.uniform(0, 2 * np.pi)
        h = rng.uniform(.16, .42)
        root = np.array((np.cos(a), 0, np.sin(a))) * rng.uniform(0, .26)
        lean = np.array((np.cos(a), 0, np.sin(a))) * rng.uniform(.1, .3)
        side = np.array((-np.sin(a), 0, np.cos(a)))
        width = rng.uniform(.03, .05)
        points = [root + lean * t * t + (0, h * t, 0) for t in np.linspace(0, 1, 5)]
        widths = width * np.array((1, .9, .72, .45, .08))
        for k in range(4):
            p, q = points[k], points[k + 1]
            c0 = base_color * (1 - k / 4) + tip_color * (k / 4)
            c1 = base_color * (1 - (k + 1) / 4) + tip_color * ((k + 1) / 4)
            color = (c0 + c1) / 2
            batch.add([[p - side * widths[k], p + side * widths[k], q + side * widths[k + 1]],
                       [p - side * widths[k], q + side * widths[k + 1], q - side * widths[k + 1]]],
                      np.clip(color, 0, 1))
    if flowers:
        palette = ((.93, .91, .84), (.95, .86, .52), (.74, .66, .86))
        petal = np.asarray(palette[variant % 3])
        for _ in range(int(rng.integers(2, 5))):
            a = rng.uniform(0, 2 * np.pi)
            root = np.array((np.cos(a), 0, np.sin(a))) * rng.uniform(0, .25)
            top = root + (rng.uniform(-.05, .05), rng.uniform(.3, .5), rng.uniform(-.05, .05))
            batch.current_material = FOLIAGE
            batch.tube(root, top, .008, (.24, .34, .14), sides=4)
            for k in range(5):
                b = k * 2 * np.pi / 5
                d = np.array((np.cos(b), 0, np.sin(b))) * .055
                e = np.array((-np.sin(b), 0, np.cos(b))) * .022
                batch.add([[top, top + d + e, top + d * 1.4], [top, top + d * 1.4, top + d - e]],
                          np.clip(petal * rng.uniform(.92, 1.04), 0, 1))
    batch.current_material = 0
    return _pack(batch)


CARPET_VARIANTS = 8


@lru_cache(maxsize=16)
def carpet_template(variant):
    """Short meadow grass clump for Blender instancing near the camera."""
    rng = np.random.default_rng(7717 + variant)
    batch = MeshBatch()
    batch.current_material = FOLIAGE
    hue = rng.uniform(0, 1)
    base = np.array((.25, .35, .14)) * (1 - .25 * hue) + np.array((.32, .34, .16)) * .25 * hue
    tip = np.array((.44, .53, .25)) * (1 - .3 * hue) + np.array((.56, .55, .31)) * .3 * hue
    for _ in range(int(rng.integers(9, 15))):
        a = rng.uniform(0, 2 * np.pi)
        h = rng.uniform(.09, .24)
        root = np.array((np.cos(a), 0, np.sin(a))) * rng.uniform(0, .14)
        lean = np.array((np.cos(a), 0, np.sin(a))) * rng.uniform(.03, .12)
        side = np.array((-np.sin(a), 0, np.cos(a)))
        width = rng.uniform(.018, .03)
        p0, p1, p2 = root, root + lean * .35 + (0, h * .55, 0), root + lean + (0, h, 0)
        batch.add([[p0 - side * width, p0 + side * width, p1 + side * width * .6],
                   [p0 - side * width, p1 + side * width * .6, p1 - side * width * .6]], np.clip(base * rng.uniform(.9, 1.1), 0, 1))
        batch.add([[p1 - side * width * .6, p1 + side * width * .6, p2]], np.clip(tip * rng.uniform(.9, 1.1), 0, 1))
    batch.current_material = 0
    return _pack(batch)


# ---------------------------------------------------------------------------
# Kits: how each environment draws trees and bushes in its own style
# ---------------------------------------------------------------------------

def _teyvat_tree(species, seed, detail):
    return _prop_template(species, seed, detail)


@lru_cache(maxsize=256)
def _voxel_tree(species, seed, detail):
    from . import models
    batch = models.ModelBatch()
    rng = np.random.default_rng(seed)
    models.tree(batch, (0, 0, 0), float(rng.uniform(5.5, 8.5)), species, float(rng.uniform()), detail)
    batch.current_material = 0
    return _pack(batch)


@dataclass(frozen=True)
class Kit:
    name: str
    tree: Callable
    species: tuple            # regional species, chosen by low-frequency noise
    meadow: tuple = (.31, .43, .18)


TEYVAT_KIT = Kit("teyvat", _teyvat_tree, ("broadleaf", "sakura", "pine", "broadleaf"))
VOXEL_KIT = Kit("voxel", _voxel_tree, ("oak", "pine", "oak", "birch"))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@dataclass
class Element:
    template: tuple           # ("structure", kind) | ("tree", species) | ("bush",) | ("tuft", flowers)
    seed: int
    offset: tuple             # local (x, z) relative to the piece
    yaw: float
    scale: float = 1.0
    sink: float = 0.0


@dataclass
class Piece:
    identity: str
    kind: str
    xz: np.ndarray
    yaw: float
    radius: float
    elements: list = field(default_factory=list)


@dataclass
class Layout:
    """Everything a presentation needs to dress one world; built once, then cached."""
    identity: tuple
    kit: Kit
    ground: Callable          # vectorised (x, z) -> y
    region: tuple             # (xmin, xmax, zmin, zmax) for set pieces
    routes: np.ndarray        # (n, 2, 2) protected straight routes in XZ
    points: np.ndarray        # (m, 2) protected entity positions
    obstacles: list           # [(xz, radius)] buildings and other solid things
    sites: list               # [(identity, xz, role)]
    paths: list = field(default_factory=list)    # [(n, 2) polyline]
    courts: list = field(default_factory=list)   # [(xz, half_extent, yaw, shape)]
    allowed: Callable | None = None              # optional (n, 2) -> bool mask
    route_margin: float = 3.0
    point_margin: float = 4.5
    footprint: float = .8        # fraction of a piece radius sampled for ground checks
    relief: float = .35          # allowed ground relief, as a fraction of the radius
    pieces: list = field(default_factory=list)
    mask: dict | None = None


def _segment_distance(points, segments):
    """Distance from each XZ point to the nearest segment."""
    points = np.asarray(points, float).reshape(-1, 2)
    if not len(segments):
        return np.full(len(points), np.inf)
    best = np.full(len(points), np.inf)
    a, b = segments[:, 0], segments[:, 1]
    d = b - a
    length = np.maximum((d * d).sum(axis=1), 1e-9)
    for start in range(0, len(segments), 256):
        sa, sd, sl = a[start:start + 256], d[start:start + 256], length[start:start + 256]
        rel = points[:, None, :] - sa[None]
        t = np.clip((rel * sd[None]).sum(-1) / sl[None], 0, 1)
        dist = np.linalg.norm(rel - t[..., None] * sd[None], axis=-1)
        best = np.minimum(best, dist.min(axis=1))
    return best


def _clear(layout, xz, radius, placed):
    xz = np.asarray(xz, float)
    xmin, xmax, zmin, zmax = layout.region
    if not (xmin + radius < xz[0] < xmax - radius and zmin + radius < xz[1] < zmax - radius):
        return False
    if _segment_distance(xz, layout.routes)[0] < radius + layout.route_margin:
        return False
    if len(layout.points) and np.min(np.linalg.norm(layout.points - xz, axis=1)) < radius + layout.point_margin:
        return False
    for center, r in layout.obstacles:
        if np.linalg.norm(np.asarray(center) - xz) < radius + r + 1.5:
            return False
    for piece in placed:
        if np.linalg.norm(piece.xz - xz) < radius + piece.radius + 1.0:
            return False
    ring = xz + radius * layout.footprint * np.stack((np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)), 1)
    footprint = np.vstack((xz, ring))
    if layout.allowed is not None and not np.all(layout.allowed(footprint)):
        return False
    heights = np.asarray(layout.ground(footprint[:, 0], footprint[:, 1]), float)
    return bool(np.ptp(heights) < max(1.2, radius * layout.relief))


def _scatter(rng, piece, template, count, inner, outer, **kwargs):
    for _ in range(count):
        a, r = rng.uniform(0, 2 * np.pi), rng.uniform(inner, outer)
        piece.elements.append(Element(template, int(rng.integers(2**31)), (np.cos(a) * r, np.sin(a) * r),
                                      float(rng.uniform(0, 2 * np.pi)), **kwargs))


def _region_species(layout, xz):
    n = value_noise(xz[0], xz[1], 140, 4049)
    return layout.kit.species[int(n * len(layout.kit.species) * .999)]


def _build_piece(layout, identity, kind, xz, yaw):
    """Compose a piece from templates; a stable function of its identity."""
    rng = _rng(layout.identity, identity, kind)
    seed = lambda: int(rng.integers(2**31))
    radius = dict(ruin_wall=5.2, ruin_stair=4.8, column_pair=4.0, cairn=4.2, outcrop=4.4,
                  boulders=3.2, grove=6.5, thicket=3.4)[kind]
    piece = Piece(identity, kind, np.asarray(xz, float), float(yaw), radius)
    species = _region_species(layout, xz)
    if kind in ("ruin_wall", "ruin_stair", "column_pair", "cairn", "outcrop", "boulders"):
        piece.elements.append(Element(("structure", kind), seed(), (0, 0), 0.0, sink=.18))
        _scatter(rng, piece, ("bush",), int(rng.integers(1, 4)), radius * .45, radius * .9, scale=float(rng.uniform(.8, 1.3)))
        _scatter(rng, piece, ("tuft", False), int(rng.integers(8, 15)), radius * .4, radius * 1.1)
        _scatter(rng, piece, ("tuft", True), int(rng.integers(2, 6)), radius * .5, radius * 1.15)
        if kind in ("outcrop", "ruin_wall") and rng.random() < .55:
            a = rng.uniform(0, 2 * np.pi)
            piece.elements.append(Element(("tree", species), seed(), (np.cos(a) * radius * .85, np.sin(a) * radius * .85),
                                          float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(.85, 1.15))))
    elif kind == "grove":
        trees = int(rng.integers(2, 5))
        for k in range(trees):
            a = k * 2 * np.pi / trees + rng.uniform(-.5, .5)
            r = rng.uniform(1.0, 4.2) if trees > 1 else 0
            piece.elements.append(Element(("tree", species), seed(), (np.cos(a) * r, np.sin(a) * r),
                                          float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(.8, 1.2))))
        piece.elements.append(Element(("structure", "boulders"), seed(), tuple(rng.uniform(-2, 2, 2)), float(rng.uniform(0, 6.3)), .6, .1))
        _scatter(rng, piece, ("bush",), int(rng.integers(2, 5)), 1.5, radius, scale=float(rng.uniform(.8, 1.3)))
        _scatter(rng, piece, ("tuft", False), int(rng.integers(10, 18)), .8, radius * 1.1)
        _scatter(rng, piece, ("tuft", True), int(rng.integers(2, 6)), 2, radius * 1.1)
    elif kind == "thicket":
        _scatter(rng, piece, ("bush",), int(rng.integers(3, 7)), 0, radius * .8, scale=float(rng.uniform(.9, 1.5)))
        piece.elements.append(Element(("structure", "boulders"), seed(), (0, 0), float(rng.uniform(0, 6.3)), .55, .1))
        _scatter(rng, piece, ("tuft", True), int(rng.integers(4, 9)), .5, radius * 1.1)
    return piece


# Role cues: each task kind gets a recognisable surrounding location.
SITE_VIGNETTES = {
    "relay": ("ruin_wall", "ruin_wall", "outcrop"),
    "lesson": ("grove", "thicket"), "exchange": ("grove", "boulders"),
    "sentinel": ("column_pair", "ruin_wall"), "fight": ("column_pair", "boulders"),
    "crossing": ("outcrop", "outcrop"), "survey": ("cairn", "thicket"),
    "collect": ("boulders", "thicket"), "dive": ("outcrop", "thicket"),
    "lift": ("ruin_stair", "grove"), "waypoint": ("cairn",), "workshop": ("ruin_wall", "grove"),
}


def plan_pieces(layout, ambient_spacing=26.0, ambient_density=.42):
    """Place vignettes around sites, then clustered ambient features."""
    placed = []
    for identity, xz, role in layout.sites:
        xz = np.asarray(xz, float)
        rng = _rng(layout.identity, "site", identity)
        phase = rng.uniform(0, 2 * np.pi)
        for n, kind in enumerate(SITE_VIGNETTES.get(role, ("outcrop",))):
            probe = _build_piece(layout, f"{identity}:{n}", kind, xz, 0.0)
            for distance in (11, 13.5, 16, 19, 23, 28):
                done = False
                for k in range(16):
                    a = phase + n * 2.1 + k * np.pi / 8
                    p = xz + distance * np.array((np.cos(a), np.sin(a)))
                    if _clear(layout, p, probe.radius, placed):
                        toward = xz - p  # local +Z faces the task site
                        yaw = float(np.arctan2(toward[0], toward[1]))
                        placed.append(_build_piece(layout, f"{identity}:{n}", kind, p, yaw))
                        done = True
                        break
                if done:
                    break
    xmin, xmax, zmin, zmax = layout.region
    for gx in np.arange(xmin, xmax, ambient_spacing):
        for gz in np.arange(zmin, zmax, ambient_spacing):
            cell = (int(round(gx / ambient_spacing)), int(round(gz / ambient_spacing)))
            jitter = np.array((_hash(cell[0], cell[1], 77), _hash(cell[0], cell[1], 78))) - .5
            p = np.array((gx, gz)) + ambient_spacing * (.5 + .8 * jitter)
            clump = fbm(p[0], p[1], 90, 3, salt=611)
            if clump < 1 - ambient_density:
                continue
            choice = value_noise(p[0], p[1], 37, 919)
            kind = "grove" if choice < .42 else "outcrop" if choice < .62 else "thicket" if choice < .84 else "boulders"
            if value_noise(p[0], p[1], 53, 123) > .9:
                kind = "ruin_wall"
            probe = _build_piece(layout, f"ambient:{cell}", kind, p, 0.0)
            if _clear(layout, p, probe.radius, placed):
                yaw = float(_hash(cell[0], cell[1], 79) * 2 * np.pi)
                placed.append(_build_piece(layout, f"ambient:{cell}", kind, p, yaw))
    layout.pieces = placed
    return placed


# ---------------------------------------------------------------------------
# Ground detail texture
# ---------------------------------------------------------------------------

def paint_ground(layout, resolution=2048, margin=24.0):
    """RGBA uint8: path wear, soil, paving, exposed rock over the dressed region."""
    xmin, xmax, zmin, zmax = layout.region
    x0, z0 = xmin - margin, zmin - margin
    size = max(xmax - xmin, zmax - zmin) + 2 * margin
    axis = x0 + (np.arange(resolution) + .5) * size / resolution
    zaxis = z0 + (np.arange(resolution) + .5) * size / resolution
    x, z = np.meshgrid(axis, zaxis)  # rows follow +Z
    path = np.zeros(x.shape, np.float32)
    soil = np.zeros(x.shape, np.float32)
    paving = np.zeros(x.shape, np.float32)
    rock = np.zeros(x.shape, np.float32)
    wobble = (value_noise(x, z, 9, 301) - .5) * 1.6 + (value_noise(x, z, 2.2, 302) - .5) * .5
    edge_noise = (value_noise(x, z, 3.0, 305) - .5) * 2.2
    intact_all = _smooth(.3, .42, value_noise(x, z, 1.9, 306)).astype(np.float32)
    jag_all = (value_noise(x, z, 2.5, 307) - .5) * 2.4
    pixel = size / resolution

    def window(center, reach):
        c0 = max(0, int((center[0] - reach - x0) / pixel)); c1 = min(resolution, int((center[0] + reach - x0) / pixel) + 1)
        r0 = max(0, int((center[1] - reach - z0) / pixel)); r1 = min(resolution, int((center[1] + reach - z0) / pixel) + 1)
        return np.s_[r0:r1, c0:c1]
    for line in layout.paths:
        line = np.asarray(line, float)
        segs = np.stack((line[:-1], line[1:]), 1)
        segs = segs[np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1) > 1e-6]
        if not len(segs):
            continue
        lo, hi = segs.reshape(-1, 2).min(0) - 6, segs.reshape(-1, 2).max(0) + 6
        cols = np.flatnonzero((axis > lo[0]) & (axis < hi[0]))
        rows = np.flatnonzero((zaxis > lo[1]) & (zaxis < hi[1]))
        if not len(cols) or not len(rows):
            continue
        sub = np.s_[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        pts = np.stack((x[sub].ravel(), z[sub].ravel()), 1)
        d = _segment_distance(pts, segs).reshape(x[sub].shape) + wobble[sub]
        width = .8 + .3 * value_noise(x[sub], z[sub], 23, 303)
        wear = _smooth(width + .45, width - .25, d) * (.72 + .28 * value_noise(x[sub], z[sub], 1.3, 304))
        path[sub] = np.maximum(path[sub], wear)
        soil[sub] = np.maximum(soil[sub], .4 * _smooth(width + 1.4, width + .2, d))
    for xz, half, yaw, shape in layout.courts:
        w = window(xz, half * 1.5 + 4)
        c, s = np.cos(yaw), np.sin(yaw)
        lx = (x[w] - xz[0]) * c - (z[w] - xz[1]) * s
        lz = (x[w] - xz[0]) * s + (z[w] - xz[1]) * c
        if shape == "square":
            d = np.maximum(np.abs(lx), np.abs(lz)) - half
        else:
            d = np.hypot(lx, lz) - half
        edge = d + edge_noise[w]
        intact = intact_all[w]
        paving[w] = np.maximum(paving[w], _smooth(.6, -.4, edge) * intact)
        soil[w] = np.maximum(soil[w], .5 * _smooth(1.2, -.2, edge) * (1 - intact))
    for piece in layout.pieces:
        w = window(piece.xz, piece.radius * 1.3 + 3)
        d = np.hypot(x[w] - piece.xz[0], z[w] - piece.xz[1]) + jag_all[w]
        if piece.kind in ("outcrop", "boulders", "cairn"):
            rock[w] = np.maximum(rock[w], .85 * _smooth(piece.radius * .75, piece.radius * .35, d))
        soil[w] = np.maximum(soil[w], .62 * _smooth(piece.radius * 1.05, piece.radius * .4, d))
    # Restrained dry patches and rock exposures across open ground.
    rock = np.maximum(rock, .7 * _smooth(.83, .9, fbm(x, z, 60, 3, salt=308)))
    channels = np.stack((path, soil, paving, rock), axis=-1)
    image = np.clip(np.round(channels * 255), 0, 255).astype(np.uint8)
    layout.mask = dict(origin=[float(x0), float(z0)], size=float(size), resolution=resolution,
                       channels=["path", "soil", "paving", "rock"], pixel_metres=float(pixel))
    return image


# ---------------------------------------------------------------------------
# Horizon ridges (outside the walkable world)
# ---------------------------------------------------------------------------

def horizon_ridges(batch, *, center, half_extent, inner_gap, depth, floor, peak, identity,
                   gaps=True, snow=None, angular=720, radial=40, shoreline=None):
    """A ring of ridged terrain hugging a square world from outside.

    ``half_extent`` is the half-size of the walkable square; the ring starts
    ``inner_gap`` metres beyond it. Ridges are never collision geometry.
    """
    salt = appearance_seed(("ridge", identity)) % 50000
    theta = np.linspace(0, 2 * np.pi, angular, endpoint=False)
    # A superellipse follows the square world without sharp corners.
    c, s = np.cos(theta), np.sin(theta)
    p = 32.0  # nearly square: the ring clears the world's corners too
    scale = half_extent / (np.abs(c) ** p + np.abs(s) ** p) ** (1 / p)
    t = np.linspace(0, 1, radial)
    r = scale[None, :] + inner_gap + (t[:, None] ** 1.3) * depth
    x = center[0] + r * c[None, :]
    z = center[1] + r * s[None, :]
    rise = _smooth(0, .35, t)[:, None] * (1 - .25 * _smooth(.7, 1, t)[:, None])
    shape = fbm(x, z, depth * .55, 5, salt=salt, ridged=True)
    massif = fbm(x, z, depth * 1.6, 2, salt=salt + 1)
    opening = _smooth(.28, .55, value_noise(np.cos(theta) * 4, np.sin(theta) * 4, 2.2, salt + 2))[None, :] if gaps else 1
    height = floor + (peak - floor) * rise * (.25 + .75 * shape) * (.45 + .75 * massif) * opening
    if shoreline is not None:
        height[0] = np.minimum(height[0], shoreline)
    points = np.stack((x, height, z), -1)
    # Close the loop around the ring.
    points = np.concatenate((points, points[:, :1]), axis=1)
    a, b, cc, d = points[:-1, :-1], points[:-1, 1:], points[1:, 1:], points[1:, :-1]
    triangles = np.stack((np.stack((a, cc, b), -2), np.stack((a, d, cc), -2)), -3).reshape(-1, 3, 3)
    # Smooth vertex normals from accumulated face normals.
    rows, cols = points.shape[:2]
    face = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face *= np.where(face[:, 1:2] < 0, -1, 1)
    index = np.arange(rows * cols).reshape(rows, cols)
    ia, ib, ic, idd = index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]
    corners = np.stack((np.stack((ia, ic, ib), -1), np.stack((ia, idd, ic), -1)), -2).reshape(-1, 3)
    accumulated = np.zeros((rows * cols, 3))
    for k in range(3):
        np.add.at(accumulated, corners[:, k], face)
    # Seam vertices share the first column's normal.
    accumulated = accumulated.reshape(rows, cols, 3)
    accumulated[:, 0] += accumulated[:, -1]
    accumulated[:, -1] = accumulated[:, 0]
    accumulated = accumulated.reshape(-1, 3)
    vertex_normals = accumulated / np.maximum(np.linalg.norm(accumulated, axis=1, keepdims=True), 1e-9)
    normals = vertex_normals[corners]
    # Fix winding to face upward/outward along the averaged normal.
    flip = (np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) * normals.mean(1)).sum(1) < 0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    normals[flip] = normals[flip][:, [0, 2, 1]]
    centre = triangles.mean(axis=1)
    slope = np.arccos(np.clip(normals.mean(1)[:, 1], -1, 1))
    grass = np.array((.25, .35, .15)) * (.85 + .3 * value_noise(centre[:, 0], centre[:, 2], 70, salt + 3))[:, None]
    forest = np.array((.15, .25, .13))
    rock_color = np.array((.44, .42, .39)) * (.9 + .2 * value_noise(centre[:, 0], centre[:, 2], 25, salt + 4))[:, None]
    woods = _smooth(.45, .62, fbm(centre[:, 0], centre[:, 2], 90, 3, salt=salt + 5))[:, None]
    color = grass * (1 - woods) + forest * woods
    steep = _smooth(np.radians(26), np.radians(42), slope)[:, None]
    color = color * (1 - steep) + rock_color * steep
    if shoreline is not None:
        beach = _smooth(shoreline + 3.5, shoreline + .5, centre[:, 1])[:, None]
        color = color * (1 - beach) + np.array((.62, .56, .42)) * beach
    if snow is not None:
        cover = (_smooth(snow - 12, snow + 8, centre[:, 1]) * (1 - _smooth(np.radians(40), np.radians(55), slope)))[:, None]
        color = color * (1 - cover) + np.array((.90, .92, .95)) * cover
    batch.add(triangles.astype(np.float32), np.clip(color, 0, 1).astype(np.float32),
              vertex_normals=normals.astype(np.float32), material=TERRAIN)
    return len(triangles)


# ---------------------------------------------------------------------------
# Emitting pieces into a mesh batch
# ---------------------------------------------------------------------------

def _rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)))


def _template(layout, element, detail):
    kind = element.template[0]
    if kind == "structure":
        return structure_template(element.template[1], element.seed)
    if kind == "tree":
        return layout.kit.tree(element.template[1], element.seed, detail)
    if kind == "bush":
        return _prop_template("bush", element.seed, detail)
    if kind == "tuft":
        return tuft_template(element.seed % 24, bool(element.template[1]))
    raise ValueError(kind)


def element_world(piece, element):
    c, s = np.cos(piece.yaw), np.sin(piece.yaw)
    ox, oz = element.offset
    return piece.xz + np.array((c * ox + s * oz, -s * ox + c * oz))


OBJ_PIECE, OBJ_FADED = 11, 12


def emit_pieces(layout, batch, eye, forward, *, max_distance=340.0, near_detail=90.0,
                tufts_within=60.0, occludes=None):
    """Add visible pieces. Those for which ``occludes(piece)`` is true get the
    faded-occluder object id so the presentation can fade them for the camera."""
    eye = np.asarray(eye, float)
    horizontal = np.asarray(forward, float)[[0, 2]]
    horizontal /= max(np.linalg.norm(horizontal), 1e-6)
    stats = dict(pieces=0, elements=0)
    for piece in layout.pieces:
        delta = piece.xz - eye[[0, 2]]
        distance = float(np.linalg.norm(delta))
        if distance > max_distance + piece.radius:
            continue
        if distance > 45 and delta @ horizontal < -.25 * distance:
            continue
        stats["pieces"] += 1
        for element in piece.elements:
            if element.template[0] == "tuft" and distance > tufts_within:
                continue
            xz = element_world(piece, element)
            detail = int(np.linalg.norm(xz - eye[[0, 2]]) < near_detail)
            vertices, colors, normals, materials, emission = _template(layout, element, detail)
            if element.template[0] == "structure":
                # Sit on the lowest ground under the footprint so nothing floats.
                extent = max(np.ptp(vertices[..., 0]), np.ptp(vertices[..., 2])) * .4 * element.scale
                ring = xz + extent * np.stack((np.cos(np.arange(6) * 1.05), np.sin(np.arange(6) * 1.05)), 1)
                y = float(np.min(layout.ground(np.r_[xz[0], ring[:, 0]], np.r_[xz[1], ring[:, 1]])))
            else:
                y = float(layout.ground(np.array([xz[0]]), np.array([xz[1]]))[0])
            yaw = piece.yaw + element.yaw
            rotation = _rotation(yaw)
            position = np.array((xz[0], y - element.sink, xz[1]))
            obj_id = OBJ_PIECE
            if occludes is not None and element.template[0] in ("structure", "tree"):
                extent = np.ptp(vertices[..., [0, 2]].reshape(-1, 2), axis=0).max() * .45 * element.scale
                top = float(vertices[..., 1].max()) * element.scale
                if occludes(xz, y, top, extent):
                    obj_id = OBJ_FADED
            batch.add(vertices @ rotation * element.scale + position, colors, obj_id=obj_id,
                      vertex_normals=normals @ rotation, material=materials)
            batch.emissive[-1] = np.asarray(emission).copy()
            stats["elements"] += 1
    return stats


def mask_at(layout, x, z):
    """Sample the painted ground-detail channels at world XZ (nearest pixel)."""
    image = getattr(layout, "mask_image", None)
    if image is None or layout.mask is None:
        return np.zeros(np.shape(x) + (4,), np.float32)
    origin, size, res = layout.mask["origin"], layout.mask["size"], layout.mask["resolution"]
    col = np.clip(((np.asarray(x) - origin[0]) / size * res).astype(int), 0, res - 1)
    row = np.clip(((np.asarray(z) - origin[1]) / size * res).astype(int), 0, res - 1)
    return image[row, col].astype(np.float32) / 255


def meadow_tufts(layout, batch, eye, *, radius=42.0, spacing=1.35, salt=0):
    """World-anchored grass and flower patches near the camera.

    Density follows a clumping field and is suppressed on paths and paving,
    so open ground reads as meadow patches rather than an even scatter.
    """
    cx, cz = np.floor(np.asarray(eye)[[0, 2]] / spacing).astype(int)
    span = int(radius / spacing)
    ix, iz = np.meshgrid(np.arange(cx - span, cx + span + 1), np.arange(cz - span, cz + span + 1))
    ix, iz = ix.ravel(), iz.ravel()
    x = (ix + _hash(ix, iz, 501 + salt)) * spacing
    z = (iz + _hash(ix, iz, 502 + salt)) * spacing
    near = np.hypot(x - eye[0], z - eye[2]) < radius
    clump = fbm(x, z, 11, 2, salt=503 + salt)
    keep = near & (clump > .64) & (_hash(ix, iz, 504 + salt) < (clump - .64) * 5.5)
    if layout.allowed is not None and keep.any():
        keep[keep] = layout.allowed(np.stack((x[keep], z[keep]), 1))
    if keep.any():
        m = mask_at(layout, x[keep], z[keep])
        keep[keep] = (m[:, 0] < .25) & (m[:, 2] < .2)
    x, z, ix, iz = x[keep], z[keep], ix[keep], iz[keep]
    if not len(x):
        return 0
    y = np.asarray(layout.ground(x, z), float)
    flowers = value_noise(x, z, 17, 505 + salt) > .72
    for k in range(len(x)):
        variant = int(_hash(ix[k], iz[k], 506 + salt) * 24) % 24
        vertices, colors, normals, materials, emission = tuft_template(variant, bool(flowers[k]))
        yaw = float(_hash(ix[k], iz[k], 507 + salt) * 2 * np.pi)
        scale = .8 + .5 * float(_hash(ix[k], iz[k], 508 + salt))
        rotation = _rotation(yaw)
        batch.add(vertices @ rotation * scale + (x[k], y[k] - .02, z[k]), colors, obj_id=OBJ_PIECE,
                  vertex_normals=normals @ rotation, material=materials)
        batch.emissive[-1] = np.asarray(emission).copy()
    return len(x)


# ---------------------------------------------------------------------------
# Environment adapters
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        if len(_CACHE) > 8:
            _CACHE.clear()
        _CACHE[key] = build()
    return _CACHE[key]


def teyvat_course_layout(world):
    """Layout for the compact Teyvat reference course (authored plateau)."""
    course = world.course
    terrain = world.terrain
    entities = list(world.entities.values())
    routes = getattr(world.architecture, "protected_routes", None)
    if routes is None:
        routes = np.empty((0, 2, 2))
    points = np.array([e.home[[0, 2]] for e in entities], float).reshape(-1, 2)
    obstacles = []
    if world.architecture is not None:
        from open_world_bench.architecture import RADII
        obstacles = [(np.asarray(b.pos)[[0, 2]], RADII[b.kind]) for b in world.architecture.buildings]
    sites, path, courts = [], [], []
    seen = set()
    for milestone in course.milestones:
        for op in milestone.operations:
            site = np.asarray(op.state["site"], float)
            if not path or np.linalg.norm(path[-1] - site) > 1e-6:
                path.append(site)
            key = tuple(np.round(site, 2))
            if key in seen:
                continue
            seen.add(key)
            sites.append((op.id, site, op.kind))
            if op.kind == "relay":
                courts.append((site + (0, .5), 7.0, 0.0, "square"))
            elif op.kind in ("lesson", "exchange", "survey", "collect"):
                courts.append((site, 3.6, 0.0, "round"))
            elif op.kind in ("sentinel", "fight"):
                courts.append((site, 6.0, 0.0, "round"))
            elif op.kind == "lift":
                courts.append((site, 4.0, .3, "square"))
    half = terrain.cfg.size / 2

    def allowed(xz):
        # Only the authored plateau; never the cliffs or the shore.
        h = terrain.height_at(xz[:, 0], xz[:, 1])
        return h > terrain.cfg.sea_level + 6

    seed = int(world.cfg.seed)
    layout = Layout(identity=("teyvat-course", seed, course.task, LAYOUT_VERSION), kit=TEYVAT_KIT,
                    ground=terrain.height_at, region=(-half + 70, half - 70, -half + 70, half - 70),
                    routes=np.asarray(routes, float).reshape(-1, 2, 2), points=points, obstacles=obstacles,
                    sites=sites, paths=[np.asarray(path)], courts=courts, allowed=allowed,
                    # The reference controller strays up to ~5 m from straight
                    # routes (combat pursuit, turning); keep a wider berth.
                    route_margin=6.5, point_margin=6.0)
    plan_pieces(layout, ambient_spacing=27.0, ambient_density=.5)
    return layout


def course_layout(world):
    key = ("teyvat-course", int(world.cfg.seed), world.course.task, len(world.entities),
           None if world.architecture is None else len(world.architecture.buildings))
    return _cached(key, lambda: _with_mask(teyvat_course_layout(world)))


def _with_mask(layout):
    layout.mask_image = paint_ground(layout)
    return layout


def voxel_layout(native, *, identity, sites, routes, region, allowed, paths=(), courts=()):
    """Layout for a voxel world (Wildlands/Skylands) from its top-surface heights."""
    world = native.world

    def ground(x, z):
        x = np.clip(np.floor(np.asarray(x)).astype(int), 0, world.size_x - 1)
        z = np.clip(np.floor(np.asarray(z)).astype(int), 0, world.size_z - 1)
        return world.height[x, z].astype(float) + 1.0

    points = np.array([np.asarray(xz, float) for _, xz, _ in sites]).reshape(-1, 2)
    layout = Layout(identity=identity, kit=VOXEL_KIT, ground=ground, region=region,
                    routes=np.asarray(routes, float).reshape(-1, 2, 2), points=points, obstacles=[],
                    sites=list(sites), paths=list(paths), courts=list(courts), allowed=allowed,
                    route_margin=2.5, point_margin=9.0)
    return layout


def layout_summary(layout):
    counts = {}
    for piece in layout.pieces:
        counts[piece.kind] = counts.get(piece.kind, 0) + 1
    return dict(version=LAYOUT_VERSION, kit=layout.kit.name, pieces=len(layout.pieces), kinds=counts,
                route_margin=layout.route_margin, point_margin=layout.point_margin,
                collision=False, observation=False)


def min_clearance(layout, positions):
    """Smallest XZ distance from any trajectory position to any set-piece footprint."""
    positions = np.asarray(positions, float).reshape(-1, 2)
    best = np.inf
    for piece in layout.pieces:
        d = np.linalg.norm(positions - piece.xz, axis=1) - piece.radius
        best = min(best, float(d.min()))
    return best


def voxel_allowed(world, forbid, *, sea_level, min_clear=3):
    """Natural, dry, open ground outside ``forbid`` (a boolean XZ mask)."""
    from ..core import blocks as B
    natural = np.zeros(256, bool)
    natural[[B.GRASS, B.DIRT, B.MOSS_STONE, B.STONE, B.SAND, B.GRAVEL]
            if hasattr(B, "GRAVEL") else [B.GRASS, B.DIRT, B.MOSS_STONE, B.STONE, B.SAND]] = True
    canopy = np.isin(world.vox, (B.LOG, B.LEAVES))

    def allowed(xz):
        x = np.clip(np.floor(xz[:, 0]).astype(int), 0, world.size_x - 1)
        z = np.clip(np.floor(xz[:, 1]).astype(int), 0, world.size_z - 1)
        y = world.height[x, z].astype(int)
        top = world.vox[x, np.clip(y, 0, world.size_y - 1), z]
        ok = (~forbid[x, z]) & natural[top] & (y >= sea_level + 1) & (world.water_height[x, z] < y)
        above = np.clip(y[:, None] + np.arange(1, min_clear + 1)[None, :], 0, world.size_y - 1)
        ok &= ~canopy[x[:, None], above, z[:, None]].any(axis=1)
        return ok
    allowed.forbid = forbid
    return allowed


def wildlands_reference_layout(native):
    """Set pieces on the natural ground around the Wildlands reference course."""
    from ..env import wildlands_reference as W
    world = native.world
    coords = [p for _, _, p in (W.A if native.task == "a" else W.B_TASK)]
    forbid = np.zeros((world.size_x, world.size_z), bool)
    for x, z in coords:
        forbid[max(0, x - 13):x + 17, max(0, z - 12):z + 13] = True
    for z in (24, 60, 96, 128):
        forbid[17:153, z - 4:z + 5] = True
    for x in (24, 52, 80, 108, 136):
        forbid[x - 4:x + 5, 17:136] = True
    forbid[119:145, 85:103] = True
    segments = [((20, z), (149, z)) for z in (24, 60, 96, 128)] + [((x, 20), (x, 132)) for x in (24, 52, 80, 108, 136)]
    roles = {"clue": "survey", "balance": "lift", "thermal": "crossing", "exchange": "exchange",
             "flow": "workshop", "cook": "lesson", "calibrate": "relay", "report": "survey"}
    sites, seen = [], set()
    for m in native.milestones:
        if tuple(m.site) in seen:
            continue
        seen.add(tuple(m.site))
        sites.append((m.id, (m.site[0] + 1.5, m.site[1]), roles.get(m.kind, "workshop")))
    layout = voxel_layout(native, identity=("wildlands-reference", int(native._seed), native.task, LAYOUT_VERSION),
                          sites=sites, routes=np.asarray(segments, float),
                          region=(4, world.size_x - 4, 4, world.size_z - 4),
                          allowed=voxel_allowed(world, forbid, sea_level=world.sea_level))
    layout.point_margin = 12.0
    layout.footprint, layout.relief = .45, 1.2
    plan_pieces(layout, ambient_spacing=12.0, ambient_density=.62)
    layout.mask_image = paint_ground(layout, resolution=1024, margin=8)
    return layout


def wildlands_course_trim(native, batch):
    """Bases, capitals and cornices around the courtyard gate voxels.

    The voxel columns and lintel are the collision geometry and stay as they
    are; this trim only wraps them so the gate reads as built architecture.
    """
    from ..env import wildlands_reference as W
    coords = {p for _, _, p in (W.A if native.task == "a" else W.B_TASK)}
    batch.current_material = STONE
    for x, z in sorted(coords):
        if (x, z) == (90, 96):
            continue
        rng = _rng("wildlands-gate", int(native._seed), x, z)
        base = (.56, .54, .48)
        for dx in (-5, 5):
            cx, cz = x + dx + 1.0, z + 7.0
            batch.box((cx, 29.18, cz), (2.9, .36, 2.9), _stone(rng, base, .05))
            batch.box((cx, 29.5, cz), (2.5, .28, 2.5), _stone(rng, base, .05))
            batch.box((cx, 33.78, cz), (2.6, .3, 2.6), _stone(rng, base, .05))
            batch.box((cx, 33.55, cz), (2.25, .18, 2.25), _stone(rng, (.48, .46, .41), .05))
            for k in range(3):  # banding on the shaft
                batch.box((cx, 30.6 + k * 1.05, cz), (2.08, .12, 2.08), _stone(rng, (.45, .44, .39), .04))
        batch.box((x + .5, 35.1, z + 7.0), (12.9, .26, 2.5), _stone(rng, base, .05))
        batch.box((x + .5, 35.32, z + 7.0), (12.4, .18, 2.2), _stone(rng, (.48, .46, .41), .05))
        batch.box((x + .5, 34.5, z + 5.94), (1.0, 1.1, .14), _stone(rng, (.62, .58, .50), .05))
    batch.current_material = 0


def skylands_campaign_layout(env):
    """Surface-layer set pieces for a Skylands campaign world (static, whole world)."""
    world = env.world
    surface = env.levels[1]
    forbid = np.zeros((world.size_x, world.size_z), bool)
    road = getattr(env, "_road_mask", None)
    if road is not None:
        grown = road.copy()
        for axis in (0, 1):
            for shift in (-3, -2, -1, 1, 2, 3):
                grown |= np.roll(road, shift, axis=axis)
        forbid |= grown
    sites = []
    for m in env.milestones:
        x, z = int(m.x), int(m.z)
        forbid[max(0, x - 14):x + 15, max(0, z - 14):z + 15] = True
        if abs(m.elevation - surface) < 2:
            sites.append((m.id, (x + .5, z + .5), "workshop"))
    for x, z in (env.vertical_xy, env.descent_xy):
        forbid[max(0, x - 16):x + 17, max(0, z - 16):z + 17] = True
    forbid |= world.height > surface + 8   # sky islands overhead; keep the surface only
    roles = ("relay", "survey", "lift", "workshop", "sentinel", "crossing")
    sites = [(sid, xz, roles[k % len(roles)]) for k, (sid, xz, _) in enumerate(sites)]
    layout = voxel_layout(env, identity=("skylands-campaign", int(env._seed), env.task, LAYOUT_VERSION),
                          sites=sites, routes=np.empty((0, 2, 2)),
                          region=(12, world.size_x - 12, 12, world.size_z - 12),
                          allowed=voxel_allowed(world, forbid, sea_level=surface - 6))
    layout.point_margin = 10.0
    layout.footprint, layout.relief = .6, .8
    plan_pieces(layout, ambient_spacing=19.0, ambient_density=.55)
    layout.mask_image = paint_ground(layout, resolution=2048, margin=8)
    return layout
