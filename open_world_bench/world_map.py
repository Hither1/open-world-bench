"""Top-down base map of a Teyvat island for the clickable fog-of-war map.

The UI, fog of war and click handling live in :mod:`owb.sim.world_map`; this
module only paints the ground truth that explored regions reveal: biome
albedo with hillshading, shallow-to-deep water, trees and rocks, and building
footprints from the collision architecture. Colour variation is keyed to
world position, never to a simulation RNG.
"""
from __future__ import annotations

import numpy as np

from .render.style import BIOME_ALBEDO
from .scatter import PINE, BROADLEAF, SAKURA, BAMBOO, ROCK, BOULDER, CRYSTAL

_PROP_COLORS = {
    PINE: (0.16, 0.36, 0.22), BROADLEAF: (0.24, 0.46, 0.22), BAMBOO: (0.38, 0.58, 0.26),
    SAKURA: (0.92, 0.66, 0.76), ROCK: (0.56, 0.55, 0.57), BOULDER: (0.48, 0.47, 0.50),
    CRYSTAL: (0.58, 0.80, 0.96),
}


def _architecture(architecture):
    """Static oriented solid boxes (centers, halves, rotations) or None."""
    base = getattr(architecture, "base", architecture)   # campaign gates wrap a base
    centers = getattr(base, "centers", None)
    if centers is None or not len(centers):
        return None
    return base.centers, base.halves, base.rotations


def terrain_surface(terrain, props=None, architecture=None, resolution: int = 512):
    """Return (rgb uint8 (n, n, 3) indexed [x, z], bounds (x0, z0, x1, z1))."""
    half = float(terrain._half)
    n = int(resolution)
    cell = 2 * half / n
    axis = -half + (np.arange(n) + 0.5) * cell
    gx, gz = np.meshgrid(axis, axis, indexing="ij")
    h = np.asarray(terrain.height_at(gx, gz), np.float32)
    albedo = BIOME_ALBEDO[np.asarray(terrain.biome_at(gx, gz))].astype(np.float32)

    # Lambert hillshade with a north-west sun.
    dhx, dhz = np.gradient(h, cell)
    normal = np.stack([-dhx, np.ones_like(h), -dhz], -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    sun = np.array((-0.55, 0.72, -0.42), np.float32)
    sun /= np.linalg.norm(sun)
    shade = 0.62 + 0.5 * np.clip(normal @ sun, 0, 1)
    rgb = albedo * shade[..., None] * (0.92 + 0.12 * np.clip(h / max(terrain.cfg.max_height, 1), 0, 1))[..., None]

    depth = terrain.cfg.sea_level - h
    water = depth > 0
    t = np.clip(depth / 14.0, 0, 1)[..., None]
    sea = np.array((0.36, 0.72, 0.80), np.float32) * (1 - t) + np.array((0.09, 0.27, 0.48), np.float32) * t
    foam = water & (depth < 0.7)
    rgb = np.where(water[..., None], sea, rgb)
    rgb[foam] = rgb[foam] * 0.5 + 0.5

    if props is not None and len(getattr(props, "pos", ())):
        idx = np.floor((np.asarray(props.pos)[:, [0, 2]] + half) / cell).astype(int)
        ok = (idx >= 0).all(1) & (idx < n).all(1)
        for kind, color in _PROP_COLORS.items():
            sel = ok & (np.asarray(props.kind) == kind)
            if sel.any():
                i, j = idx[sel, 0], idx[sel, 1]
                rgb[i, j] = rgb[i, j] * 0.25 + np.asarray(color, np.float32) * 0.75

    boxes = _architecture(architecture)
    if boxes is not None:
        centers, halves, rotations = boxes
        tops = centers[:, 1] + halves[:, 1]
        for k in np.argsort(tops):   # roofs over walls
            c, hf, rot = centers[k], halves[k], rotations[k]
            reach = float(np.hypot(hf[0], hf[2]))
            i0, i1 = np.clip(np.floor((c[[0, 0]] + (-reach, reach) + half) / cell).astype(int) + (0, 1), 0, n)
            j0, j1 = np.clip(np.floor((c[[2, 2]] + (-reach, reach) + half) / cell).astype(int) + (0, 1), 0, n)
            if i0 >= i1 or j0 >= j1:
                continue
            px, pz = np.meshgrid(axis[i0:i1] - c[0], axis[j0:j1] - c[2], indexing="ij")
            local_x = px * rot[0, 0] + pz * rot[2, 0]
            local_z = px * rot[0, 2] + pz * rot[2, 2]
            inside = (np.abs(local_x) <= hf[0] + cell * 0.5) & (np.abs(local_z) <= hf[2] + cell * 0.5)
            lift = np.clip((tops[k] - terrain.height_at(c[0], c[2])) / 12.0, 0, 1)
            color = np.array((0.50, 0.40, 0.34), np.float32) * (1 - lift) + np.array((0.70, 0.36, 0.28), np.float32) * lift
            block = rgb[i0:i1, j0:j1]
            block[inside] = color
    return (rgb * 255).clip(0, 255).astype(np.uint8), (-half, -half, half, half)
