"""Original, seeded three-layer construction course in a single voxel world."""
from __future__ import annotations

import numpy as np

from ..core import blocks as B
from .voxel import VoxelWorld
from .worldgen import Landmark, WorldGenResult


def _island(w, cx, cz, rx, rz, top, depth, phase, bounds=None):
    """An irregular meadow cap over a tapered, stratified stone mass."""
    for x in range(max(1, int(cx - rx - 1)), min(w.size_x - 1, int(cx + rx + 2))):
        for z in range(max(1, int(cz - rz - 1)), min(w.size_z - 1, int(cz + rz + 2))):
            if bounds and not bounds[0] <= x <= bounds[1]:
                continue
            dx, dz = (x + 0.5 - cx) / rx, (z + 0.5 - cz) / rz
            angle = np.arctan2(dz, dx)
            radius = np.hypot(dx, dz) / (1 + 0.06 * np.sin(angle * 5 + phase))
            if radius > 1:
                continue
            bottom = max(1, top - int(2 + depth * (1 - radius) ** 0.7))
            w.vox[x, bottom:top, z] = B.MOSS_STONE
            w.vox[x, bottom:bottom + 2, z] = B.STONE
            w.vox[x, top, z] = B.GRASS


def _tree(w, x, y, z, height=4):
    w.fill_box((x - 2, y + height - 1, z - 2), (x + 2, y + height + 1, z + 2), B.LEAVES)
    w.fill_box((x, y, z), (x, y + height - 1, z), B.LOG)


def generate(cfg):
    if min(cfg.size_x, cfg.size_z) < 32 or cfg.size_y < 40:
        raise ValueError("Skylands requires at least 32 x 40 x 32 voxels (x, y, z)")
    rng = np.random.default_rng(cfg.seed)
    sx, sy, sz = cfg.size_x, cfg.size_y, cfg.size_z
    w = VoxelWorld(sx, sy, sz, sea_level=0)
    surface, sky = max(14, sy // 3), sy - 10
    edge, z = sx // 2 - 3, sz // 2 + int(rng.integers(-2, 3))
    # The surface is a thin roof above a walkable cavern, not a filled column.
    w.fill_box((0, 0, 0), (sx - 1, 3, sz - 1), B.BASALT)
    w.fill_box((0, surface - 2, 0), (sx - 1, surface, sz - 1), B.STONE)
    w.fill_box((0, surface, 0), (sx - 1, surface, sz - 1), B.GRASS)
    for lo, hi in (((0, 4, 0), (1, surface - 3, sz - 1)),
                   ((sx - 2, 4, 0), (sx - 1, surface - 3, sz - 1)),
                   ((0, 4, 0), (sx - 1, surface - 3, 1)),
                   ((0, 4, sz - 2), (sx - 1, surface - 3, sz - 1))):
        w.fill_box(lo, hi, B.BASALT)
    # A real opening connects the surface and depths; gliding controls the fall.
    w.fill_box((4, 4, z + 7), (8, surface, z + 11), B.AIR)
    for _ in range(max(8, sx // 3)):
        x, zz = int(rng.integers(2, sx - 2)), int(rng.integers(2, sz - 2))
        w.set_block(x, 4, zz, B.CRYSTAL)
        if abs(zz - z) > 7:
            w.set_block(x, surface + 1, zz, B.FLOWER)
    # Irregular floating meadows, with a guaranteed clear construction lane.
    for lo, hi in ((4, edge - 1), (edge + 6, sx - 5)):
        _island(w, (lo + hi + 1) / 2, z + 0.5, (hi - lo + 1) / 2, 9,
                sky, 10, float(rng.uniform(0, 6)), (lo, hi))
    for lo, hi in ((edge - 6, edge - 1), (edge + 6, edge + 11)):
        w.fill_box((lo, sky - 2, z - 4), (hi, sky, z + 4), B.MOSS_STONE)
        w.fill_box((lo, sky, z - 4), (hi, sky, z + 4), B.GRASS)
    # Satellite islands create a layered skyline and gliding destinations.
    satellites = [(sx * 0.30, z - 16, 5, 4, sky - 5),
                  (sx * 0.70, z - 19, 7, 5, sky + 1),
                  (sx * 0.87, z + 16, 5, 4, sky - 9),
                  (sx * 0.18, z + 16, 4, 4, sky - 3)]
    for cx, cz, rx, rz, top in satellites:
        if 4 < cz < sz - 4:
            _island(w, cx, cz, rx, rz, top, 9, float(rng.uniform(0, 6)))
            _tree(w, int(cx), top + 1, int(cz), 3)
    # Golden trees, worn paths, broken pillars and hanging vines frame the lane.
    for x, zz in ((edge - 7, z - 4), (edge - 8, z + 4), (edge + 10, z - 4), (sx - 9, z + 5)):
        if 2 <= x < sx - 2 and w.get(x, sky, zz) == B.GRASS:
            _tree(w, x, sky + 1, zz)
    for x in range(5, sx - 4):
        if w.get(x, sky, z) == B.GRASS and x % 3 != 0:
            w.set_block(x, sky, z, B.BRICK)
    for x, zz in ((edge - 1, z - 4), (edge + 6, z + 4)):
        for yy in range(sky - 7, sky):
            w.set_block(x, yy, zz, B.VINE)
    for _ in range(45):
        x, zz = int(rng.integers(3, sx - 3)), int(rng.integers(max(2, z - 8), min(sz - 2, z + 9)))
        if abs(zz - z) > 1 and w.get(x, sky, zz) == B.GRASS and w.get(x, sky + 1, zz) == B.AIR:
            w.set_block(x, sky + 1, zz, B.FLOWER if rng.random() < 0.3 else B.TALL_GRASS)
    arch_x = edge + 10
    for zz in (z - 3, z + 3):
        w.fill_box((arch_x, sky + 1, zz), (arch_x + 1, sky + 5, zz), B.BRICK)
        w.fill_box((arch_x - 1, sky + 1, zz - 1), (arch_x + 1, sky + 1, zz + 1), B.MOSS_STONE)
    w.fill_box((arch_x, sky + 6, z - 2), (arch_x + 1, sky + 6, z + 2), B.BRICK)
    w.set_block(arch_x, sky + 5, z - 2, B.BRICK)
    w.set_block(arch_x, sky + 5, z + 2, B.BRICK)
    beacon = (edge + 8, sky + 1, z)
    w.set_block(*beacon, B.BEACON)
    # A surface shelter provides a short, safe ascent before the sky ascent.
    w.fill_box((4, surface + 5, z - 4), (8, surface + 5, z - 1), B.BRICK)
    if sky - surface > 36:
        w.fill_box((edge - 6, surface + 24, z - 2), (edge - 2, surface + 24, z + 2), B.MOSS_STONE)
    # Surface copses and a small turquoise basin are visible far below the sky.
    for x, zz in ((sx // 4, z + 13), (sx // 2, z - 10), (sx - 8, z - 10), (10, z - 12)):
        if 3 <= zz < sz - 3:
            _tree(w, x, surface + 1, zz, 5)
    w.fill_box((sx // 2 - 5, surface - 1, z + 12), (sx // 2 + 5, surface, z + 18), B.WATER)
    w.temperature[:] = 0.15  # normalized climate driver, not degrees Celsius
    w.moisture[:] = 0.5
    w.set_region_names({0: "Aster Reach"})
    w.meta.update(theme="skylands", layered=True, layer_heights=(3, surface, sky), bridge=(edge, sky, z, 6),
                  beacon=beacon,
                  parts=[dict(anchor=(edge - 4, sky + 1, z - 3)),
                         dict(anchor=(edge - 4, sky + 1, z + 3))],
                  scenery=[dict(kind="sky_ring", pos=(arch_x + 0.4, sky + 4.3, z + 0.5),
                                radius=2.2, height=5, semantic=B.SEM_INTERACTIVE, color=(102, 215, 190))])
    # Cavern pillars and luminous fungi give the depths its own silhouette.
    for x, zz in ((12, z + 3), (20, z + 8), (sx - 7, z - 6), (sx // 2, z - 12)):
        if 3 <= x < sx - 3 and 3 <= zz < sz - 3:
            w.fill_box((x, 4, zz), (x + 1, surface - 3, zz + 1), B.BASALT)
            w.set_block(x - 1, 4, zz, B.CRYSTAL)
    for x, zz, height in ((10, z + 6, 3), (16, z + 11, 4), (22, z + 4, 3), (9, z - 4, 2.5)):
        if 3 <= zz < sz - 3:
            w.set_block(x, 4, zz, B.CRYSTAL)
            w.meta["scenery"].append(dict(kind="sky_mushroom", pos=(x + 0.5, 4, zz + 0.5),
                                           radius=height * 0.55, height=height,
                                           semantic=B.SEM_PLANT, color=(84, 187, 183)))
    w.meta["crystal_lights"] = [tuple(p + 0.5) for p in np.argwhere(w.vox == B.CRYSTAL)]
    # Keep the two construction parts' footprints empty after dressing the scene.
    for part in w.meta["parts"]:
        x, y, zz = part["anchor"]
        w.fill_box((x, y, zz), (x + 2, y + 2, zz), B.AIR)
    w.recompute_heightmaps()
    spawn = (edge - 3.5, sky + 1.01, z + 0.5)
    return WorldGenResult(w, spawn, landmarks=[
        Landmark("sky", "Aster workshop", spawn, radius=3),
        Landmark("surface", "Roof garden", (6.5, surface + 1, z - 2.5), radius=3),
        Landmark("depths", "Crystal hollow", (6.5, 4, z + 8.5), radius=4),
    ])
