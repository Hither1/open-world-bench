"""The voxel world container.

Dense ``uint8`` grid of block ids with cached surface heightmaps, fast
vectorised queries, a DDA raycaster for interaction/targeting, and in-place
edits (mining, placing, explosions, fire).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import blocks as B


@dataclass
class RayHit:
    hit: bool
    block: int = 0
    pos: tuple[int, int, int] = (0, 0, 0)       # voxel coordinate hit
    point: tuple[float, float, float] = (0.0, 0.0, 0.0)  # exact surface point
    normal: tuple[int, int, int] = (0, 0, 0)    # face normal
    dist: float = 0.0
    prev: tuple[int, int, int] = (0, 0, 0)      # last empty voxel before the hit


class VoxelWorld:
    """A finite voxel world of shape ``(size_x, size_y, size_z)``.

    Out-of-bounds columns behave like an infinite stone wall at the map edge
    and empty air above the sky, so agents cannot leave the playable area.
    """

    __slots__ = ("vox", "size_x", "size_y", "size_z", "sea_level", "height",
                 "water_height", "biome", "temperature", "moisture", "slope",
                 "_region_names", "meta", "rev", "_mip_cache")

    def __init__(self, size_x: int, size_y: int, size_z: int, sea_level: int = 18):
        self.size_x, self.size_y, self.size_z = int(size_x), int(size_y), int(size_z)
        self.sea_level = int(sea_level)
        self.vox = np.zeros((self.size_x, self.size_y, self.size_z), dtype=np.uint8)
        self.height = np.zeros((self.size_x, self.size_z), dtype=np.int16)
        self.water_height = np.full((self.size_x, self.size_z), -1, dtype=np.int16)
        self.biome = np.zeros((self.size_x, self.size_z), dtype=np.uint8)
        self.temperature = np.zeros((self.size_x, self.size_z), dtype=np.float32)
        self.moisture = np.zeros((self.size_x, self.size_z), dtype=np.float32)
        self.slope = np.zeros((self.size_x, self.size_z), dtype=np.float32)
        self._region_names: dict[int, str] = {}
        self.meta: dict = {}
        #: bumped on every edit; caches keyed on it stay coherent
        self.rev = 0
        self._mip_cache: dict[int, tuple[int, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # basic access
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.vox.shape

    def in_bounds(self, x: int, y: int, z: int) -> bool:
        return 0 <= x < self.size_x and 0 <= y < self.size_y and 0 <= z < self.size_z

    def in_bounds_xz(self, x: int, z: int) -> bool:
        return 0 <= x < self.size_x and 0 <= z < self.size_z

    def get(self, x: int, y: int, z: int) -> int:
        """Block id at an integer voxel; out of bounds -> STONE walls / AIR sky."""
        x, y, z = int(x), int(y), int(z)
        if not (0 <= x < self.size_x and 0 <= z < self.size_z):
            return B.STONE if 0 <= y < self.size_y else B.AIR
        if y < 0:
            return B.STONE
        if y >= self.size_y:
            return B.AIR
        return int(self.vox[x, y, z])

    def get_f(self, px: float, py: float, pz: float) -> int:
        return self.get(int(np.floor(px)), int(np.floor(py)), int(np.floor(pz)))

    def set_block(self, x: int, y: int, z: int, block: int,
                  update_height: bool = True) -> bool:
        x, y, z = int(x), int(y), int(z)
        if not self.in_bounds(x, y, z):
            return False
        self.vox[x, y, z] = block
        self.rev += 1
        if update_height:
            self._update_column(x, z)
        return True

    def _update_column(self, x: int, z: int) -> None:
        col = self.vox[x, :, z]
        solid = B.SOLID[col]
        idx = np.flatnonzero(solid)
        self.height[x, z] = idx[-1] if idx.size else -1
        wat = np.flatnonzero(col == B.WATER)
        self.water_height[x, z] = wat[-1] if wat.size else -1

    def height_mip(self, tile: int = 8) -> np.ndarray:
        """Max column height over ``tile`` x ``tile`` blocks, cached.

        The renderer uses this to skip empty space: a ray above a tile's
        maximum height cannot hit anything inside that tile, so it can jump
        straight across it instead of stepping voxel by voxel.
        """
        got = self._mip_cache.get(tile)
        if got is not None and got[0] == self.rev:
            return got[1]
        nx = (self.size_x + tile - 1) // tile
        nz = (self.size_z + tile - 1) // tile
        pad = np.full((nx * tile, nz * tile), -1, dtype=np.int16)
        pad[:self.size_x, :self.size_z] = self.height
        mip = pad.reshape(nx, tile, nz, tile).max(axis=(1, 3)).astype(np.int16)
        self._mip_cache[tile] = (self.rev, mip)
        return mip

    def recompute_heightmaps(self) -> None:
        """Recompute both cached heightmaps from scratch (vectorised)."""
        self.rev += 1
        solid = B.SOLID[self.vox]                      # (X, Y, Z)
        ys = np.arange(self.size_y, dtype=np.int16)[None, :, None]
        self.height = np.where(solid, ys, np.int16(-1)).max(axis=1).astype(np.int16)
        water = self.vox == B.WATER
        self.water_height = np.where(water, ys, np.int16(-1)).max(axis=1).astype(np.int16)
        self.recompute_slope()

    def recompute_slope(self) -> None:
        h = np.maximum(self.height, 0).astype(np.float32)
        gx = np.gradient(h, axis=0)
        gz = np.gradient(h, axis=1)
        self.slope = np.sqrt(gx * gx + gz * gz).astype(np.float32)

    # ------------------------------------------------------------------
    # vectorised queries
    # ------------------------------------------------------------------
    def blocks_at(self, pts: np.ndarray) -> np.ndarray:
        """``pts`` is ``(N, 3)`` float world positions -> ``(N,)`` block ids."""
        v = np.floor(np.asarray(pts, dtype=np.float32)).astype(np.int64)
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        out = np.zeros(len(v), dtype=np.uint8)
        oob_xz = (x < 0) | (x >= self.size_x) | (z < 0) | (z >= self.size_z)
        below = y < 0
        above = y >= self.size_y
        ok = ~(oob_xz | below | above)
        if ok.any():
            out[ok] = self.vox[x[ok], y[ok], z[ok]]
        out[oob_xz & ~above] = B.STONE
        out[below] = B.STONE
        out[above] = B.AIR
        return out

    def solid_at(self, pts: np.ndarray) -> np.ndarray:
        return B.SOLID[self.blocks_at(pts)]

    def surface_y(self, x: int, z: int) -> int:
        """Topmost solid y of a column (``-1`` if the column is empty)."""
        x, z = int(x), int(z)
        if not self.in_bounds_xz(x, z):
            return self.size_y - 1
        return int(self.height[x, z])

    def ground_y(self, x: float, z: float) -> int:
        return self.surface_y(int(np.floor(x)), int(np.floor(z)))

    def spawn_y(self, x: float, z: float) -> float:
        """A safe standing height for a column (above water, above ground)."""
        xi, zi = int(np.floor(x)), int(np.floor(z))
        top = self.surface_y(xi, zi)
        wat = int(self.water_height[xi, zi]) if self.in_bounds_xz(xi, zi) else -1
        return float(max(top, wat) + 1)

    def is_solid(self, x: int, y: int, z: int) -> bool:
        return bool(B.SOLID[self.get(x, y, z)])

    def is_liquid(self, x: int, y: int, z: int) -> bool:
        return bool(B.LIQUID[self.get(x, y, z)])

    def box_blocks(self, center: np.ndarray, half: np.ndarray) -> np.ndarray:
        """All block ids overlapped by an axis-aligned box."""
        lo = np.floor(center - half).astype(np.int64)
        hi = np.floor(center + half).astype(np.int64)
        xs = np.arange(lo[0], hi[0] + 1)
        ys = np.arange(lo[1], hi[1] + 1)
        zs = np.arange(lo[2], hi[2] + 1)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
        return self.blocks_at(pts + 0.5)

    def box_collides(self, center: np.ndarray, half: np.ndarray) -> bool:
        return bool(B.SOLID[self.box_blocks(center, half)].any())

    def box_touches(self, center: np.ndarray, half: np.ndarray, block: int) -> bool:
        return bool((self.box_blocks(center, half) == block).any())

    # ------------------------------------------------------------------
    # raycasting (single ray, Amanatides & Woo DDA)
    # ------------------------------------------------------------------
    def raycast(self, origin, direction, max_dist: float = 64.0,
                stop_on_liquid: bool = False,
                ignore: tuple[int, ...] = ()) -> RayHit:
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        n = np.linalg.norm(d)
        if n < 1e-9:
            return RayHit(False)
        d = d / n

        vx, vy, vz = (int(np.floor(o[0])), int(np.floor(o[1])), int(np.floor(o[2])))
        step = np.sign(d).astype(np.int64)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_delta = np.where(d != 0, np.abs(1.0 / d), np.inf)
            nxt = np.array([vx, vy, vz], dtype=np.float64) + (step > 0).astype(np.float64)
            t_max = np.where(d != 0, (nxt - o) / np.where(d != 0, d, 1.0), np.inf)
        t = 0.0
        prev = (vx, vy, vz)
        axis = 0
        while t <= max_dist:
            blk = self.get(vx, vy, vz)
            solid = bool(B.SOLID[blk]) or (stop_on_liquid and bool(B.LIQUID[blk]))
            if solid and blk not in ignore:
                normal = [0, 0, 0]
                normal[axis] = int(-step[axis])
                point = o + d * t
                return RayHit(True, int(blk), (vx, vy, vz),
                              (float(point[0]), float(point[1]), float(point[2])),
                              tuple(normal), float(t), prev)
            prev = (vx, vy, vz)
            axis = int(np.argmin(t_max))
            t = float(t_max[axis])
            if axis == 0:
                vx += int(step[0])
            elif axis == 1:
                vy += int(step[1])
            else:
                vz += int(step[2])
            t_max[axis] += t_delta[axis]
            if vy < -1 or vy > self.size_y + 1:
                break
        return RayHit(False, dist=float(min(t, max_dist)), prev=prev)

    def ray_from_camera(self, eye, yaw: float, pitch: float, max_dist: float = 8.0,
                        **kw) -> RayHit:
        d = (np.cos(pitch) * np.cos(yaw), np.sin(pitch), np.cos(pitch) * np.sin(yaw))
        return self.raycast(eye, d, max_dist, **kw)

    def line_of_sight(self, a, b) -> bool:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        d = b - a
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return True
        hit = self.raycast(a, d / dist, dist)
        return not hit.hit

    # ------------------------------------------------------------------
    # bulk edits
    # ------------------------------------------------------------------
    def fill_box(self, lo, hi, block: int, only_if=None) -> int:
        x0, y0, z0 = (max(0, int(lo[0])), max(0, int(lo[1])), max(0, int(lo[2])))
        x1 = min(self.size_x - 1, int(hi[0]))
        y1 = min(self.size_y - 1, int(hi[1]))
        z1 = min(self.size_z - 1, int(hi[2]))
        if x0 > x1 or y0 > y1 or z0 > z1:
            return 0
        sub = self.vox[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        if only_if is None:
            n = sub.size
            sub[...] = block
        else:
            mask = np.isin(sub, np.asarray(only_if, dtype=np.uint8))
            n = int(mask.sum())
            sub[mask] = block
        self._refresh_region(x0, x1, z0, z1)
        return n

    def destroy_sphere(self, center, radius: float,
                       spare_indestructible: bool = True) -> list[tuple[int, int, int, int]]:
        """Carve a sphere; returns the ``(x, y, z, old_block)`` list removed."""
        cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
        r = int(np.ceil(radius))
        x0, x1 = max(0, int(cx) - r), min(self.size_x - 1, int(cx) + r)
        y0, y1 = max(0, int(cy) - r), min(self.size_y - 1, int(cy) + r)
        z0, z1 = max(0, int(cz) - r), min(self.size_z - 1, int(cz) + r)
        if x0 > x1 or y0 > y1 or z0 > z1:
            return []
        xs = np.arange(x0, x1 + 1)[:, None, None] + 0.5 - cx
        ys = np.arange(y0, y1 + 1)[None, :, None] + 0.5 - cy
        zs = np.arange(z0, z1 + 1)[None, None, :] + 0.5 - cz
        d2 = xs ** 2 + ys ** 2 + zs ** 2
        sub = self.vox[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        mask = (d2 <= radius * radius) & (sub != B.AIR)
        if spare_indestructible:
            mask &= ~B.INDESTRUCTIBLE[sub]
        if not mask.any():
            return []
        idx = np.argwhere(mask)
        removed = [(int(x0 + i), int(y0 + j), int(z0 + k), int(sub[i, j, k]))
                   for i, j, k in idx]
        sub[mask] = B.AIR
        self._refresh_region(x0, x1, z0, z1)
        return removed

    def _refresh_region(self, x0: int, x1: int, z0: int, z1: int) -> None:
        self.rev += 1
        sub = self.vox[x0:x1 + 1, :, z0:z1 + 1]
        solid = B.SOLID[sub]
        ys = np.arange(self.size_y, dtype=np.int16)[None, :, None]
        self.height[x0:x1 + 1, z0:z1 + 1] = np.where(solid, ys, np.int16(-1)).max(axis=1)
        water = sub == B.WATER
        self.water_height[x0:x1 + 1, z0:z1 + 1] = np.where(water, ys, np.int16(-1)).max(axis=1)

    # ------------------------------------------------------------------
    # water / connectivity helpers
    # ------------------------------------------------------------------
    def water_body(self, start, limit: int = 4000) -> set[tuple[int, int, int]]:
        """Flood fill a connected body of water (used by shock conduction)."""
        sx, sy, sz = (int(start[0]), int(start[1]), int(start[2]))
        if self.get(sx, sy, sz) != B.WATER:
            return set()
        seen = {(sx, sy, sz)}
        stack = [(sx, sy, sz)]
        while stack and len(seen) < limit:
            x, y, z = stack.pop()
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                               (0, 0, 1), (0, 0, -1)):
                p = (x + dx, y + dy, z + dz)
                if p not in seen and self.get(*p) == B.WATER:
                    seen.add(p)
                    stack.append(p)
        return seen

    def region_name(self, x: float, z: float) -> str:
        xi, zi = int(np.clip(x, 0, self.size_x - 1)), int(np.clip(z, 0, self.size_z - 1))
        return self._region_names.get(int(self.biome[xi, zi]), "wilds")

    def set_region_names(self, mapping: dict[int, str]) -> None:
        self._region_names = dict(mapping)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        ids, counts = np.unique(self.vox, return_counts=True)
        return {B.name_of(int(i)): int(c) for i, c in zip(ids, counts)}

    def __repr__(self) -> str:  # pragma: no cover
        return (f"VoxelWorld({self.size_x}x{self.size_y}x{self.size_z}, "
                f"sea={self.sea_level})")
