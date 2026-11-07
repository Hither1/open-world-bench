"""Procedural island terrain: heightmap synthesis, biomes and sampling queries.

The world is a square island of `size` meters centred on the origin, with the
water plane at y = 0. A heightmap is baked once at construction so that the
physics, the renderer and the quest logic all agree on the same surface.

Coordinates: x / z are horizontal, y is up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import math3d as mth
from . import noise as nz

# Biome ids (kept small and dense so they can index a colour table).
OCEAN, BEACH, MEADOW, FOREST, HIGHLAND, CLIFF, SNOW = range(7)
BIOME_NAMES = ["ocean", "beach", "meadow", "forest", "highland", "cliff", "snow"]


@dataclass
class TerrainConfig:
    seed: int = 0
    size: float = 1024.0           # world extent in metres (x and z)
    resolution: int = 513          # heightmap samples per axis (power of two + 1)
    max_height: float = 265.0      # roughly the summit height
    sea_level: float = 0.0
    shore_falloff: float = 0.95    # fraction of the half-extent where land ends
    terrace_step: float = 7.0      # stylized stepped cliffs (0 disables)
    terrace_strength: float = 0.33
    snow_line: float = 178.0
    tree_line: float = 126.0


@dataclass
class Terrain:
    cfg: TerrainConfig = field(default_factory=TerrainConfig)

    def __post_init__(self) -> None:
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        c = self.cfg
        n = c.resolution
        half = c.size * 0.5
        self._half = half
        self._cell = c.size / (n - 1)
        axis = np.linspace(-half, half, n, dtype=np.float32)
        gx, gz = np.meshgrid(axis, axis, indexing="ij")
        s = c.seed

        def unit(f, gain=2.9):
            """Remap an fbm field from its narrow spread to a full [0, 1]."""
            return np.clip(f * gain * 0.5 + 0.5, 0.0, 1.0)

        # Continental shape: large-scale warped fbm, highest near the centre.
        wx, wz = nz.domain_warp(gx, gz, s, strength=60.0, scale=1.0 / 520.0)
        cont = unit(nz.fbm(wx / 430.0, wz / 430.0, s + 11, octaves=5))
        hills = unit(nz.fbm(gx / 86.0, gz / 86.0, s + 23, octaves=5))
        detail = nz.fbm(gx / 23.0, gz / 23.0, s + 37, octaves=3)
        crest = nz.ridged(wx / 215.0, wz / 215.0, s + 53, octaves=6)
        crest = np.clip((crest - 0.3) / 0.7, 0.0, 1.0)

        # Radial island mask with a noisy coastline.
        r = np.sqrt(gx * gx + gz * gz) / half
        coast = 1.0 + 0.18 * nz.fbm(gx / 150.0, gz / 150.0, s + 71, octaves=4)
        mask = np.clip(1.0 - (r / (c.shore_falloff * coast)) ** 3.0, 0.0, 1.0) ** 0.85

        # Where mountains may grow: inland, and only on continental highs.
        mtn = mth.smoothstep(0.46, 0.88, cont) * np.clip(1.0 - r / 0.80, 0.0, 1.0)
        mtn = (mtn ** 1.15) * mask

        land = 10.0 + 52.0 * cont + 20.0 * hills + 3.4 * detail
        h = land * mask + crest * mtn * c.max_height
        h -= 16.0 * (1.0 - mask) ** 1.5      # carve the sea floor offshore
        h = self._terrace(h, mtn)
        h = h.astype(np.float32)

        self.height = h
        self.mountain_mask = mtn.astype(np.float32)
        self.moisture = (nz.fbm(gx / 240.0, gz / 240.0, s + 97, octaves=4) * 1.6 + 0.5).astype(np.float32)
        self.moisture = np.clip(self.moisture, 0.0, 1.0)
        self._compute_normals()
        self._compute_biomes()

    def _terrace(self, h: np.ndarray, mtn: np.ndarray) -> np.ndarray:
        """Quantize height toward flat plateaus -- the stylized cliff look."""
        c = self.cfg
        if c.terrace_step <= 0.0 or c.terrace_strength <= 0.0:
            return h
        step = c.terrace_step
        stepped = np.round(h / step) * step
        # Stronger terracing on highlands, none under water.
        w = c.terrace_strength * np.clip(mtn * 1.4 + 0.25, 0.0, 1.0) * (h > 1.0)
        return h * (1.0 - w) + stepped * w

    def _compute_normals(self) -> None:
        h = self.height
        d = self._cell
        dhdx = np.zeros_like(h)
        dhdz = np.zeros_like(h)
        dhdx[1:-1, :] = (h[2:, :] - h[:-2, :]) / (2.0 * d)
        dhdz[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2.0 * d)
        dhdx[0, :], dhdx[-1, :] = dhdx[1, :], dhdx[-2, :]
        dhdz[:, 0], dhdz[:, -1] = dhdz[:, 1], dhdz[:, -2]
        nrm = np.stack([-dhdx, np.ones_like(h), -dhdz], axis=-1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-8)
        self.normals = nrm.astype(np.float32)
        self.slope = np.arccos(np.clip(nrm[..., 1], -1.0, 1.0)).astype(np.float32)

    def _compute_biomes(self) -> None:
        c = self.cfg
        h, m, slope = self.height, self.moisture, self.slope
        b = np.full(h.shape, MEADOW, dtype=np.int8)
        b[h > c.tree_line] = HIGHLAND
        b[(h <= c.tree_line) & (h > 6.0) & (m > 0.52)] = FOREST
        b[h > c.snow_line] = SNOW
        b[(slope > np.radians(46.0)) & (h > 2.0)] = CLIFF
        b[h <= 3.5] = BEACH
        b[h <= c.sea_level] = OCEAN
        self.biome = b

    # ----------------------------------------------------------------- query
    def _to_grid(self, x, z):
        """World metres -> continuous grid indices, clamped to the map."""
        n = self.cfg.resolution
        gi = (np.asarray(x, dtype=np.float32) + self._half) / self._cell
        gj = (np.asarray(z, dtype=np.float32) + self._half) / self._cell
        return np.clip(gi, 0.0, n - 1.001), np.clip(gj, 0.0, n - 1.001)

    def _bilinear(self, field: np.ndarray, x, z):
        gi, gj = self._to_grid(x, z)
        i0 = np.floor(gi).astype(np.int32)
        j0 = np.floor(gj).astype(np.int32)
        i1, j1 = i0 + 1, j0 + 1
        tx = (gi - i0)[..., None] if field.ndim == 3 else gi - i0
        tz = (gj - j0)[..., None] if field.ndim == 3 else gj - j0
        f00, f10 = field[i0, j0], field[i1, j0]
        f01, f11 = field[i0, j1], field[i1, j1]
        a = f00 + (f10 - f00) * tx
        b = f01 + (f11 - f01) * tx
        return a + (b - a) * tz

    def height_at(self, x, z):
        """Ground height in metres; scalar in, scalar out (array in, array out)."""
        return self._bilinear(self.height, x, z)

    def normal_at(self, x, z):
        nrm = self._bilinear(self.normals, x, z)
        return nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-8)

    def slope_at(self, x, z):
        """Surface slope in radians (0 = flat)."""
        n = self.normal_at(x, z)
        return np.arccos(np.clip(n[..., 1], -1.0, 1.0))

    def biome_at(self, x, z):
        gi, gj = self._to_grid(x, z)
        return self.biome[np.rint(gi).astype(np.int32), np.rint(gj).astype(np.int32)]

    def moisture_at(self, x, z):
        return self._bilinear(self.moisture, x, z)

    def is_water(self, x, z) -> bool | np.ndarray:
        return self.height_at(x, z) < self.cfg.sea_level

    def water_depth(self, x, z):
        return np.maximum(self.cfg.sea_level - self.height_at(x, z), 0.0)

    def in_bounds(self, x, z, margin: float = 0.0) -> bool:
        lim = self._half - margin
        return bool(abs(float(x)) <= lim and abs(float(z)) <= lim)

    def clamp_to_bounds(self, x: float, z: float, margin: float = 2.0):
        lim = self._half - margin
        return float(np.clip(x, -lim, lim)), float(np.clip(z, -lim, lim))

    # --------------------------------------------------------------- sampling
    def sample_land(
        self,
        rng: np.random.Generator,
        n: int = 1,
        min_height: float = 2.0,
        max_height: float = 1e9,
        max_slope_deg: float = 30.0,
        radius: float | None = None,
        center: tuple[float, float] = (0.0, 0.0),
        tries: int = 4000,
    ) -> np.ndarray:
        """Rejection-sample `n` walkable (x, z) positions matching the filters."""
        lim = self._half * 0.95 if radius is None else radius
        out = []
        for _ in range(tries):
            if len(out) >= n:
                break
            x = center[0] + float(rng.uniform(-lim, lim))
            z = center[1] + float(rng.uniform(-lim, lim))
            if not self.in_bounds(x, z, margin=6.0):
                continue
            h = float(self.height_at(x, z))
            if not (min_height <= h <= max_height):
                continue
            if float(self.slope_at(x, z)) > np.radians(max_slope_deg):
                continue
            out.append((x, z))
        if not out:  # degenerate world; fall back to the highest point we know
            idx = np.unravel_index(int(np.argmax(self.height)), self.height.shape)
            x = -self._half + idx[0] * self._cell
            z = -self._half + idx[1] * self._cell
            out = [(float(x), float(z))] * n
        while len(out) < n:
            out.append(out[len(out) % max(1, len(out))])
        return np.asarray(out[:n], dtype=np.float32)

    def highest_point(self) -> np.ndarray:
        idx = np.unravel_index(int(np.argmax(self.height)), self.height.shape)
        return np.array(
            [
                -self._half + idx[0] * self._cell,
                float(self.height[idx]),
                -self._half + idx[1] * self._cell,
            ],
            dtype=np.float32,
        )

    def deepest_water_point(self, min_depth: float = 4.0) -> np.ndarray | None:
        """Centre of a deep-water pocket, used for the underwater quest site."""
        mask = self.height < -min_depth
        if not mask.any():
            return None
        idx = np.unravel_index(int(np.argmin(np.where(mask, self.height, 1e9))), mask.shape)
        return np.array(
            [
                -self._half + idx[0] * self._cell,
                float(self.height[idx]),
                -self._half + idx[1] * self._cell,
            ],
            dtype=np.float32,
        )

    # --------------------------------------------------------------- raycast
    def raymarch(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_dist: float = 600.0,
        steps: int = 96,
        refine: int = 6,
    ):
        """March a ray (or a batch of rays) against the heightmap.

        `origin` is (..., 3) and `direction` is (..., 3) normalized. Returns
        (hit, distance) where distance is `max_dist` where nothing was hit.
        """
        origin = np.asarray(origin, dtype=np.float32)
        direction = np.asarray(direction, dtype=np.float32)
        shape = np.broadcast_shapes(origin.shape[:-1], direction.shape[:-1])
        t = np.zeros(shape, dtype=np.float32)
        hit = np.zeros(shape, dtype=bool)
        prev_t = np.zeros(shape, dtype=np.float32)
        # Quadratic step growth keeps near-field detail without many samples.
        ramp = (np.arange(1, steps + 1, dtype=np.float32) / steps) ** 1.7 * max_dist
        for ti in ramp:
            p = origin + direction * ti
            gh = self.height_at(p[..., 0], p[..., 2])
            below = (p[..., 1] <= gh) & ~hit
            t = np.where(below, ti, t)
            prev_t = np.where(~hit & ~below, ti, prev_t)
            hit |= below
            if bool(np.all(hit)):
                break
        # Binary refinement between the last miss and the first hit.
        lo, hi = prev_t.copy(), t.copy()
        for _ in range(refine):
            mid = (lo + hi) * 0.5
            p = origin + direction * mid[..., None]
            gh = self.height_at(p[..., 0], p[..., 2])
            under = p[..., 1] <= gh
            hi = np.where(under, mid, hi)
            lo = np.where(under, lo, mid)
        dist = np.where(hit, hi, np.float32(max_dist))
        return hit, dist.astype(np.float32)
