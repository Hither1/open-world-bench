"""Vegetation / rock scattering plus a uniform spatial hash for fast lookups."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .terrain import BEACH, CLIFF, FOREST, HIGHLAND, MEADOW, SNOW, Terrain

# Prop kinds.
PINE, BROADLEAF, SAKURA, BAMBOO, ROCK, BOULDER, BUSH, GRASS, CRYSTAL = range(9)
PROP_NAMES = ["pine", "broadleaf", "sakura", "bamboo", "rock", "boulder", "bush", "grass", "crystal"]

# Collision radius in metres; 0 means you can walk straight through it.
PROP_RADIUS = np.array([0.55, 0.65, 0.6, 0.25, 0.0, 1.25, 0.0, 0.0, 0.45], dtype=np.float32)
# Rough drawn height in metres, used by the renderer and for camera occlusion.
PROP_HEIGHT = np.array([11.0, 9.0, 7.5, 6.0, 0.9, 2.6, 0.8, 0.35, 2.2], dtype=np.float32)

# Per-biome (kind, weight) tables.
_BIOME_TABLE = {
    MEADOW: [(GRASS, 0.52), (BUSH, 0.16), (BROADLEAF, 0.1), (SAKURA, 0.04), (ROCK, 0.14), (BOULDER, 0.04)],
    FOREST: [(PINE, 0.38), (BROADLEAF, 0.22), (BAMBOO, 0.1), (BUSH, 0.14), (GRASS, 0.12), (ROCK, 0.04)],
    HIGHLAND: [(ROCK, 0.42), (BOULDER, 0.2), (GRASS, 0.2), (PINE, 0.1), (CRYSTAL, 0.08)],
    SNOW: [(ROCK, 0.45), (BOULDER, 0.3), (CRYSTAL, 0.25)],
    BEACH: [(GRASS, 0.5), (ROCK, 0.4), (BOULDER, 0.1)],
    CLIFF: [(ROCK, 0.7), (BOULDER, 0.3)],
}
# Props per 100 m^2 per biome.
_BIOME_DENSITY = {MEADOW: 0.55, FOREST: 1.25, HIGHLAND: 0.3, SNOW: 0.12, BEACH: 0.18, CLIFF: 0.08}


@dataclass
class PropField:
    """Immutable scattered props with a spatial hash over the xz plane."""

    pos: np.ndarray       # (N, 3) float32, y = ground height
    yaw: np.ndarray       # (N,) float32
    scale: np.ndarray     # (N,) float32
    kind: np.ndarray      # (N,) int8
    cell_size: float = 8.0

    def __post_init__(self) -> None:
        self.radius = PROP_RADIUS[self.kind] * self.scale
        self.height = PROP_HEIGHT[self.kind] * self.scale
        self._build_hash()

    def _build_hash(self) -> None:
        if len(self.pos) == 0:
            self._cells: dict[tuple[int, int], np.ndarray] = {}
            return
        keys = np.floor(self.pos[:, [0, 2]] / self.cell_size).astype(np.int32)
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        sorted_keys = keys[order]
        # Group indices by cell key.
        boundaries = np.any(np.diff(sorted_keys, axis=0) != 0, axis=1).nonzero()[0] + 1
        groups = np.split(order, boundaries)
        starts = np.concatenate(([0], boundaries))
        self._cells = {tuple(sorted_keys[s].tolist()): g for s, g in zip(starts, groups)}

    def __len__(self) -> int:
        return int(len(self.pos))

    def query_radius(self, x: float, z: float, r: float) -> np.ndarray:
        """Indices of props whose centre is within `r` metres of (x, z)."""
        if not self._cells:
            return np.empty(0, dtype=np.int64)
        cs = self.cell_size
        i0, i1 = int(np.floor((x - r) / cs)), int(np.floor((x + r) / cs))
        j0, j1 = int(np.floor((z - r) / cs)), int(np.floor((z + r) / cs))
        buckets = [self._cells[(i, j)] for i in range(i0, i1 + 1) for j in range(j0, j1 + 1) if (i, j) in self._cells]
        if not buckets:
            return np.empty(0, dtype=np.int64)
        cand = np.concatenate(buckets)
        d = self.pos[cand][:, [0, 2]] - np.array([x, z], dtype=np.float32)
        return cand[(d[:, 0] ** 2 + d[:, 1] ** 2) <= r * r]

    def solid_near(self, x: float, z: float, agent_radius: float) -> np.ndarray:
        """Indices of *colliding* props overlapping a circle at (x, z)."""
        idx = self.query_radius(x, z, agent_radius + 1.6)
        if len(idx) == 0:
            return idx
        idx = idx[self.radius[idx] > 0.0]
        if len(idx) == 0:
            return idx
        d = self.pos[idx][:, [0, 2]] - np.array([x, z], dtype=np.float32)
        dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        return idx[dist < (self.radius[idx] + agent_radius)]

    def visible_subset(self, eye: np.ndarray, forward: np.ndarray, max_dist: float, fov_cos: float = -0.1):
        """Indices of props in front of the camera and within `max_dist`."""
        if len(self.pos) == 0:
            return np.empty(0, dtype=np.int64)
        d = self.pos - eye[None, :]
        dist2 = (d[:, 0] ** 2 + d[:, 2] ** 2)
        near = dist2 < max_dist * max_dist
        idx = near.nonzero()[0]
        if len(idx) == 0:
            return idx
        dn = d[idx] / np.maximum(np.linalg.norm(d[idx], axis=1, keepdims=True), 1e-6)
        return idx[(dn @ forward) > fov_cos]


def scatter_props(terrain: Terrain, seed: int = 0, density_scale: float = 1.0) -> PropField:
    """Jittered-grid scatter: deterministic, biome-weighted, slope-aware."""
    rng = np.random.default_rng(seed * 7919 + 13)
    cfg = terrain.cfg
    half = cfg.size * 0.5 - 6.0
    cell = 10.0  # metres per candidate cell -> 100 m^2, matching _BIOME_DENSITY
    coords = np.arange(-half, half, cell, dtype=np.float32)
    gx, gz = np.meshgrid(coords, coords, indexing="ij")
    jitter = rng.uniform(0.0, cell, size=gx.shape + (2,)).astype(np.float32)
    px = (gx + jitter[..., 0]).ravel()
    pz = (gz + jitter[..., 1]).ravel()

    biome = terrain.biome_at(px, pz)
    height = terrain.height_at(px, pz)
    slope = terrain.slope_at(px, pz)

    keep_pos, keep_kind = [], []
    u = rng.random(px.shape[0])
    for b, table in _BIOME_TABLE.items():
        sel = (biome == b) & (height > 0.3) & (slope < np.radians(58.0))
        if not sel.any():
            continue
        dens = _BIOME_DENSITY[b] * density_scale
        # Expected props per cell: thin the candidate set, then expand dense cells.
        n_cell = np.full(sel.sum(), dens, dtype=np.float32)
        counts = rng.poisson(n_cell)
        sub_x, sub_z = px[sel], pz[sel]
        kinds, weights = zip(*table)
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        for n_extra in range(1, int(counts.max()) + 1 if len(counts) else 1):
            m = counts >= n_extra
            if not m.any():
                continue
            cx = sub_x[m] + rng.uniform(-cell * 0.45, cell * 0.45, m.sum()).astype(np.float32)
            cz = sub_z[m] + rng.uniform(-cell * 0.45, cell * 0.45, m.sum()).astype(np.float32)
            keep_pos.append(np.stack([cx, cz], axis=1))
            keep_kind.append(rng.choice(np.asarray(kinds, dtype=np.int8), size=m.sum(), p=weights))
    del u

    if not keep_pos:
        empty3 = np.zeros((0, 3), dtype=np.float32)
        return PropField(empty3, np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.int8))

    xz = np.concatenate(keep_pos).astype(np.float32)
    kind = np.concatenate(keep_kind).astype(np.int8)
    # Drop anything that landed in water or on a near-vertical face.
    h = terrain.height_at(xz[:, 0], xz[:, 1])
    s = terrain.slope_at(xz[:, 0], xz[:, 1])
    ok = (h > 0.25) & (s < np.radians(60.0))
    # Trees refuse to grow on steep ground or above the tree line.
    tree = np.isin(kind, [PINE, BROADLEAF, SAKURA, BAMBOO])
    ok &= ~(tree & ((s > np.radians(34.0)) | (h > cfg.tree_line + 8.0)))
    xz, kind, h = xz[ok], kind[ok], h[ok]

    pos = np.stack([xz[:, 0], h, xz[:, 1]], axis=1).astype(np.float32)
    yaw = rng.uniform(0.0, 2.0 * np.pi, len(pos)).astype(np.float32)
    scale = (0.78 + 0.45 * rng.random(len(pos))).astype(np.float32)
    # Sakura and crystals read better slightly larger; grass smaller.
    scale[kind == SAKURA] *= 1.15
    scale[kind == GRASS] *= 0.9
    return PropField(pos, yaw, scale, kind)
