"""Vectorised procedural noise (Perlin / fBm / ridged / Worley).

Everything takes and returns NumPy arrays so a whole 512x512 heightmap is one
call.  Noise is fully determined by ``seed``, which keeps worlds reproducible
across machines (no reliance on Python's hash randomisation).
"""

from __future__ import annotations

import numpy as np

_MASK = np.uint32(0xFFFFFFFF)


def _hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Integer hash -> uint32, deterministic and cheap."""
    h = (ix.astype(np.uint32) * np.uint32(374761393)) ^ \
        (iy.astype(np.uint32) * np.uint32(668265263)) ^ np.uint32(seed & 0xFFFFFFFF)
    h ^= h >> np.uint32(13)
    h = (h * np.uint32(1274126177)) & _MASK
    h ^= h >> np.uint32(16)
    return h


def _hash3(ix, iy, iz, seed: int) -> np.ndarray:
    h = (ix.astype(np.uint32) * np.uint32(374761393)) ^ \
        (iy.astype(np.uint32) * np.uint32(668265263)) ^ \
        (iz.astype(np.uint32) * np.uint32(2147483647)) ^ np.uint32(seed & 0xFFFFFFFF)
    h ^= h >> np.uint32(13)
    h = (h * np.uint32(1274126177)) & _MASK
    h ^= h >> np.uint32(16)
    return h


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def hash_unit(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic uniform value in [0, 1) per integer lattice point."""
    return (_hash2(ix, iy, seed) & np.uint32(0xFFFFFF)).astype(np.float32) / 16777216.0


def hash_unit3(ix, iy, iz, seed: int) -> np.ndarray:
    return (_hash3(ix, iy, iz, seed) & np.uint32(0xFFFFFF)).astype(np.float32) / 16777216.0


def perlin2(x: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """2-D gradient (Perlin) noise in roughly [-1, 1]."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    u, v = _fade(fx), _fade(fy)

    def grad_dot(ix, iy, dx, dy):
        h = _hash2(ix, iy, seed)
        ang = (h & np.uint32(0xFFFF)).astype(np.float32) * (np.float32(2 * np.pi) / 65536.0)
        return np.cos(ang) * dx + np.sin(ang) * dy

    n00 = grad_dot(x0, y0, fx, fy)
    n10 = grad_dot(x0 + 1, y0, fx - 1.0, fy)
    n01 = grad_dot(x0, y0 + 1, fx, fy - 1.0)
    n11 = grad_dot(x0 + 1, y0 + 1, fx - 1.0, fy - 1.0)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return (nx0 + v * (nx1 - nx0)).astype(np.float32)


def perlin3(x, y, z, seed: int = 0) -> np.ndarray:
    """3-D gradient noise in roughly [-1, 1] (used for caves and clouds)."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    fx, fy, fz = x - x0, y - y0, z - z0
    u, v, w = _fade(fx), _fade(fy), _fade(fz)

    def grad_dot(ix, iy, iz, dx, dy, dz):
        h = _hash3(ix, iy, iz, seed)
        th = (h & np.uint32(0xFFFF)).astype(np.float32) * (np.float32(2 * np.pi) / 65536.0)
        ph = ((h >> np.uint32(16)) & np.uint32(0xFFFF)).astype(np.float32) * (np.float32(np.pi) / 65536.0)
        sp = np.sin(ph)
        return np.cos(th) * sp * dx + np.cos(ph) * dy + np.sin(th) * sp * dz

    c = [
        grad_dot(x0, y0, z0, fx, fy, fz),
        grad_dot(x0 + 1, y0, z0, fx - 1, fy, fz),
        grad_dot(x0, y0 + 1, z0, fx, fy - 1, fz),
        grad_dot(x0 + 1, y0 + 1, z0, fx - 1, fy - 1, fz),
        grad_dot(x0, y0, z0 + 1, fx, fy, fz - 1),
        grad_dot(x0 + 1, y0, z0 + 1, fx - 1, fy, fz - 1),
        grad_dot(x0, y0 + 1, z0 + 1, fx, fy - 1, fz - 1),
        grad_dot(x0 + 1, y0 + 1, z0 + 1, fx - 1, fy - 1, fz - 1),
    ]
    x00 = c[0] + u * (c[1] - c[0])
    x10 = c[2] + u * (c[3] - c[2])
    x01 = c[4] + u * (c[5] - c[4])
    x11 = c[6] + u * (c[7] - c[6])
    y0i = x00 + v * (x10 - x00)
    y1i = x01 + v * (x11 - x01)
    return (y0i + w * (y1i - y0i)).astype(np.float32)


def fbm2(x, y, seed: int = 0, octaves: int = 5, lacunarity: float = 2.0,
         gain: float = 0.5) -> np.ndarray:
    """Fractal Brownian motion, normalised to about [-1, 1]."""
    total = np.zeros(np.broadcast(x, y).shape, dtype=np.float32)
    amp, freq, norm = 1.0, 1.0, 0.0
    for o in range(octaves):
        total += amp * perlin2(x * freq, y * freq, seed + o * 1013)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / max(norm, 1e-6)


def ridged2(x, y, seed: int = 0, octaves: int = 5, lacunarity: float = 2.0,
            gain: float = 0.5) -> np.ndarray:
    """Ridged multifractal in [0, 1] -- sharp mountain crests."""
    total = np.zeros(np.broadcast(x, y).shape, dtype=np.float32)
    amp, freq, norm = 1.0, 1.0, 0.0
    for o in range(octaves):
        n = 1.0 - np.abs(perlin2(x * freq, y * freq, seed + o * 2203))
        total += amp * (n * n)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / max(norm, 1e-6)


def warp2(x, y, seed: int = 0, strength: float = 4.0, freq: float = 0.6):
    """Domain warp: returns displaced coordinates for organic-looking terrain."""
    dx = perlin2(x * freq + 5.2, y * freq + 1.3, seed + 7717)
    dy = perlin2(x * freq - 3.1, y * freq + 8.7, seed + 3331)
    return x + strength * dx, y + strength * dy


def worley2(x, y, seed: int = 0, jitter: float = 0.8):
    """Worley / cellular noise.

    Returns ``(f1, cell_id)`` where ``f1`` is the distance to the nearest
    feature point and ``cell_id`` is a stable integer id for that point --
    handy for biome regions and landmark placement.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    best = np.full(x.shape, 1e9, dtype=np.float32)
    best_id = np.zeros(x.shape, dtype=np.int64)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx, cy = xi + ox, yi + oy
            px = cx + jitter * hash_unit(cx, cy, seed)
            py = cy + jitter * hash_unit(cx, cy, seed + 9199)
            d = (px - x) ** 2 + (py - y) ** 2
            upd = d < best
            best = np.where(upd, d, best)
            cid = (cx * np.int64(73856093)) ^ (cy * np.int64(19349663))
            best_id = np.where(upd, cid, best_id)
    return np.sqrt(best), best_id


def smoothstep(a: float, b: float, t):
    """Smooth 0->1 ramp between ``a`` and ``b``.

    ``a > b`` is allowed and gives a descending ramp (1 below ``b``, 0 above
    ``a``), which is how the river and volcanic masks are written.
    """
    denom = float(b) - float(a)
    t = np.asarray(t, dtype=np.float32)
    if abs(denom) < 1e-9:
        return (t >= float(b)).astype(np.float32)
    t = np.clip((t - float(a)) / denom, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def box_blur(a: np.ndarray, radius: int = 1, iterations: int = 1) -> np.ndarray:
    """Separable box blur with edge clamping (used to smooth heightmaps)."""
    out = a.astype(np.float32)
    k = 2 * radius + 1
    for _ in range(iterations):
        pad = np.pad(out, radius, mode="edge")
        acc = np.zeros_like(out)
        for d in range(k):
            acc += pad[d:d + out.shape[0], radius:radius + out.shape[1]]
        out = acc / k
        pad = np.pad(out, radius, mode="edge")
        acc = np.zeros_like(out)
        for d in range(k):
            acc += pad[radius:radius + out.shape[0], d:d + out.shape[1]]
        out = acc / k
    return out
