"""Deterministic, fully vectorized gradient noise.

Everything here is seeded integer hashing, so a given seed reproduces the exact
same world on any machine and any numpy version -- important for a benchmark.
"""
from __future__ import annotations

import numpy as np

_M1 = np.int64(374761393)
_M2 = np.int64(668265263)
_M3 = np.int64(1274126177)


def _hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Hash a 2D integer lattice to uint32-ish values in [0, 1)."""
    h = ix.astype(np.int64) * _M1 + iy.astype(np.int64) * _M2 + np.int64(seed) * np.int64(1013904223)
    h = (h ^ (h >> np.int64(13))) * _M3
    h = h ^ (h >> np.int64(16))
    return (h & np.int64(0xFFFFFF)).astype(np.float32) / np.float32(0xFFFFFF)


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def gradient_noise(x: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Perlin-style gradient noise in roughly [-1, 1]. Shape-preserving."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    fx = x - ix
    fy = y - iy

    def dot_grad(cx, cy, dx, dy):
        ang = _hash2(cx, cy, seed) * np.float32(2.0 * np.pi)
        return np.cos(ang) * dx + np.sin(ang) * dy

    n00 = dot_grad(ix, iy, fx, fy)
    n10 = dot_grad(ix + 1, iy, fx - 1.0, fy)
    n01 = dot_grad(ix, iy + 1, fx, fy - 1.0)
    n11 = dot_grad(ix + 1, iy + 1, fx - 1.0, fy - 1.0)

    ux, uy = _fade(fx), _fade(fy)
    nx0 = n00 + (n10 - n00) * ux
    nx1 = n01 + (n11 - n01) * ux
    return (nx0 + (nx1 - nx0) * uy) * np.float32(1.4142135)


def fbm(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    octaves: int = 5,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Fractional Brownian motion; output normalized to about [-1, 1]."""
    total = np.zeros(np.broadcast(x, y).shape, dtype=np.float32)
    amp, freq, norm = 1.0, 1.0, 0.0
    for o in range(octaves):
        total += amp * gradient_noise(x * freq, y * freq, seed + o * 1013)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / max(norm, 1e-6)


def ridged(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    octaves: int = 5,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Ridged multifractal in [0, 1] -- the source of sharp anime-style crests."""
    total = np.zeros(np.broadcast(x, y).shape, dtype=np.float32)
    amp, freq, norm = 1.0, 1.0, 0.0
    for o in range(octaves):
        n = 1.0 - np.abs(gradient_noise(x * freq, y * freq, seed + o * 7919))
        total += amp * (n * n)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / max(norm, 1e-6)


def domain_warp(x: np.ndarray, y: np.ndarray, seed: int, strength: float, scale: float):
    """Offset sample coordinates by a noise field: gives terrain flowing valleys."""
    wx = fbm(x * scale, y * scale, seed + 5501, octaves=3)
    wy = fbm(x * scale, y * scale, seed + 9013, octaves=3)
    return x + wx * strength, y + wy * strength


def hash_scalar(*args: float, seed: int = 0) -> float:
    """Cheap deterministic scalar hash for per-entity jitter. Returns [0, 1)."""
    mask = 0xFFFFFFFF
    acc = (int(seed) * 2654435761) & mask
    for a in args:
        acc = (acc ^ (int(a * 1024.0) & mask)) & mask
        acc = (acc * 1274126177) & mask
        acc ^= acc >> 15
    return (acc & 0xFFFFFF) / float(0x1000000)
