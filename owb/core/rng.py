"""Seeded randomness helpers.

The simulator keeps one :class:`numpy.random.Generator` so that an episode is
bit-reproducible from ``(world_seed, episode_seed)``.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(int(seed) & 0x7FFFFFFF))


def derive(seed: int, *salt: int) -> int:
    """Derive a stable child seed from a parent seed and integer salts."""
    mask = (1 << 64) - 1
    h = (int(seed) ^ 0x9E3779B97F4A7C15) & mask
    for s in salt:
        h = (h ^ (int(s) & mask)) & mask
        h = (h * 0xBF58476D1CE4E5B9) & mask
        h ^= h >> 27
    return h & 0x7FFFFFFF


def choice_weighted(rng: np.random.Generator, options, weights) -> object:
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total <= 0:
        return options[int(rng.integers(len(options)))]
    return options[int(rng.choice(len(options), p=w / total))]
