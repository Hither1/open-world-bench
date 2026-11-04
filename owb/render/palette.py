"""Colour tables, sky/sun model and per-voxel texture hashing.

The renderer is a painterly voxel raycaster rather than a PBR engine: colours
come from the block table, light is one directional sun/moon plus a handful of
point lights, and the atmosphere is a two-stop gradient with height fog.  The
look is deliberately flat and readable -- an agent's policy should be able to
tell grass from stone from water at 96x96.
"""

from __future__ import annotations

import numpy as np

from ..core import blocks as B

# --------------------------------------------------------------------------
# render-time opacity (distinct from physical solidity)
# --------------------------------------------------------------------------
#: 1.0 = stops the ray, <1 = alpha-accumulates and the ray continues.
ALPHA = np.where(B.SOLID, 1.0, 0.0).astype(np.float32)
ALPHA[B.TALL_GRASS] = 0.55
ALPHA[B.FLOWER] = 0.50
ALPHA[B.BUSH] = 0.70
ALPHA[B.VINE] = 0.65
ALPHA[B.FIRE] = 0.55
ALPHA[B.TORCH_WALL] = 0.70
ALPHA[B.CAMPFIRE_LIT] = 0.60
ALPHA[B.SIGNPOST] = 0.80
ALPHA[B.GLASS] = 0.30
ALPHA[B.ICE] = 0.82
ALPHA[B.CRYSTAL] = 0.90
ALPHA[B.PLATE] = 1.0
ALPHA[B.AIR] = 0.0
ALPHA[B.WATER] = 0.0      # volumetric, handled separately
ALPHA[B.LAVA] = 0.0       # volumetric + emissive

#: blocks whose colour is accumulated volumetrically per metre travelled
VOLUMETRIC = np.zeros(B.NUM_BLOCKS, dtype=bool)
VOLUMETRIC[B.WATER] = True
VOLUMETRIC[B.LAVA] = True

COLOR_F = (B.COLOR.astype(np.float32) / 255.0)

WATER_TINT = np.array([0.14, 0.34, 0.50], dtype=np.float32)
WATER_EXTINCTION = 0.22          # per metre
LAVA_TINT = np.array([1.0, 0.42, 0.14], dtype=np.float32)


# --------------------------------------------------------------------------
# sun / sky
# --------------------------------------------------------------------------
def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def sun_state(time_of_day: float) -> dict:
    """Light direction, colour and intensity for a time of day.

    Between dusk and dawn the "sun" becomes a dim blue moon, which is what
    makes torches and luminous crystals matter at night.
    """
    t = float(time_of_day) % 1.0
    ang = (t - 0.25) * 2.0 * np.pi
    elev = float(np.sin(ang))
    # The light direction's elevation is capped well short of vertical: with a
    # true overhead sun every wall in a voxel world gets zero direct light and
    # the scene reads as flat silhouettes.  `elev` keeps the true value, which
    # is what day length and sky colour are driven from.
    dirv = _norm(np.array([np.cos(ang) * 0.92, min(elev, 0.72), 0.42],
                          dtype=np.float32))
    day = max(0.0, elev)
    night = max(0.0, -elev)
    if elev > -0.06:
        # warm near the horizon, white at noon; the band just below zero keeps
        # sunrise and sunset lit rather than snapping to moonlight
        warm = np.array([1.0, 0.72, 0.42], dtype=np.float32)
        white = np.array([1.0, 0.97, 0.90], dtype=np.float32)
        k = min(1.0, day * 2.4)
        color = warm * (1 - k) + white * k
        intensity = 0.22 + 1.08 * day ** 0.55
        ambient = 0.26 + 0.34 * day ** 0.5
    else:
        color = np.array([0.55, 0.66, 1.0], dtype=np.float32)
        moon_dir = _norm(np.array([-np.cos(ang) * 0.92, min(night, 0.70), -0.42],
                                  dtype=np.float32))
        dirv = moon_dir
        intensity = 0.16 + 0.12 * night
        ambient = 0.10 + 0.05 * night
    return dict(dir=dirv.astype(np.float32), color=color.astype(np.float32),
                intensity=float(intensity), ambient=float(ambient),
                elevation=float(elev), is_day=elev > -0.05)


def sky_colors(time_of_day: float, weather: str = "clear") -> dict:
    """Zenith / horizon / sun-glow colours."""
    st = sun_state(time_of_day)
    e = st["elevation"]
    if e > 0.22:
        zenith = np.array([0.28, 0.50, 0.86], dtype=np.float32)
        horizon = np.array([0.69, 0.82, 0.94], dtype=np.float32)
    elif e > -0.02:
        k = (e + 0.02) / 0.24
        zenith = (np.array([0.16, 0.22, 0.47]) * (1 - k)
                  + np.array([0.28, 0.50, 0.86]) * k).astype(np.float32)
        horizon = (np.array([0.96, 0.55, 0.32]) * (1 - k)
                   + np.array([0.69, 0.82, 0.94]) * k).astype(np.float32)
    else:
        night = min(1.0, -e / 0.3)
        zenith = (np.array([0.16, 0.22, 0.47]) * (1 - night)
                  + np.array([0.025, 0.035, 0.09]) * night).astype(np.float32)
        horizon = (np.array([0.96, 0.55, 0.32]) * (1 - night)
                   + np.array([0.06, 0.08, 0.17]) * night).astype(np.float32)
    glow = st["color"] * (0.8 + 0.6 * max(e, 0.0))
    if weather in ("rain", "storm"):
        grey = np.array([0.42, 0.45, 0.50], dtype=np.float32) * (0.35 + 0.65 * max(e, 0.0) + 0.08)
        zenith = 0.35 * zenith + 0.65 * grey
        horizon = 0.40 * horizon + 0.60 * grey
        glow *= 0.4
    elif weather == "snow":
        pale = np.array([0.72, 0.76, 0.82], dtype=np.float32) * (0.35 + 0.65 * max(e, 0.0) + 0.1)
        zenith = 0.45 * zenith + 0.55 * pale
        horizon = 0.35 * horizon + 0.65 * pale
        glow *= 0.5
    elif weather == "fog":
        pale = np.array([0.66, 0.68, 0.70], dtype=np.float32) * (0.3 + 0.7 * max(e, 0.0) + 0.1)
        zenith = 0.4 * zenith + 0.6 * pale
        horizon = 0.25 * horizon + 0.75 * pale
        glow *= 0.5
    return dict(zenith=zenith, horizon=horizon, glow=glow.astype(np.float32),
                sun=st)


FOG_DISTANCE = {"clear": 150.0, "cloud": 120.0, "rain": 62.0, "storm": 48.0,
                "snow": 55.0, "fog": 26.0}


# --------------------------------------------------------------------------
# per-voxel texture hash
# --------------------------------------------------------------------------
def voxel_hash(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
    """Stable per-voxel value in [-1, 1] -- cheap stand-in for a texture."""
    h = (vx.astype(np.uint32) * np.uint32(0x9E3779B1)) ^ \
        (vy.astype(np.uint32) * np.uint32(0x85EBCA77)) ^ \
        (vz.astype(np.uint32) * np.uint32(0xC2B2AE3D))
    h ^= h >> np.uint32(15)
    h = (h * np.uint32(0x27D4EB2F)) & np.uint32(0xFFFFFFFF)
    h ^= h >> np.uint32(13)
    return (h & np.uint32(0xFFFF)).astype(np.float32) * (2.0 / 65535.0) - 1.0


#: minimap colour per block, used by the top-down map renderer
def map_color_table() -> np.ndarray:
    tab = B.COLOR.astype(np.float32).copy()
    tab[B.WATER] = (60, 118, 186)
    tab[B.AIR] = (20, 22, 26)
    return tab.astype(np.uint8)
