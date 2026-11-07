"""Art direction: the palette, the toon ramp, the sky and the ink lines.

Every colour the renderers use lives here, so the OpenGL path and the numpy
path stay visually consistent. Values are linear-ish sRGB in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..terrain import BEACH, CLIFF, FOREST, HIGHLAND, MEADOW, OCEAN, SNOW

# Base albedo per biome, in a bright anime palette: saturated greens, warm
# sand, lilac-grey rock, near-white snow.
BIOME_ALBEDO = np.zeros((7, 3), dtype=np.float32)
BIOME_ALBEDO[OCEAN] = (0.17, 0.42, 0.56)
BIOME_ALBEDO[BEACH] = (0.90, 0.84, 0.66)
BIOME_ALBEDO[MEADOW] = (0.52, 0.72, 0.32)
BIOME_ALBEDO[FOREST] = (0.34, 0.61, 0.34)
BIOME_ALBEDO[HIGHLAND] = (0.56, 0.60, 0.47)
BIOME_ALBEDO[CLIFF] = (0.52, 0.49, 0.53)
BIOME_ALBEDO[SNOW] = (0.93, 0.95, 0.98)

# Second, cooler tone each biome shifts toward in shadow.
BIOME_SHADOW = np.zeros((7, 3), dtype=np.float32)
BIOME_SHADOW[OCEAN] = (0.08, 0.22, 0.38)
BIOME_SHADOW[BEACH] = (0.55, 0.52, 0.52)
BIOME_SHADOW[MEADOW] = (0.20, 0.38, 0.34)
BIOME_SHADOW[FOREST] = (0.12, 0.28, 0.27)
BIOME_SHADOW[HIGHLAND] = (0.31, 0.34, 0.37)
BIOME_SHADOW[CLIFF] = (0.27, 0.26, 0.34)
BIOME_SHADOW[SNOW] = (0.60, 0.68, 0.86)

PROP_ALBEDO = {
    "pine": (0.22, 0.48, 0.36), "broadleaf": (0.40, 0.65, 0.30), "sakura": (0.98, 0.72, 0.78),
    "bamboo": (0.52, 0.72, 0.33), "rock": (0.50, 0.48, 0.52), "boulder": (0.44, 0.43, 0.48),
    "bush": (0.30, 0.54, 0.30), "grass": (0.50, 0.74, 0.36), "crystal": (0.55, 0.82, 0.95),
    "trunk": (0.43, 0.34, 0.27),
}

ENTITY_ALBEDO = {
    "npc": (0.86, 0.46, 0.42), "chest": (0.74, 0.55, 0.26), "beacon": (0.45, 0.78, 0.92),
    "pillar": (0.62, 0.60, 0.68), "collectible": (0.70, 0.84, 0.98), "enemy": (0.58, 0.34, 0.52),
    "boss": (0.40, 0.22, 0.46), "altar": (0.82, 0.78, 0.60), "waypoint": (0.52, 0.86, 0.80),
}
# Emissive colour once an entity has been lit / opened / picked.
ENTITY_EMISSIVE = {
    "beacon": (1.0, 0.86, 0.45), "pillar": (0.65, 0.92, 1.0), "waypoint": (0.75, 1.0, 0.95),
    "altar": (1.0, 0.93, 0.70), "collectible": (0.85, 0.95, 1.0),
}

CHARACTER_PALETTE = {
    "hair": (0.96, 0.84, 0.52),
    "skin": (0.99, 0.87, 0.78),
    "tunic": (0.33, 0.52, 0.86),
    "trim": (0.95, 0.95, 0.98),
    "boots": (0.30, 0.26, 0.34),
    "cape": (0.86, 0.36, 0.38),
    "glider": (0.95, 0.92, 0.86),
}


@dataclass
class SkyKeys:
    """Sky gradient control points for one time of day."""

    zenith: tuple[float, float, float]
    horizon: tuple[float, float, float]
    sun: tuple[float, float, float]
    fog: tuple[float, float, float]
    ambient: float
    sun_strength: float


# Keyframes at midnight, dawn, midday, dusk. Interpolated by time of day.
SKY_KEYS: dict[float, SkyKeys] = {
    0.00: SkyKeys((0.03, 0.05, 0.14), (0.09, 0.11, 0.24), (0.55, 0.60, 0.80), (0.07, 0.09, 0.19), 0.30, 0.18),
    0.24: SkyKeys((0.33, 0.47, 0.78), (0.99, 0.72, 0.55), (1.00, 0.84, 0.58), (0.84, 0.72, 0.68), 0.46, 0.85),
    0.32: SkyKeys((0.25, 0.59, 0.89), (0.78, 0.89, 0.94), (1.00, 0.95, 0.82), (0.70, 0.83, 0.85), 0.64, 1.00),
    0.50: SkyKeys((0.27, 0.56, 0.92), (0.74, 0.88, 0.98), (1.00, 0.98, 0.90), (0.76, 0.86, 0.95), 0.55, 1.00),
    0.76: SkyKeys((0.26, 0.31, 0.63), (1.00, 0.60, 0.42), (1.00, 0.74, 0.46), (0.80, 0.60, 0.56), 0.44, 0.75),
    1.00: SkyKeys((0.03, 0.05, 0.14), (0.09, 0.11, 0.24), (0.55, 0.60, 0.80), (0.07, 0.09, 0.19), 0.30, 0.18),
}


def sky_keys_at(time_of_day: float) -> SkyKeys:
    """Linearly blend the sky keyframes at `time_of_day` in [0, 1)."""
    t = float(time_of_day) % 1.0
    stops = sorted(SKY_KEYS)
    for lo, hi in zip(stops, stops[1:]):
        if lo <= t <= hi:
            a, b = SKY_KEYS[lo], SKY_KEYS[hi]
            w = (t - lo) / max(hi - lo, 1e-6)

            def mix(x, y):
                return tuple(float(xi + (yi - xi) * w) for xi, yi in zip(x, y))

            return SkyKeys(
                zenith=mix(a.zenith, b.zenith), horizon=mix(a.horizon, b.horizon),
                sun=mix(a.sun, b.sun), fog=mix(a.fog, b.fog),
                ambient=float(a.ambient + (b.ambient - a.ambient) * w),
                sun_strength=float(a.sun_strength + (b.sun_strength - a.sun_strength) * w),
            )
    return SKY_KEYS[0.0]


@dataclass
class ToonStyle:
    """Parameters of the cel-shading ramp and the ink pass."""

    bands: int = 3                    # number of lighting steps
    band_floor: float = 0.58          # lifted, cool shadows
    band_softness: float = 0.10       # softer painted light transitions
    rim_strength: float = 0.18
    rim_power: float = 2.6
    specular_strength: float = 0.25
    sky_bounce: float = 0.22          # cool light from the sky on upward faces
    outline_strength: float = 0.24
    outline_depth_threshold: float = 0.022
    outline_normal_threshold: float = 0.45
    saturation: float = 1.04
    contrast: float = 0.98
    exposure: float = 1.08
    fog_start: float = 65.0
    fog_end: float = 620.0
    fog_max: float = 0.88
    fog_height_falloff: float = 0.0032
    vignette: float = 0.045
    water_shallow: tuple[float, float, float] = (0.35, 0.78, 0.80)
    water_deep: tuple[float, float, float] = (0.06, 0.26, 0.45)
    water_foam: tuple[float, float, float] = (0.95, 0.99, 1.0)
    cloud_cover: float = 0.48
    cloud_sharpness: float = 1.8
    extras: dict = field(default_factory=dict)


def toon_ramp(ndotl: np.ndarray, style: ToonStyle) -> np.ndarray:
    """Quantize diffuse lighting into `bands` soft steps."""
    x = np.clip(ndotl, 0.0, 1.0)
    bands = max(1, int(style.bands))
    scaled = x * bands
    idx = np.floor(scaled)
    frac = scaled - idx
    # Smooth only right at a band edge, so the look stays crisp but not aliased.
    soft = np.clip((frac - (1.0 - style.band_softness * bands)) / max(style.band_softness * bands, 1e-6), 0.0, 1.0)
    level = (idx + soft) / bands
    return (style.band_floor + (1.0 - style.band_floor) * level).astype(np.float32)


def grade(rgb: np.ndarray, style: ToonStyle) -> np.ndarray:
    """Final colour grade: exposure, saturation, gentle contrast, clamp."""
    out = np.asarray(rgb, dtype=np.float32) * style.exposure
    lum = out @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    out = lum[..., None] + (out - lum[..., None]) * style.saturation
    out = np.clip((out - 0.5) * style.contrast + 0.5, 0.0, 1.0)
    return out
