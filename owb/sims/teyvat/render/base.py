"""Renderer interface shared by the numpy and OpenGL backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .style import ToonStyle


@dataclass
class RenderSettings:
    width: int = 192
    height: int = 144
    march_steps: int = 112          # terrain ray steps (numpy backend)
    refine_steps: int = 6
    max_view_distance: float = 620.0
    shadow_steps: int = 10          # 0 disables sun shadows
    draw_props: bool = True
    ground_cover: bool = True      # decorative grass and meadow flowers
    prop_draw_distance: float = 240.0
    draw_entities: bool = True
    draw_character: bool = True
    outlines: bool = True
    clouds: bool = True
    water: bool = True
    hud: bool = False               # overlay quest text / bars (videos, not obs)
    supersample: int = 1            # render at NxN then box-filter down
    style: ToonStyle = field(default_factory=ToonStyle)

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1)


@dataclass
class Frame:
    """One rendered frame plus the buffers the post passes need."""

    color: np.ndarray                      # (H, W, 3) float32 in [0, 1]
    depth: np.ndarray                      # (H, W) float32 metres, inf for sky
    normal: np.ndarray                     # (H, W, 3) float32
    ids: np.ndarray | None = None          # (H, W) int32 object ids, for ink lines

    def to_uint8(self) -> np.ndarray:
        return (np.clip(self.color, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


class Renderer(ABC):
    """Anything that can turn (world, character, camera) into pixels."""

    def __init__(self, settings: RenderSettings | None = None):
        self.settings = settings or RenderSettings()

    @abstractmethod
    def render(self, world, char, cam) -> np.ndarray:
        """Return an (H, W, 3) uint8 image."""

    def close(self) -> None:  # pragma: no cover - backends override if needed
        pass

    @property
    def name(self) -> str:
        return type(self).__name__
