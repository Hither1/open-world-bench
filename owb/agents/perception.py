"""Observation -> model input parts. Reads only the common observation schema.

Field descriptions come from the ``owb.spaces`` modality declarations; this module
imports no simulator vocabulary.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import json

import numpy as np

from .. import spaces as owb_spaces
from ..bench.framework import PROPRIO_NAMES
from ..core import jsonable

TEXT = "text"
IMAGE_PNG_BASE64 = "image_png_base64"
PART_KINDS = (TEXT, IMAGE_PNG_BASE64)


@dataclass(frozen=True)
class Part:
    """One model input part: UTF-8 text, or a base64-encoded PNG (no data-URL prefix)."""
    kind: str
    data: str

    def __post_init__(self):
        if self.kind not in PART_KINDS:
            raise ValueError(f"unsupported part kind: {self.kind}")
        if not isinstance(self.data, str):
            raise TypeError("part data must be a string")


def _named(names, values):
    return dict(zip(names, jsonable(values)))


#: Description order of the schema-v2 modalities (fixes the prompt's JSON key order);
#: modalities registered later follow in registration order.
_DESCRIBE_ORDER = ("inventory", "survival", "construction", "build_state", "party", "minimap", "goal")


def _describe_order(registry):
    names = registry.modality_names()
    first = [n for n in _DESCRIBE_ORDER if n in names]
    return [registry.modality(n) for n in first + [n for n in names if n not in _DESCRIBE_ORDER]]


def describe_observation(observation):
    """Compact, named sensor data; never accepts simulator info or an env handle."""
    registry = owb_spaces.REGISTRY
    result = {
        "instruction": observation["instruction"],
        "proprio": _named(PROPRIO_NAMES, observation["proprio"]),
        "available_commands": [n for n, flag in zip(
            registry.control("command").vocab, observation["command_mask"]) if flag],
        "available_buttons": [n for n, flag in zip(
            registry.control("buttons").vocab, observation["button_mask"]) if flag],
        "available_swap_slots": np.flatnonzero(observation["swap_mask"]).tolist(),
    }
    enabled = {name for name, flag in zip(registry.modality_names(), observation["modality_mask"]) if flag}
    for modality in _describe_order(registry):
        if modality.name in enabled and modality.describe is not None:
            result.update(modality.describe(observation[modality.name], observation))
    return result


class Perception:
    """Builds ``[text, image]`` parts: JSON sensors/memory/history, then the RGB frame as PNG.

    Pillow is imported on construction (optional ``[agent]`` dependency).
    """
    def __init__(self):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError('LLM policies require pip install -e ".[agent]"') from exc
        self._image = Image

    def describe(self, observation):
        return describe_observation(observation)

    def encode_png(self, rgb):
        buffer = io.BytesIO()
        self._image.fromarray(rgb).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def parts(self, observation, *, step, decision_seconds, memory, recent_history, sensors=None):
        sensors = self.describe(observation) if sensors is None else sensors
        prompt = dict(step=step, decision_seconds=decision_seconds,
                      observation=sensors, memory=memory, recent_history=recent_history)
        return [Part(TEXT, json.dumps(prompt, allow_nan=False)),
                Part(IMAGE_PNG_BASE64, self.encode_png(observation["rgb"]))]
