"""Observation -> model input parts. Reads only the common observation schema.

This is the only module in ``owb.agents`` that imports simulator vocabularies.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import json

import numpy as np

from ..bench.framework import BUTTONS, COMMANDS, MODALITIES, PROPRIO_NAMES
from ..core import jsonable
from ..sims.voxel.env import INVENTORY_COLUMNS, ITEM_IDS, SURVIVAL_NAMES
from ..sims.voxel.sim.construction import BUILD_STATE_NAMES, PART_COLUMNS

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


def describe_observation(observation):
    """Compact, named sensor data; never accepts simulator info or an env handle."""
    result = {
        "instruction": observation["instruction"],
        "proprio": _named(PROPRIO_NAMES, observation["proprio"]),
        "available_commands": [n for n, flag in zip(COMMANDS, observation["command_mask"]) if flag],
        "available_buttons": [n for n, flag in zip(BUTTONS, observation["button_mask"]) if flag],
        "available_swap_slots": np.flatnonzero(observation["swap_mask"]).tolist(),
    }
    enabled = {name for name, flag in zip(MODALITIES, observation["modality_mask"]) if flag}
    if "inventory" in enabled:
        result["inventory"] = []
        for slot, row in enumerate(observation["inventory"]):
            if row[0] > 0 and row[1] > 0:
                entry = dict(slot=slot, **_named(INVENTORY_COLUMNS, row))
                entry["item_name"] = ITEM_IDS[int(row[0]) - 1]
                result["inventory"].append(entry)
    if "survival" in enabled:
        result["survival"] = _named(SURVIVAL_NAMES, observation["survival"])
    if "construction" in enabled:
        result["construction"] = [dict(part=i + 1, **_named(PART_COLUMNS, row))
                                  for i, row in enumerate(observation["construction"]) if row[0]]
    if "build_state" in enabled:
        result["build_state"] = _named(BUILD_STATE_NAMES, observation["build_state"])
    if "party" in enabled:
        names = ("active", "health_fraction", "energy_fraction", "skill_ready",
                 "burst_cooldown_ready", "element_id_div_6", "alive")
        party = observation["party"]
        result["party"] = [dict(slot=i + 1, **_named(names, party[i * 7:(i + 1) * 7]))
                           for i in range(4) if observation["swap_mask"][i + 1]]
        result["party_status"] = _named(("shield_fraction", "swap_ready"), party[28:])
    if "minimap" in enabled:
        result["visited_map"] = {"shape": list(observation["minimap"].shape),
                                 "nonzero_cells": np.argwhere(observation["minimap"] > 0).tolist()}
    if "goal" in enabled:
        result["oracle_goal_hint"] = jsonable(observation["goal"])
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
