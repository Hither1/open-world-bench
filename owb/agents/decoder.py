"""Strict JSON action schema and decoder for the common action space."""
from __future__ import annotations

import math

import numpy as np

from ..bench.framework import BUTTONS, COMMANDS, neutral_action


def action_schema(observation, memory_chars):
    def vector(size, kind="number", low=-1, high=1):
        return dict(type="array", minItems=size, maxItems=size,
                    items=dict(type=kind, minimum=low, maximum=high))

    available_buttons = [n for n, flag in zip(BUTTONS, observation["button_mask"]) if flag]
    properties = {
        "move": vector(2), "look": vector(2),
        "buttons": {"type": "array", "maxItems": len(available_buttons),
                    "items": {"type": "string", "enum": available_buttons or list(BUTTONS)}},
        "command": {"type": "string", "enum": [n for n, flag in zip(
            COMMANDS, observation["command_mask"]) if flag]},
        "swap": {"type": "integer", "enum": np.flatnonzero(observation["swap_mask"]).tolist()},
        "slot": {"type": "integer", "minimum": 0, "maximum": 31},
        "ingredients": vector(5, "integer", 0, 32),
        "part": {"type": "integer", "minimum": 0, "maximum": 8},
        "build_move": vector(3),
        "memory": {"type": "string", "maxLength": memory_chars},
    }
    return dict(type="object", properties=properties, required=list(properties), additionalProperties=False)


def decode_action(payload, observation, memory_chars):
    """Reject malformed/unsupported model controls before the simulator advances."""
    schema = action_schema(observation, memory_chars)
    if not isinstance(payload, dict) or set(payload) != set(schema["required"]):
        raise ValueError("model output must contain exactly the action fields and memory")
    action = neutral_action()
    for key in ("move", "look", "build_move", "ingredients"):
        spec, value = schema["properties"][key], payload[key]
        if not isinstance(value, list) or len(value) != spec["minItems"]:
            raise ValueError(f"invalid {key} shape")
        item = spec["items"]
        kinds = (int,) if key == "ingredients" else (int, float)
        if any(type(v) not in kinds or not math.isfinite(v)
               or not item["minimum"] <= v <= item["maximum"] for v in value):
            raise ValueError(f"invalid {key} values")
        action[key] = np.asarray(value, dtype=action[key].dtype)
    for key in ("slot", "part", "swap"):
        value, spec = payload[key], schema["properties"][key]
        if type(value) is not int or (value not in spec["enum"] if "enum" in spec
                                     else not spec["minimum"] <= value <= spec["maximum"]):
            raise ValueError(f"invalid {key}")
        action[key] = value
    if payload["command"] not in schema["properties"]["command"]["enum"]:
        raise ValueError("unsupported command")
    action["command"] = COMMANDS.index(payload["command"])
    buttons = payload["buttons"]
    if not isinstance(buttons, list) or len(buttons) > len(BUTTONS):
        raise ValueError("invalid buttons")
    for button in buttons:
        if button not in BUTTONS or not observation["button_mask"][BUTTONS.index(button)]:
            raise ValueError("unsupported button")
        action["buttons"][BUTTONS.index(button)] = 1
    if not isinstance(payload["memory"], str) or len(payload["memory"]) > memory_chars:
        raise ValueError("invalid memory")
    return action


class ActionDecoder:
    """Observation-masked schema plus validation; raises ValueError on any bad field."""
    def __init__(self, memory_chars):
        if type(memory_chars) is not int or memory_chars <= 0:
            raise ValueError("memory_chars must be a positive integer")
        self.memory_chars = memory_chars

    def schema(self, observation):
        return action_schema(observation, self.memory_chars)

    def decode(self, payload, observation):
        return decode_action(payload, observation, self.memory_chars)
