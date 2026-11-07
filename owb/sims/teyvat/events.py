"""World events, the single channel between the simulation and the quest logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """Something the world reported during one step."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def __repr__(self) -> str:  # compact, for rollout logs
        inner = ", ".join(f"{k}={v!r}" for k, v in self.payload.items() if k != "pos")
        return f"Event({self.type}{', ' + inner if inner else ''})"
