"""Declarative observation modalities and action controls of the common schema.

A backend adapter declares the ``Modality`` objects it can supply and any extra
``Control`` objects it understands; ``BenchmarkEnv`` assembles its Dict spaces from
the shared fields plus the union of everything registered, in registration order
(append-only, so the index of an existing modality in ``modality_mask`` never
moves). ``describe_observation`` and the recorder's ``config.json`` vocabularies
are generated from the same declarations.

The built-in declarations live in :mod:`owb.spaces.builtin` and are registered
when :mod:`owb.bench.framework` is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import gymnasium as gym

__all__ = ["Control", "Modality", "REGISTRY", "SpaceRegistry", "controls", "modalities",
           "modality_names", "register_control", "register_modality"]

#: keys of config.json["schema"] that belong to the shared fields, not to a modality
SHARED_VOCAB_KEYS = ("buttons", "commands", "modalities", "proprio")


@dataclass(frozen=True)
class Modality:
    """One optional observation field.

    ``describe(value, observation) -> dict`` returns named, JSON-compatible entries
    merged into ``describe_observation`` (``None``: not described to models).
    ``vocab`` maps config.json schema keys to the field's column/item names.
    ``version`` is the modality's own version; ``SCHEMA_VERSION`` only covers the
    shared fields.
    """
    name: str
    space: gym.Space
    describe: Callable[[Any, Mapping], dict] | None
    vocab: Mapping[str, tuple] | None = None
    version: int = 1


@dataclass(frozen=True)
class Control:
    """One field of the common action dictionary; ``vocab`` names discrete/binary entries."""
    name: str
    space: gym.Space
    doc: str
    vocab: tuple[str, ...] | None = None


class SpaceRegistry:
    """Ordered, append-only registry. Re-registering an equal declaration is a no-op."""

    def __init__(self):
        self._modalities: dict[str, Modality] = {}
        self._controls: dict[str, Control] = {}

    def copy(self) -> "SpaceRegistry":
        other = SpaceRegistry()
        other._modalities, other._controls = dict(self._modalities), dict(self._controls)
        return other

    # -- validation without mutation, so a backend registers all-or-nothing
    def check_modality(self, modality: Modality) -> bool:
        """Return True if ``modality`` is new, False if already registered; raise on a conflict."""
        if not isinstance(modality, Modality):
            raise TypeError("expected an owb.spaces.Modality")
        existing = self._modalities.get(modality.name)
        if existing is not None:
            if existing is not modality and existing != modality:
                raise ValueError(f"modality {modality.name!r} is already registered with a different declaration")
            return False
        if modality.name in self._controls or modality.name in ("rgb", "proprio", "instruction") \
                or modality.name.endswith("_mask"):
            raise ValueError(f"modality name {modality.name!r} collides with a shared field")
        taken = set(SHARED_VOCAB_KEYS).union(*(m.vocab or {} for m in self._modalities.values()))
        clash = taken.intersection(modality.vocab or {})
        if clash:
            raise ValueError(f"modality {modality.name!r} vocabulary keys {sorted(clash)} are already used")
        return True

    def check_control(self, control: Control) -> bool:
        if not isinstance(control, Control):
            raise TypeError("expected an owb.spaces.Control")
        existing = self._controls.get(control.name)
        if existing is not None:
            if existing is not control and existing != control:
                raise ValueError(f"control {control.name!r} is already registered with a different declaration")
            return False
        return True

    def register_modality(self, modality: Modality) -> None:
        if self.check_modality(modality):
            self._modalities[modality.name] = modality

    def register_control(self, control: Control) -> None:
        if self.check_control(control):
            self._controls[control.name] = control

    def modalities(self) -> tuple[Modality, ...]:
        return tuple(self._modalities.values())

    def controls(self) -> tuple[Control, ...]:
        return tuple(self._controls.values())

    def modality_names(self) -> tuple[str, ...]:
        return tuple(self._modalities)

    def modality(self, name: str) -> Modality:
        return self._modalities[name]

    def control(self, name: str) -> Control:
        return self._controls[name]


#: the process-wide registry used by ``BenchmarkEnv`` and ``describe_observation``
REGISTRY = SpaceRegistry()


def register_modality(modality: Modality) -> None:
    REGISTRY.register_modality(modality)


def register_control(control: Control) -> None:
    REGISTRY.register_control(control)


def modalities() -> tuple[Modality, ...]:
    return REGISTRY.modalities()


def controls() -> tuple[Control, ...]:
    return REGISTRY.controls()


def modality_names() -> tuple[str, ...]:
    return REGISTRY.modality_names()
