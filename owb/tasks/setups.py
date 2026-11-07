"""Setup (placement) builders, registered by the name a ``TaskSpec.setup`` refers to.

A builder has the signature ``setup(world_hooks, rng, params) -> SetupResult``. It places
entities and edits the level through the sim-specific ``world_hooks``, draws every seeded
assignment from ``rng`` (in a fixed order, so the same seed reproduces the same course),
and reports what it built. ``SetupResult.record`` is the JSON-ready summary of the sampled
assignments, meant for run manifests.

Only the reference courses register builders so far (``teyvat.reference_course``,
``wildlands.reference_course``); ``TaskEnv`` does not run setups yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

SETUPS: dict[str, Callable[..., "SetupResult"]] = {}


@dataclass
class SetupResult:
    #: the placed task structure (sim-specific, e.g. the course's milestone objects)
    milestones: list
    #: seeded rule bindings the mechanisms consult (sim-specific objects)
    rules: dict = field(default_factory=dict)
    #: JSON-ready record of the sampled assignments and the layout
    record: dict = field(default_factory=dict)
    #: anything else the hosting env needs (sim-specific)
    extras: dict = field(default_factory=dict)


def register_setup(name: str):
    def wrap(fn):
        if name in SETUPS:
            raise ValueError(f"setup {name!r} is already registered")
        SETUPS[name] = fn
        return fn
    return wrap


def get_setup(name: str):
    if name not in SETUPS:
        # builders register on import of their sim's setup module
        import importlib
        sim = name.split(".", 1)[0]
        try:
            importlib.import_module(f"owb.tasks.{sim}.setup")
        except ModuleNotFoundError:
            pass
    try:
        return SETUPS[name]
    except KeyError:
        raise KeyError(f"unknown setup {name!r}; registered: {sorted(SETUPS)}") from None


def run_setup(name: str, world_hooks: Any, rng, params: dict | None = None) -> SetupResult:
    return get_setup(name)(world_hooks, rng, dict(params or {}))


__all__ = ["SETUPS", "SetupResult", "get_setup", "register_setup", "run_setup"]
