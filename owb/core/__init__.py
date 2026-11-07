"""Simulator-agnostic helpers shared by every OWB layer.

``owb.core`` used to alias the voxel simulator's ``core`` package.  Those old
submodule paths (``owb.core.blocks``, ``owb.core.config``, ``owb.core.items``,
``owb.core.noise``, ``owb.core.rng``) still resolve to ``owb.sims.voxel.core.*``
with a DeprecationWarning through ``owb._compat``; the names defined here are new.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

#: Version of the shared common-schema fields (``move``/``look``/``proprio``/``instruction`` ...).
SCHEMA_VERSION = 2

#: deprecated submodules that ``owb.core`` aliased before it became a real package
_LEGACY_SUBMODULES = ("blocks", "config", "items", "noise", "rng")

__all__ = ["SCHEMA_VERSION", "event_type", "jsonable"]


def jsonable(value):
    """Detach simulator values and convert them to JSON-compatible data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return deepcopy(value)


def event_type(event) -> str:
    """Type name of an event: ``.type`` of a Teyvat ``Event`` or ``["type"]`` of a dict record."""
    if isinstance(event, dict):
        return event["type"]
    return event.type


def __getattr__(name):
    # ``import owb.core; owb.core.config`` worked while owb.core aliased the voxel core package.
    if name in _LEGACY_SUBMODULES:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
