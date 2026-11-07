"""State machines of the Wildlands reference courses, one module per mechanism family.

``MECHANISMS`` maps a milestone kind (balance, thermal, flow/cool/prepare, calibrate,
station, exchange, clue, cook, report) to its stateless mechanism object;
:mod:`.commands` implements the course commands (inspect, operate, grip, drag, drop,
install) that route to them.
"""
from __future__ import annotations

from .base import Mechanism, fail

MECHANISMS: dict[str, Mechanism] = {}


def register(cls):
    instance = cls()
    for kind in cls.kinds:
        MECHANISMS[kind] = instance
    return cls


def mechanism(kind: str) -> Mechanism:
    return MECHANISMS[kind]


from . import counterweight, flow, stations, thermal  # noqa: E402,F401


__all__ = ["MECHANISMS", "Mechanism", "fail", "mechanism", "register"]
