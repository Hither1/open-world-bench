"""State machines of the Teyvat reference courses, one module per mechanism family.

``MECHANISMS`` maps an operation kind to its (stateless) mechanism object; the course
runtime (:mod:`owb.tasks.teyvat.course`) forwards native interactions, elemental
applications, hits and ticks to the current operation's mechanism.
"""
from __future__ import annotations

from .base import Mechanism

MECHANISMS: dict[str, Mechanism] = {}


def register(cls):
    MECHANISMS[cls.kind] = cls()
    return cls


def mechanism(kind: str) -> Mechanism:
    return MECHANISMS[kind]


from . import relay  # noqa: E402,F401

__all__ = ["MECHANISMS", "Mechanism", "mechanism", "register"]
