"""Course data of the Wildlands reference courses A/B: the milestone record.

``Milestone.state`` is the mechanism state that ``state_digest`` serialises;
``targets`` maps control role -> target id in ``env.targets``; ``prereqs`` is the
assessment DAG (``env.eligible``); ``consumes``/``gives`` are the repair-satchel
transfers. ``requires`` is never set or read (dead field kept for compatibility).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Milestone:
    id: str
    title: str
    kind: str
    site: tuple
    requires: tuple = ()
    gives: tuple = ()
    consumes: tuple = ()
    state: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)
    prereqs: tuple = ()


__all__ = ["Milestone"]
