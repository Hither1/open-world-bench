"""Submerged recovery: the chest only opens to a swimming character below -2.8 m."""
from __future__ import annotations

from owb.sims.teyvat.character import MoveState
from owb.sims.teyvat.world import EntityKind

from . import register
from .stations import Station


@register
class Dive(Station):
    kind = "dive"

    def place(self, hooks, op, p, rng, rules):
        hooks.flat_patch(p, -4.5, radius=5.0)
        hooks.spawn(op, "chest", EntityKind.CHEST, p, label=op.title)

    def ready(self, course, op, char):
        if char.pos[1] > -2.8 or char.state != MoveState.SWIMMING:
            course.feedback("Dive to the chest; surface interaction cannot recover it.")
            return False
        return True
