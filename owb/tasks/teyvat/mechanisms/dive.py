"""Submerged recovery: the chest only opens to a swimming character below -2.8 m."""
from __future__ import annotations

from owb.sims.teyvat.character import MoveState

from . import register
from .stations import Station


@register
class Dive(Station):
    kind = "dive"

    def ready(self, course, op, char):
        if char.pos[1] > -2.8 or char.state != MoveState.SWIMMING:
            course.feedback("Dive to the chest; surface interaction cannot recover it.")
            return False
        return True
