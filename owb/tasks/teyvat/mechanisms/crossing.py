"""Vent-assisted crossing (procedure T): two charged updraft legs across three shelves.

Charging the current leg's vent with the course's vent element arms it; jumping off the
vent during the safe bright pulse (1.0-3.5 s of every 4 s) launches the character
upward; a landing on the next shelf counts the leg (and whether it was glided). A ground
contact below the next shelf disarms the leg (``crossing_retry``). The second landing
completes the operation.
"""
from __future__ import annotations

import numpy as np

from owb.sims.teyvat.character import MoveState

from . import register
from .base import Mechanism


@register
class Crossing(Mechanism):
    kind = "crossing"
    accepts_elements = True

    def target(self, course, op):
        k = op.state["leg"] + int(op.state["launched"])
        return course.world.get(op.entities[min(k, 2)])

    def apply_element(self, course, op, char, element, entity):
        s = op.state
        if entity.id == op.entities[s["leg"]] and element == course.vent_element:
            s["charged"] = True
            s["charged_at"] = course.world.elapsed
            entity.used = True
            course.feedback("Vent charged. Jump during the bright pulse.")
            course.emit("vent_charged", operation=op.id, leg=s["leg"], element=element.name.lower())

    def tick(self, course, op, char, dt):
        s = op.state
        leg = s["leg"]
        start = course.world.get(op.entities[leg]).pos
        end = course.world.get(op.entities[leg + 1]).pos
        near = np.linalg.norm((char.pos - start)[[0, 2]]) < 2.3
        safe = 1.0 <= course.world.elapsed % 4.0 <= 3.5
        if s["charged"] and not s["launched"] and near and safe and char.vel[1] > 1 and char.state == MoveState.AIRBORNE:
            char.vel[1] = 20.0
            s.update(launched=True, glided=False)
            s["launches"] += 1
            course.emit("vent_launch", operation=op.id, leg=leg)
        if s["launched"] and char.state == MoveState.GLIDING:
            s["glided"] = True
        if s["launched"] and np.linalg.norm((char.pos - end)[[0, 2]]) < 2.0 and char.grounded and char.pos[1] >= end[1] - 0.4:
            s["landings"] += 1
            s["glide_legs"] += int(s["glided"])
            course.emit("vent_landed", operation=op.id, leg=leg, glided=s["glided"])
            if leg == 1:
                course.finish_operation()
            else:
                s.update(leg=1, charged=False, launched=False, glided=False)
        elif s["launched"] and char.grounded and not near and char.pos[1] < end[1] - 1:
            s.update(charged=False, launched=False, glided=False)
            course.emit("crossing_retry", operation=op.id, leg=leg)

    def verify(self, course, milestone, op):
        if op.state["landings"] != 2 or op.state["glide_legs"] != 2:
            return [f"{milestone.id}: two demonstrated vent/glide landings not verified"]
        return []
