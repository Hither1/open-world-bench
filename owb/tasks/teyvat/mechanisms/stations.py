"""Interaction stations: lesson, survey, collect and exchange.

Interacting with the station's entity completes the operation, provided the mission
inventory holds the operation's ``requires`` items. The use consumes those items (unless
``keep``), grants the operation's ``item`` if any, and marks the entity used. A lesson
first shows its dialogue; that is exposure, not a demonstration of the taught procedure.
"""
from __future__ import annotations

from . import register
from .base import Mechanism


class Station(Mechanism):
    def interact(self, course, op, char, entity):
        if not course.can_use_items(op):
            course.feedback("Required mission items are missing.")
            return
        if not self.ready(course, op, char):
            return
        self.use(course, op, char, entity)

    def ready(self, course, op, char):
        return True

    def use(self, course, op, char, entity):
        course.consume_items(op)
        if op.state.get("item"):
            item = op.state["item"]
            course.inventory[item] = course.inventory.get(item, 0) + 1
            course.emit("item_acquired", item=item, entity=entity.id)
        entity.used = True
        course.finish_operation()


@register
class Lesson(Station):
    kind = "lesson"

    def use(self, course, op, char, entity):
        course.dialogue = op.state["text"]
        course.emit("talk", entity=entity.id, text=course.dialogue)
        course.feedback(course.dialogue)
        super().use(course, op, char, entity)


@register
class Survey(Station):
    kind = "survey"


@register
class Collect(Station):
    kind = "collect"


@register
class Exchange(Station):
    kind = "exchange"
