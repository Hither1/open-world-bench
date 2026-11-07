"""Elemental relay (procedure R): single-ring (version 1) and double-ring (version 2).

Roles source / vane / drain / core. The drain clears an overload (and, on a double ring,
the residual charge that must be drained before priming); the source primes
(progress 0 -> 1) and may trigger the one authored interruption, which re-overloads the
relay while preserving progress; the vane routes (1 -> 2); interacting with the core of a
routed, fault-free relay activates it and completes the operation.
"""
from __future__ import annotations

from . import register
from .base import Mechanism


@register
class Relay(Mechanism):
    kind = "relay"
    accepts_elements = True

    def target(self, course, op):
        s = op.state
        role = "drain" if s["overloaded"] or (s["version"] == 2 and not s["drained"]) else (
               "source" if s["progress"] == 0 else "vane" if s["progress"] == 1 else "core")
        return course.world.get(s["roles"][role])

    def interact(self, course, op, char, entity):
        s = op.state
        if entity.id != s["roles"]["core"]:
            course.feedback("Elemental receiver: cast a skill to apply an element.")
        elif s["progress"] == 2 and not s["overloaded"] and (s["version"] == 1 or s["drained"]):
            s["active"] = True
            entity.used = True
            course.emit("relay_active", operation=op.id, version=s["version"])
            course.finish_operation()
        else:
            course.feedback("Core blocked: inspect progress and residual charge.")

    def apply_element(self, course, op, char, element, entity):
        s = op.state
        role = next(k for k, v in s["roles"].items() if v == entity.id)
        valid = False
        if role == "drain" and element == course.relay_elements["drain"]:
            s["overloaded"] = False
            s["drained"] = True
            valid = True
        elif (role == "source" and element == course.relay_elements["source"] and not s["overloaded"]
              and (s["version"] == 1 or s["drained"]) and s["progress"] == 0):
            s["progress"] = 1
            valid = True
            if s.get("interrupt") and not s["interrupted"]:
                s["interrupted"] = True
                s["overloaded"] = True
                course.emit("relay_interrupted", operation=op.id, preserved_progress=1)
        elif role == "vane" and element == course.relay_elements["vane"] and s["progress"] == 1 and not s["overloaded"]:
            s["progress"] = 2
            valid = True
        if valid:
            entity.used = True
            course.emit("relay_application", operation=op.id, role=role, element=element.name.lower(),
                        progress=s["progress"], overloaded=s["overloaded"])
            course.feedback(f"{role}: accepted; progress={s['progress']}; overload={s['overloaded']}")
        else:
            course.feedback("Application rejected: inspect role, stage, and residual charge.")
            course.emit("invalid_application", operation=op.id, role=role, element=element.name.lower())

    def verify(self, course, milestone, op):
        return [] if op.state.get("active") else [f"{milestone.id}: relay inactive"]
