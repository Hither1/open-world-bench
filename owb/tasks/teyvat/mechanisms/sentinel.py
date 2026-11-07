"""Enemy encounters: ordinary camp fights and the phase sentinel (procedure G).

A fight completes when its enemy is defeated by native combat. A sentinel holds its
authored position and cycles shield -> exposed -> windup: the first guard element breaks
the shield, the second interrupts the wind-up (once re-forming the shield on a
``reform`` sentinel), and only physical attacks damage it while vulnerable. An
uninterrupted wind-up resets it to the shield phase after 18 s.
"""
from __future__ import annotations

from owb.sims.teyvat.elements import Element
from owb.sims.teyvat.world import EntityKind

from . import register
from .base import Mechanism


@register
class Fight(Mechanism):
    kind = "fight"

    def place(self, hooks, op, p, rng, rules):
        kind = op.kind
        state = op.state
        eid = hooks.spawn(op, "guard", EntityKind.ENEMY, p,
                          hp=84.0 if kind == "sentinel" else 60.0,
                          damage=2.0, label="Phase Sentinel" if kind == "sentinel" else "Camp Sentry")
        op.state.update(phase=state.get("phase", "shield"), timer=0.0, reformed=False)
        if kind == "sentinel":
            hooks.get(eid).elemental.shield = 50.0
            hooks.get(eid).elemental.shield_element = Element(int(rules["guard_elements"][0]))

    def tick(self, course, op, char, dt):
        e = course.world.get(op.entities[0])
        if not e.alive:
            course.finish_operation()
        else:
            self.tick_alive(course, op, e, dt)

    def tick_alive(self, course, op, enemy, dt):
        pass

    def verify(self, course, milestone, op):
        return [f"{milestone.id}: enemy alive"] if course.world.get(op.entities[0]).alive else []


@register
class Sentinel(Fight):
    kind = "sentinel"
    holds_position = True

    def hit(self, course, op, enemy, damage, element):
        s = op.state
        if s["phase"] == "shield" and element == course.guard_elements[0]:
            s.update(phase="exposed", timer=0.0)
            enemy.elemental.shield = 0.0
            course.emit("sentinel_exposed", operation=op.id)
        elif s["phase"] == "windup" and element == course.guard_elements[1]:
            if s.get("reform") and not s["reformed"]:
                s.update(phase="shield", timer=0.0, reformed=True)
                enemy.elemental.shield = 50.0
                course.emit("sentinel_reformed", operation=op.id)
            else:
                s.update(phase="vulnerable", timer=0.0)
                course.emit("sentinel_interrupted", operation=op.id)
        elif s["phase"] == "vulnerable" and element == Element.NONE:
            enemy.hp = max(0.0, enemy.hp - damage)
            course.emit("enemy_hit", entity=enemy.id, damage=damage, hp=enemy.hp)
            if enemy.hp == 0:
                enemy.alive = False
                course.emit("defeat", entity=enemy.id)
        else:
            course.emit("sentinel_blocked", operation=op.id, phase=s["phase"])
        return []

    def tick_alive(self, course, op, enemy, dt):
        s = op.state
        s["timer"] += dt
        if s["phase"] == "exposed":
            enemy.elemental.shield = 0
            if s["timer"] >= 0.8:
                s.update(phase="windup", timer=0.0)
                course.emit("sentinel_windup", operation=op.id)
        elif s["phase"] == "windup" and s["timer"] >= 18:
            s.update(phase="shield", timer=0.0)
            enemy.elemental.shield = 50
            course.emit("sentinel_reset", operation=op.id)
