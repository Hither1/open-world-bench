"""Service lift: installing the required part starts a ride that carries the character
and platform up at 2 m/s; at the target height the landing is levelled into the terrain
and the operation completes."""
from __future__ import annotations

from owb.sims.teyvat.character import MoveState

from . import register
from .stations import Station


@register
class Lift(Station):
    kind = "lift"

    def use(self, course, op, char, entity):
        if not op.state["riding"]:
            course.consume_items(op)
            op.state["riding"] = True
            course.emit("lift_started", operation=op.id)

    def tick(self, course, op, char, dt):
        s = op.state
        if not s["riding"]:
            return
        s["height"] = min(s["target_height"], s["height"] + dt * 2.0)
        char.pos[1] = s["height"]
        char.vel[:] = 0
        char.grounded = True
        char.state = MoveState.GROUNDED
        course.world.get(op.entities[0]).pos[1] = s["height"]
        if s["height"] >= s["target_height"]:
            course.flat_patch(s["site"], s["height"], radius=5)
            course.world.terrain._compute_normals()
            course.world.terrain._compute_biomes()
            course.emit("lift_arrived", operation=op.id, height=s["height"])
            course.finish_operation()
