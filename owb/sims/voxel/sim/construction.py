"""Finite grid parts: remote manipulation and static, face-connected placement.

Placed parts occupy actual collision voxels. Held previews do not. This is a
deterministic construction prototype, not a rigid-body joint/vehicle solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core import blocks as B

MAX_PARTS = 8
PART_COLUMNS = ("visible", "held", "relative_x", "relative_y", "relative_z",
                "size_x", "size_y", "size_z", "quarter_turn")
BUILD_STATE_NAMES = ("held_part", "cursor_x", "cursor_y", "cursor_z", "can_place", "layer")
BUILD_COMMANDS = ("grab", "place", "rotate", "ascend")


@dataclass
class Part:
    id: int
    anchor: np.ndarray
    turn: int = 0

    @property
    def size(self):
        return np.array((3, 1, 1) if self.turn % 2 == 0 else (1, 1, 3))

    @property
    def center(self):
        return self.anchor + self.size / 2

    def cells(self):
        return [tuple(self.anchor + offset) for offset in np.ndindex(*self.size)]


class Construction:
    reach = 9.0
    sight = 14.0

    def __init__(self, world, agent, cfg, specs):
        self.world, self.agent, self.cfg = world, agent, cfg
        self.parts = {i + 1: Part(i + 1, np.array(p["anchor"], dtype=int))
                      for i, p in enumerate(specs)}
        if len(self.parts) > MAX_PARTS:
            raise ValueError("too many construction parts")
        self.held = None
        for part in self.parts.values():
            self._write(part, B.BRIDGE)

    def _write(self, part, block):
        for cell in part.cells():
            self.world.set_block(*cell, block)

    def _visible(self, part, reach):
        eye = self.agent.eye
        if np.linalg.norm(part.center - eye) > reach:
            return False
        # Ray to the nearest face, stopping just before the part itself.
        target = np.clip(eye, part.anchor, part.anchor + part.size)
        delta = target - eye
        distance = np.linalg.norm(delta)
        if distance < 1e-5:
            return True
        return not self.world.raycast(eye, delta, max(0, distance - 0.01)).hit

    def _overlaps_agent(self, part):
        a, c = self.agent, self.cfg
        low = a.pos - (c.agent_radius, 0, c.agent_radius)
        high = a.pos + (c.agent_radius, c.agent_height, c.agent_radius)
        return bool(np.all(part.anchor < high - 0.001)
                    and np.all(part.anchor + part.size > low + 0.001))

    def placement_error(self, part):
        cells = set(part.cells())
        if any(not self.world.in_bounds(*p) for p in cells):
            return "out_of_bounds"
        if np.linalg.norm(part.center - self.agent.eye) > self.reach:
            return "out_of_reach"
        if any(self.world.get(*p) != B.AIR for p in cells):
            return "occupied"
        if self._overlaps_agent(part):
            return "agent_overlap"
        if not self._visible(part, self.reach):
            return "occluded"
        for cell in cells:
            for axis in range(3):
                for sign in (-1, 1):
                    neighbor = list(cell)
                    neighbor[axis] += sign
                    if tuple(neighbor) not in cells and self.world.in_bounds(*neighbor):
                        if self.world.is_solid(*neighbor):
                            return None
        return "unsupported"

    def dispatch(self, command, part_id, delta):
        events = []
        if self.held is not None and np.any(delta):
            part = self.parts[self.held]
            shift = np.where(np.abs(delta) >= 0.5, np.sign(delta), 0).astype(int)
            if shift.any():
                candidate = Part(part.id, part.anchor + shift, part.turn)
                if (np.linalg.norm(candidate.center - self.agent.eye) <= self.reach
                        and all(self.world.in_bounds(*p) for p in candidate.cells())):
                    part.anchor = candidate.anchor
                    events.append(dict(type="part_moved", part=part.id))
                else:
                    events.append(dict(type="action_failed", command="build_move", reason="out_of_reach"))
        if command not in BUILD_COMMANDS or command == "ascend":
            return events

        def fail(reason):
            return events + [dict(type="action_failed", command=command, reason=reason)]

        if command == "grab":
            if self.held is not None:
                return fail("already_holding")
            candidates = [p for p in self.parts.values()
                          if (part_id == 0 or p.id == part_id) and self._visible(p, self.reach)]
            if not candidates:
                return fail("no_reachable_part")
            part = min(candidates, key=lambda p: (np.linalg.norm(p.center - self.agent.eye), p.id))
            if any(self.world.get(*p) != B.BRIDGE for p in part.cells()):
                return fail("part_damaged")
            self._write(part, B.AIR)
            self.held = part.id
            events.append(dict(type="part_grabbed", part=part.id))
        elif self.held is None:
            return fail("not_holding")
        elif command == "rotate":
            part = self.parts[self.held]
            candidate = Part(part.id, part.anchor, (part.turn + 1) % 4)
            if (np.linalg.norm(candidate.center - self.agent.eye) > self.reach
                    or any(not self.world.in_bounds(*p) for p in candidate.cells())):
                return fail("out_of_reach")
            part.turn = candidate.turn
            events.append(dict(type="part_rotated", part=part.id, quarter_turn=part.turn))
        elif command == "place":
            part = self.parts[self.held]
            error = self.placement_error(part)
            if error:
                return fail(error)
            self._write(part, B.BRIDGE)
            self.held = None
            events.append(dict(type="part_placed", part=part.id))
        return events

    def observe(self, layer):
        parts = np.zeros((MAX_PARTS, len(PART_COLUMNS)), np.float32)
        state = np.zeros(len(BUILD_STATE_NAMES), np.float32)
        state[-1] = layer
        for part in self.parts.values():
            held = part.id == self.held
            if held or self._visible(part, self.sight):
                parts[part.id - 1] = (1, held, *(part.center - self.agent.pos), *part.size, part.turn)
            if held:
                state[:5] = (part.id, *(part.anchor - self.agent.pos), self.placement_error(part) is None)
        return parts, state

    def render_list(self):
        if self.held is None:
            return []
        part = self.parts[self.held]
        color = (84, 226, 190) if self.placement_error(part) is None else (240, 128, 76)
        return [dict(pos=tuple(np.asarray(cell) + 0.5), radius=0.48, height=0.96,
                     color=color, semantic=B.SEM_STRUCTURE, kind="build_preview")
                for cell in part.cells()]
