"""Placement of the Teyvat reference courses A/B (``setup: teyvat.reference_course``).

``reference_course(hooks, rng, {"course": "a"|"b"})`` draws the course's rule bindings
(relay role elements, sentinel guard elements, vent element), lays the operations out on
a serpentine 8-wide grid of 46 m cells, asks each operation's mechanism to place its
local fixture, and returns the milestone list. ``rng`` is the course RNG
(``default_rng(seed + 739)``); the draw order is: three rule permutations/choice, one
``shuffle`` per relay in build order, and (course B only) the outpost relay version.

Level construction around the course: :func:`author_plateau` (coastal plateau, before the
world exists) and :func:`settle_entities` (after placement).
"""
from __future__ import annotations

import numpy as np

from owb.sims.teyvat.elements import Element

from ..setups import SetupResult, register_setup
from .course import Milestone, Operation
from .mechanisms import MECHANISMS

COURSE_SIZES = {"a": 12, "b": 24}


def flat_patch(terrain, xz, height, radius=5.0):
    """Blend the terrain to ``height`` over a disc of ``radius`` (+3 m falloff)."""
    t = terrain
    grid = np.linspace(-t._half, t._half, t.cfg.resolution)
    x, z = np.meshgrid(grid, grid, indexing="ij")
    distance = np.hypot(x - xz[0], z - xz[1])
    w = np.clip((radius + 3.0 - distance) / 3.0, 0, 1)
    t.height[:] = t.height * (1-w) + height*w


class WorldHooks:
    """The placement surface a Teyvat setup may use: spawn, look up, flatten terrain."""

    def __init__(self, world):
        self.world = world

    def spawn(self, op, suffix, kind, xz, **kwargs):
        eid = f"{op.id}:{suffix}"
        self.world.spawn_at(eid, kind, *xz, radius=2.1, **kwargs)
        op.entities.append(eid)
        return eid

    def get(self, eid):
        return self.world.get(eid)

    def flat_patch(self, xz, height, radius=5.0):
        flat_patch(self.world.terrain, xz, height, radius)


def draw_rules(rng):
    elements = [Element.EMBER, Element.TIDE, Element.GALE]
    relay_elements = dict(zip(("source", "vane", "drain"), rng.permutation(elements)))
    guard_elements = tuple(rng.permutation(elements)[:2])
    vent_element = Element(int(rng.choice(elements)))
    return {"relay_elements": relay_elements, "guard_elements": guard_elements,
            "vent_element": vent_element}


class _Builder:
    def __init__(self, hooks, rng, task, rules):
        self.hooks, self.rng, self.task, self.rules = hooks, rng, task, rules
        self.milestones: list[Milestone] = []
        self.cell = 0

    def site(self):
        # A serpentine course keeps physical walking continuous and auditable.
        i, self.cell = self.cell, self.cell + 1
        r, c = divmod(i, 8)
        if r % 2:
            c = 7 - c
        return np.array([-168.0 + c * 46.0, -168.0 + r * 46.0])

    def make(self, kind, title, **state):
        op = Operation(f"op{self.cell:03d}_{len(self.milestones):02d}", kind, title, state=state)
        # The default is evaluated even when a site is given: every operation takes a cell.
        p = np.asarray(state.pop("site", self.site()), dtype=float)
        op.state["site"] = p.tolist()
        MECHANISMS[kind].place(self.hooks, op, p, self.rng, self.rules)
        return op

    def add(self, title, *ops):
        mid = f"{self.task.upper()}{len(self.milestones)+1:02d}"
        self.milestones.append(Milestone(mid, title, list(ops)))

    def lesson(self, which):
        relay_elements, guard_elements = self.rules["relay_elements"], self.rules["guard_elements"]
        if which == "R":
            e = {k: Element(int(v)).name.lower() for k, v in relay_elements.items()}
            message = (f"Relay lesson: inspect progress. If overloaded, drain with {e['drain']}. "
                       f"Prime source with {e['source']}; route vane with {e['vane']}; interact with core. "
                       "After interruption, drain and resume; primed stages survive.")
        elif which == "G":
            a, b = [Element(int(e)).name.lower() for e in guard_elements]
            message = f"Sentinel lesson: {a} breaks the shield. At WIND-UP use {b}, then attack. Re-diagnose if the shield reforms."
        else:
            message = f"Wind lesson: charge with {self.rules['vent_element'].name.lower()}, jump on the bright pulse, hold jump to glide, land and rest."
        return self.make("lesson", f"Learn {which}", text=message)

    def build_a(self):
        self.add("Guided relay repair", self.lesson("R"), self.make("relay","Training relay"))
        self.add("Repair without coaching", self.make("relay","Rearranged relay"))
        self.add("Recover from a fault", self.make("relay","Faulty training relay",overloaded=True,interrupt=True))
        cache = self.site()
        self.add("Survey the marsh route", self.make("survey","Register route junction"),
                 self.make("survey","Remember the supply cache",site=cache))
        self.add("Recover the lift casing", self.make("fight","Clear the camp"),
                 self.make("collect","Lift casing",item="casing"))
        self.add("Repair the ferry lift", self.make("exchange","Ferry mechanic",requires={"casing":1},grants="ferry_open"))
        self.add("Supply the outpost", self.make("collect","Supply cache",site=cache,item="supplies",requires={"ferry_open":1},keep=True),
                 self.make("exchange","Highland quartermaster",requires={"supplies":1},grants="corridor_open"))
        self.add("Restore a field relay", self.make("relay","First field relay"))
        self.add("Stabilize a primed relay", self.make("relay","Second field relay",progress=1,overloaded=True))
        self.add("Recover the service cap", self.make("fight","Clear the ruin"), self.make("collect","Service cap",item="cap"))
        self.add("Reopen summit access", self.make("lift","Install cap and ride",requires={"cap":1}))
        self.add("Commission the summit beacon", self.make("relay","Summit beacon",interrupt=True))

    def build_b(self):
        self.add("Learn relay repair",self.lesson("R"),self.make("relay","Training relay"))
        self.add("Recover a faulty relay",self.make("relay","Faulty workshop relay",overloaded=True,interrupt=True))
        self.add("Learn sentinel combat",self.lesson("G"),self.make("sentinel","Training sentinel"))
        self.add("Handle shield reformation",self.make("sentinel","Reforming sentinel",reform=True))
        self.add("Learn the vent crossing",self.lesson("T"),self.make("crossing","Guided vent crossing"))
        self.add("Reach a new overlook",self.make("crossing","Second overlook"))
        depot=self.site()
        self.add("Survey the supply depot",self.make("survey","Reserve depot landmark",site=depot))
        self.add("Recover the repair kit",self.make("fight","Camp sentry"),self.make("collect","Repair kit",item="kit"))
        self.add("Dive for the coupling",self.make("dive","Submerged coupling",item="coupling"))
        self.add("Reopen the causeway",self.make("exchange","Causeway mechanic",requires={"kit":1,"coupling":1},grants="causeway_open"),
                 self.make("survey","Cross the repaired causeway"))
        self.add("Restore the signal station",self.make("relay","Valley signal station"))
        self.add("Reach the cliff post",self.make("crossing","Cliff vent route"))
        self.add("Recover the field seal",self.make("sentinel","Exposed field sentinel",phase="exposed",grants="field_seal"))
        self.add("Restore reserve power",self.make("exchange","Unlock the service panel",requires={"field_seal":1}),
                 self.make("relay","Reserve relay",progress=1,overloaded=True,grants="reserve_power"))
        self.add("Retrieve service tools",self.make("collect","Reserve service tools",site=depot,item="tools",requires={"reserve_power":1},keep=True))
        self.add("Open the sky-lift route",self.make("exchange","Open sky-lift controls",requires={"tools":1},keep=True),
                 self.make("relay","Sky-lift relay"),self.make("crossing","Powered sky-lift crossing"))
        self.add("Restore the guarded gate",self.make("sentinel","Gate sentinel"),self.make("relay","Causeway gate relay"))
        self.add("Recover the shrine seal",self.make("crossing","High shrine approach"),
                 self.make("sentinel","Shrine sentinel",grants="shrine_seal"))
        self.add("Discover the relay update",self.make("exchange","Shrine seal gate",requires={"shrine_seal":1}),
                 self.make("lesson","Inspect double-ring service plate",text="Double-ring inlet retains residual charge. Drain first even without an overload. Single-ring rules are unchanged."),
                 self.make("relay","First double-ring relay",version=2))
        self.add("Establish a landing route",self.make("crossing","Remote wind route"),self.make("survey","Activate landing signal"))
        self.add("Reuse the updated rule",self.make("relay","Second double-ring relay",version=2))
        self.add("Repair an older station",self.make("relay","Original single-ring station",version=1))
        self.add("Recover the beacon heart",self.make("relay","Summit approach relay",version=2),
                 self.make("crossing","Summit approach"),self.make("sentinel","Reliquary sentinel",grants="heart"))
        self.add("Commission a distant outpost",self.make("sentinel","Outpost sentinel"),
                 self.make("crossing","Outpost approach"),self.make("exchange","Install beacon heart",requires={"heart":1}),
                 self.make("relay","Outpost beacon",version=int(self.rng.integers(1,3))))


@register_setup("teyvat.reference_course")
def reference_course(hooks, rng, params) -> SetupResult:
    task = params["course"]
    if task not in COURSE_SIZES:
        raise ValueError("task must be 'a' or 'b'")
    rules = draw_rules(rng)
    builder = _Builder(hooks, rng, task, rules)
    builder.build_a() if task == "a" else builder.build_b()
    assert len(builder.milestones) == COURSE_SIZES[task]
    record = {"course": task,
              "relay_elements": {k: Element(int(v)).name.lower() for k, v in rules["relay_elements"].items()},
              "guard_elements": [Element(int(v)).name.lower() for v in rules["guard_elements"]],
              "vent_element": rules["vent_element"].name.lower(),
              "milestones": [{"id": m.id, "operations": [[o.id, o.kind] for o in m.operations]}
                             for m in builder.milestones]}
    return SetupResult(builder.milestones, rules, record, {"cells": builder.cell})


def author_plateau(terrain):
    """Author a coastal plateau; this is level construction at reset, not a policy action."""
    axis = np.linspace(-terrain._half, terrain._half, terrain.cfg.resolution)
    x, z = np.meshgrid(axis, axis, indexing="ij")
    edge = np.clip((np.maximum(np.abs(x), np.abs(z))-255)/50, 0, 1)
    terrain.height[:] = 12.0-edge*24


def settle_entities(world, terrain):
    """Recompute terrain derivatives and drop every entity onto the patched terrain."""
    terrain._compute_normals(); terrain._compute_biomes()
    # Refresh heights after all authored shelf/pool patches are constructed.
    for entity in world.entities.values():
        entity.pos[1] = terrain.height_at(entity.pos[0], entity.pos[2])
        entity.home = entity.pos.copy()


__all__ = ["WorldHooks", "author_plateau", "draw_rules", "flat_patch", "reference_course",
           "settle_entities"]
