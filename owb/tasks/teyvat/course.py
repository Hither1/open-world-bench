"""Course runtime of the Teyvat reference courses A/B (moved from reference_tasks.py).

``Course`` holds the course's private state: the seeded rule bindings, the milestone /
operation list, the mission inventory, and the completion ledger (``completed``). It
forwards native inputs (interact, elemental application, hits, ticks) for the current
operation to that operation's mechanism (:mod:`owb.tasks.teyvat.mechanisms`), and
completes operations and milestones when a mechanism calls :meth:`Course.finish_operation`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from owb.sims.teyvat.events import Event
from owb.sims.teyvat.world import World

from .mechanisms import MECHANISMS


@dataclass
class Operation:
    id: str
    kind: str
    title: str
    entities: list[str] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    complete: bool = False


@dataclass
class Milestone:
    id: str
    title: str
    operations: list[Operation]


class Course:
    """Private task/evaluator state. Only world actions may change operations."""

    def __init__(self, world: World, task: str, seed: int):
        if task not in ("a", "b"):
            raise ValueError("task must be 'a' or 'b'")
        self.world, self.task = world, task
        self.rng = np.random.default_rng(seed + 739)
        from owb.tasks.setups import run_setup
        from .setup import WorldHooks
        self.setup = run_setup("teyvat.reference_course", WorldHooks(world), self.rng, {"course": task})
        self.relay_elements = self.setup.rules["relay_elements"]
        self.guard_elements = self.setup.rules["guard_elements"]
        self.vent_element = self.setup.rules["vent_element"]
        self.inventory: dict[str, int] = {}
        self.milestones: list[Milestone] = self.setup.milestones
        self.index = 0
        self.operation_index = 0
        self.cell = self.setup.extras["cells"]
        self.events: list[Event] = []
        self.completed: list[dict] = []
        self.last_feedback = "Reference course initialized."
        self.dialogue = ""
        self.vent_glides = 0  # unused (kept for compatibility; nothing reads or writes it)

    @property
    def current(self):
        return self.milestones[self.index] if self.index < len(self.milestones) else None

    @property
    def op(self):
        return self.current.operations[self.operation_index] if self.current else None

    def mechanism(self, op):
        """The state machine that runs ``op``."""
        return MECHANISMS[op.kind]

    def emit(self, event_type, **payload):
        self.events.append(Event(event_type, payload))

    def feedback(self, message, **payload):
        self.last_feedback = message
        self.emit("mechanism_feedback", text=message, **payload)

    def flat_patch(self, xz, height, radius=5.0):
        from .setup import flat_patch
        flat_patch(self.world.terrain, xz, height, radius)

    def target(self):
        op=self.op
        if op is None:
            return None
        return MECHANISMS[op.kind].target(self,op)

    def local_entities(self):
        return self.op.entities if self.op else []

    def finish_operation(self):
        op=self.op
        if op is None or op.complete:
            return
        op.complete=True
        if op.state.get("grants"):
            item=op.state["grants"]
            self.inventory[item]=self.inventory.get(item,0)+1
            self.emit("item_granted",item=item,operation=op.id)
        self.emit("operation_complete",operation=op.id,kind=op.kind)
        self.operation_index+=1
        if self.operation_index==len(self.current.operations):
            milestone=self.current
            self.completed.append({"id":milestone.id,"time":round(self.world.elapsed,6),
                                   "operations":[o.id for o in milestone.operations]})
            self.emit("activate",entity=milestone.id)
            self.emit("milestone_complete",milestone=milestone.id,title=milestone.title)
            self.feedback(f"{milestone.id} complete: {milestone.title}")
            self.index+=1
            self.operation_index=0
            if self.current and self.current.id in {"A08","A12","B11","B21","B24"}:
                self.emit("designed_context_boundary",milestone=self.current.id,
                          note="Oracle reference controller; this is not a model-memory evaluation.")

    def can_use_items(self, op):
        return all(self.inventory.get(k,0)>=n for k,n in op.state.get("requires",{}).items())

    def consume_items(self, op):
        if not op.state.get("keep"):
            for k,n in op.state.get("requires",{}).items():
                self.inventory[k]-=n
                self.emit("item_consumed",item=k,amount=n,operation=op.id)

    def interact(self,char):
        op=self.op
        if op is None:
            return
        candidates=[self.world.get(eid) for eid in op.entities]
        candidates=[e for e in candidates if np.linalg.norm((e.pos-char.pos)[[0,2]])<=e.radius and abs(float(e.pos[1]-char.pos[1]))<2.8]
        if not candidates:
            return
        entity=min(candidates,key=lambda e:float(np.linalg.norm(e.pos-char.pos)))
        self.emit("interact",entity=entity.id,operation=op.id)
        MECHANISMS[op.kind].interact(self,op,char,entity)

    def apply_element(self,char,element):
        op=self.op
        if op is None or not MECHANISMS[op.kind].accepts_elements:
            return
        candidates=[self.world.get(e) for e in op.entities]
        candidates=[e for e in candidates if np.linalg.norm((e.pos-char.pos)[[0,2]])<2.6 and abs(float(e.pos[1]-char.pos[1]))<4.0]
        if not candidates:
            return
        entity=min(candidates,key=lambda e:float(np.linalg.norm(e.pos-char.pos)))
        MECHANISMS[op.kind].apply_element(self,op,char,element,entity)

    def sentinel_hit(self,enemy,damage,element):
        op=self.op
        if op is None or enemy.id not in op.entities:
            return []
        return MECHANISMS[op.kind].hit(self,op,enemy,damage,element)

    def tick(self,char,dt):
        op=self.op
        if op is None:
            return
        MECHANISMS[op.kind].tick(self,op,char,dt)

    def state(self):
        return {"task":self.task,"milestone_index":self.index,"operation_index":self.operation_index,
                "inventory":dict(self.inventory),"completed":list(self.completed),
                "operations":{o.id:{"kind":o.kind,"complete":o.complete,"state":o.state}
                              for m in self.milestones for o in m.operations}}

    def verify(self):
        failures=[]
        for m in self.milestones:
            for op in m.operations:
                if not op.complete:
                    failures.append(f"{m.id}: {op.id} incomplete")
                failures.extend(MECHANISMS[op.kind].verify(self,m,op))
        if len(self.completed)!=len(self.milestones):
            failures.append("milestone count mismatch")
        return {"success":not failures,"failures":failures,"completed":len(self.completed),
                "required":len(self.milestones),"scope":"compact native-simulator reference prototype"}


__all__ = ["Course", "Milestone", "Operation"]
