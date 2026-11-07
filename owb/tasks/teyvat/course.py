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

from owb.sims.teyvat.character import MoveState
from owb.sims.teyvat.elements import Element
from owb.sims.teyvat.events import Event
from owb.sims.teyvat.world import EntityKind, World

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
        elements = [Element.EMBER, Element.TIDE, Element.GALE]
        self.relay_elements = dict(zip(("source", "vane", "drain"), self.rng.permutation(elements)))
        self.guard_elements = tuple(self.rng.permutation(elements)[:2])
        self.vent_element = Element(int(self.rng.choice(elements)))
        self.inventory: dict[str, int] = {}
        self.milestones: list[Milestone] = []
        self.index = 0
        self.operation_index = 0
        self.cell = 0
        self.events: list[Event] = []
        self.completed: list[dict] = []
        self.last_feedback = "Reference course initialized."
        self.dialogue = ""
        self.vent_glides = 0
        self._build_a() if task == "a" else self._build_b()
        assert len(self.milestones) == (12 if task == "a" else 24)

    @property
    def current(self):
        return self.milestones[self.index] if self.index < len(self.milestones) else None

    @property
    def op(self):
        return self.current.operations[self.operation_index] if self.current else None

    def mechanism(self, op):
        """The state machine that runs ``op`` (``None`` while a kind is still inline)."""
        return MECHANISMS.get(op.kind)

    def emit(self, event_type, **payload):
        self.events.append(Event(event_type, payload))

    def feedback(self, message, **payload):
        self.last_feedback = message
        self.emit("mechanism_feedback", text=message, **payload)

    def site(self):
        # A serpentine course keeps physical walking continuous and auditable.
        i, self.cell = self.cell, self.cell + 1
        r, c = divmod(i, 8)
        if r % 2:
            c = 7 - c
        return np.array([-168.0 + c * 46.0, -168.0 + r * 46.0])

    def entity(self, op, suffix, kind, xz, **kwargs):
        eid = f"{op.id}:{suffix}"
        self.world.spawn_at(eid, kind, *xz, radius=2.1, **kwargs)
        op.entities.append(eid)
        return eid

    def flat_patch(self, xz, height, radius=5.0):
        t = self.world.terrain
        grid = np.linspace(-t._half, t._half, t.cfg.resolution)
        x, z = np.meshgrid(grid, grid, indexing="ij")
        distance = np.hypot(x - xz[0], z - xz[1])
        w = np.clip((radius + 3.0 - distance) / 3.0, 0, 1)
        t.height[:] = t.height * (1-w) + height*w

    def make(self, kind, title, **state):
        op = Operation(f"op{self.cell:03d}_{len(self.milestones):02d}", kind, title, state=state)
        p = np.asarray(state.pop("site", self.site()), dtype=float)
        op.state["site"] = p.tolist()
        if kind == "relay":
            op.state.update(progress=int(state.get("progress", 0)),
                            overloaded=bool(state.get("overloaded", False)),
                            drained=False, interrupted=False, active=False,
                            version=int(state.get("version", 1)))
            offsets = np.array([[-5,-4], [5,-4], [-5,5], [5,5]])
            self.rng.shuffle(offsets)
            op.state["roles"] = {}
            for role, off in zip(("source", "vane", "drain", "core"), offsets):
                kind_ = EntityKind.BEACON if role == "core" else EntityKind.PILLAR
                eid = self.entity(op, role, kind_, p+off,
                                  label=f"{'Double' if op.state['version']==2 else 'Single'}-ring {role}",
                                  visual_role=role, visual_version=op.state["version"])
                op.state["roles"][role] = eid
                if role == "source" and op.state["progress"] >= 1:
                    self.world.get(eid).used = True
        elif kind in ("sentinel", "fight"):
            eid = self.entity(op, "guard", EntityKind.ENEMY, p,
                              hp=84.0 if kind == "sentinel" else 60.0,
                              damage=2.0, label="Phase Sentinel" if kind == "sentinel" else "Camp Sentry")
            op.state.update(phase=state.get("phase", "shield"), timer=0.0, reformed=False)
            if kind == "sentinel":
                self.world.get(eid).elemental.shield = 50.0
                self.world.get(eid).elemental.shield_element = Element(int(self.guard_elements[0]))
        elif kind == "crossing":
            op.state.update(leg=0, charged=False, launched=False, glided=False,
                            landings=0, launches=0, glide_legs=0, charged_at=0.0)
            # Two genuine upward flights with native gliding and landings.
            points = [p + (-12,0), p, p + (12,0)]
            for k, point in enumerate(points):
                self.flat_patch(point, 12.0 + k*3.0, radius=4.0)
                self.entity(op, f"shelf{k}", EntityKind.WAYPOINT, point, label=f"Wind shelf {k+1}",
                            visual_role="vent")
            op.state["points"] = [v.tolist() for v in points]
        elif kind == "dive":
            self.flat_patch(p, -4.5, radius=5.0)
            self.entity(op, "chest", EntityKind.CHEST, p, label=title)
        elif kind == "lift":
            self.entity(op, "platform", EntityKind.ALTAR, p, label="Service lift")
            op.state.update(riding=False, height=12.0, target_height=18.0)
        else:
            ek = EntityKind.NPC if kind in ("lesson", "exchange") else (
                 EntityKind.CHEST if kind == "collect" else EntityKind.WAYPOINT)
            self.entity(op, "target", ek, p, label=title)
        return op

    def add(self, title, *ops):
        mid = f"{self.task.upper()}{len(self.milestones)+1:02d}"
        self.milestones.append(Milestone(mid, title, list(ops)))

    def lesson(self, which):
        if which == "R":
            e = {k: Element(int(v)).name.lower() for k,v in self.relay_elements.items()}
            message = (f"Relay lesson: inspect progress. If overloaded, drain with {e['drain']}. "
                       f"Prime source with {e['source']}; route vane with {e['vane']}; interact with core. "
                       "After interruption, drain and resume; primed stages survive.")
        elif which == "G":
            a,b = [Element(int(e)).name.lower() for e in self.guard_elements]
            message = f"Sentinel lesson: {a} breaks the shield. At WIND-UP use {b}, then attack. Re-diagnose if the shield reforms."
        else:
            message = f"Wind lesson: charge with {self.vent_element.name.lower()}, jump on the bright pulse, hold jump to glide, land and rest."
        return self.make("lesson", f"Learn {which}", text=message)

    def _build_a(self):
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

    def _build_b(self):
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

    def target(self):
        op=self.op
        if op is None:
            return None
        if op.kind in MECHANISMS:
            return MECHANISMS[op.kind].target(self,op)
        if op.kind=="crossing":
            k=op.state["leg"] + int(op.state["launched"])
            return self.world.get(op.entities[min(k,2)])
        return self.world.get(op.entities[0])

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
        if op.kind in MECHANISMS:
            MECHANISMS[op.kind].interact(self,op,char,entity)
        elif op.kind in ("lesson","survey","collect","exchange","dive","lift"):
            if not self.can_use_items(op):
                self.feedback("Required mission items are missing.")
                return
            if op.kind=="dive" and (char.pos[1]>-2.8 or char.state!=MoveState.SWIMMING):
                self.feedback("Dive to the chest; surface interaction cannot recover it.")
                return
            if op.kind=="lesson":
                self.dialogue=op.state["text"]
                self.emit("talk",entity=entity.id,text=self.dialogue)
                self.feedback(self.dialogue)
            if op.kind=="lift":
                if not op.state["riding"]:
                    self.consume_items(op)
                    op.state["riding"]=True
                    self.emit("lift_started",operation=op.id)
                return
            self.consume_items(op)
            if op.state.get("item"):
                item=op.state["item"]
                self.inventory[item]=self.inventory.get(item,0)+1
                self.emit("item_acquired",item=item,entity=entity.id)
            entity.used=True
            self.finish_operation()

    def apply_element(self,char,element):
        op=self.op
        if op is None or op.kind not in ("relay","crossing"):
            return
        candidates=[self.world.get(e) for e in op.entities]
        candidates=[e for e in candidates if np.linalg.norm((e.pos-char.pos)[[0,2]])<2.6 and abs(float(e.pos[1]-char.pos[1]))<4.0]
        if not candidates:
            return
        entity=min(candidates,key=lambda e:float(np.linalg.norm(e.pos-char.pos)))
        s=op.state
        if op.kind=="crossing":
            if entity.id==op.entities[s["leg"]] and element==self.vent_element:
                s["charged"]=True
                s["charged_at"]=self.world.elapsed
                entity.used=True
                self.feedback("Vent charged. Jump during the bright pulse.")
                self.emit("vent_charged",operation=op.id,leg=s["leg"],element=element.name.lower())
            return
        MECHANISMS[op.kind].apply_element(self,op,char,element,entity)

    def sentinel_hit(self,enemy,damage,element):
        op=self.op
        if op is None or enemy.id not in op.entities:
            return []
        return MECHANISMS[op.kind].hit(self,op,enemy,damage,element) if op.kind in MECHANISMS else None

    def tick(self,char,dt):
        op=self.op
        if op is None:
            return
        s=op.state
        if op.kind in MECHANISMS:
            MECHANISMS[op.kind].tick(self,op,char,dt)
        elif op.kind=="crossing":
            leg=s["leg"]
            start=self.world.get(op.entities[leg]).pos
            end=self.world.get(op.entities[leg+1]).pos
            near=np.linalg.norm((char.pos-start)[[0,2]])<2.3
            safe=1.0 <= self.world.elapsed % 4.0 <= 3.5
            if s["charged"] and not s["launched"] and near and safe and char.vel[1]>1 and char.state==MoveState.AIRBORNE:
                char.vel[1]=20.0
                s.update(launched=True,glided=False)
                s["launches"]+=1
                self.emit("vent_launch",operation=op.id,leg=leg)
            if s["launched"] and char.state==MoveState.GLIDING:
                s["glided"]=True
            if s["launched"] and np.linalg.norm((char.pos-end)[[0,2]])<2.0 and char.grounded and char.pos[1]>=end[1]-0.4:
                s["landings"]+=1
                s["glide_legs"]+=int(s["glided"])
                self.emit("vent_landed",operation=op.id,leg=leg,glided=s["glided"])
                if leg==1:
                    self.finish_operation()
                else:
                    s.update(leg=1,charged=False,launched=False,glided=False)
            elif s["launched"] and char.grounded and not near and char.pos[1]<end[1]-1:
                s.update(charged=False,launched=False,glided=False)
                self.emit("crossing_retry",operation=op.id,leg=leg)
        elif op.kind=="lift" and s["riding"]:
            s["height"]=min(s["target_height"],s["height"]+dt*2.0)
            char.pos[1]=s["height"]
            char.vel[:]=0
            char.grounded=True
            char.state=MoveState.GROUNDED
            self.world.get(op.entities[0]).pos[1]=s["height"]
            if s["height"]>=s["target_height"]:
                self.flat_patch(s["site"],s["height"],radius=5)
                self.world.terrain._compute_normals()
                self.world.terrain._compute_biomes()
                self.emit("lift_arrived",operation=op.id,height=s["height"])
                self.finish_operation()

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
                if op.kind in MECHANISMS:
                    failures.extend(MECHANISMS[op.kind].verify(self,m,op))
                if op.kind=="crossing" and (op.state["landings"]!=2 or op.state["glide_legs"]!=2):
                    failures.append(f"{m.id}: two demonstrated vent/glide landings not verified")
        if len(self.completed)!=len(self.milestones):
            failures.append("milestone count mismatch")
        return {"success":not failures,"failures":failures,"completed":len(self.completed),
                "required":len(self.milestones),"scope":"compact native-simulator reference prototype"}


__all__ = ["Course", "Milestone", "Operation"]
