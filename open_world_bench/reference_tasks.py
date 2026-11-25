"""Compact, executable reference courses for the two Teyvat task designs.

This is a demonstration prototype, not the long-horizon evaluation suite.
It reuses native movement, party cooldowns, combat, terrain, and rendering.
Mission travel is shortened; a privileged reference controller is kept separate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json

import numpy as np

from .camera import CameraConfig, OrbitCamera
from .character import CharacterController, MoveState
from .elements import Element
from .env import EnvConfig, OpenWorldEnv
from .events import Event
from .party import Party
from .quests import ObjKind, Objective, Quest, QuestEngine, QuestGraph, Stage
from .render import RenderSettings
from .scatter import PropField
from .terrain import Terrain, TerrainConfig
from .world import CombatSystem, EntityKind, World, WorldConfig
from . import working_memory_probes as wmp

# Working-memory fixtures on the compact plateau (x, z). Each use site is more
# than the 160 m camera range from its source; see docs/teyvat-reference-demos.md.
WM_SITES = {"a": {"levers": (120.0, 55.0)},
            "b": {"boatman": (-230.0, -40.0), "winch": (-215.0, 40.0)}}
WM_POOL_RADIUS = 8.0
WM_CHEST_RING = 4.5
WM_MARK_RANGE = 3.0
WM_PANEL_SPACING = 3.0


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

    def __init__(self, world: World, task: str, seed: int, *, working_memory: bool = True, wm_control: bool = False):
        if task not in ("a", "b"):
            raise ValueError("task must be 'a' or 'b'")
        self.world, self.task = world, task
        self.seed = seed
        # Working-memory probes draw from their own generators, before and apart from self.rng.
        self.working_memory = bool(working_memory)
        self.wm = wmp.make_probes(task, seed, control=wm_control) if working_memory else {}
        self.wm_control = bool(wm_control) and self.working_memory
        self.wm_cue = wmp.CueChannel()
        self.wm_sources = {}        # probe id -> source entity id
        self.wm_armed = set()       # probes whose cue has been given at least once by the course
        self.wm_marks = {}          # chest entity id -> mark tuple
        self.decisions = lambda: 0  # set by the env: decisions taken so far
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
        elif kind == "dive" and self.working_memory:
            # WM-TB09: six marked chests on a wider pool floor; one holds the item.
            self.flat_patch(p, -4.5, radius=WM_POOL_RADIUS)
            probe = self.wm[wmp.TB09]
            marks, target = wmp.chest_marks(probe, self.seed)
            for k, mark in enumerate(marks):
                a = k * np.pi / 3
                eid = self.entity(op, f"chest{k+1}", EntityKind.CHEST,
                                  p + WM_CHEST_RING * np.array([np.cos(a), np.sin(a)]),
                                  label="Sunken chest", visual_role="wm_marked", visual_version=wmp.mark_code(mark))
                self.wm_marks[eid] = mark
            probe.set_expected((op.entities[target],))
            op.state["wm"] = wmp.TB09
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

    def make_wm(self, kind, title, probe_id, site, **state):
        """A working-memory operation at an explicit site; consumes no grid cell and no course RNG."""
        op = Operation(f"wm{len(self.milestones)+1:02d}_{kind}", kind, title, state=dict(state))
        p = np.asarray(site, dtype=float)
        op.state.update(site=p.tolist(), wm=probe_id)
        probe = self.wm[probe_id]
        if kind == "cue":
            eid = self.entity(op, "source", EntityKind.NPC, p, label=title, visual_role="wm_boatman")
            self.wm_sources[probe_id] = eid
        elif kind == "panel":
            symbols = wmp.LEVER_COLOURS if probe_id == wmp.TA06 else wmp.PAWL_ELEMENTS
            role = "wm_lever" if probe_id == wmp.TA06 else "wm_pawl"
            noun = "lift lever" if probe_id == wmp.TA06 else "winch pawl"
            op.state["symbols"] = {}
            for k, symbol in enumerate(symbols):
                xz = p + ((k - (len(symbols) - 1) / 2) * WM_PANEL_SPACING, 0.0)
                eid = self.entity(op, f"{role[3:]}_{symbol}", EntityKind.PILLAR, xz,
                                  label=f"{symbol.title()} {noun}", visual_role=role, visual_version=k + 1)
                op.state["symbols"][eid] = symbol
            # The housing names the fixture; it is not an entry and not in the operation's entities.
            housing = f"{op.id}:housing"
            kind_ = EntityKind.ALTAR if probe_id == wmp.TA06 else EntityKind.PILLAR
            self.world.spawn_at(housing, kind_, float(p[0]), float(p[1] + 4.0), radius=2.1,
                                label="Lift test station" if probe_id == wmp.TA06 else "Causeway winch",
                                visual_role="wm_station" if probe_id == wmp.TA06 else "wm_winch")
            op.state["housing"] = housing
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
        if self.working_memory:
            mechanic=self.make("exchange","Ferry mechanic",requires={"casing":1},wm_cue=wmp.TA06)
            self.add("Repair the ferry lift",mechanic,
                     self.make_wm("panel","Lift lever bank",wmp.TA06,WM_SITES["a"]["levers"],grants="ferry_open"))
            self.wm_sources[wmp.TA06]=mechanic.entities[0]
        else:
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
        if self.working_memory:
            self.add("Dive for the coupling",self.make_wm("cue","Ferry boatman",wmp.TB09,WM_SITES["b"]["boatman"]),
                     self.make("dive","Submerged coupling",item="coupling"))
            mechanic=self.make("exchange","Causeway mechanic",requires={"kit":1,"coupling":1},grants="causeway_open",wm_cue=wmp.TB10)
            self.add("Reopen the causeway",mechanic,self.make("survey","Cross the repaired causeway"),
                     self.make_wm("panel","Causeway winch",wmp.TB10,WM_SITES["b"]["winch"]))
            self.wm_sources[wmp.TB10]=mechanic.entities[0]
        else:
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
        if op.kind=="relay":
            s=op.state
            role="drain" if s["overloaded"] or (s["version"]==2 and not s["drained"]) else (
                 "source" if s["progress"]==0 else "vane" if s["progress"]==1 else "core")
            return self.world.get(s["roles"][role])
        if op.kind=="crossing":
            k=op.state["leg"] + int(op.state["launched"])
            return self.world.get(op.entities[min(k,2)])
        if op.kind=="panel":
            # Privileged: the next lever/pawl of the expected entry (the reference reads this).
            probe=self.wm[op.state["wm"]]
            if not probe.solved:
                want=probe.expected[len(probe.buffer)]
                return self.world.get(next(e for e,v in op.state["symbols"].items() if v==want))
        if op.kind=="dive" and op.state.get("wm"):
            return self.world.get(self.wm[op.state["wm"]].expected[0])
        return self.world.get(op.entities[0])

    def local_entities(self):
        return self.op.entities if self.op else []

    # Observable step description (no coordinates or rule tables).
    _STEP_PHRASES={"lesson":"talk to the instructor","relay":"restore the relay","survey":"visit the landmark",
                   "collect":"recover the item","exchange":"hand over or use mission items",
                   "fight":"defeat the guard","sentinel":"defeat the sentinel","crossing":"cross by wind vent",
                   "dive":"dive to recover the item","lift":"ride the service lift",
                   "cue":"get directions","panel":"set the mechanism"}
    _LESSON_TITLES={"Learn R":"relay lesson","Learn G":"sentinel lesson","Learn T":"wind lesson"}

    def step_text(self):
        op=self.op
        if op is None:
            return ""
        title=self._LESSON_TITLES.get(op.title,op.title)
        return (f"Step {self.operation_index+1}/{len(self.current.operations)}: "
                f"{self._STEP_PHRASES[op.kind]} ({title})")

    # Same proximity tests that interact/apply_element use below.
    @staticmethod
    def _in_reach(entity,char,radius,height):
        d=entity.pos-char.pos
        return float(np.linalg.norm(d[[0,2]]))<=radius and abs(float(d[1]))<height

    def interact_hint(self,char):
        """Label of a current-step entity that interact would act on, else None."""
        op=self.op
        if op is None or op.kind not in ("relay","lesson","survey","collect","exchange","dive","lift","cue","panel"):
            return None
        near=[self.world.get(eid) for eid in op.entities]
        near=[e for e in near if self._in_reach(e,char,e.radius,2.8)]
        return min(near,key=lambda e:float(np.linalg.norm(e.pos-char.pos))).label if near else None

    def _off_step_entity(self,char,radius=None,height=2.8,kinds=None):
        """Nearest course entity in reach that belongs to a different operation."""
        current=set(self.op.entities) if self.op else set()
        near=[]
        for m in self.milestones:
            for op in m.operations:
                if kinds is not None and op.kind not in kinds:
                    continue
                for eid in op.entities:
                    e=self.world.get(eid)
                    if eid not in current and e.alive and self._in_reach(e,char,e.radius if radius is None else radius,height):
                        near.append(e)
        return min(near,key=lambda e:float(np.linalg.norm(e.pos-char.pos))) if near else None

    def off_step_feedback(self,entity,what):
        self.feedback(f"{entity.label}: no response; {what} is not part of the current step.",
                      entity=entity.id,reason="off_step")

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
            if self._wm_reread(char):
                return
            other=self._off_step_entity(char)
            if other is not None:
                self.off_step_feedback(other,"this object")
            return
        entity=min(candidates,key=lambda e:float(np.linalg.norm(e.pos-char.pos)))
        self.emit("interact",entity=entity.id,operation=op.id)
        if op.kind=="cue":
            self.feedback(f"{entity.label} speaks.")
            self._wm_read(op.state["wm"],entity.id)
            entity.used=True
            self.finish_operation()
            return
        if op.kind=="panel":
            self._wm_enter(op,entity)
            return
        if op.kind=="relay":
            s=op.state
            if entity.id!=s["roles"]["core"]:
                self.feedback("Elemental receiver: cast a skill to apply an element.")
            elif s["progress"]==2 and not s["overloaded"] and (s["version"]==1 or s["drained"]):
                s["active"]=True
                entity.used=True
                self.emit("relay_active",operation=op.id,version=s["version"])
                self.finish_operation()
            else:
                self.feedback("Core blocked: inspect progress and residual charge.")
        elif op.kind in ("lesson","survey","collect","exchange","dive","lift"):
            if not self.can_use_items(op):
                self.feedback("Required mission items are missing.")
                return
            if op.kind=="dive" and (char.pos[1]>-2.8 or char.state!=MoveState.SWIMMING):
                self.feedback("Dive to the chest; surface interaction cannot recover it.")
                return
            if op.kind=="dive" and op.state.get("wm"):
                self._wm_open_chest(op,entity)
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
            if op.state.get("wm_cue"):
                # The exchange is done; the mechanic now gives the cue for the next step.
                self.feedback(f"{entity.label} takes the mission items and speaks.")
                self._wm_read(op.state["wm_cue"],entity.id)

    # ------------------------------------------------------- working memory
    def _wm_emit(self,payloads):
        for payload in payloads:
            self.events.append(wmp.wm_event(payload))

    def _wm_read(self,probe_id,eid):
        probe=self.wm[probe_id]
        self.wm_armed.add(probe_id)
        self._wm_emit([probe.read_cue(self.decisions(),self.world.elapsed)])
        self.wm_cue.post(probe_id,eid,wmp.cue_text(probe,compact=True),self.world.elapsed)

    def _wm_reread(self,char):
        """The source can always be revisited while its probe is unsolved."""
        for pid in sorted(self.wm_armed):
            probe=self.wm[pid]
            e=self.world.get(self.wm_sources[pid])
            if not probe.solved and self._in_reach(e,char,e.radius,2.8):
                self.emit("interact",entity=e.id,operation=None)
                self.feedback(f"{e.label} speaks again.")
                self._wm_read(pid,e.id)
                return True
        return False

    def _wm_panel_reset(self,op,probe):
        for eid,symbol in op.state["symbols"].items():
            self.world.get(eid).used=probe.solved and symbol in probe.expected

    def _wm_enter(self,op,entity):
        pid=op.state["wm"]
        probe=self.wm[pid]
        symbol=op.state["symbols"].get(entity.id)
        if symbol is None:
            return
        events=probe.enter(symbol,self.decisions(),self.world.elapsed)
        self._wm_emit(events)
        entity.used=True
        house=self.world.get(op.state["housing"]).label
        if not any(e["type"]=="wm_attempt" for e in events):
            self.feedback(wmp.neutral_entry(entity.label,len(probe.buffer)),entity=entity.id)
            return
        self._wm_panel_reset(op,probe)
        if probe.solved:
            done={wmp.TA06:"the engine turns over; the ferry lift runs.",
                  wmp.TB10:"the drum turns and locks the causeway open."}[pid]
            self.feedback(f"{house}: {done}")
            self.finish_operation()
        else:
            failed={wmp.TA06:"the levers spring back; the engine does not turn over.",
                    wmp.TB10:"the pawls drop back; the drum does not turn."}[pid]
            self.feedback(f"{house}: {failed}")

    def _wm_open_chest(self,op,entity):
        probe=self.wm[op.state["wm"]]
        if entity.used:
            self.feedback(f"{entity.label}: open and empty.")
            return
        entity.used=True
        self._wm_emit(probe.choose(0,entity.id,self.decisions(),self.world.elapsed))
        if not probe.solved:
            self.feedback(f"{entity.label}: empty.")
            return
        item=op.state["item"]
        self.inventory[item]=self.inventory.get(item,0)+1
        self.emit("item_acquired",item=item,entity=entity.id)
        self.feedback(f"{entity.label}: the ferry coupling is inside.")
        self.finish_operation()

    def _wm_use_site(self,pid):
        op=next(o for m in self.milestones for o in m.operations
                if o.state.get("wm")==pid and o.kind in ("panel","dive"))
        return op

    def _wm_tick(self,char):
        self.wm_cue.update(self.world,char,self.world.elapsed)
        op=self.op
        if op is not None and op.kind=="panel":
            probe=self.wm[op.state["wm"]]
            if probe.buffer and np.hypot(*(char.pos[[0,2]]-np.asarray(op.state["site"])))>wmp.WALK_AWAY:
                probe.clear()                      # walked away: the partial entry springs back
                self._wm_panel_reset(op,probe)

    def local_text(self,char):
        """Policy-visible local text: feedback, plus the cue and marks while they are in reach."""
        if not self.working_memory:
            return self.last_feedback
        parts=[self.last_feedback]
        cue=self.wm_cue.text(self.world,char,self.world.elapsed)
        if cue:
            parts.append(cue)
        near=[]
        for eid,mark in self.wm_marks.items():
            e=self.world.get(eid)
            if self._in_reach(e,char,WM_MARK_RANGE,2.8):
                near.append((float(np.linalg.norm(e.pos-char.pos)),f"{e.label} mark: {wmp.mark_text(mark)}"))
        op=self.op
        if op is not None and op.kind=="panel":
            for eid in op.state["symbols"]:
                e=self.world.get(eid)
                if self._in_reach(e,char,e.radius,2.8):
                    near.append((float(np.linalg.norm(e.pos-char.pos)),f"Within reach: {e.label}"))
        if near:
            parts.append(min(near)[1])
        if self.wm_control:
            for pid,probe in self.wm.items():
                site=np.asarray(self._wm_use_site(pid).state["site"])
                if not probe.solved and np.hypot(*(char.pos[[0,2]]-site))<=10.0:
                    parts.append(wmp.control_text(probe,compact=True))
        return " | ".join(p for p in parts if p)

    def wm_summary(self):
        """Evaluator-only: per-probe scores plus where the source and use site are."""
        out={}
        for pid,probe in self.wm.items():
            source=self.world.get(self.wm_sources[pid]).pos
            site=np.asarray(self._wm_use_site(pid).state["site"])
            use=np.array([site[0],float(self.world.terrain.height_at(*site)),site[1]])
            out[pid]=dict(probe.summary(),source=[round(float(v),3) for v in source],
                          use_site=[round(float(v),3) for v in use],
                          distance=round(float(np.linalg.norm((use-source)[[0,2]])),2),
                          content=list(probe.items),expected=list(probe.expected))
        return out

    def apply_element(self,char,element):
        op=self.op
        current=op is not None and op.kind in ("relay","crossing")
        candidates=[self.world.get(e) for e in op.entities] if current else []
        candidates=[e for e in candidates if np.linalg.norm((e.pos-char.pos)[[0,2]])<2.6 and abs(float(e.pos[1]-char.pos[1]))<4.0]
        if not candidates:
            # Elemental receivers of other steps respond visibly instead of silently ignoring casts.
            other=self._off_step_entity(char,radius=2.6,height=4.0,kinds=("relay","crossing"))
            if other is not None and op is not None:
                self.off_step_feedback(other,"this receiver")
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
        role=next(k for k,v in s["roles"].items() if v==entity.id)
        valid=False
        if role=="drain" and element==self.relay_elements["drain"]:
            s["overloaded"]=False
            s["drained"]=True
            valid=True
        elif role=="source" and element==self.relay_elements["source"] and not s["overloaded"] and (s["version"]==1 or s["drained"]) and s["progress"]==0:
            s["progress"]=1
            valid=True
            if s.get("interrupt") and not s["interrupted"]:
                s["interrupted"]=True
                s["overloaded"]=True
                self.emit("relay_interrupted",operation=op.id,preserved_progress=1)
        elif role=="vane" and element==self.relay_elements["vane"] and s["progress"]==1 and not s["overloaded"]:
            s["progress"]=2
            valid=True
        if valid:
            entity.used=True
            self.emit("relay_application",operation=op.id,role=role,element=element.name.lower(),progress=s["progress"],overloaded=s["overloaded"])
            self.feedback(f"{role}: accepted; progress={s['progress']}; overload={s['overloaded']}")
        else:
            self.feedback("Application rejected: inspect role, stage, and residual charge.")
            self.emit("invalid_application",operation=op.id,role=role,element=element.name.lower())

    def sentinel_hit(self,enemy,damage,element):
        op=self.op
        if op is None or enemy.id not in op.entities:
            return []
        if op.kind!="sentinel":
            return None
        s=op.state
        if s["phase"]=="shield" and element==self.guard_elements[0]:
            s.update(phase="exposed",timer=0.0)
            enemy.elemental.shield=0.0
            self.emit("sentinel_exposed",operation=op.id)
        elif s["phase"]=="windup" and element==self.guard_elements[1]:
            if s.get("reform") and not s["reformed"]:
                s.update(phase="shield",timer=0.0,reformed=True)
                enemy.elemental.shield=50.0
                self.emit("sentinel_reformed",operation=op.id)
            else:
                s.update(phase="vulnerable",timer=0.0)
                self.emit("sentinel_interrupted",operation=op.id)
        elif s["phase"]=="vulnerable" and element==Element.NONE:
            enemy.hp=max(0.0,enemy.hp-damage)
            self.emit("enemy_hit",entity=enemy.id,damage=damage,hp=enemy.hp)
            if enemy.hp==0:
                enemy.alive=False
                self.emit("defeat",entity=enemy.id)
        else:
            self.emit("sentinel_blocked",operation=op.id,phase=s["phase"])
        return []

    def tick(self,char,dt):
        if self.working_memory:
            self._wm_tick(char)
        op=self.op
        if op is None:
            return
        s=op.state
        if op.kind in ("sentinel","fight"):
            e=self.world.get(op.entities[0])
            if not e.alive:
                self.finish_operation()
            elif op.kind=="sentinel":
                s["timer"]+=dt
                if s["phase"]=="exposed":
                    e.elemental.shield=0
                    if s["timer"]>=0.8:
                        s.update(phase="windup",timer=0.0)
                        self.emit("sentinel_windup",operation=op.id)
                elif s["phase"]=="windup" and s["timer"]>=18:
                    s.update(phase="shield",timer=0.0)
                    e.elemental.shield=50
                    self.emit("sentinel_reset",operation=op.id)
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
        state={"task":self.task,"milestone_index":self.index,"operation_index":self.operation_index,
               "inventory":dict(self.inventory),"completed":list(self.completed),
               "operations":{o.id:{"kind":o.kind,"complete":o.complete,"state":o.state}
                             for m in self.milestones for o in m.operations}}
        if self.working_memory:
            state["working_memory"]={pid:p.state() for pid,p in self.wm.items()}
            state["wm_cue"]=self.wm_cue.state()
            state["wm_armed"]=sorted(self.wm_armed)
        return state

    def verify(self):
        failures=[]
        for m in self.milestones:
            for op in m.operations:
                if not op.complete:
                    failures.append(f"{m.id}: {op.id} incomplete")
                if op.kind=="relay" and not op.state.get("active"):
                    failures.append(f"{m.id}: relay inactive")
                if op.kind in ("fight","sentinel") and self.world.get(op.entities[0]).alive:
                    failures.append(f"{m.id}: enemy alive")
                if op.kind=="crossing" and (op.state["landings"]!=2 or op.state["glide_legs"]!=2):
                    failures.append(f"{m.id}: two demonstrated vent/glide landings not verified")
        for pid,probe in self.wm.items():
            if not probe.solved:
                failures.append(f"{probe.milestone}: working-memory probe {pid} unsolved")
        if len(self.completed)!=len(self.milestones):
            failures.append("milestone count mismatch")
        return {"success":not failures,"failures":failures,"completed":len(self.completed),
                "required":len(self.milestones),"scope":"compact native-simulator reference prototype"}


class CourseCombat(CombatSystem):
    def cast(self,char,controller,*,burst=False):
        events=super().cast(char,controller,burst=burst)
        for ev in events:
            if ev.type in ("skill","burst"):
                self.world.course.apply_element(char,Element[ev.get("element").upper()])
        return events

    def _hit(self,enemy,char,damage,element,charge=1.0):
        result=self.world.course.sentinel_hit(enemy,damage,element)
        if result is not None:
            return result
        return super()._hit(enemy,char,damage,element,charge)


class CourseWorld(World):
    def interact(self,char):
        self.course.interact(char)
        return []

    def _hint(self,char):
        # The proprio interaction hint means "interact would act on the current step here";
        # objects of other steps no longer raise it.
        label=self.course.interact_hint(char)
        return f"Interact: {label}" if label else None

    def _update_enemy(self,e,char,dt):
        op=self.course.op
        if not op or e.id not in op.entities:
            return []
        # Sentinels hold an authored arena position; ordinary camp enemies pursue.
        if op.kind=="sentinel":
            return []
        return super()._update_enemy(e,char,dt)

    def update(self,char,dt,char_cfg):
        events=super().update(char,dt,char_cfg)
        self.course.tick(char,dt)
        events.extend(self.course.events)
        self.course.events.clear()
        return events


class ReferenceTaskEnv(OpenWorldEnv):
    """Native-action demonstration env with compact authored course geometry."""

    def __init__(self,task="a",seed=7,*,render_enabled=True,width=384,height=216,working_memory=True,wm_control=False):
        self.task=task
        self.render_enabled=render_enabled
        # working_memory=False rebuilds the pre-2026-11-24 course, so archived traces still replay.
        self.working_memory=bool(working_memory)
        self.wm_control=bool(wm_control)
        cfg=EnvConfig(seed=seed,text_obs=True,backend="soft",action_repeat=6,max_episode_steps=30000,
                      world=WorldConfig(terrain=TerrainConfig(size=640,resolution=321,max_height=65),
                                        prop_density=0,day_length=100000,start_time_of_day=.43),
                      camera=CameraConfig(distance=9,far=160),
                      render=RenderSettings(width=width,height=height,march_steps=48,refine_steps=4,
                                            shadow_steps=0,prop_draw_distance=65,max_view_distance=160,
                                            ground_cover=False,clouds=False))
        cfg.reward.shaping=0
        cfg.reward.discovery=0
        super().__init__(cfg)
        from gymnasium import spaces
        import string
        self.observation_space["local_text"]=spaces.Text(max_length=1024,charset=string.printable)

    def _build(self,seed):
        self.rng=np.random.default_rng(seed)
        terrain=Terrain(replace(self.cfg.world.terrain,seed=seed))
        # Author a coastal plateau; this is level construction at reset, not a policy action.
        axis=np.linspace(-terrain._half,terrain._half,terrain.cfg.resolution)
        x,z=np.meshgrid(axis,axis,indexing="ij")
        edge=np.clip((np.maximum(np.abs(x),np.abs(z))-255)/50,0,1)
        terrain.height[:]=12.0-edge*24
        self.party=Party(self.cfg.party)
        empty=PropField(np.empty((0,3),np.float32),np.empty(0,np.float32),np.empty(0,np.float32),np.empty(0,np.int8))
        self.world=CourseWorld(replace(self.cfg.world,seed=seed),terrain=terrain,props=empty,party=self.party)
        self.course=Course(self.world,self.task,seed,working_memory=self.working_memory,wm_control=self.wm_control)
        self.course.decisions=self._decisions
        self.world.course=self.course
        terrain._compute_normals(); terrain._compute_biomes()
        # Refresh heights after all authored shelf/pool patches are constructed.
        for entity in self.world.entities.values():
            entity.pos[1]=terrain.height_at(entity.pos[0],entity.pos[2])
            entity.home=entity.pos.copy()
        self.world.combat=CourseCombat(self.world,self.party)
        stages=[Stage([Objective(m.id,ObjKind.ACTIVATE,m.title,targets=(m.id,))]) for m in self.course.milestones]
        self.graph=QuestGraph([Quest(f"reference_{self.task}",f"Task {self.task.upper()}","Reference course",stages,reward=0)])
        self.quests=QuestEngine(self.graph)
        self.controller=CharacterController(terrain,empty,replace(self.cfg.character))
        self.camera_rig=OrbitCamera(terrain,empty,self.cfg.camera)
        target=self.course.target()
        self.char=self.controller.spawn(float(target.pos[0]),float(target.pos[2])+5)
        from .architecture import populate_architecture
        architecture=populate_architecture(self.world,self.char.pos)
        self.controller.architecture=self.camera_rig.architecture=architecture
        self.cam=self.camera_rig.reset(self.char)
        self.cam.pitch=.34
        self.steps=0
        self._prev_dist=None
        self._used_entities=set()
        self._episode_reward=0.0
        self._last_events=[]
        self._last_frame=None

    def _decisions(self):
        return self.steps

    def _observe(self):
        if self.render_enabled:
            obs=super()._observe()
        else:
            obs={"rgb":np.zeros((self.cfg.render.height,self.cfg.render.width,3),np.uint8),
                 "proprio":self._proprio(),"task":self._task_vector(),"party":self.party.as_vector(),
                 "instruction":self.quests.instruction()}
        # Local dialogue/feedback is observable, but rule tables and target coordinates are private.
        obs["local_text"]=self.course.local_text(self.char)
        step=self.course.step_text()
        if step and "instruction" in obs:
            obs["instruction"]=f"{obs['instruction']} | {step}"
        return obs

    def _info(self,events=None):
        result=super()._info(events)
        result["inventory"]=dict(self.course.inventory)
        result["reference_course"]={"milestone_index":self.course.index,"operation_index":self.course.operation_index}
        return result

    def state_digest(self):
        state={"course":self.course.state(),"position":self.char.pos.tolist(),
               "party":self.party.summary(),"time":self.world.elapsed,"quest":self.quests.snapshot(),
               "entities":{e.id:{"position":e.pos.tolist(),"hp":e.hp,"alive":e.alive,"used":e.used}
                           for e in self.world.entities.values()}}
        return hashlib.sha256(json.dumps(state,sort_keys=True,default=lambda v:int(v),allow_nan=False).encode()).hexdigest()
