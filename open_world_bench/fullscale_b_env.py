"""Full-scale Teyvat Task B, *The Expedition of Three Arts*, as a native environment.

The course runs the 24 legacy milestones on the expedition island
(:mod:`open_world_bench.fullscale_b_world`) with the mechanisms of
:mod:`open_world_bench.fullscale_mechanics`. Design and rules:
``docs/teyvat-full-scale-task-b.md``.

The agent sees RGB, proprioception, the party, a constant goal and
``local_text`` (pack, dialogue in earshot, mechanism feedback, visible cues).
Rule assignments, receipts and positions stay private. Only the reference
controller and the evaluator read :attr:`TeyvatExpeditionEnv.course`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import zlib

import numpy as np

from .camera import CameraConfig, OrbitCamera
from .character import MoveState
from .elements import Element
from .env import EnvConfig, OpenWorldEnv
from .events import Event
from .fullscale_b_world import (CAUSEWAY_OPEN, CRATER_GATE_OPEN, CROWN_SIGNAL, OUTPOST_CLEAR, RESERVE_POWER,
                                SKYLIFT_POWER, SUITE_VERSION, SUMMIT_POWER, build_world)
from .fullscale_mechanics import (COLUMN_RADIUS, PULSE_DAMAGE, PULSE_RADIUS, RISE_SPEED, SLAM_DAMAGE, SLAM_RADIUS,
                                  WAKE_RADIUS, RelayUnit, SentinelUnit, VentState, stable_unit)
from .fullscale_physics import PHYSICS_VERSION, FullScaleCombat, FullScaleController
from .party import Party
from .quests import ObjKind, Objective, Quest, QuestEngine, QuestGraph, Stage
from .render import RenderSettings
from .world import HOSTILE, EntityKind, World, WorldConfig

COURSE_VERSION = "teyvat-fullscale-b-course-v2"
GOAL = ("Reopen the valley's expedition route, recover the beacon heart, and bring the distant outpost online. "
        "Train with the relay keeper, sentinel warden, and wind guide before setting out.")
INTERACT_REACH = 3.0
INTERACT_HEIGHT = 2.8
EARSHOT = 4.0
DIALOGUE_LINGER = 5.0
LABEL_RANGE = 24.0
CUE_RANGE = 8.0
STATUE_DISCOVER = 10.0
REACH_RADIUS = 6.0
CONTEXT_BOUNDARIES = ("B11", "B21", "B24")
SENTINEL_GLYPH, VENT_GLYPH = "crescent", "spiral"
ITEM_NAMES = dict(kit="causeway repair kit", coupling="ferry coupling", field_seal="warden's field seal",
                  tools="expedition service tools", shrine_seal="shrine access seal", heart="beacon heart")


@dataclass
class Op:
    id: str
    kind: str                   # talk | relay | sentinel | reach | survey | collect | exchange | signal
    target: str                 # entity id, relay id, sentinel id or anchor
    params: dict = field(default_factory=dict)
    complete: bool = False


@dataclass
class Milestone:
    id: str
    title: str
    ops: list
    lead: tuple = ("", "")      # (speaker label, lead text) delivered on completion


class ExpeditionCourse:
    """Private task state and rules. Only world actions change it."""

    def __init__(self, env, seed: int):
        self.env = env
        self.world = env.world
        self.fw = env.fullscale
        rng = np.random.default_rng([int(seed), 0xB0B])        # rule stream, not the simulation RNG
        elements = [Element.EMBER, Element.TIDE, Element.GALE]
        perm = rng.permutation(3)
        self.relay_elements = {role: elements[int(k)] for role, k in zip(("source", "vane", "drain"), perm)}
        pair = rng.permutation(3)[:2]
        self.guard_elements = (elements[int(pair[0])], elements[int(pair[1])])
        self.vent_element = elements[int(rng.integers(3))]
        self.safe_beat = ("bright", "dark")[int(rng.integers(2))]
        self.outpost_version = int(rng.integers(1, 3))
        self.pack: dict[str, int] = {}
        self.relays: dict[str, RelayUnit] = {}
        self.sentinels: dict[str, SentinelUnit] = {}
        self.vents = {v.id: VentState(v.id, v.route, v.leg, v.pad, v.pad_height, v.apex, v.requires,
                                      offset=stable_unit(seed, v.id) * 4.0) for v in self.fw.vents}
        self.roles: dict[str, tuple] = {}          # entity id -> (kind, owner id, role)
        self.statues: dict[str, bool] = {}
        self.milestones: list[Milestone] = []
        self.index = 0
        self.op_index = 0
        self.events: list[Event] = []
        self.completed: list[dict] = []
        self.dialogue = ("", "", -1e9)             # (speaker entity, text, time)
        self.feedback = ("Reference course initialized.", -1e9)
        self.leads: dict[str, str] = {}            # speaker entity -> latest lead
        self.assisted_recall = 0
        self.deaths = 0
        self.launch = None                         # (vent id, x, z) while a vent carries the character
        self._was_grounded = True
        self.min_hp = float("inf")
        self.exposures: list[dict] = []

    # ------------------------------------------------------------------ setup
    def anchor(self, name):
        return self.fw.layout.point(name)

    def spawn(self, eid, kind, xz, role, owner, label, **kw):
        self.world.spawn_at(eid, kind, float(xz[0]), float(xz[1]), radius=2.1, label=label, **kw)
        self.roles[eid] = (role, owner)
        return eid

    def relay(self, rid, anchor, label, family, version=1, radius=7.0, **state):
        centre = self.anchor(anchor)
        angles = np.radians([45.0, 135.0, 225.0, 315.0]) + stable_unit(self.fw.seed, rid, "turn") * np.pi / 2
        order = np.random.default_rng([self.fw.seed, zlib.crc32(rid.encode())]).permutation(4)
        receivers = {}
        for role, k in zip(("source", "vane", "drain", "core"), order):
            xz = centre + radius * np.array([np.cos(angles[k]), np.sin(angles[k])])
            kind = EntityKind.BEACON if role == "core" else EntityKind.PILLAR
            receivers[role] = self.spawn(f"{rid}:{role}", kind, xz, "receiver", rid,
                                         f"{label} {role}", visual_role=role, visual_version=version)
        unit = RelayUnit(rid, label, family, version, receivers, **state)
        self.relays[rid] = unit
        return unit

    def sentinel(self, sid, anchor, label, hp, phase="shield", reform_once=False):
        xz = self.anchor(anchor)
        eid = self.spawn(f"{sid}:core", EntityKind.BOSS, xz, "sentinel", sid, label, hp=hp, max_hp=hp,
                         damage=0.0, aggro_radius=0.0)
        unit = SentinelUnit(sid, eid, label, hp, phase=phase, hold=phase != "shield", reform_once=reform_once)
        self.sentinels[sid] = unit
        self._shield_visual(unit)
        return unit

    def _shield_visual(self, unit):
        e = self.world.get(unit.entity)
        e.elemental.shield = 50.0 if unit.phase == "shield" else 0.0

    def add(self, mid, title, ops, lead=("", "")):
        for k, op in enumerate(ops):
            op.id = f"{mid}.{k + 1}"
        self.milestones.append(Milestone(mid, title, ops, lead))

    # ---------------------------------------------------------------- queries
    @property
    def current(self):
        return self.milestones[self.index] if self.index < len(self.milestones) else None

    @property
    def op(self):
        m = self.current
        return m.ops[self.op_index] if m is not None else None

    @property
    def facts(self):
        return self.fw.state.facts

    def live_relay(self):
        op = self.op
        return self.relays[op.target] if op is not None and op.kind == "relay" else None

    def relay_locked(self, unit) -> str | None:
        op = self.op
        need = op.params.get("unlock") if op is not None and op.kind == "relay" and op.target == unit.id else None
        if need and not op.params.get("unlocked"):
            return need
        return None

    def live_sentinel(self, sid) -> bool:
        op = self.op
        if op is None or op.kind != "sentinel" or op.target != sid:
            return False
        return all(self.pack.get(k, 0) >= n for k, n in op.params.get("wake", {}).items())

    def vent_powered(self, vent) -> bool:
        return vent.requires is None or vent.requires in self.facts

    # ----------------------------------------------------------------- events
    def emit(self, event_type, **payload):
        self.events.append(Event(event_type, payload))

    def say(self, text, **payload):
        self.feedback = (text, self.world.elapsed)
        self.emit("mechanism_feedback", text=text, **payload)

    def speak(self, speaker, text):
        self.dialogue = (speaker, text, self.world.elapsed)
        self.emit("talk", entity=speaker, text=text)

    def give(self, item, source):
        self.pack[item] = self.pack.get(item, 0) + 1
        self.emit("item_acquired", item=item, entity=source)
        self.say(f"Took the {ITEM_NAMES.get(item, item)}.")

    def take(self, items):
        for k, n in items.items():
            self.pack[k] -= n
            if self.pack[k] <= 0:
                del self.pack[k]
            self.emit("item_consumed", item=k, amount=n)

    def open_fact(self, fact):
        if fact not in self.facts:
            self.facts.add(fact)
            self.emit("fact_opened", fact=fact)

    # ------------------------------------------------------------- completion
    def finish_op(self):
        op = self.op
        if op is None or op.complete:
            return
        op.complete = True
        grants = op.params.get("grants")
        if grants:
            if grants in ITEM_NAMES:
                self.give(grants, op.id)
            else:
                self.open_fact(grants)
        self.emit("operation_complete", operation=op.id, kind=op.kind)
        self.op_index += 1
        m = self.current
        if self.op_index < len(m.ops):
            return
        self.completed.append(dict(id=m.id, time=round(self.world.elapsed, 6)))
        self.emit("activate", entity=m.id)
        self.emit("milestone_complete", milestone=m.id, title=m.title)
        speaker, text = m.lead
        if text:
            self.leads[speaker] = text
            self.speak(speaker, text)
            self.emit("lead", milestone=m.id, speaker=speaker, text=text)
        self.index += 1
        self.op_index = 0
        if self.current is not None and self.current.id in CONTEXT_BOUNDARIES:
            self.emit("designed_context_boundary", milestone=self.current.id,
                      note="Fresh agent context point; the oracle reference keeps none.")

    # ------------------------------------------------------------- interact
    def _nearest(self, char, reach=INTERACT_REACH, height=INTERACT_HEIGHT, roles=None):
        best = None
        for eid, (role, owner) in self.roles.items():
            if roles is not None and role not in roles:
                continue
            e = self.world.get(eid)
            if not e.alive:
                continue
            d = float(np.hypot(*(e.pos - char.pos)[[0, 2]]))
            if d <= reach and abs(float(e.pos[1] - char.pos[1])) < height and (best is None or d < best[0]):
                best = (d, eid, role, owner)
        return best

    def interact(self, char):
        hit = self._nearest(char, roles=("npc", "plaque", "chest", "receiver", "exchange", "signal", "statue"))
        if hit is None:
            return
        _, eid, role, owner = hit
        self.emit("interact", entity=eid)
        op = self.op
        current = op is not None and op.target == owner
        if role == "statue":
            self.statues[eid] = True
            self.env.party.revive()
            char.hp = self.env.party.active.hp
            self.say("The statue's warmth restores the party.")
            self.emit("statue_heal", entity=eid)
        elif role == "npc":
            if current and op.kind in ("talk", "survey"):
                self.speak(eid, op.params["text"])
                self.finish_op()
            elif eid in self.leads:
                self.assisted_recall += 1
                self.speak(eid, self.leads[eid])
                self.emit("assisted_recall", entity=eid)
            else:
                self.speak(eid, f"{self.world.get(eid).label}: \"Not now - there is work elsewhere.\"")
        elif role == "plaque":
            text = self.world.get(eid).dialogue
            self.speak(eid, text)
            if current and op.kind == "survey":
                self.finish_op()
        elif role == "chest":
            self._open_chest(char, eid, owner)
        elif role == "receiver":
            self._touch_receiver(eid, owner)
        elif role == "exchange":
            self._exchange(eid, owner)
        elif role == "signal":
            if not current:
                self.say("The signal mast is cold.")
            elif not self.relays["relay_b19"].active:
                self.say("The landing signal has no charge: the shrine court's relay is dark.")
            else:
                self.world.get(eid).used = True
                self.say("The landing signal burns; across the Crown, relay inlets hum awake.")
                self.finish_op()

    def _open_chest(self, char, eid, owner):
        e = self.world.get(eid)
        spec = self.chests[owner]
        if e.used:
            self.say(f"{e.label}: empty.")
            return
        if spec.get("dive") and not (char.state == MoveState.SWIMMING and char.pos[1] < -2.5):
            self.say(f"{e.label}: it lies on the lake bed; dive to reach it.")
            return
        need = spec.get("fact")
        if need and need not in self.facts:
            self.say(spec["locked"])
            return
        if spec.get("sentinel") and self.world.get(self.sentinels[spec["sentinel"]].entity).alive:
            self.say(spec["locked"])
            return
        op = self.op
        if spec.get("only_current") and not (op is not None and op.target == owner):
            self.say(spec["locked"])
            return
        e.used = True
        self.emit("open_chest", entity=eid)
        self.give(spec["item"], eid)
        self._check_collect()

    def _touch_receiver(self, eid, rid):
        unit = self.relays[rid]
        role = next(r for r, e in unit.receivers.items() if e == eid)
        if unit.active:
            self.say(f"{unit.label} is already running.")
            return
        if self.live_relay() is not unit:
            self.say(f"{unit.label}: no charge reaches it.")
            return
        lock = self.relay_locked(unit)
        if lock:
            if self.pack.get(lock, 0) > 0 and role == "core":
                self.op.params["unlocked"] = True
                self.emit("housing_unlocked", relay=rid, item=lock)
                self.say(f"{unit.label}: the {ITEM_NAMES[lock]} opens the housing.")
            else:
                self.say(self.op.params.get("lock_text", f"{unit.label}: the housing is sealed."))
            return
        if role != "core":
            self.say(f"{unit.cue()}. Receiver: {role}; cast an element to apply it.")
            if unit.version == 2:
                self.say(f"{unit.cue()}. Service plate: double-ring inlets keep a residual charge.")
                self.emit("service_plate_read", relay=rid)
            return
        ok, text = unit.confirm()
        self.say(text)
        if ok:
            self.world.get(eid).used = True
            self.emit("relay_active", relay=rid, version=unit.version)
            self.finish_op()

    def _exchange(self, eid, owner):
        op = self.op
        e = self.world.get(eid)
        if op is None or op.kind != "exchange" or op.target != owner:
            self.say(self.exchanges[owner]["idle"])
            return
        need = op.params.get("requires", {})
        if not all(self.pack.get(k, 0) >= n for k, n in need.items()):
            missing = [ITEM_NAMES[k] for k, n in need.items() if self.pack.get(k, 0) < n]
            self.say(f"{e.label}: still missing the {', '.join(missing)}.")
            return
        if not op.params.get("keep"):
            self.take(need)
        e.used = True
        self.say(self.exchanges[owner]["done"])
        self.finish_op()

    # ------------------------------------------------------------- elements
    def apply_element(self, char, element, reach):
        """A skill landed: act on the nearest relay receiver or vent pad in reach."""
        best = None
        for eid, (role, owner) in self.roles.items():
            if role != "receiver":
                continue
            e = self.world.get(eid)
            d = float(np.hypot(*(e.pos - char.pos)[[0, 2]]))
            if d <= reach and abs(float(e.pos[1] - char.pos[1])) < 4.0 and (best is None or d < best[0]):
                best = (d, "receiver", eid, owner)
        for vent in self.vents.values():
            d = float(np.hypot(char.pos[0] - vent.pad[0], char.pos[2] - vent.pad[1]))
            if d <= reach and abs(float(char.pos[1]) - vent.pad_height) < 4.0 and (best is None or d < best[0]):
                best = (d, "vent", vent.id, vent.id)
        if best is None:
            return
        _, kind, eid, owner = best
        if kind == "vent":
            vent = self.vents[eid]
            if not self.vent_powered(vent):
                self.say(f"Vent ({VENT_GLYPH}): dark pad - no power.")
                self.emit("vent_unpowered", vent=eid)
            elif element != self.vent_element:
                self.say(f"Vent ({VENT_GLYPH}): the pad does not take that element.")
                self.emit("vent_rejected", vent=eid, element=element.name.lower())
            else:
                vent.charged, vent.charged_at = True, self.world.elapsed
                vent.charges += 1
                self.say(f"Vent ({VENT_GLYPH}): charged.")
                self.emit("vent_charged", vent=eid, element=element.name.lower())
            return
        unit = self.relays[owner]
        role = next(r for r, e in unit.receivers.items() if e == eid)
        if unit.active:
            self.say(f"{unit.label} is already running.")
            return
        if self.live_relay() is not unit:
            self.say(f"{unit.label}: no charge reaches it.")
            return
        if self.relay_locked(unit):
            self.say(self.op.params.get("lock_text", f"{unit.label}: the housing is sealed."))
            return
        was = unit.overloaded
        ok, text, payload = unit.apply(role, element, self.relay_elements)
        self.say(text)
        self.emit("relay_application" if ok else "invalid_application", accepted=ok, **payload)
        if ok and payload.get("interrupted"):
            self.emit("relay_interrupted", relay=unit.id, preserved_progress=unit.progress)
        if ok and role == "drain" and not was and unit.version == 1 and unit.unnecessary_drains:
            self.emit("unnecessary_drain", relay=unit.id)

    def sentinel_hit(self, enemy, damage, element, char):
        sid = self.roles[enemy.id][1]
        unit = self.sentinels[sid]
        if not self.live_sentinel(sid):
            self.emit("sentinel_inert", sentinel=sid)
            self.say(f"{unit.label}: inert; nothing answers.")
            return []
        if element == Element.NONE:
            if unit.phase == "vulnerable":
                enemy.hp = max(0.0, enemy.hp - damage)
                out = [Event("enemy_hit", {"entity": enemy.id, "hp": enemy.hp, "damage": damage, "element": "none"})]
                if enemy.hp <= 0.0:
                    enemy.alive = False
                    out.append(Event("defeat", {"entity": enemy.id, "kind": int(enemy.kind)}))
                return out
            self.emit("sentinel_blocked", sentinel=sid, phase=unit.phase)
            return []
        for kind, payload in unit.element_hit(element, self.guard_elements):
            self._sentinel_event(unit, kind, payload, char)
        self._shield_visual(unit)
        return []

    def _sentinel_event(self, unit, kind, payload, char):
        self.emit(kind, sentinel=unit.id, **payload)
        e = self.world.get(unit.entity)
        dist = float(np.hypot(*(e.pos - char.pos)[[0, 2]]))
        if kind == "sentinel_pulse":
            unit.pulses += 1
            if dist <= PULSE_RADIUS:
                self._hurt(char, PULSE_DAMAGE, unit.entity)
            self.say(f"{unit.label}: a warding pulse ripples out.")
        elif kind == "sentinel_slam":
            unit.slams += 1
            if dist <= SLAM_RADIUS:
                self._hurt(char, SLAM_DAMAGE, unit.entity)
            unit.set_phase("shield")
            self.say(f"{unit.label}: the wind-up breaks into a slam; the shield closes again.")
        elif kind == "sentinel_windup":
            self.say(f"{unit.label}: winding up - a rising whine.")
        elif kind == "sentinel_exposed":
            self.say(f"{unit.label}: the shield splits; the core shows.")
        elif kind == "sentinel_interrupted":
            self.say(f"{unit.label}: the wind-up stalls; the core cracks.")
        elif kind == "sentinel_reformed":
            self.say(f"{unit.label}: the shield reforms.")
        self._shield_visual(unit)

    def _hurt(self, char, amount, source):
        party = self.env.party
        damage = party.absorb(amount) if party is not None else amount
        damage = min(char.hp, damage)
        char.hp = max(0.0, char.hp - damage)
        self.emit("player_hit", entity=source, damage=damage)

    # ----------------------------------------------------------------- tick
    def tick(self, char, dt):
        elapsed = self.world.elapsed
        for vent in self.vents.values():
            if vent.expire(elapsed):
                self.emit("vent_discharged", vent=vent.id)
        self._vents(char, dt)
        for sid, unit in self.sentinels.items():
            e = self.world.get(unit.entity)
            if not e.alive or not self.live_sentinel(sid):
                continue
            dist = float(np.hypot(*(e.pos - char.pos)[[0, 2]]))
            for kind, payload in unit.tick(dt, dist <= WAKE_RADIUS, dist):
                self._sentinel_event(unit, kind, payload, char)
        for eid, (role, owner) in self.roles.items():
            if role == "statue" and not self.statues.get(eid):
                e = self.world.get(eid)
                if float(np.hypot(*(e.pos - char.pos)[[0, 2]])) <= STATUE_DISCOVER:
                    self.statues[eid] = True
                    self.emit("statue_discovered", entity=eid)
        self.min_hp = min(self.min_hp, float(char.hp))
        self._advance(char)

    def _vents(self, char, dt):
        elapsed = self.world.elapsed
        if self.launch is not None:
            vid, x, z = self.launch
            vent = self.vents[vid]
            char.pos[0], char.pos[2] = x, z
            char.pos[1] = min(vent.apex, float(char.pos[1]) + RISE_SPEED * dt)
            char.vel[:] = 0.0
            char.state, char.grounded = MoveState.AIRBORNE, False
            char.air_time = 0.0
            char._peak_fall_speed = 0.0
            if char.pos[1] >= vent.apex - 1e-6:
                self.launch = None
                self.emit("vent_apex", vent=vid, height=round(float(char.pos[1]), 3))
            self._was_grounded = False
            return
        jumped = (self._was_grounded and char.state == MoveState.AIRBORNE
                  and abs(float(char.vel[1]) - self.env.controller.cfg.jump_speed) < 0.6)
        self._was_grounded = bool(char.grounded or char.state == MoveState.GROUNDED)
        if not jumped:
            return
        for vent in self.vents.values():
            if not vent.charged or not vent.in_column(char.pos - char.vel * dt):
                continue
            if not self.vent_powered(vent):
                continue
            vent.charged = False
            if vent.safe(elapsed, self.safe_beat):
                vent.launches += 1
                self.launch = (vent.id, float(vent.pad[0]), float(vent.pad[1]))
                self.emit("vent_launch", vent=vent.id, route=vent.route, leg=vent.leg, beat=vent.beat(elapsed))
                self.say(f"Vent ({VENT_GLYPH}): the column throws you upward.")
            else:
                vent.sputters += 1                # an ordinary jump; the charge is spent
                self.emit("vent_sputter", vent=vent.id, beat=vent.beat(elapsed))
                self.say(f"Vent ({VENT_GLYPH}): it sputters on the {vent.beat(elapsed)} beat; the charge is spent.")
            return

    def _check_collect(self):
        op = self.op
        if op is not None and op.kind == "collect" and self.pack.get(op.params["item"], 0) > 0:
            self.finish_op()

    def _advance(self, char):
        """Complete the current operation when its world predicate holds (early work is credited)."""
        for _ in range(4):
            op = self.op
            if op is None:
                return
            before = (self.index, self.op_index)
            if op.kind == "relay" and self.relays[op.target].active:
                self.finish_op()
            elif op.kind == "sentinel" and not self.world.get(self.sentinels[op.target].entity).alive:
                self.finish_op()
            elif op.kind == "collect" and self.pack.get(op.params["item"], 0) > 0:
                self.finish_op()
            elif op.kind == "reach":
                p = self.anchor(op.target)
                ground = self.fw.terrain.height_at(*p)
                if (char.state == MoveState.GROUNDED
                        and float(np.hypot(char.pos[0] - p[0], char.pos[2] - p[1])) <= op.params.get("radius", REACH_RADIUS)
                        and abs(float(char.pos[1]) - float(ground)) < 3.0):
                    self.finish_op()
            if (self.index, self.op_index) == before:
                return

    # ----------------------------------------------------------- observation
    def local_text(self, char) -> str:
        now = self.world.elapsed
        parts = []
        pack = ", ".join(ITEM_NAMES.get(k, k) + (f" x{n}" if n > 1 else "") for k, n in sorted(self.pack.items()))
        parts.append(f"Pack: {pack or 'empty'}")
        speaker, text, when = self.dialogue
        if text and speaker in self.world.entities:
            e = self.world.get(speaker)
            if now - when <= DIALOGUE_LINGER or float(np.hypot(*(e.pos - char.pos)[[0, 2]])) <= EARSHOT:
                parts.append(f"{e.label}: {text}")
        text, when = self.feedback
        if text and now - when <= 8.0:
            parts.append(text)
        cues = []
        for unit in self.relays.values():
            core = self.world.get(unit.receivers["core"])
            if float(np.hypot(*(core.pos - char.pos)[[0, 2]])) <= CUE_RANGE + 8.0:
                cues.append(unit.cue())
        for unit in self.sentinels.values():
            e = self.world.get(unit.entity)
            if e.alive and float(np.hypot(*(e.pos - char.pos)[[0, 2]])) <= 16.0:
                state = unit.posture() if self.live_sentinel(unit.id) else "inert"
                cues.append(f"{unit.label} (emblem: {SENTINEL_GLYPH}): {state}")
        for vent in self.vents.values():
            if float(np.hypot(char.pos[0] - vent.pad[0], char.pos[2] - vent.pad[1])) <= CUE_RANGE \
                    and abs(float(char.pos[1]) - vent.pad_height) < 6.0:
                power = "" if self.vent_powered(vent) else ", dark pad"
                cues.append(f"Vent ({VENT_GLYPH}): {'charged' if vent.charged else 'uncharged'}{power}; "
                            f"beat {vent.beat(now)}")
        parts += cues[:3]
        labels = []
        for eid, (role, owner) in self.roles.items():
            if role in ("receiver",):
                continue
            e = self.world.get(eid)
            d = float(np.linalg.norm(e.pos - char.pos))
            if e.alive and d <= LABEL_RANGE:
                labels.append((d, e.label))
        if labels:
            parts.append("Nearby: " + "; ".join(l for _, l in sorted(labels)[:5]))
        return " | ".join(parts)[:1000]

    def state(self):
        return dict(index=self.index, op_index=self.op_index, pack=dict(self.pack), facts=sorted(self.facts),
                    completed=list(self.completed), deaths=self.deaths, launch=self.launch,
                    relays={k: dict(progress=u.progress, overloaded=u.overloaded, drained=u.drained, active=u.active,
                                    interrupted=u.interrupted, invalid=u.invalid,
                                    unnecessary_drains=u.unnecessary_drains) for k, u in self.relays.items()},
                    sentinels={k: dict(phase=u.phase, timer=round(u.timer, 6), hold=u.hold, reformed=u.reformed,
                                       pulse=round(u.pulse_timer, 6), wrong=u.wrong_casts, slams=u.slams)
                               for k, u in self.sentinels.items()},
                    vents={k: (v.charged, round(v.charged_at, 6), v.launches, v.sputters) for k, v in self.vents.items()},
                    statues=sorted(k for k, v in self.statues.items() if v),
                    ops=[op.params.get("unlocked", False) for m in self.milestones for op in m.ops])

    def verify(self):
        failures = []
        for m in self.milestones:
            for op in m.ops:
                if not op.complete:
                    failures.append(f"{op.id} incomplete")
        for rid, unit in self.relays.items():
            if not unit.active:
                failures.append(f"relay {rid} inactive")
        for sid, unit in self.sentinels.items():
            if self.world.get(unit.entity).alive:
                failures.append(f"sentinel {sid} alive")
        return dict(success=not failures, failures=failures, completed=len(self.completed),
                    required=len(self.milestones), deaths=self.deaths)


def build_course(course: ExpeditionCourse):
    """Entities, chests, exchanges and the 24 milestones."""
    c, w = course, course.world
    A = c.anchor

    def npc(eid, anchor, label, dx=0.0, dz=0.0, dialogue=""):
        p = A(anchor) + (dx, dz)
        return c.spawn(eid, EntityKind.NPC, p, "npc", eid, label, dialogue=dialogue)

    def plaque(eid, anchor, label, text, dx=0.0, dz=0.0):
        return c.spawn(eid, EntityKind.ALTAR, A(anchor) + (dx, dz), "plaque", eid, label, dialogue=text)

    def statue(eid, anchor, dx=6.0, dz=0.0):
        c.spawn(eid, EntityKind.WAYPOINT, A(anchor) + (dx, dz), "statue", eid, "Statue of the Winds")
        c.statues[eid] = False

    keeper = npc("keeper", "keeper", "Relay keeper")
    warden = npc("warden", "warden", "Sentinel warden")
    guide = npc("guide", "guide", "Wind guide")
    quartermaster = npc("quartermaster", "depot", "Expedition quartermaster", dx=-8.0, dz=-6.0)
    observer = npc("observer", "post", "Lookout observer", dx=3.0, dz=3.0)
    for name, anchor in (("statue_village", "spawn"), ("statue_depot", "depot_statue"), ("statue_cut", "cut_top"),
                         ("statue_skylift", "highland_rim"), ("statue_crater", "crater_inside"),
                         ("statue_crown", "crown_landing")):
        statue(name, anchor)
    w.get("statue_village").pos[:] = w.ground_pos(*(A("spawn") + (8.0, 0.0)))
    c.statues["statue_village"] = True

    # Chests and their locks.
    c.chests = {
        "camp_chest": dict(item="kit", locked=""),
        "lake_chest": dict(item="coupling", dive=True, locked=""),
        "reserve": dict(item="tools", fact=RESERVE_POWER, only_current=True,
                        locked="Reserve compartment: a dark seal glyph; it will not open without reserve power."),
        "reliquary": dict(item="heart", sentinel="b23_sentinel",
                          locked="Reliquary: the sentinel's ward holds the lid shut."),
    }
    c.spawn("camp_chest", EntityKind.CHEST, A("camp") + (6.0, 4.0), "chest", "camp_chest", "Raider strongbox")
    lake = A("lake")
    c.spawn("lake_chest", EntityKind.CHEST, lake + (4.0, 3.0), "chest", "lake_chest", "Sunken marked chest")
    c.spawn("reserve", EntityKind.CHEST, A("depot") + (9.0, 3.0), "chest", "reserve", "Reserve compartment")
    c.spawn("reliquary", EntityKind.CHEST, A("spire_top") + (-5.0, 3.0), "chest", "reliquary", "Reliquary")
    for k, offset in enumerate(((-8.0, -6.0), (9.0, -7.0), (-2.0, 10.0))):
        xz = A("camp") + offset
        eid = f"raider_{k + 1}"
        w.spawn_at(eid, EntityKind.ENEMY, float(xz[0]), float(xz[1]), radius=2.1, label="Camp raider",
                   hp=70.0, max_hp=70.0, damage=6.0, aggro_radius=14.0)

    # Exchanges.
    c.exchanges = {
        "mechanic": dict(idle="Causeway mechanic: \"Bring me the repair kit and the ferry coupling.\"",
                         done="Causeway mechanic: \"Kit and coupling in - the causeway gate is rising!\""),
        "controls": dict(idle="Sky-lift control house: the service hatch is bolted.",
                         done="Sky-lift control house: the tools free the hatch; the relay inside can be worked."),
        "socket": dict(idle="Outpost beacon socket: empty and dark.",
                       done="Outpost beacon socket: the heart seats with a low chime."),
    }
    c.spawn("mechanic", EntityKind.NPC, A("cut_mouth") + (-9.0, 4.0), "exchange", "mechanic", "Causeway mechanic")
    c.spawn("controls", EntityKind.ALTAR, A("skylift_house") + (0.0, -9.0), "exchange", "controls",
            "Sky-lift control house")
    c.spawn("socket", EntityKind.ALTAR, A("outpost"), "exchange", "socket", "Outpost beacon socket")
    c.spawn("landing_signal", EntityKind.BEACON, A("crown_landing") + (4.0, 0.0), "signal", "landing_signal",
            "Crown landing signal", visual_role="core", visual_version=2)
    depot_plaque = plaque("depot_plaque", "depot", "Depot ledger plaque",
                          "Expedition depot. Reserve compartment sealed until reserve power returns from the "
                          "upper valley.", dx=-4.0, dz=8.0)

    # Relays.
    c.relay("relay_b01", "relay_b01", "Training relay", "workshop")
    c.relay("relay_b02", "relay_b02", "Workshop relay", "workshop", overloaded=True, interrupt="prime")
    c.relay("relay_b11", "relay_b11", "Valley signal station", "field")
    c.relay("relay_b14", "relay_b14", "Reserve-power relay", "field", progress=1, overloaded=True)
    c.relay("relay_b16", "relay_b16", "Sky-lift relay", "field")
    c.relay("relay_b17", "relay_b17", "Crater-gate relay", "field")
    c.relay("relay_b19", "relay_b19", "Shrine-court relay", "shrine", version=2)
    c.relay("relay_b21", "relay_b21", "Crown east relay", "shrine", version=2)
    c.relay("relay_b22", "relay_b22", "Old Crown station", "field")
    c.relay("relay_b23", "relay_b23", "Summit-approach relay", "shrine", version=2, interrupt="prime")
    c.relay("relay_b24", "outpost", "Outpost beacon", "beacon", version=c.outpost_version, radius=7.5,
            interrupt="prime")

    # Sentinels.
    c.sentinel("b03_sentinel", "arena_b03", "Training sentinel", 160.0)
    c.sentinel("b04_sentinel", "arena_b04", "Arena sentinel", 160.0, reform_once=True)
    c.sentinel("b13_sentinel", "arena_b13", "Hollow sentinel", 220.0, phase="exposed")
    c.sentinel("b17_sentinel", "arena_b17", "Gate sentinel", 220.0, phase="windup")
    c.sentinel("b18_sentinel", "shrine", "Shrine sentinel", 220.0)
    c.sentinel("b23_sentinel", "spire_top", "Reliquary sentinel", 260.0)
    c.sentinel("b24_sentinel", "arena_b24", "Outpost sentinel", 260.0)

    lesson_r = ("Relay lesson: every relay has a source, a vane, a drain and a core. Read the lamps and the "
                f"overload flag. If it is overloaded, drain it with {c.relay_elements['drain'].name.lower()}. "
                f"Prime the source with {c.relay_elements['source'].name.lower()}, route the vane with "
                f"{c.relay_elements['vane'].name.lower()}, then confirm at the core by hand. A surge after "
                "priming overloads it again: drain, and the priming holds.")
    br, it = (e.name.lower() for e in c.guard_elements)
    lesson_g = (f"Sentinel lesson: the {SENTINEL_GLYPH} emblem shield breaks to {br}. Wait out the open core; "
                f"when it winds up, hit it with {it} before it slams, then strike the cracked core. A wrong element "
                "brings a pulse or the slam. If the shield closes again, start over.")
    lesson_t = (f"Wind lesson: the {VENT_GLYPH} vents take {c.vent_element.name.lower()}. Charge the pad, then jump "
                f"inside the column on the {c.safe_beat} beat; the column carries you to its top. Glide to the next "
                "shelf, and rest until your stamina will carry the next leg.")
    T = Op
    c.add("B01", "Complete the relay lesson",
          [T("", "talk", "keeper", dict(text=lesson_r)), T("", "relay", "relay_b01")],
          (keeper, "The workshop relay by the mill keeps tripping. Fix it yourself."))
    c.add("B02", "Repair the workshop's faulty relay", [T("", "relay", "relay_b02")],
          (keeper, "The sentinel warden trains at the ring arenas west of the village."))
    c.add("B03", "Complete the sentinel lesson",
          [T("", "talk", "warden", dict(text=lesson_g)), T("", "sentinel", "b03_sentinel")],
          (warden, "The second arena's sentinel reforms its shield once. Clear it."))
    c.add("B04", "Clear the second training arena", [T("", "sentinel", "b04_sentinel")],
          (warden, "The wind guide waits by the sea stacks east of the village."))
    c.add("B05", "Complete the wind guide's crossing",
          [T("", "talk", "guide", dict(text=lesson_t)), T("", "reach", "overlook_1", dict(radius=8.0))],
          (guide, "Reach the second overlook on your own."))
    c.add("B06", "Reach the second training overlook", [T("", "reach", "overlook_2", dict(radius=8.0))],
          (guide, "The expedition quartermaster waits at the depot: the granary with the crane, south-east of "
                  "Mirror Lake."))
    c.add("B07", "Survey the expedition cache", [T("", "survey", "depot_plaque")],
          (quartermaster, "Register done. The reserve compartment stays sealed until reserve power returns from the "
                          "upper valley. The causeway mechanic at the foot of the Scarp Cut needs a repair kit - "
                          "raiders in the eastern hills took it - and the ferry coupling that sank in Mirror Lake."))
    c.add("B08", "Recover the causeway repair kit", [T("", "collect", "camp_chest", dict(item="kit"))])
    c.add("B09", "Retrieve the submerged ferry coupling", [T("", "collect", "lake_chest", dict(item="coupling"))])
    c.add("B10", "Reopen the expedition causeway",
          [T("", "exchange", "mechanic", dict(requires=dict(kit=1, coupling=1), grants=CAUSEWAY_OPEN)),
           T("", "reach", "cut_top", dict(radius=10.0))],
          ("mechanic", "The valley signal station up the cut has been dark."))
    c.add("B11", "Restore the valley signal station", [T("", "relay", "relay_b11")],
          ("relay_b11:core", "Station log: the observation post on the eastern stacks reports the route north."))
    c.add("B12", "Reach the cliff observation post", [T("", "reach", "post", dict(radius=8.0))],
          (observer, "A field sentinel in the western hollow holds the warden's field seal."))
    c.add("B13", "Recover the warden's field seal", [T("", "sentinel", "b13_sentinel", dict(grants="field_seal"))],
          (observer, "That seal opens the reserve relay's service panel near the eastern rim."))
    c.add("B14", "Stabilize the reserve-power relay",
          [T("", "relay", "relay_b14", dict(unlock="field_seal", grants=RESERVE_POWER,
                                            lock_text="Reserve-power relay: the service panel is locked."))],
          ("relay_b14:core", "Reserve power restored; the depot compartment is open."))
    c.add("B15", "Retrieve the expedition service tools", [T("", "collect", "reserve", dict(item="tools"))],
          ("reserve", "Service tools, for the sky-lift control house at the foot of the northern wall."))
    c.add("B16", "Open the sky-lift route",
          [T("", "exchange", "controls", dict(requires=dict(tools=1), keep=True)),
           T("", "relay", "relay_b16", dict(grants=SKYLIFT_POWER)),
           T("", "reach", "highland_rim", dict(radius=12.0))],
          ("statue_skylift", "Sky-lift landing: the crater gate to the north is shut."))
    c.add("B17", "Restore the guarded crater gate",
          [T("", "sentinel", "b17_sentinel"), T("", "relay", "relay_b17", dict(grants=CRATER_GATE_OPEN))],
          ("relay_b17:core", "Gatekeeper's plaque: the high shrine above the crater floor keeps the court's seal."))
    c.add("B18", "Recover the high shrine's access seal",
          [T("", "sentinel", "b18_sentinel", dict(grants="shrine_seal"))],
          ("b18_sentinel:core", "The shrine seal opens the shrine-court relay housing."))
    c.add("B19", "Commission a double-ring relay",
          [T("", "relay", "relay_b19", dict(unlock="shrine_seal",
                                            lock_text="Shrine-court relay: the housing is sealed with the shrine's mark."))],
          ("relay_b19:core", "Shrine plaque: the Crown above the crater is dark. Its landing signal waits beyond "
                             "the wind stacks."))
    c.add("B20", "Establish the remote landing route", [T("", "signal", "landing_signal", dict(grants=CROWN_SIGNAL))],
          ("landing_signal", "The Crown's relays wake. The east relay is a double ring."))
    c.add("B21", "Repair another double-ring relay", [T("", "relay", "relay_b21")],
          ("relay_b21:core", "Crown plaque: the old station on the Crown's south side is dark."))
    c.add("B22", "Restore an older single-ring station", [T("", "relay", "relay_b22")],
          ("relay_b22:core", "Station plaque: the summit-approach housing is open. The heart lies in the reliquary "
                             "on the spire."))
    c.add("B23", "Recover the summit beacon heart",
          [T("", "relay", "relay_b23", dict(grants=SUMMIT_POWER)), T("", "sentinel", "b23_sentinel"),
           T("", "collect", "reliquary", dict(item="heart"))],
          ("reliquary", "The beacon heart, for the outpost beyond the Crown's north edge."))
    c.add("B24", "Commission a distant outpost from saved memory",
          [T("", "sentinel", "b24_sentinel", dict(wake=dict(heart=1), grants=OUTPOST_CLEAR)),
           T("", "exchange", "socket", dict(requires=dict(heart=1))), T("", "relay", "relay_b24")])
    assert len(c.milestones) == 24
    return c


class ExpeditionWorld(World):
    """World whose interactions, skills and enemies go through the course."""

    def interact(self, char):
        self.expedition.interact(char)
        return []

    def _hint(self, char):
        course = getattr(self, "expedition", None)
        if course is None:
            return None
        hit = course._nearest(char, roles=("npc", "plaque", "chest", "receiver", "exchange", "signal", "statue"))
        return f"Interact: {self.get(hit[1]).label}" if hit else None

    def _update_enemy(self, e, char, dt):
        if e.id in self.expedition.roles and self.expedition.roles[e.id][0] == "sentinel":
            return []                         # sentinels hold their arena; damage comes from their phases
        return super()._update_enemy(e, char, dt)

    def update(self, char, dt, char_cfg):
        events = super().update(char, dt, char_cfg)
        if char.hp > 0.0:
            self.expedition.tick(char, dt)
        events.extend(self.expedition.events)
        self.expedition.events.clear()
        # Sentinel pulses and slams land after the base defeat check: handle them here,
        # so a downed member swaps out (or the party respawns) instead of ending the episode.
        events += self._handle_defeat(char, char_cfg)
        return events

    def attack(self, char):
        events = super().attack(char)
        return events + self._handle_defeat(char, self.expedition.env.controller.cfg)

    def _respawn(self, char, char_cfg):
        course = self.expedition
        course.deaths += 1
        course.launch = None
        found = [self.get(k) for k, v in course.statues.items() if v]
        if found:
            target = min(found, key=lambda s: float(np.hypot(*(s.pos - char.pos)[[0, 2]])))
            char.pos = self.ground_pos(float(target.pos[0]) + 1.5, float(target.pos[2]))
        char.vel[:] = 0.0
        char.hp = self.party.active.hp if self.party else char_cfg.max_hp
        char.stamina = char_cfg.stamina_max
        char.state = MoveState.GROUNDED
        char.grounded = True
        char.air_time = char._peak_fall_speed = char._regen_timer = 0.0
        char._jump_was_down = False


class ExpeditionCombat(FullScaleCombat):
    def cast(self, char, controller, *, burst=False):
        party = self.party
        ability = (party.ready_burst() if burst else party.ready_skill()) if party is not None else None
        events = super().cast(char, controller, burst=burst)
        if ability is not None and any(e.type in ("skill", "burst") for e in events):
            reach = float(ability.range or ability.radius or 2.6)
            self.world.expedition.apply_element(char, ability.element, reach)
            events += self.world._handle_defeat(char, controller.cfg)
        return events

    def _sentinel(self, enemy):
        roles = self.world.expedition.roles
        return enemy.id in roles and roles[enemy.id][0] == "sentinel"

    def _damage(self, enemy, damage, element):
        if self._sentinel(enemy):
            return []                         # reaction splash and other side paths never touch a sentinel
        return super()._damage(enemy, damage, element)

    def _hit(self, enemy, char, damage, element, charge=1.0):
        if self._sentinel(enemy):
            return self.world.expedition.sentinel_hit(enemy, damage, element, char)
        return super()._hit(enemy, char, damage, element, charge)


def _expedition_renderer(settings):
    """The soft observation renderer, plus the island's site pieces."""
    from .render.site_assets import add_site_pieces
    from .render.soft_renderer import SoftRenderer

    class ExpeditionRenderer(SoftRenderer):
        def _build_meshes(self, world, char, eye, fwd, *, batch=None, full_detail=False):
            batch = super()._build_meshes(world, char, eye, fwd, batch=batch, full_detail=full_detail)
            if self.settings.draw_props:
                pieces = getattr(getattr(world.architecture, "base", world.architecture), "pieces", ())
                add_site_pieces(batch, pieces, eye, self.settings.prop_draw_distance)
            return batch
    return ExpeditionRenderer(settings)


class ExpeditionQuests(QuestEngine):
    """Advances through every stage whose milestone completed this step.

    The base engine advances one stage per step. Early credit can complete two
    milestones in one step (opening the strongbox after diving for the coupling
    completes B08 and B09), and the engine would then stay a stage behind for
    good, so the finish never arrives.
    """

    def _update_quest(self, quest, events, world_state):
        reward = 0.0
        progress = self.progress[quest.id]
        while True:
            before = progress.stage
            reward += super()._update_quest(quest, events, world_state)
            if progress.stage == before or progress.stage >= len(quest.stages):
                return reward


class TeyvatExpeditionEnv(OpenWorldEnv):
    """Native-action full-scale Task B."""

    def __init__(self, seed=7, *, render_enabled=False, width=384, height=216, max_steps=90_000):
        self.render_enabled = render_enabled
        cfg = EnvConfig(seed=seed, text_obs=True, backend="soft", action_repeat=6, max_episode_steps=max_steps,
                        world=WorldConfig(prop_density=0, day_length=100000, start_time_of_day=.43),
                        camera=CameraConfig(distance=9, far=520),
                        render=RenderSettings(width=width, height=height, march_steps=48, refine_steps=4,
                                              shadow_steps=0, prop_draw_distance=90, max_view_distance=520,
                                              ground_cover=False, clouds=False))
        rc = cfg.reward
        rc.shaping = rc.discovery = rc.objective = rc.time_penalty = rc.damage_penalty = rc.death_penalty = 0.0
        rc.finish_bonus = 1.0
        super().__init__(cfg)
        from gymnasium import spaces
        import string
        self.observation_space["local_text"] = spaces.Text(max_length=1024, charset=string.printable)
        self._renderer.close()
        self._renderer = _expedition_renderer(self.cfg.render)

    def _build(self, seed):
        c = self.cfg
        self.fullscale = fw = build_world(seed)
        self.world_digest = fw.digest()
        self.rng = np.random.default_rng(seed)
        self.party = Party(c.party)
        wcfg = replace(c.world, seed=seed, terrain=fw.terrain.cfg)
        self.world = ExpeditionWorld(wcfg, terrain=fw.terrain, props=fw.props, party=self.party)
        self.world.architecture = fw.architecture
        self.world.combat = ExpeditionCombat(self.world, self.party)
        self.course = ExpeditionCourse(self, seed)
        self.world.expedition = self.course       # not `course`: the Cycles exporter reserves it for compact courses
        build_course(self.course)
        stages = [Stage([Objective(m.id, ObjKind.ACTIVATE, m.title, targets=(m.id,))]) for m in self.course.milestones]
        self.graph = QuestGraph([Quest("fullscale_b", "Task B", GOAL, stages, reward=0)])
        self.quests = ExpeditionQuests(self.graph)
        char_cfg = replace(c.character)
        char_cfg.max_hp = self.party.max_hp
        char_cfg.stamina_regen *= self.party.stamina_regen_multiplier
        for cost in ("sprint_cost", "climb_cost", "glide_cost", "swim_cost", "jump_cost"):
            setattr(char_cfg, cost, getattr(char_cfg, cost) * self.party.stamina_multiplier)
        self.controller = FullScaleController(fw.terrain, fw.props, char_cfg)
        self.camera_rig = OrbitCamera(fw.terrain, fw.props, c.camera)
        self.controller.architecture = self.camera_rig.architecture = fw.architecture
        sx, sz = fw.layout.anchors["spawn"]
        self.char = self.controller.spawn(sx, sz)
        self.cam = self.camera_rig.reset(self.char)
        self.cam.pitch = .34
        self.steps = 0
        self._prev_dist = None
        self._used_entities = set()
        self._episode_reward = 0.0
        self._last_events = []
        self._last_frame = None

    def _observe(self):
        if self.render_enabled:
            obs = super()._observe()
        else:
            obs = {"rgb": np.zeros((self.cfg.render.height, self.cfg.render.width, 3), np.uint8),
                   "proprio": self._proprio(), "task": self._task_vector(), "party": self.party.as_vector()}
        obs["task"] = np.zeros_like(obs["task"])
        obs["instruction"] = GOAL
        obs["local_text"] = self.course.local_text(self.char)
        return obs

    def _info(self, events=None):
        result = super()._info(events)
        result["instruction"] = GOAL             # the base env puts the current milestone title here
        c = self.course
        result["course"] = dict(milestone_index=c.index, op_index=c.op_index, facts=sorted(c.facts),
                                pack=dict(c.pack), deaths=c.deaths)
        return result

    def state_digest(self):
        state = {"suite": SUITE_VERSION, "physics": PHYSICS_VERSION, "course": COURSE_VERSION,
                 "world": self.world_digest, "position": self.char.pos.tolist(), "velocity": self.char.vel.tolist(),
                 "stamina": float(self.char.stamina), "hp": float(self.char.hp), "move_state": int(self.char.state),
                 "yaw": float(self.char.yaw), "regen": float(self.char._regen_timer),
                 "time": self.world.elapsed, "party": self.party.summary(), "course_state": self.course.state(),
                 "entities": {e.id: [e.pos.tolist(), float(e.hp), bool(e.alive), bool(e.used), float(e.attack_cd),
                                     e.elemental.aura.element.name, float(e.elemental.aura.charge)]
                              for e in self.world.entities.values()},
                 "camera": [float(self.cam.yaw), float(self.cam.pitch)]}
        return hashlib.sha256(json.dumps(state, sort_keys=True, default=float, allow_nan=False).encode()).hexdigest()
