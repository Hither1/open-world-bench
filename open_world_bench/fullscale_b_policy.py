"""Privileged reference controller for full-scale Task B.

It reads private state (anchors, receiver roles, the sampled rule
assignments, sentinel phases, vent beats) to decide, and acts only through
native actions: camera-relative movement, look, sprint, jump/glide, interact,
attack, skill and party swap. It never writes simulator state.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .character import MoveState
from .fullscale_b_world import STACKS
from .fullscale_mechanics import COLUMN_RADIUS, PULSE_RADIUS, SLAM_RADIUS
from .fullscale_nav import NavGrid

WAYPOINT_REACHED = 2.0
STUCK_WINDOW = 15
STUCK_DISTANCE = 1.0
SPRINT_ON, SPRINT_OFF = 55.0, 15.0
STANCE = 3.0                # metres from a sentinel core
SAFE_DISTANCE = 10.5        # outside the slam
PAD_TOLERANCE = 0.7
BUTTON = dict(jump=0, sprint=1, interact=2, attack=3, climb=4, skill=5, burst=6)


class ExpeditionReference:
    def __init__(self, env):
        self.env = env
        self.explanation = ""
        self.mode = "walk"
        self._nav = {}
        self._path, self._goal, self._facts, self._k = None, None, None, 0
        self._history = deque(maxlen=STUCK_WINDOW)
        self._sprinting = True
        self.replans = 0
        self.route = None           # (route name, leg index, phase)
        self.descending = False
        self.route_restarts = 0

    @staticmethod
    def neutral():
        return {"move": np.zeros(2, np.float32), "look": np.zeros(2, np.float32),
                "buttons": np.zeros(7, np.int8), "swap": 0, "pointer": np.zeros(2, np.float32), "click": 0}

    @property
    def course(self):
        return self.env.course

    def nav(self) -> NavGrid:
        facts = frozenset(self.env.fullscale.state.facts)
        if facts not in self._nav:
            self._nav[facts] = NavGrid(self.env.fullscale, facts)
        return self._nav[facts]

    def anchor(self, name):
        return self.env.fullscale.layout.point(name)

    def pos_xz(self):
        return self.env.char.pos[[0, 2]].astype(float)

    # ================================================================ decide
    def act(self):
        env, c = self.env, self.course
        a = self.neutral()
        if c.launch is not None:
            self.mode, self.explanation = "vent", "Ride the vent column"
            a["buttons"][BUTTON["jump"]] = 1
            return a
        if self._keep_gliding(a):
            return a
        if self.route is not None:
            self._fly(a)
            return a
        op = c.op
        if op is None:
            self.mode, self.explanation = "done", "All 24 milestones complete"
            return a
        if self._heal(a):
            return a
        work = self._work_point(op)
        if work is not None and not self._go(a, work, op):
            return a
        self._do(a, op)
        return a

    def _keep_gliding(self, a):
        """Never fall from height: outside a route's glide leg, a falling character opens the glider."""
        env = self.env
        st = env.char
        if st.state not in (MoveState.AIRBORNE, MoveState.GLIDING) or st.vel[1] >= 0.0:
            return False
        if self.route is not None and self.route[2] in ("launch", "glide"):
            return False
        clearance = float(st.pos[1]) - float(env.fullscale.terrain.height_at(st.pos[0], st.pos[2]))
        if clearance < 4.0:
            return False
        op = self.course.op
        target = self._work_point(op) if op is not None else None
        if target is not None:
            self._move_toward(a, target, stop=0.0, sprint=False)
        a["buttons"][BUTTON["jump"]] = 1
        self.mode, self.explanation = "glide", "Glide down"
        return True

    def _heal(self, a):
        """A downed member (or a badly hurt party) is restored at the nearest reachable statue."""
        env, c = self.env, self.course
        party = env.party
        hurt = any(not m.alive for m in party.members) or env.char.hp < 0.3 * party.max_hp
        if not hurt or env.char.state not in (MoveState.GROUNDED, MoveState.SWIMMING):
            return False
        nav = self.nav()
        here = self.pos_xz()
        statues = [env.world.get(k) for k, v in c.statues.items() if v]
        statues = [s for s in statues if nav.connected(here, s.pos[[0, 2]])]
        if not statues:
            return False
        s = min(statues, key=lambda e: float(np.linalg.norm(e.pos[[0, 2]] - here)))
        if np.linalg.norm(s.pos[[0, 2]] - here) > 2.2:
            self._walk(a, s.pos[[0, 2]], "Walk to the statue to restore the party", stop=1.8)
        else:
            a["buttons"][BUTTON["interact"]] = 1
            self.mode, self.explanation = "mechanism", "Restore the party at the statue"
        return True

    # ---------------------------------------------------------- work points
    def _entity(self, eid):
        return self.env.world.get(eid)

    def _work_point(self, op):
        """Where the character must stand for ``op`` (x, z), or None to act in place."""
        c = self.course
        if op.kind in ("talk", "survey", "exchange", "signal"):
            return self._entity(op.target).pos[[0, 2]]
        if op.kind == "collect":
            return self._entity(op.target).pos[[0, 2]]
        if op.kind == "reach":
            return self.anchor(op.target)
        if op.kind == "relay":
            unit = c.relays[op.target]
            if c.relay_locked(unit):
                return self._entity(unit.receivers["core"]).pos[[0, 2]]
            return self._entity(unit.receivers[unit.needed_role()]).pos[[0, 2]]
        if op.kind == "sentinel":
            return self._entity(c.sentinels[op.target].entity).pos[[0, 2]]
        return None

    # --------------------------------------------------------------- travel
    def _go(self, a, target, op):
        """Move toward ``target``; True once close enough to act."""
        env, c = self.env, self.course
        nav = self.nav()
        here = self.pos_xz()
        target = np.asarray(target, float)
        stop = self._stop_distance(op)
        if np.linalg.norm(target - here) <= stop and self._same_level(target):
            return True
        if nav.connected(here, target):
            self.descending = False
            self._walk(a, target, f"Walk to {self._label(op)}", stop)
            return False
        route = self._route_to(target)
        if route is not None:
            self.route = (route, 0, "approach")
            self._fly(a)
            return False
        # On a shelf or mesa with no route onward: glide down toward the target.
        self._descend(a, target, op)
        return False

    def _same_level(self, target):
        ground = float(self.env.fullscale.terrain.height_at(*target))
        return abs(float(self.env.char.pos[1]) - ground) < 4.0 or self.env.char.state == MoveState.SWIMMING

    def _stop_distance(self, op):
        if op.kind == "sentinel":
            return STANCE + 0.6
        if op.kind == "relay":
            return 1.6
        if op.kind == "reach":
            return 3.0
        if op.kind == "collect" and op.target == "lake_chest":
            return 1.2
        return 2.0

    def _route_to(self, target):
        """A vent route whose first pad is reachable on foot and whose end connects to ``target``."""
        nav = self.nav()
        here = self.pos_xz()
        for name in ("b05", "b06", "b12", "b16", "b18", "b20", "b23", "b24"):
            legs = self.env.fullscale.route(name)
            if any(v.requires and v.requires not in self.course.facts for v in legs):
                continue
            first, last = legs[0], legs[-1]
            if nav.connected(here, first.pad) and nav.connected(last.target_xz, target) \
                    and not nav.connected(here, target):
                return name
        return None

    def _label(self, op):
        c = self.course
        if op.kind == "relay":
            return c.relays[op.target].label
        if op.kind == "sentinel":
            return c.sentinels[op.target].label
        if op.kind == "reach":
            return op.target.replace("_", " ")
        return self._entity(op.target).label

    def _walk(self, a, target, explanation, stop=0.6):
        env = self.env
        pos = env.char.pos
        facts = frozenset(env.fullscale.state.facts)
        goal = np.asarray(target, float)
        stuck = self._stuck()
        if self._path is None or self._facts != facts or np.linalg.norm(goal - self._goal) > 1.0 or stuck:
            self._path = self.nav().plan(pos[[0, 2]], goal)
            self._goal, self._facts, self._k = goal, facts, 1
            if stuck:
                self.replans += 1
                self._history.clear()
        path = self._path
        if path is None:
            waypoint = goal
        else:
            while self._k < len(path) - 1 and np.linalg.norm(pos[[0, 2]] - path[self._k]) < WAYPOINT_REACHED:
                self._k += 1
            waypoint = path[min(self._k, len(path) - 1)]
        precise = np.linalg.norm(goal - pos[[0, 2]]) < 4.0
        self._move_toward(a, waypoint, stop=stop if precise else 0.6, slow=precise)
        if stuck:
            a["buttons"][BUTTON["jump"]] = 1
        swimming = env.char.state == MoveState.SWIMMING
        self.mode = "swim" if swimming else "walk"
        self.explanation = ("Swim: " if swimming else "") + explanation

    def _descend(self, a, target, op):
        env = self.env
        st = env.char
        full = env.controller.cfg.stamina_max
        self.mode = "glide"
        if st.state == MoveState.GROUNDED and st.stamina < 0.97 * full:
            self._aim_camera(a, np.array([target[0] - st.pos[0], 0.0, target[1] - st.pos[2]]))
            self._history.clear()
            self.explanation = "Rest before gliding down"
            return
        self.descending = True
        d = np.asarray(target, float) - self.pos_xz()
        direction = d / max(float(np.linalg.norm(d)), 1e-6)
        self._move_toward(a, target, stop=0.0, sprint=False)
        terrain = env.fullscale.terrain
        if st.state in (MoveState.AIRBORNE, MoveState.GLIDING):
            a["buttons"][BUTTON["jump"]] = 1                    # hold: the glider opens while falling
        else:
            # Jump off the rim rather than sliding down the face: press when the ground ahead drops away.
            ahead = [float(terrain.height_at(*(self.pos_xz() + direction * r))) for r in (1.5, 3.0, 4.5)]
            if min(ahead) < float(st.pos[1]) - 2.0:
                a["buttons"][BUTTON["jump"]] = int(not st._jump_was_down)
        self.explanation = f"Glide down toward {self._label(op)}"

    # ---------------------------------------------------------- vent routes
    def _fly(self, a):
        env, c = self.env, self.course
        st = env.char
        name, k, phase = self.route
        legs = env.fullscale.route(name)
        vent = legs[k]
        state = c.vents[vent.id]
        pad = np.asarray(vent.pad, float)
        self.mode = "vent"
        if phase == "approach":
            if not self.nav().connected(self.pos_xz(), pad):
                # Fell off the route: start it again from wherever is reachable.
                self.route = None
                self.route_restarts += 1
                return
            d = float(np.linalg.norm(pad - self.pos_xz()))
            if d > PAD_TOLERANCE or st.state != MoveState.GROUNDED:
                if d > 4.0:
                    self._walk(a, pad, f"Walk to vent {k + 1} of the {name.upper()} route", stop=PAD_TOLERANCE)
                else:
                    self._move_toward(a, pad, stop=PAD_TOLERANCE * 0.6, sprint=False, slow=True)
                    self.explanation = f"Step onto vent {k + 1}"
                self.mode = "vent"
                return
            full = env.controller.cfg.stamina_max
            if st.stamina < 0.98 * full:
                self.explanation = f"Rest on the pad until stamina is full (leg {k + 1}/{len(legs)})"
                self._aim_camera(a, np.array([vent.target_xz[0] - pad[0], 0.0, vent.target_xz[1] - pad[1]]))
                return
            if not state.charged:
                self._cast(a, c.vent_element, f"Charge the vent with {c.vent_element.name.lower()}")
                return
            now = env.world.elapsed
            window_ok = all(state.safe(now + dt, c.safe_beat) for dt in (0.0, 1 / 30, 0.1, 0.2, 0.3))
            if window_ok and not st._jump_was_down:
                a["buttons"][BUTTON["jump"]] = 1
                self.route = (name, k, "launch")
                self.explanation = f"Jump on the {c.safe_beat} beat"
            else:
                self.explanation = f"Wait for the {c.safe_beat} beat"
            return
        if phase == "launch":
            if c.launch is not None or st.state in (MoveState.AIRBORNE, MoveState.GLIDING):
                self.route = (name, k, "glide")
                a["buttons"][BUTTON["jump"]] = 1
                self.explanation = "Ride the column"
                return
            self.route = (name, k, "approach")       # sputtered or missed: back to the pad
            return
        if phase == "glide":
            target = np.asarray(vent.target_xz, float)
            d = float(np.linalg.norm(target - self.pos_xz()))
            if st.state in (MoveState.GROUNDED, MoveState.SWIMMING) and c.launch is None:
                on_shelf = d < 8.0 and float(st.pos[1]) > vent.target_top - 2.0
                if on_shelf and k + 1 < len(legs):
                    self.route = (name, k + 1, "approach")
                elif on_shelf:
                    self.route = None
                else:
                    self.route = None
                    self.route_restarts += 1
                return
            a["buttons"][BUTTON["jump"]] = 1
            if d > 2.5:
                self._move_toward(a, target, stop=0.0, sprint=False)
            else:
                self._aim_camera(a, np.array([target[0] - st.pos[0], 0.0, target[1] - st.pos[2]]))
            self.explanation = f"Glide to {vent.target.replace('_', ' ')} (leg {k + 1}/{len(legs)})"

    # ------------------------------------------------------------ mechanisms
    def _do(self, a, op):
        c = self.course
        self.mode = "mechanism"
        if op.kind in ("talk", "survey", "exchange", "signal"):
            a["buttons"][BUTTON["interact"]] = int(not self._interact_was_down())
            self.explanation = f"Interact: {self._label(op)}"
        elif op.kind == "collect":
            self._collect(a, op)
        elif op.kind == "reach":
            self.explanation = f"Stand at {self._label(op)}"
        elif op.kind == "relay":
            self._relay(a, op)
        elif op.kind == "sentinel":
            self._sentinel(a, op)

    def _interact_was_down(self):
        return False

    def _collect(self, a, op):
        env = self.env
        st = env.char
        if op.target == "lake_chest":
            a["buttons"][BUTTON["jump"]] = 1          # hold to dive and stay down
            chest = self._entity(op.target)
            self.mode = "swim"
            if float(st.pos[1]) < float(chest.pos[1]) + 2.4:
                a["buttons"][BUTTON["interact"]] = 1
                self.explanation = "Open the sunken chest on the lake bed"
            else:
                self.explanation = "Dive to the marked chest"
            return
        a["buttons"][BUTTON["interact"]] = 1
        self.explanation = f"Open the {self._label(op).lower()}"

    def _relay(self, a, op):
        c = self.course
        unit = c.relays[op.target]
        if c.relay_locked(unit):
            a["buttons"][BUTTON["interact"]] = 1
            self.explanation = f"Unlock the {unit.label} housing"
            return
        role = unit.needed_role()
        if role == "core":
            a["buttons"][BUTTON["interact"]] = 1
            self.explanation = f"Confirm the {unit.label} at its core"
            return
        element = c.relay_elements[role]
        reason = "residual charge" if (role == "drain" and not unit.overloaded) else role
        self._cast(a, element, f"{unit.label}: {element.name.lower()} on the {role}"
                               + (" (double ring: drain first)" if reason == "residual charge" else ""))

    def _sentinel(self, a, op):
        env, c = self.env, self.course
        st = env.char
        unit = c.sentinels[op.target]
        core = self._entity(unit.entity)
        d = float(np.linalg.norm(core.pos[[0, 2]] - self.pos_xz()))
        breaker, interrupter = c.guard_elements
        party = env.party
        self.mode = "fight"
        if unit.hold:
            self._move_toward(a, core.pos[[0, 2]], stop=STANCE, sprint=False, slow=True)
            self.explanation = f"{unit.label}: approach ({unit.posture()})"
            return
        if unit.phase == "shield":
            if self._ready(breaker):
                if d > STANCE + 0.8:
                    self._move_toward(a, core.pos[[0, 2]], stop=STANCE, sprint=False, slow=True)
                    self.explanation = f"{unit.label}: close in to break the shield"
                    return
                self._cast(a, breaker, f"{unit.label}: break the shield with {breaker.name.lower()}")
            else:
                self._keep_out(a, core, PULSE_RADIUS + 1.5, f"{unit.label}: wait outside the pulse for "
                                                           f"{breaker.name.lower()}")
            return
        if unit.phase == "exposed":
            self._preswap(a, interrupter)
            self.explanation = f"{unit.label}: core open; ready {interrupter.name.lower()}"
            return
        if unit.phase == "windup":
            if self._ready(interrupter, swap_ok=True):
                if d > STANCE + 0.8:
                    self._move_toward(a, core.pos[[0, 2]], stop=STANCE, sprint=False, slow=True)
                    return
                self._cast(a, interrupter, f"{unit.label}: interrupt the wind-up with {interrupter.name.lower()}")
            else:
                self._keep_out(a, core, SLAM_RADIUS + 1.5, f"{unit.label}: back out of the slam")
            return
        # Vulnerable: strike the cracked core from a stance just outside it.
        if st.state != MoveState.GROUNDED:
            self.explanation = f"{unit.label}: land, then strike"
            return
        if d < 1.3:
            self._keep_out(a, core, 1.9, f"{unit.label}: step back to strike the core")
            return
        if d > 2.5:
            self._move_toward(a, core.pos[[0, 2]], stop=1.9, sprint=False, slow=True)
            self.explanation = f"{unit.label}: close in on the cracked core"
            return
        if self._face(a, core.pos):
            a["buttons"][BUTTON["attack"]] = 1
            self.explanation = f"{unit.label}: strike the cracked core"

    def _keep_out(self, a, core, radius, text):
        here = self.pos_xz()
        away = here - core.pos[[0, 2]]
        dist = float(np.linalg.norm(away))
        if dist < radius:
            target = core.pos[[0, 2]] + away / max(dist, 1e-6) * (radius + 0.5)
            self._move_toward(a, target, stop=0.3, sprint=False)
        self.explanation = text

    def _face(self, a, target):
        env = self.env
        d = np.asarray(target) - env.char.pos
        dist = float(np.linalg.norm(d[[0, 2]]))
        desired = float(np.arctan2(d[0], -d[2]))
        error = (desired - env.char.yaw + np.pi) % (2 * np.pi) - np.pi
        if dist > .4 and abs(error) > .25:
            cy, sy = np.cos(env.cam.yaw), np.sin(env.cam.yaw)
            dx, dz = d[[0, 2]] / dist
            a["move"][:] = np.array([dx * cy + dz * sy, dx * sy - dz * cy]) * .4
            return False
        return True

    # ------------------------------------------------------------- casting
    def _member(self, element):
        return next(i for i, m in enumerate(self.env.party.members) if m.spec.element == element)

    def _ready(self, element, swap_ok=True):
        party = self.env.party
        m = party.members[self._member(element)]
        return m.skill_cd <= 1e-5 and m.alive

    def _preswap(self, a, element):
        party = self.env.party
        idx = self._member(element)
        if idx != party.active_index and party.swap_timer <= 0:
            a["swap"] = idx + 1

    def _cast(self, a, element, text):
        party = self.env.party
        idx = self._member(element)
        member = party.members[idx]
        st = self.env.char
        self.explanation = text
        if idx != party.active_index:
            if party.swap_timer > 0:
                self.explanation = text + " (swap cooling down)"
                return
            a["swap"] = idx + 1
            return
        if member.skill_cd > 1e-5:
            self.explanation = text + f" (waiting {member.skill_cd:.1f} s for the skill)"
            return
        if st.state != MoveState.GROUNDED:
            return
        a["buttons"][BUTTON["skill"]] = 1

    # ------------------------------------------------------------- movement
    def _move_toward(self, a, target_xz, stop=1.0, sprint=True, slow=False):
        env = self.env
        d = np.asarray(target_xz, float) - env.char.pos[[0, 2]]
        dist = float(np.linalg.norm(d))
        self._history.append(env.char.pos[[0, 2]].copy())
        if dist < stop:
            return True
        self._aim_camera(a, np.array([d[0], 0.0, d[1]]))
        direction = d / max(dist, 1e-6)
        cy, sy = np.cos(env.cam.yaw), np.sin(env.cam.yaw)
        a["move"][:] = (direction[0] * cy + direction[1] * sy, direction[0] * sy - direction[1] * cy)
        if slow:
            a["move"] *= float(np.clip((dist - stop) / 2.0 + 0.35, 0.35, 1.0))
        np.clip(a["move"], -1.0, 1.0, out=a["move"])
        stamina = float(env.char.stamina)
        self._sprinting = stamina > SPRINT_ON if not self._sprinting else stamina > SPRINT_OFF
        if sprint and not slow and self._sprinting and env.char.state == MoveState.GROUNDED:
            a["buttons"][BUTTON["sprint"]] = 1
        return False

    def _aim_camera(self, a, direction):
        env = self.env
        desired = float(np.arctan2(direction[0], -direction[2]))
        delta = (desired - env.cam.yaw + np.pi) % (2 * np.pi) - np.pi
        a["look"][0] = np.clip(delta / (env.cfg.camera.yaw_rate * .2), -1, 1)
        a["look"][1] = np.clip((.34 - env.cam.pitch) / .34, -1, 1)

    def _stuck(self):
        h = self._history
        if len(h) < STUCK_WINDOW:
            return False
        return float(np.linalg.norm(h[-1] - h[0])) < STUCK_DISTANCE
