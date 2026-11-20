"""Privileged route-preview controller for the full-scale Teyvat world.

It reads private state (anchors, lift paths, open facts) to plan, and acts only
through native actions: camera-relative movement, camera look, sprint, jump
(glider), and interact (to board a lift). It never writes simulator state.
Phase 3 replaces the checkpoint visits with the real milestone procedures.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .character import MoveState
from .fullscale_nav import NavGrid

LOWLAND, PLATEAU, RIDGE, SUMMIT = range(4)
TIER_NAMES = ("lowland", "plateau", "ridge", "summit")
WAYPOINT_REACHED = 2.0
STUCK_WINDOW = 15           # decisions (3 s)
STUCK_DISTANCE = 1.0
SPRINT_ON, SPRINT_OFF = 55.0, 15.0


def tier_of(y: float) -> int:
    return LOWLAND if y < 35.0 else PLATEAU if y < 85.0 else RIDGE if y < 135.0 else SUMMIT


class RoutePreviewPolicy:
    def __init__(self, env):
        self.env = env
        self.explanation = ""
        self._nav = {}
        self._path, self._goal, self._facts, self._k = None, None, None, 0
        self._history = deque(maxlen=STUCK_WINDOW)
        self._sprinting = True
        self._gliding_leg = False
        self._replans = 0

    @staticmethod
    def neutral():
        return {"move": np.zeros(2, np.float32), "look": np.zeros(2, np.float32),
                "buttons": np.zeros(7, np.int8), "swap": 0, "pointer": np.zeros(2, np.float32), "click": 0}

    def nav(self) -> NavGrid:
        facts = frozenset(self.env.fullscale.state.facts)
        if facts not in self._nav:
            self._nav[facts] = NavGrid(self.env.fullscale, facts)
        return self._nav[facts]

    # ------------------------------------------------------------ actions
    def act(self):
        env, world = self.env, self.env.world
        action = self.neutral()
        milestone, checkpoint = world.current_checkpoint()
        if checkpoint is None:
            self.explanation = "Route complete"
            return action
        lift = world.riding()
        if lift is not None:
            self.explanation = f"Ride the {lift.id.replace('_', ' ')} {'up' if lift.riding > 0 else 'down'}"
            self._aim_camera(action, lift.path[2] - lift.path[1] if lift.riding > 0 else lift.path[1] - lift.path[2])
            self._history.clear()
            return action
        target = world.place(checkpoint.name)
        here = tier_of(float(env.char.pos[1]))
        goal = tier_of(float(target[1]))
        if self._gliding_leg or (checkpoint.name == "ruin" and here == PLATEAU):
            return self._glide(action, target, checkpoint.label)
        board = self._next_lift(here, goal)
        if board is not None:
            lift, end = board
            end_xz = end[[0, 2]]
            if np.linalg.norm(env.char.pos[[0, 2]] - end_xz) < 2.5 and env.char.state == MoveState.GROUNDED:
                action["buttons"][2] = 1
                self.explanation = f"Board the {lift.id.replace('_', ' ')}"
                return action
            self._walk(action, end, f"Walk to the {lift.id.replace('_', ' ')}")
            return action
        self._walk(action, target, f"Walk to {checkpoint.label}")
        return action

    def _next_lift(self, here, goal):
        """The first lift between two tiers, as (lift, boarding end), or None to walk.

        Plateau and ridge connect on foot through the ravine, so a boarding end
        on another tier is reached by walking there first.
        """
        lifts = {l.id: l for l in self.env.world.lifts}
        step = 1 if goal > here else -1
        for tier in range(here, goal, step):
            pair = {tier, tier + step}
            lift = lifts["ferry_lift"] if pair == {LOWLAND, PLATEAU} else \
                lifts["summit_lift"] if pair == {RIDGE, SUMMIT} else None
            if lift is not None:
                return lift, (lift.point(0.0) if step > 0 else lift.point(lift.length))
        return None

    # ------------------------------------------------------------- moving
    def _walk(self, action, target, explanation):
        env = self.env
        pos = env.char.pos
        facts = frozenset(env.fullscale.state.facts)
        goal_xz = np.asarray(target, float)[[0, 2]]
        stuck = self._stuck()
        if (self._path is None or self._facts != facts or np.linalg.norm(goal_xz - self._goal) > 1.0 or stuck):
            self._path = self.nav().plan(pos[[0, 2]], goal_xz)
            self._goal, self._facts, self._k = goal_xz, facts, 1
            if stuck:
                self._replans += 1
                self._history.clear()
        path = self._path
        if path is None:
            waypoint = goal_xz
        else:
            while self._k < len(path) - 1 and np.linalg.norm(pos[[0, 2]] - path[self._k]) < WAYPOINT_REACHED:
                self._k += 1
            waypoint = path[min(self._k, len(path) - 1)]
        self._move_toward(action, waypoint, stop=0.6)
        if stuck:
            action["buttons"][0] = 1            # hop over a small snag
        swimming = env.char.state == MoveState.SWIMMING
        self.explanation = ("Swim: " if swimming else "") + explanation

    def _glide(self, action, target, label):
        env = self.env
        st = env.char
        self._gliding_leg = True
        if st.state in (MoveState.GROUNDED, MoveState.SWIMMING) and tier_of(float(st.pos[1])) == LOWLAND:
            self._gliding_leg = False
            self._path = None
            self._walk(action, target, f"Walk to {label}")
            return action
        full = env.controller.cfg.stamina_max
        if st.state == MoveState.GROUNDED and st.stamina < 0.98 * full:
            # A full bar covers the whole descent; a partial one can run out mid-air.
            self._aim_camera(action, np.array([target[0] - st.pos[0], 0.0, target[2] - st.pos[2]]))
            self._history.clear()
            self.explanation = "Rest at the edge until stamina is full"
            return action
        self._move_toward(action, np.asarray(target, float)[[0, 2]], stop=0.0, sprint=False)
        if st.state == MoveState.GLIDING or (st.state == MoveState.AIRBORNE and st.vel[1] < 0.0):
            action["buttons"][0] = 1
            self.explanation = "Glide down to the lowland ruin"
        else:
            self.explanation = "Walk off the west edge"
        return action

    def _move_toward(self, action, target_xz, stop=1.0, sprint=True):
        env = self.env
        d = np.asarray(target_xz, float) - env.char.pos[[0, 2]]
        dist = float(np.linalg.norm(d))
        self._history.append(env.char.pos[[0, 2]].copy())
        if dist < stop:
            return True
        self._aim_camera(action, np.array([d[0], 0.0, d[1]]))
        direction = d / max(dist, 1e-6)
        cy, sy = np.cos(env.cam.yaw), np.sin(env.cam.yaw)
        action["move"][:] = (direction[0] * cy + direction[1] * sy, direction[0] * sy - direction[1] * cy)
        np.clip(action["move"], -1.0, 1.0, out=action["move"])
        stamina = float(env.char.stamina)
        self._sprinting = stamina > SPRINT_ON if not self._sprinting else stamina > SPRINT_OFF
        if sprint and self._sprinting and env.char.state == MoveState.GROUNDED:
            action["buttons"][1] = 1
        return False

    def _aim_camera(self, action, direction):
        env = self.env
        desired = float(np.arctan2(direction[0], -direction[2]))
        delta = (desired - env.cam.yaw + np.pi) % (2 * np.pi) - np.pi
        action["look"][0] = np.clip(delta / (env.cfg.camera.yaw_rate * .2), -1, 1)
        action["look"][1] = np.clip((.34 - env.cam.pitch) / .34, -1, 1)

    def _stuck(self):
        h = self._history
        if len(h) < STUCK_WINDOW:
            return False
        return float(np.linalg.norm(h[-1] - h[0])) < STUCK_DISTANCE
